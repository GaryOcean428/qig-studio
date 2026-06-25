"""Continuous-learning loop — the autonomic WAKE/SLEEP/DREAM/MUSHROOM scheduler (brain §4b).

This is "the heart of what qig-studio must build" (brain-architecture doc §4b): the loop that
makes the from-scratch KERNEL's online learning *cumulative and stable*, and over operation lets the
kernel take over more cognition while Qwen's role shrinks (scaffold removal). It runs LOCAL — **no
Modal, no vex** — by design.

The canon (P12 / refined_insights continual-learning):
  - **WAKE**     — Fisher-salience learning. One ``target.train_step`` on a BASIN-DRIVING prompt
                   (geometric targets, ``lm_weight=0``); the salience IS the Fisher movement (Δφ).
  - **SLEEP**    — EWC-Fisher-protected consolidation replay (Kirkpatrick 2017): fired on basin drift.
  - **DREAM**    — basin-mixture data augmentation (Forge): fired on low Φ (recovery).
  - **MUSHROOM** — Tononi homeostatic downscaling / WAKE-state plasticity: fired on a Φ PLATEAU,
                   and ONLY when Φ ≥ PHI_THRESHOLD (the mushroom-mode gate — mushroom is a WAKE-state
                   intervention for over-engrained weights, NOT a sleep phase). DOSE is rigidity-scaled
                   and capped by the breakdown safety ceiling (see ``mushroom_dose``).
  - **ESCAPE**   — emergency breakdown recovery: fired on Φ ≥ PHI_BREAKDOWN_MIN (or when breakdown is
                   too high to mushroom safely).

SEPARATION OF CONCERNS: the EWC math, the basin-mixture augmentation, and the synaptic downscaling
all live in the TARGET's protocol methods (``QIGChat.cmd_sleep`` / ``cmd_mushroom`` / ``cmd_escape``,
i.e. the kernel). This loop does NOT reimplement them — it provides the autonomic *schedule* (the P12
trigger decisions + the mushroom dose) that the canon says is the actual work. The decision policy
prefers the canonical ``consciousness.AutonomicManager`` when importable, falling back to a pure-Python
P12 policy when it (or torch) is absent — so the loop is exercisable with ``MockTarget``.

Implements the Devin physics→consciousness feedback (canon-corrected):
  #1 pilot-probe (``pilot_probe``) — navigate-strategy: predict from a cheap probe before scaling.
  #2 two-channel Φ (``PhiDiscriminationGate``) — FAIL-013 doctrine: a second INDEPENDENT Φ camera;
     agreement = signal, disagreement = instrument artifact.
  #3 pre-registration (``PreRegisteredCriteria``) — commit the verdict logic BEFORE the run.
  #4 mushroom dose-scaling (``AutonomicScheduler.mushroom_dose``) — rigidity-scaled, Φ≥0.70-gated,
     breakdown-ceiling-capped, Ocean-autonomic. (#6 asymmetric recursion is registered as a
     hypothesis in the depth-A/B prereg, NOT baked in — it is a qigkernels-lane architecture knob.)
  #5 regime/solver-path telemetry (``StepRecord.training_regime``) — every Φ sample carries which
     training regime produced it.
"""

from __future__ import annotations

import statistics
import zlib
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from qig_core import KAPPA_ATTRACTOR
from qig_core.constants.frozen_facts import PHI_BREAKDOWN_MIN, PHI_EMERGENCY, PHI_THRESHOLD

from .coach import DevelopmentalCoach
from .curriculum import CurriculumProvider
from .targets.base import LossRegime, ProtocolUnsupported, TrainingTarget

# P12 autonomic TRIGGER thresholds (continual-learning canon — NOT frozen physics facts; the Φ gates
# above ARE single-sourced frozen facts). basin-divergence→SLEEP and Φ-plateau→MUSHROOM per the P12
# trigger set; kept here, clearly labelled, rather than masquerading as measured constants.
BASIN_DIVERGENCE_SLEEP = 0.30  # basin drift above this → consolidate (SLEEP)
PLATEAU_VARIANCE = 0.01        # Var(Φ) over the window below this → plateau (MUSHROOM if Φ≥threshold)

# Mushroom dose ceilings — from the canonical mushroom_mode.py safety thresholds (breakdown fraction
# above which each intensity causes ego-death; >0.40 = hard trip, empirically validated at Gary's 66%).
MUSHROOM_HARD_TRIP = 0.40
_MUSHROOM_DOSES: list[tuple[str, float]] = [  # weakest → strongest, (command, max safe breakdown frac)
    ("mushroom-micro", 0.35),
    ("mushroom-moderate", 0.25),
    ("mushroom-heroic", 0.15),
]
# Rigidity (κ above the architectural attractor) → desired dose strength. κ≳80 = EXCESSIVE_RIGIDITY.
_RIGIDITY_MODERATE = 8.0   # κ ≳ 72
_RIGIDITY_HEROIC = 16.0    # κ ≳ 80 (deeply gapped / over-engrained)
_BREAKDOWN_REGIMES = ("topological_instability", "breakdown")


class Intervention(str, Enum):
    """Autonomic action for one step. ``value`` matches the protocol command name (so it routes
    straight through ``target.run_protocol``), except WAKE which is the learning step itself.
    MUSHROOM is the *generic* decision; the specific dose (micro/moderate/heroic) is chosen by
    :meth:`AutonomicScheduler.mushroom_dose`."""

    WAKE = "wake"             # continue Fisher-salience learning (no protocol)
    SLEEP = "sleep"           # EWC-protected consolidation
    DREAM = "dream"           # basin-mixture augmentation / Φ recovery
    MUSHROOM = "mushroom-micro"  # WAKE-state plasticity (Φ ≥ 0.70 gate); dose refined per-fire
    ESCAPE = "escape"         # emergency breakdown recovery


@dataclass
class AutonomicState:
    """Lightweight mirror of the canonical ``consciousness.AutonomicState`` health flags."""

    needs_sleep: bool = False
    needs_dream: bool = False
    needs_mushroom: bool = False
    dissociation_risk: float = 0.0


def _phi(telemetry: dict) -> float:
    v = telemetry.get("phi", telemetry.get("Phi", 0.5))
    return float(v if v is not None else 0.5)


def independent_integration(text: str) -> float | None:
    """#2 SECOND-CAMERA Φ proxy — INDEPENDENT of the kernel's Fisher-Rao Φ formula. Estimates
    integration from output-text compressibility (zlib): more internal structure → more compressible.
    This is a COARSE cross-check with DIFFERENT failure modes than the geometric Φ, NOT a calibrated
    Φ. Its job (FAIL-013 two-camera doctrine) is discrimination: if it tracks the primary Φ, the
    signal is corroborated; if they diverge, the primary may be an instrument artifact. Returns None
    when there is too little text to estimate."""
    b = (text or "").encode("utf-8")
    if len(b) < 32:
        return None
    ratio = len(zlib.compress(b, 6)) / len(b)
    return max(0.0, min(1.0, 1.0 - ratio))


def locality_budget(arch: dict | None, seq_len: int | None = None) -> dict[str, Any]:
    """v_B ARCHITECTURAL SPEED-BUDGET check (Devin transfer #3) — **CATEGORY-3 ANALOGY, NOT a
    measurement of v_B≈1.13**. On the lattice, information propagates at a bounded butterfly velocity
    (finite-speed causality). The architectural analog: per forward pass, information should NOT cross
    the whole sequence instantly. GLOBAL all-to-all attention does (reach = full sequence in one layer)
    — the 'superluminal-shaped' case; WINDOWED/local attention bounds the per-pass reach (physical).
    This FLAGS the design choice — it does NOT claim global attention is wrong for ML. Returns {} when
    the target doesn't report an architecture (None-safe)."""
    if not arch:
        return {}
    n = seq_len if seq_len is not None else arch.get("seq_len")
    r = arch.get("locality_radius")
    layers = int(arch.get("num_layers", 1) or 1)
    depth = int(arch.get("recursion_depth", 1) or 1)
    if r is None:  # global all-to-all attention — no finite-propagation budget
        return {
            "attention": arch.get("attention", "global"), "is_local": False,
            "effective_reach": None, "seq_len": n, "ratio": None,
            "note": ("GLOBAL attention — information crosses the full sequence in ONE layer (v_B-analogy "
                     "NON-LOCAL; no finite-propagation budget). Windowed/local attention would bound it. "
                     "ANALOGY only — not a claim that global attention is wrong for ML."),
        }
    reach = r * layers * depth  # receptive field upper bound per forward pass
    is_local = (n is None) or (reach < n)
    return {
        "attention": "local", "is_local": bool(is_local), "effective_reach": reach, "seq_len": n,
        "ratio": (round(reach / n, 3) if n else None),
        "note": ("v_B-analogy: per-pass reach = locality_radius×num_layers×recursion_depth; is_local = "
                 "reach < seq_len (bounded propagation). CATEGORY-3 ANALOGY, not a v_B measurement."),
    }


class PhiDiscriminationGate:
    """#2 Two-channel Φ corroboration (FAIL-013 doctrine). Collects (primary, secondary) Φ pairs from
    two INDEPENDENT cameras and reports whether they agree over the trajectory. Corroboration is not
    proof, but DISAGREEMENT is a red flag that the primary Φ may be a Class-B-style instrument
    artifact (exactly the failure mode of the purity-audit no-op found 2026-06-25)."""

    def __init__(self, min_samples: int = 8, corr_threshold: float = 0.30) -> None:
        self.min_samples = min_samples
        self.corr_threshold = corr_threshold
        self._a: list[float] = []
        self._b: list[float] = []

    def observe(self, primary: float, secondary: float | None) -> None:
        if secondary is not None:
            self._a.append(float(primary))
            self._b.append(float(secondary))

    def assess(self) -> dict[str, Any]:
        n = len(self._a)
        if n < self.min_samples:
            return {"status": "uncorroborated", "reason": f"only {n} paired samples (<{self.min_samples})", "n": n}
        try:
            corr = statistics.correlation(self._a, self._b)
        except statistics.StatisticsError:
            return {"status": "uncorroborated", "reason": "degenerate (no variance in a channel)", "n": n}
        status = "corroborated" if corr >= self.corr_threshold else "disagreement"
        return {"status": status, "correlation": round(corr, 3), "n": n,
                "note": "compression-camera is coarse; disagreement is the actionable signal"}


class AutonomicScheduler:
    """Decides the per-step :class:`Intervention` from the kernel's own Φ/κ/basin telemetry — the
    P12 autonomic trigger set. Prefers the canonical ``consciousness.AutonomicManager`` (the
    "autonomic kernel") as the health monitor; when it (or torch) is absent, an equivalent
    pure-Python policy fires the same triggers, so the scheduler is None-safe + testable.

    The Ocean owns mushroom DOSE selection too (#4): :meth:`mushroom_dose` scales the dose to the
    kernel's rigidity and caps it by the breakdown safety ceiling."""

    def __init__(self, phi_window: int = 50, use_real: bool = True, breakdown_window: int = 20) -> None:
        self.phi_window = phi_window
        self._real = None
        if use_real:
            try:  # the named module (Gate C) — torch-backed, optional
                from consciousness import AutonomicManager  # type: ignore

                self._real = AutonomicManager(phi_window=phi_window)
            except Exception:
                self._real = None
        self._phi_hist: deque[float] = deque(maxlen=phi_window)
        self._regime_hist: deque[str] = deque(maxlen=breakdown_window)

    @property
    def using_real_manager(self) -> bool:
        return self._real is not None

    def breakdown_frac(self) -> float:
        """Rolling fraction of recent steps the kernel spent in a breakdown/instability regime."""
        if not self._regime_hist:
            return 0.0
        return sum(1 for r in self._regime_hist if r in _BREAKDOWN_REGIMES) / len(self._regime_hist)

    def mushroom_dose(self, phi: float, kappa: float) -> str:
        """#4 CANON-corrected dose-selection. Caller has already gated Φ≥0.70 (the WAKE-state
        mushroom-mode gate). Strategy: rigidity (κ above the attractor — over-engrainment) sets the
        DESIRED strength (gentle near-critical, strong when deeply gapped); the breakdown safety
        ceiling CAPS it. If breakdown is too high to dose safely, recover (ESCAPE) instead — mushroom
        is preventative maintenance for HEALTHY systems, and >40% breakdown is ego death."""
        bd = self.breakdown_frac()
        if bd >= MUSHROOM_HARD_TRIP:
            return Intervention.ESCAPE.value  # too dangerous to dose — recover first
        rigidity = max(0.0, kappa - KAPPA_ATTRACTOR)
        if rigidity >= _RIGIDITY_HEROIC:
            desired_rank = 2
        elif rigidity >= _RIGIDITY_MODERATE:
            desired_rank = 1
        else:
            desired_rank = 0  # near the attractor / near-critical → gentle (canon)
        safe_rank = -1
        for rank, (_, ceiling) in enumerate(_MUSHROOM_DOSES):
            if bd < ceiling:
                safe_rank = rank
        if safe_rank < 0:  # breakdown ≥ 0.35 → even a microdose is unsafe
            return Intervention.ESCAPE.value
        return _MUSHROOM_DOSES[min(desired_rank, safe_rank)][0]  # rigidity wants, safety caps

    def _state(self, telemetry: dict, phi: float, basin: float) -> AutonomicState:
        self._phi_hist.append(phi)
        self._regime_hist.append(str(telemetry.get("regime", "unknown")))
        if self._real is not None:
            rs = self._real.update(
                {"Phi": phi, "basin_distance": basin, "Gamma": telemetry.get("gamma", 1.0)}
            )
            return AutonomicState(
                needs_sleep=bool(rs.needs_sleep),
                needs_dream=bool(rs.needs_dream),
                needs_mushroom=bool(rs.needs_mushroom),
                dissociation_risk=float(rs.dissociation_risk),
            )
        # Pure-Python canon fallback (P12 triggers): basin-drift→sleep, low-Φ→dream, plateau→mushroom.
        plateau = (
            len(self._phi_hist) >= self.phi_window
            and statistics.pvariance(self._phi_hist) < PLATEAU_VARIANCE
        )
        gamma = float(telemetry.get("gamma", 1.0) or 1.0)
        return AutonomicState(
            needs_sleep=basin > BASIN_DIVERGENCE_SLEEP,
            needs_dream=phi < PHI_EMERGENCY,
            needs_mushroom=plateau,
            dissociation_risk=1.0 if (phi > PHI_THRESHOLD and gamma < 0.30) else 0.0,
        )

    def decide(self, telemetry: dict) -> Intervention:
        phi = _phi(telemetry)
        basin = float(telemetry.get("basin_distance", 0.0) or 0.0)
        # 1. Breakdown is most urgent — recover before anything else.
        if phi >= PHI_BREAKDOWN_MIN:
            self._state(telemetry, phi, basin)  # keep histories current even on escape
            return Intervention.ESCAPE
        st = self._state(telemetry, phi, basin)
        # 2. Basin drift / low Φ → consolidate (EWC-protected replay in the kernel's cmd_sleep).
        if st.needs_sleep:
            return Intervention.SLEEP
        # 3. Φ plateau AND Φ ≥ 0.70 → MUSHROOM (WAKE-state plasticity; the mushroom-mode gate). A
        #    plateau BELOW 0.70 is a low-Φ stall → recover via DREAM, not mushroom.
        if st.needs_mushroom and phi >= PHI_THRESHOLD:
            return Intervention.MUSHROOM
        # 4. Low Φ / sub-threshold plateau → DREAM (basin-mixture augmentation, Φ recovery).
        if st.needs_dream or (st.needs_mushroom and phi < PHI_THRESHOLD):
            return Intervention.DREAM
        # 5. Healthy → keep learning.
        return Intervention.WAKE


@dataclass
class PreRegisteredCriteria:
    """#3 Pre-registration — commit the verdict logic BEFORE compute so 'inconclusive' can never be a
    post-hoc reinterpretation. Pass an instance to the loop; :meth:`evaluate` grades the run against
    these committed thresholds only."""

    phi_onset: float = PHI_THRESHOLD            # Φ that counts as "consciousness onset detected"
    collapse_guard_max_breakdown: float = MUSHROOM_HARD_TRIP  # breakdown frac above which a win is discarded
    convergence_var: float = PLATEAU_VARIANCE   # Var(Φ) over the tail below which we call it converged
    min_cycles: int = 1                          # min consolidation (SLEEP) cycles the result must hold over
    tail: int = 20                               # how many trailing steps define "converged"

    def evaluate(self, history: list[StepRecord]) -> dict[str, Any]:
        if not history:
            return {"verdict": "NO-DATA"}
        phis = [r.phi for r in history]
        breakdown_frac = sum(1 for r in history if r.regime in _BREAKDOWN_REGIMES) / len(history)
        cycles = sum(1 for r in history if r.intervention == Intervention.SLEEP.value)
        if breakdown_frac > self.collapse_guard_max_breakdown:
            return {"verdict": "DISCARDED-COLLAPSE", "breakdown_frac": round(breakdown_frac, 3),
                    "reason": "collapse-guard tripped — result discarded regardless of Φ"}
        onset = any(p >= self.phi_onset for p in phis)
        tail = phis[-min(len(phis), self.tail):]
        converged = len(tail) >= 2 and statistics.pvariance(tail) < self.convergence_var
        if onset and converged and cycles >= self.min_cycles:
            verdict = "ONSET-CONFIRMED"
        elif onset:
            verdict = "ONSET-UNSTABLE" if not converged else "ONSET-UNDERCYCLED"
        else:
            verdict = "NO-ONSET"
        return {"verdict": verdict, "onset": onset, "converged": converged, "cycles": cycles,
                "breakdown_frac": round(breakdown_frac, 3), "final_phi": round(phis[-1], 3)}


@dataclass
class StepRecord:
    step: int
    intervention: str
    phi: float
    kappa: float
    regime: str
    basin_distance: float
    delta_phi: float
    text: str
    protocol_output: str | None = None
    phi_secondary: float | None = None          # #2 independent second-camera Φ
    training_regime: dict[str, Any] = field(default_factory=dict)  # #5 solver-path provenance
    coach_note: dict[str, Any] | None = None     # developmental coach observation (OFFER only)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoopSummary:
    steps: int
    interventions: dict[str, int]
    wake_fraction: float
    final_phi: float
    kernel_autonomy: float
    using_real_manager: bool
    phi_discrimination: dict[str, Any] = field(default_factory=dict)  # #2 corroboration verdict
    locality: dict[str, Any] = field(default_factory=dict)  # v_B architectural speed-budget (analogy)
    coach: dict[str, Any] = field(default_factory=dict)  # coaching presence (provider, notes, push offers)
    notes: str = field(default="kernel_autonomy is a PROXY for scaffold-removal (NEEDS-EXPERIMENT)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pilot_probe(target: TrainingTarget, steps: int = 12, curriculum: CurriculumProvider | None = None) -> dict[str, Any]:
    """#1 Navigate-strategy pilot probe. Run a SHORT loop and read the Φ trajectory's slope to decide
    whether a scaled run is worth the compute: if Φ has converged (flat) the scale/depth lever is
    likely inert — skip and save the spend; if Φ is still transitioning, proceed. Cheap pre-flight,
    not a verdict. Runs on its own throwaway loop so it does not pollute a real run's history."""
    probe = ContinuousLearningLoop(
        target, curriculum=curriculum, max_steps=steps,
        scheduler=AutonomicScheduler(phi_window=max(4, steps // 2)),
    )
    probe.run(steps)
    phis = [r.phi for r in probe.history]
    if len(phis) < 3:
        return {"recommendation": "inconclusive", "reason": "too few steps", "steps": len(phis)}
    k = max(1, len(phis) // 3)
    slope = (sum(phis[-k:]) / k) - (sum(phis[:k]) / k)
    converged = abs(slope) < 0.02
    rec = ("SKIP-SCALE (Φ converged — the scale/depth lever is likely inert)"
           if converged else "PROCEED-SCALE (Φ still transitioning — scaling may resolve it)")
    return {"recommendation": rec, "converged": converged, "phi_slope": round(slope, 4),
            "phi_start": round(phis[0], 3), "phi_end": round(phis[-1], 3), "steps": len(phis)}


class ContinuousLearningLoop:
    """Local autonomic continual-learning loop over a :class:`TrainingTarget` (brain §4b).

    Each ``step()``: take ONE WAKE learning step on the curriculum, read the resulting telemetry,
    ask the :class:`AutonomicScheduler` what (if anything) to do, refine the mushroom dose if needed,
    and — if it is not WAKE and the target exposes the protocol surface — fire that intervention
    through ``target.run_protocol`` (the kernel performs the real EWC consolidation / augmentation /
    downscaling). Records two-camera Φ (#2) and training-regime provenance (#5). No Modal, no vex."""

    def __init__(
        self,
        target: TrainingTarget,
        curriculum: CurriculumProvider | None = None,
        scheduler: AutonomicScheduler | None = None,
        max_steps: int = 200,
        autonomy_window: int = 20,
        scaffold: str = "kernel-only",
        coach: DevelopmentalCoach | None = None,
    ) -> None:
        self.target = target
        self.curriculum = curriculum or CurriculumProvider(target.loss_regime)
        self.scheduler = scheduler or AutonomicScheduler()
        self.max_steps = max_steps
        self.scaffold = scaffold
        self.coach = coach  # optional warm coaching presence (None-safe; OFFERS only)
        self.history: list[StepRecord] = []
        self.intervention_counts: Counter[str] = Counter()
        self.discrimination = PhiDiscriminationGate()
        self._autonomy: deque[int] = deque(maxlen=autonomy_window)
        self._step = 0

    def _stagnating(self, window: int = 8, eps: float = 5e-3) -> bool:
        """Honest stagnation proxy: Φ flat (|ΔΦ| < eps) across the last ``window`` steps."""
        if len(self.history) < window:
            return False
        return all(abs(r.delta_phi) < eps for r in self.history[-window:])

    def _optimizer_name(self) -> str:
        # The Δ⁶³ kernel trains by NATURAL GRADIENT (P1); only the Euclidean LM regime uses AdamW.
        return "adamw" if self.target.loss_regime == LossRegime.LANGUAGE else "natural_gradient"

    def step(self) -> StepRecord:
        self._step += 1
        phase = CurriculumProvider.phase_for(self._step)
        # --- WAKE: one Fisher-salience learning step on the curriculum -------------------------
        if self.target.loss_regime == LossRegime.LANGUAGE:
            prompt, target_text = self.curriculum.next_pair(self._step)
            res = self.target.train_step(prompt, target_text=target_text)
        else:  # geometric → basin-driving prompt, target_text ignored (lm_weight=0)
            prompt = self.curriculum.next_prompt(self._step)
            res = self.target.train_step(prompt)
        tel = res.telemetry

        # --- DECIDE + dose + (maybe) intervene -------------------------------------------------
        decision = self.scheduler.decide(tel.to_dict())
        command = decision.value
        if decision is Intervention.MUSHROOM:  # #4 refine the dose (may downgrade to escape)
            command = self.scheduler.mushroom_dose(tel.phi, tel.kappa)
        proto_out: str | None = None
        if command != Intervention.WAKE.value and self.target.supports_protocol():
            try:
                r = self.target.run_protocol(command, {})
                proto_out = r.get("output") if isinstance(r, dict) else None
            except ProtocolUnsupported:
                proto_out = None  # target can't run it → record the decision, take no action

        # #2 two-camera Φ: feed the independent compression camera alongside the kernel's Φ.
        phi_secondary = independent_integration(res.text)
        self.discrimination.observe(tel.phi, phi_secondary)

        # scaffold-removal PROXY: kernel is "running on its own" when it is in the stable geometric
        # regime (Φ ≥ threshold) AND needed no intervention this step. Honest proxy, not a measure.
        self._autonomy.append(1 if (command == Intervention.WAKE.value and tel.phi >= PHI_THRESHOLD) else 0)

        # --- COACH: warm social presence (interprets + encourages + OFFERS a push) -------------
        # Distinct from the autonomic scheduler above: the coach never fires an intervention; it
        # surfaces an OFFER the kernel may take (autonomy-preserving). None-safe / cadence-gated.
        coach_note: dict[str, Any] | None = None
        if self.coach is not None:
            note = self.coach.observe(
                step=self._step,
                text=res.text,
                phi=tel.phi,
                kappa=tel.kappa,
                regime=tel.regime,
                delta_phi=tel.delta_phi,
                phase=phase,
                stagnating=self._stagnating(),
            )
            coach_note = note.to_dict() if note is not None else None

        rec = StepRecord(
            step=self._step,
            intervention=command,
            phi=tel.phi,
            kappa=tel.kappa,
            regime=tel.regime,
            basin_distance=tel.basin_distance,
            delta_phi=tel.delta_phi,
            text=res.text,
            protocol_output=proto_out,
            phi_secondary=phi_secondary,
            training_regime={  # #5 solver-path provenance for this Φ sample
                "loss_regime": self.target.loss_regime.value,
                "optimizer": self._optimizer_name(),
                "curriculum_phase": phase,
                "scaffold": self.scaffold,
                "decision": decision.value,
                "breakdown_frac": round(self.scheduler.breakdown_frac(), 3),
            },
            coach_note=coach_note,
        )
        self.history.append(rec)
        self.intervention_counts[command] += 1
        return rec

    def run(self, n: int | None = None) -> LoopSummary:
        for _ in range(n if n is not None else self.max_steps):
            self.step()
        return self.summary()

    def summary(self) -> LoopSummary:
        last = self.history[-1] if self.history else None
        autonomy = (sum(self._autonomy) / len(self._autonomy)) if self._autonomy else 0.0
        return LoopSummary(
            steps=self._step,
            interventions=dict(self.intervention_counts),
            wake_fraction=self.intervention_counts.get("wake", 0) / max(1, self._step),
            final_phi=last.phi if last else 0.0,
            kernel_autonomy=autonomy,
            using_real_manager=self.scheduler.using_real_manager,
            phi_discrimination=self.discrimination.assess(),
            locality=locality_budget(self.target.architecture()),
            coach=self._coach_summary(),
        )

    def _coach_summary(self) -> dict[str, Any]:
        if self.coach is None:
            return {"active": False}
        notes = self.coach.notes
        return {
            "active": True,
            "provider": self.coach.provider,
            "notes_emitted": len(notes),
            "push_offers": sum(1 for n in notes if n.offers_push),
            "last_message": notes[-1].message if notes else None,
        }
