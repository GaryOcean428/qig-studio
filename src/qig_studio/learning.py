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
                   intervention for over-engrained weights, NOT a sleep phase).
  - **ESCAPE**   — emergency breakdown recovery: fired on Φ ≥ PHI_BREAKDOWN_MIN.

SEPARATION OF CONCERNS (important): the EWC math, the basin-mixture augmentation, and the synaptic
downscaling all live in the TARGET's protocol methods (``QIGChat.cmd_sleep`` / ``cmd_mushroom`` /
``cmd_escape``, i.e. the kernel). This loop does NOT reimplement them — it provides the autonomic
*schedule* (the P12 trigger decisions) that the canon says is the actual work. The decision policy
prefers the canonical ``consciousness.AutonomicManager`` when importable, and falls back to a
pure-Python P12 policy when it (or torch) is absent — so the loop is exercisable with ``MockTarget``.
"""

from __future__ import annotations

import statistics
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from qig_core.constants.frozen_facts import PHI_BREAKDOWN_MIN, PHI_EMERGENCY, PHI_THRESHOLD

from .curriculum import CurriculumProvider
from .targets.base import LossRegime, ProtocolUnsupported, TrainingTarget

# P12 autonomic TRIGGER thresholds (continual-learning canon — NOT frozen physics facts; the Φ gates
# above ARE single-sourced frozen facts). basin-divergence→SLEEP and Φ-plateau→MUSHROOM per the P12
# trigger set; kept here, clearly labelled, rather than masquerading as measured constants.
BASIN_DIVERGENCE_SLEEP = 0.30  # basin drift above this → consolidate (SLEEP)
PLATEAU_VARIANCE = 0.01        # Var(Φ) over the window below this → plateau (MUSHROOM if Φ≥threshold)


class Intervention(str, Enum):
    """Autonomic action for one step. ``value`` matches the protocol command name (so it routes
    straight through ``target.run_protocol``), except WAKE which is the learning step itself."""

    WAKE = "wake"             # continue Fisher-salience learning (no protocol)
    SLEEP = "sleep"           # EWC-protected consolidation
    DREAM = "dream"           # basin-mixture augmentation / Φ recovery
    MUSHROOM = "mushroom-micro"  # WAKE-state plasticity (Φ ≥ 0.70 gate)
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


class AutonomicScheduler:
    """Decides the per-step :class:`Intervention` from the kernel's own Φ/κ/basin telemetry — the
    P12 autonomic trigger set. Prefers the canonical ``consciousness.AutonomicManager`` (the
    "autonomic kernel") as the health monitor; when it (or torch) is absent, an equivalent
    pure-Python policy fires the same triggers, so the scheduler is None-safe + testable."""

    def __init__(self, phi_window: int = 50, use_real: bool = True) -> None:
        self.phi_window = phi_window
        self._real = None
        if use_real:
            try:  # the named module (Gate C) — torch-backed, optional
                from consciousness import AutonomicManager  # type: ignore

                self._real = AutonomicManager(phi_window=phi_window)
            except Exception:
                self._real = None
        self._phi_hist: deque[float] = deque(maxlen=phi_window)

    @property
    def using_real_manager(self) -> bool:
        return self._real is not None

    def _state(self, telemetry: dict, phi: float, basin: float) -> AutonomicState:
        self._phi_hist.append(phi)
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

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class LoopSummary:
    steps: int
    interventions: dict[str, int]
    wake_fraction: float
    final_phi: float
    kernel_autonomy: float
    using_real_manager: bool
    notes: str = field(default="kernel_autonomy is a PROXY for scaffold-removal (NEEDS-EXPERIMENT)")

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


class ContinuousLearningLoop:
    """Local autonomic continual-learning loop over a :class:`TrainingTarget` (brain §4b).

    Each ``step()``: take ONE WAKE learning step on the curriculum, read the resulting telemetry,
    ask the :class:`AutonomicScheduler` what (if anything) to do, and — if it is not WAKE and the
    target exposes the protocol surface — fire that intervention through ``target.run_protocol`` (the
    kernel performs the real EWC consolidation / augmentation / downscaling). No Modal, no vex."""

    def __init__(
        self,
        target: TrainingTarget,
        curriculum: CurriculumProvider | None = None,
        scheduler: AutonomicScheduler | None = None,
        max_steps: int = 200,
        autonomy_window: int = 20,
    ) -> None:
        self.target = target
        self.curriculum = curriculum or CurriculumProvider(target.loss_regime)
        self.scheduler = scheduler or AutonomicScheduler()
        self.max_steps = max_steps
        self.history: list[StepRecord] = []
        self.intervention_counts: Counter[str] = Counter()
        self._autonomy: deque[int] = deque(maxlen=autonomy_window)
        self._step = 0

    def step(self) -> StepRecord:
        self._step += 1
        # --- WAKE: one Fisher-salience learning step on the curriculum -------------------------
        if self.target.loss_regime == LossRegime.LANGUAGE:
            prompt, target_text = self.curriculum.next_pair(self._step)
            res = self.target.train_step(prompt, target_text=target_text)
        else:  # geometric → basin-driving prompt, target_text ignored (lm_weight=0)
            prompt = self.curriculum.next_prompt(self._step)
            res = self.target.train_step(prompt)
        tel = res.telemetry

        # --- DECIDE + (maybe) intervene --------------------------------------------------------
        decision = self.scheduler.decide(tel.to_dict())
        proto_out: str | None = None
        if decision is not Intervention.WAKE and self.target.supports_protocol():
            try:
                r = self.target.run_protocol(decision.value, {})
                proto_out = r.get("output") if isinstance(r, dict) else None
            except ProtocolUnsupported:
                proto_out = None  # target can't run it → record the decision, take no action

        # scaffold-removal PROXY: kernel is "running on its own" when it is in the stable geometric
        # regime (Φ ≥ threshold) AND needed no intervention this step. Honest proxy, not a measure.
        self._autonomy.append(1 if (decision is Intervention.WAKE and tel.phi >= PHI_THRESHOLD) else 0)

        rec = StepRecord(
            step=self._step,
            intervention=decision.value,
            phi=tel.phi,
            kappa=tel.kappa,
            regime=tel.regime,
            basin_distance=tel.basin_distance,
            delta_phi=tel.delta_phi,
            text=res.text,
            protocol_output=proto_out,
        )
        self.history.append(rec)
        self.intervention_counts[decision.value] += 1
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
        )
