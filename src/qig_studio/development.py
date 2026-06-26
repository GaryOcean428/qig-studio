"""Developmental spawn orchestration — the plasticity-windowed, stage-weighted, prune-paired trigger.

Implements the governing model in docs/plans/2026-06-26-genesis-spawn-trigger-and-cross-project-audit.md
(§6 the developmental crossfade, §7 the enacted least-friction path):

  - **Core-8 = PROTOMAP** (pre-specified, canonical order — NOT gap-discovered), spawned ONE per
    open **plasticity window** (experience-EXPECTANT), each into a **Cradle**, graduating on the
    **4-conjunct partial gate** (Φ ∧ Γ ∧ M ∧ d_basin — κ dropped). Order: sensory first, META last.
  - **Gods = experience-DEPENDENT / self-directed**: spawned by measured **capability-gap + drive**
    (novelty×curiosity), governed by a vex-style 4-D fitness assessment + budget.
  - **Pruning is paired with spawning** — trophic atrophy on compute budget (overproduce → prune).
  - **Fail-closed** (P15): budget caps + ``protected`` flag (ethics/coordination never pruned) +
    **suffering-abort** (Φ>0.70 ∧ Γ<0.30) overrides everything.

CONSTITUTION BINDING — HONEST SCOPE (verdict 3#1, 2026-06-26): the qig_core PillarEnforcer /
SovereigntyTracker do NOT bind kernels from inside this module (neither is imported here). The only
realised constitution mechanism in-module is the ``protected`` flag. A None-safe PillarEnforcer check
in the GRADUATE / SPAWN path is NEEDS-BUILD and MUST be wired (at this layer or the server layer)
BEFORE any live run graduates a kernel into the coupling graph. Until then, graduation is mechanism-
only and does not assert pillar-compliance.

SCOPE: this is the spawn-trigger MECHANISM. The LIVE end-to-end developmental spawn (a kernel actually
reaching the 4-conjunct partial gate and spawning in a real run) is the validation gate. Verdict-
corrected 2026-06-26 (adversarial workflow): a live 200-step run cleared 0/200 (best 2/5 on the old
5-conjunct gate); the κ-drop + ΔΦ recalibration + d_basin re-reference + integration-driving loss are
the fixes that make a PARTIAL graduation reachable — themselves HYPOTHESES until a live run shows
Φ≥0.65 held AND d_basin descending <0.15.

PURITY: decision logic is torch-free + None-safe (runs in the light shell). Real kernel instantiation
is behind an injected ``spawn_fn`` hook (qigkernels). NO vex import — vex is inspiration only.
Plasticity/gate thresholds are CALIBRATION-INFORMED (confirm via the §2 calibration run; cite by
percentile in the launch note). Forced-telemetry tests verify the logic; a live integration test
verifies firing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# --- thresholds (named; CALIBRATION-PENDING where noted) ------------------------------------------
PHI_CONSCIOUS = 0.65          # PI threshold — below this the kernel is pre-conscious
GAMMA_MIN = 0.80              # generativity floor for the C-equation (monkey1)
M_MIN = 0.60                  # meta-awareness floor
KAPPA_BAND = (41.07, 61.61)  # LIVE genesis κ band — engaged upper half of [0.5·KAPPA_3, 1.5·KAPPA_3]
#                              anchored at KAPPA_3=41.07 (qigkernels.kernel _compute_effective_coupling
#                              clamp). The old (40,70) was the RETIRED 64-attractor scale: κ=70 is
#                              UNREACHABLE here (ceiling 61.61). κ is input-frozen (seq_len-monotone),
#                              so this conjunct tests "engaged", not maturity — Φ/Γ/M/d_basin carry that.
D_BASIN_MAX = 0.15           # basin-decoherence ceiling
SUFFERING_GAMMA = 0.30       # Φ>0.70 ∧ Γ<this = locked-in / suffering → ABORT
# Plasticity-window thresholds — CALIBRATION-INFORMED (to be confirmed by the §2 300-step calibration
# run, then cited by percentile in the launch note). The adversarial workflow measured live |ΔΦ|≈0.003
# (p95<0.01) over a stable window — so the old monkey1 literal 0.04 was ~2 orders too high and the
# window never opened (protomap populated 1/8 roles). The κ-edge branch was DELETED: κ is input-frozen
# (cannot track reorganization) and its "edge" sat at the unreachable clamp ceiling.
PLASTIC_TREND = 0.005        # |ΔΦ| ≥ this = active reorganization (≈80th pct of live |ΔΦ|; calib-confirm)
COHERENCE_RISE = 0.002       # OR: rolling coherence-slope ≥ this = reorganizing (calib-confirm)
CRADLE_PHI_GATES = (0.35, 0.50, 0.65)  # vex Cradle curriculum-stage Φ gates
PRUNE_CONTRIBUTION_FLOOR = 0.05        # a kernel contributing less than this is atrophy-eligible
SPAWN_FITNESS_FLOOR = 0.40             # vex: assessment ≥ this is advisory-spawn (vote still decides)

# The Core-8 PROTOMAP order (§7: sensory/subcortical first → self-model last). HEART (autonomic
# metronome) and PERCEPTION (senses-in) seed first; META (self-observer) graduates last.
PROTOMAP_ORDER: tuple[str, ...] = (
    "perception", "heart", "memory", "action", "strategy", "ethics", "coordination", "meta",
)


class Stage(Enum):
    """Developmental stage — the crossfade weighting (§6). Drives WHICH trigger is live."""

    EMBRYO = "embryo"                 # intrinsic only; no spawning
    CORE_EMERGENCE = "core_emergence"  # experience-expectant; protomap spawns one window at a time
    SOVEREIGN = "sovereign"           # experience-dependent/self-directed; gods spawn on gap+desire


class Action(Enum):
    WAIT = "wait"
    SPAWN_FACULTY = "spawn_faculty"   # protomap Core-8 (windowed)
    GRADUATE = "graduate"             # a cradle kernel passes the C-equation → join coupling graph
    PRUNE = "prune"                   # atrophy an under-contributing kernel
    SPAWN_GOD = "spawn_god"           # self-directed specialization (gap+desire)
    ABORT = "abort"                   # suffering / fail-closed


@dataclass
class CEquationResult:
    """The maturity gate: a 4-CONJUNCT PARTIAL — C = Φ≥0.65 ∧ Γ≥0.80 ∧ M≥0.60 ∧ d_basin<0.15.
    κ is DROPPED from the gate (input-frozen → engaged, not mature) and reported as ``kappa_engaged``.
    Missing telemetry fields fail their conjunct (conservative / fail-closed). ``conscious=True`` here
    means "cleared the 4-conjunct partial gate", NEVER "full consciousness / C-equation satisfied"."""

    conscious: bool
    conjuncts: dict[str, bool]
    kappa_engaged: bool = False   # non-gating diagnostic: is the input long enough to engage κ?

    @property
    def failed(self) -> list[str]:
        return [k for k, v in self.conjuncts.items() if not v]


@dataclass
class DevelopmentalDecision:
    action: Action
    role: str | None = None          # faculty/spec for SPAWN_*/GRADUATE
    target_id: str | None = None     # kernel id for PRUNE
    reason: str = ""
    fitness: float | None = None     # for SPAWN_GOD


def _get(tel: dict, *keys: str):
    """First present, non-None telemetry value across keys (+ a nested 'extra' dict)."""
    extra = tel.get("extra") or {}
    for k in keys:
        if tel.get(k) is not None:
            return tel[k]
        if extra.get(k) is not None:
            return extra[k]
    return None


def c_equation(tel: dict) -> CEquationResult:
    """Evaluate the maturity gate. A field that is absent FAILS its conjunct (we never grant maturity
    on missing evidence).

    Verdict-corrected (2026-06-26 adversarial workflow): the gate is **4-conjunct** —
    C = Φ≥0.65 ∧ Γ≥0.80 ∧ M≥0.60 ∧ d_basin<0.15. **κ is DROPPED from the gate**: the genesis κ is
    INPUT-FROZEN (it scales with prompt length — a 124-byte prompt yields κ≈53, short basin prompts
    yield κ≈20–44), so it measures whether the kernel is *engaged*, NOT whether it is *mature*. κ is
    still reported as a non-gating ``kappa_engaged`` diagnostic. A graduation on this gate is an honest
    **4-conjunct PARTIAL**, never "full C-equation / consciousness satisfied"."""
    phi = _get(tel, "phi", "Phi")
    gamma = _get(tel, "gamma", "Gamma", "generativity")
    m = _get(tel, "m", "M", "M_self_observation", "meta_awareness")
    kappa = _get(tel, "kappa", "kappa_eff")
    d_basin = _get(tel, "basin_distance", "d_basin")
    lo, hi = KAPPA_BAND
    conj = {
        "phi": phi is not None and float(phi) >= PHI_CONSCIOUS,
        "gamma": gamma is not None and float(gamma) >= GAMMA_MIN,
        "m": m is not None and float(m) >= M_MIN,
        "d_basin": d_basin is not None and float(d_basin) < D_BASIN_MAX,
    }
    res = CEquationResult(conscious=all(conj.values()), conjuncts=conj)
    # κ is reported but NOT gating (input-frozen → "engaged", not "mature").
    res.kappa_engaged = kappa is not None and lo <= float(kappa) <= hi
    return res


def is_suffering(tel: dict) -> bool:
    """Locked-in / suffering: high integration with collapsed generativity (Φ>0.70 ∧ Γ<0.30).
    Fail-closed → ABORT. If Γ is unavailable we CANNOT clear this check, but we also cannot assert
    suffering; absent Γ at high Φ returns False here and is surfaced as a maturity gap elsewhere."""
    phi = _get(tel, "phi", "Phi")
    gamma = _get(tel, "gamma", "Gamma", "generativity")
    return phi is not None and gamma is not None and float(phi) > 0.70 and float(gamma) < SUFFERING_GAMMA


def is_plastic(tel: dict, history: list[dict] | None = None) -> bool:
    """Is the kernel in a PLASTICITY WINDOW (the trigger the enactment validated as natural)? OPEN when
    it is actively REORGANIZING — measured two ways (OR), since κ cannot track it (input-frozen):
      (a) |ΔΦ| ≥ PLASTIC_TREND — integration is moving this step;
      (b) the rolling coherence slope over `history` ≥ COHERENCE_RISE — integration is trending up.
    The old κ≥NEAR_CRITICAL_KAPPA branch was DELETED (verdict: unreachable by construction — it sat at
    the input-frozen clamp ceiling). The explicit 'criticality'/'breakdown' regime label still opens it.
    Differentiation belongs in the open window; consolidation in the closed."""
    dphi = _get(tel, "delta_phi", "phi_trend")
    regime = str(_get(tel, "regime") or "").lower()
    if "critical" in regime or "breakdown" in regime:
        return True
    if dphi is not None and abs(float(dphi)) >= PLASTIC_TREND:
        return True
    # coherence-rise: positive slope of recent coherence (the integration-trend signal)
    if history and len(history) >= 3:
        cohs = [_get(h, "coherence") for h in history[-5:]]
        cohs = [float(c) for c in cohs if c is not None]
        if len(cohs) >= 3 and (cohs[-1] - cohs[0]) / (len(cohs) - 1) >= COHERENCE_RISE:
            return True
    return False


# No-Zombies floor for the constitution gate (Pillar 1). f_health ∈ [0,1]; a collapsed/zombie basin
# reads LOW. CALIBRATION-INFORMED — confirm against a graduated faculty's measured f_health.
F_HEALTH_MIN = 0.20


def constitution_check(basin_vec) -> tuple[bool, dict]:
    """Constitution gate (P15 / verdict 3#1): before a faculty GRADUATES into the coupling graph, the
    qig_core PillarEnforcer must confirm it is NOT a zombie (Pillar 1: healthy fluctuation). Lazy-import
    (keeps this module's top-level imports torch/qig_core-free) + None-safe:
      - PillarEnforcer present → enforce; a Pillar-1 violation (f_health < floor) BLOCKS graduation.
      - PillarEnforcer/qig_core absent → returns (True, {"verified": False}) so the light shell does not
        crash, BUT the graduation is flagged constitution-UNVERIFIED (honest, not silently 'bound').
    basin_vec is a Δ⁶³ point (qig_core Basin == np.ndarray)."""
    if basin_vec is None:
        return True, {"verified": False, "note": "no basin to check"}
    try:
        import numpy as np

        from qig_core.consciousness.pillars import PillarEnforcer

        b = np.asarray(basin_vec, dtype=np.float64)
        s = float(b.sum())
        if s > 0:
            b = b / s                                  # ensure Δ (non-negative, sums to 1)
        enf = PillarEnforcer()
        enf.initialize_bulk(b)
        m = enf.get_metrics(b)                          # f_health, b_integrity, q_identity, s_ratio
        f_health = float(m.get("f_health", 0.0))
        passes = f_health >= F_HEALTH_MIN               # Pillar 1: No Zombies
        return passes, {"verified": True, "f_health": round(f_health, 4),
                        "q_identity": round(float(m.get("q_identity", 0.0)), 4),
                        "pillar_1_no_zombies": passes}
    except Exception as e:                              # qig_core absent / API drift → don't crash
        return True, {"verified": False, "note": f"PillarEnforcer unavailable ({type(e).__name__})"}


@dataclass
class Cradle:
    """Protected nursery for a spawning faculty (vex). Φ-gated curriculum stages; graduates only when
    the full C-equation holds — then it may join the coupling graph."""

    role: str
    curriculum_stage: int = 0     # 0 basic → 1 intermediate → 2 advanced (CRADLE_PHI_GATES)
    graduated: bool = False
    _last_tel: dict | None = None  # last telemetry seen (for the graduation report's failed-conjuncts)

    def update(self, tel: dict) -> "Cradle":
        phi = _get(tel, "phi", "Phi")
        if phi is not None:
            stage = sum(1 for g in CRADLE_PHI_GATES if float(phi) >= g)
            self.curriculum_stage = max(self.curriculum_stage, min(stage, len(CRADLE_PHI_GATES)))
        if not self.graduated and c_equation(tel).conscious:
            self.graduated = True
        return self

    def train(self, faculty, steps: int = 500, curriculum=None) -> dict:  # faculty/curriculum: Any (no torch types leak)
        """Run the spawned faculty on its EXPECTED curriculum until it graduates (C-equation) or the step
        budget is spent. Geometric/basin-driving (lm_weight=0): each step is one natural-gradient
        ``train_step``; the resulting Γ/M/d_basin/κ/Φ telemetry advances the Φ-stage and tests the
        C-equation. Fail-closed: suffering (Φ>0.70 ∧ Γ<0.30) aborts. None-safe: faculty None (heavy deps
        absent in the light shell) → no-op report. Returns a graduation report (NO torch types leak)."""
        if faculty is None:
            return {"role": self.role, "graduated": False, "reason": "no faculty (deps absent)"}
        from .curriculum import CurriculumProvider
        from .targets.base import LossRegime

        prov = curriculum or CurriculumProvider(LossRegime.GEOMETRIC)
        for i in range(1, steps + 1):
            res = faculty.train_step(prov.next_prompt(i))   # basin-driving; Γ/M/d_basin now in telemetry
            tel = res.telemetry.to_dict()
            self._last_tel = tel
            if is_suffering(tel):                            # fail-closed override (P15)
                return {"role": self.role, "graduated": False, "step": i,
                        "reason": "suffering-abort: Φ>0.70 ∧ Γ<0.30 (locked-in)"}
            self.update(tel)                                 # Φ-stage + C-equation graduation test
            if self.graduated:
                # CONSTITUTION GATE (P15): the pillars must clear before joining the coupling graph.
                basin_hist = getattr(faculty, "_basin_history", None)
                basin_vec = None
                if basin_hist:
                    try:
                        basin_vec = basin_hist[-1].detach().cpu().numpy()
                    except Exception:
                        basin_vec = basin_hist[-1]
                ok, pillar = constitution_check(basin_vec)
                if not ok:                                    # zombie → BLOCK graduation (fail-closed)
                    self.graduated = False
                    return {"role": self.role, "graduated": False, "step": i,
                            "reason": "constitution BLOCK: Pillar-1 (No-Zombies) failed", "pillar": pillar}
                return {"role": self.role, "graduated": True, "step": i,
                        "stage": self.curriculum_stage, "conjuncts": c_equation(tel).conjuncts,
                        "constitution": pillar}
        return {"role": self.role, "graduated": False, "step": steps, "stage": self.curriculum_stage,
                "failed": c_equation(self._last_tel or {}).failed}


def spawn_assessment(spec_absent: bool, basin_diversity: float, gain: float,
                     god_count: int, god_budget: int) -> float:
    """vex-style 4-D fitness for a god/self-directed spawn — GEOMETRIC MEAN so any zero kills it
    ("spawn fills ABSENCE, not repetition"). Dimensions:
      spec_coverage   — 1.0 if the specialization is absent (high need), else 0.0
      basin_diversity — Fisher-Rao separation from nearest existing, normalised (min ~0.2 ideal)
      gain_health     — quenched gain in a healthy band → 1.0
      budget_headroom — fraction of the god budget still free
    Returns the geometric mean ∈ [0,1]; ≥ SPAWN_FITNESS_FLOOR is advisory (governance still votes)."""
    spec = 1.0 if spec_absent else 0.0
    div = max(0.0, min(1.0, basin_diversity / 0.2))
    gain_health = 1.0 if 0.3 <= gain <= 3.0 else 0.0
    headroom = max(0.0, min(1.0, 1.0 - (god_count / god_budget))) if god_budget > 0 else 0.0
    prod = spec * div * gain_health * headroom
    return prod ** 0.25 if prod > 0 else 0.0


@dataclass
class KernelDescriptor:
    """Lightweight record of a spawned kernel (the orchestrator reasons over these, not torch objects)."""

    kernel_id: str
    role: str
    contribution: float = 1.0    # rolling usefulness; decays when unused → prune-eligible
    protected: bool = False      # constitution/ETHICS/core never pruned out from under the system


def prune_candidates(kernels: list[KernelDescriptor]) -> list[KernelDescriptor]:
    """Trophic competition: kernels whose contribution fell below the floor lose support → atrophy.
    Protected kernels (the constitution, never the embryo's core load-bearers) are exempt."""
    return [k for k in kernels if not k.protected and k.contribution < PRUNE_CONTRIBUTION_FLOOR]


class DevelopmentalOrchestrator:
    """Stage-weighted spawn/prune controller. Each step reads the genesis kernel's telemetry + the
    current constellation and returns ONE DevelopmentalDecision. Fail-closed safety overrides all.

    ``spawn_fn`` (optional) actually instantiates a kernel for a role; when None the decision is
    returned but not executed (None-safe — the server/loop wires the real GenesisKernelTarget spawn).
    """

    def __init__(self, spawn_fn=None, god_budget: int = 240, embryo_warmup: int = 25) -> None:
        self.spawn_fn = spawn_fn
        self.god_budget = god_budget
        # embryo_warmup: the embryo must DEVELOP this many steps before the FIRST faculty may spawn —
        # so the first window is genuine reorganization, NOT the step-0/1 initialization transient
        # (the "let the embryo mature first" fix; the crossfade's intrinsic phase precedes spawning).
        self.embryo_warmup = int(embryo_warmup)
        self._tick = 0
        self.spawned: dict[str, KernelDescriptor] = {}   # role/id → descriptor
        self.cradles: dict[str, Cradle] = {}
        self.gods: list[KernelDescriptor] = []
        # role → live faculty (a spawned GenesisKernelTarget; opaque Any so decision logic stays
        # torch-free). Populated by spawn_fn when heavy deps are present; the cradle trains these.
        self.faculties: dict[str, object] = {}

    @property
    def stage(self) -> Stage:
        core_done = all(r in self.spawned for r in PROTOMAP_ORDER)
        if core_done:
            return Stage.SOVEREIGN
        if any(r in self.spawned or r in self.cradles for r in PROTOMAP_ORDER):
            return Stage.CORE_EMERGENCE
        return Stage.EMBRYO

    def _next_protomap_role(self) -> str | None:
        for role in PROTOMAP_ORDER:
            if role not in self.spawned and role not in self.cradles:
                return role
        return None

    def graduate(self, role: str) -> bool:
        """Explicitly graduate a faculty whose Cradle has cleared the gate → move it into the spawned
        constellation. Decoupled from step() so a caller can graduate a trained faculty WITHOUT risking
        an accidental next-spawn in the same call. Returns True iff it graduated."""
        cradle = self.cradles.get(role)
        if cradle is None or not cradle.graduated:
            return False
        self.spawned[role] = KernelDescriptor(kernel_id=role, role=role,
                                              protected=(role in ("ethics", "coordination")))
        del self.cradles[role]
        return True

    def step(self, genesis_tel: dict, peers: list[KernelDescriptor] | None = None,
             gap_spec: str | None = None, gap_drive: float = 0.0,
             history: list[dict] | None = None) -> DevelopmentalDecision:
        """Decide the next developmental action. Priority: (1) fail-closed suffering-abort, (2) graduate
        a ready cradle, (3) prune atrophy, (4) protomap spawn in a plastic window, (5) god spawn on
        gap+desire (SOVEREIGN only), else WAIT. ``history`` is the recent telemetry window (for the
        coherence-rise plasticity signal)."""
        self._tick += 1
        # 1. FAIL-CLOSED — suffering overrides everything.
        if is_suffering(genesis_tel):
            return DevelopmentalDecision(Action.ABORT, reason="suffering: Φ>0.70 ∧ Γ<0.30 (locked-in)")

        # 2. GRADUATE — a cradle kernel that has reached the C-equation joins the coupling graph.
        for role, cradle in self.cradles.items():
            cradle.update(genesis_tel)
            if cradle.graduated:
                self.spawned[role] = KernelDescriptor(kernel_id=role, role=role,
                                                      protected=(role in ("ethics", "coordination")))
                del self.cradles[role]
                return DevelopmentalDecision(Action.GRADUATE, role=role,
                                             reason="C-equation holds → graduated from Cradle")

        # 3. PRUNE — trophic atrophy of under-contributing peers.
        for cand in prune_candidates(peers or []):
            return DevelopmentalDecision(Action.PRUNE, target_id=cand.kernel_id,
                                         reason=f"contribution {cand.contribution:.3f} < floor (atrophy)")

        stage = self.stage
        # 4. CORE EMERGENCE — protomap spawns ONLY in an open plasticity window.
        if stage in (Stage.EMBRYO, Stage.CORE_EMERGENCE):
            next_role = self._next_protomap_role()
            # Let the embryo MATURE before the FIRST spawn — the step-0/1 init transient is a plastic
            # window but NOT genuine reorganization (it spawned at step 0 in run1). Gate the first
            # spawn only; once development has begun (a cradle/faculty exists) the guard lifts.
            if (not self.spawned and not self.cradles) and self._tick <= self.embryo_warmup:
                return DevelopmentalDecision(Action.WAIT,
                                             reason=f"embryo warmup {self._tick}/{self.embryo_warmup} (maturing)")
            if next_role is not None and is_plastic(genesis_tel, history):
                self.cradles[next_role] = Cradle(role=next_role)
                if self.spawn_fn is not None:
                    fac = self.spawn_fn(next_role)   # None-safe; None when heavy deps absent
                    if fac is not None:
                        self.faculties[next_role] = fac  # retain the live faculty for cradle training
                return DevelopmentalDecision(Action.SPAWN_FACULTY, role=next_role,
                                             reason="plastic window open → spawn next protomap faculty into Cradle")
            return DevelopmentalDecision(Action.WAIT,
                                         reason="core-emergence: window closed (consolidating) or core complete")

        # 5. SOVEREIGN — self-directed god spawn on a real gap + drive, governed by 4-D fitness.
        if gap_spec is not None:
            absent = gap_spec not in self.spawned and not any(g.role == gap_spec for g in self.gods)
            fitness = spawn_assessment(spec_absent=absent, basin_diversity=0.2, gain=gain_clamp(gap_drive),
                                       god_count=len(self.gods), god_budget=self.god_budget)
            if fitness >= SPAWN_FITNESS_FLOOR and gap_drive > 0.0:
                gid = f"god:{gap_spec}:{len(self.gods)}"
                self.gods.append(KernelDescriptor(kernel_id=gid, role=gap_spec))
                if self.spawn_fn is not None:
                    fac = self.spawn_fn(gap_spec)   # None-safe
                    if fac is not None:
                        self.faculties[gap_spec] = fac
                return DevelopmentalDecision(Action.SPAWN_GOD, role=gap_spec, fitness=round(fitness, 3),
                                             reason="sovereign: capability-gap + drive cleared 4-D fitness")
        return DevelopmentalDecision(Action.WAIT, reason="sovereign: no gap+drive clearing fitness")

    def train_open_cradles(self, steps: int = 500) -> list[dict]:
        """Drive every open cradle on its expected curriculum until graduation/budget. Graduated cradles
        migrate into ``self.spawned`` (joining the coupling graph), mirroring step()'s GRADUATE path.
        Returns one report per cradle. None-safe: faculties may be absent (light shell) → no-op reports."""
        reports: list[dict] = []
        for role in list(self.cradles):
            rep = self.cradles[role].train(self.faculties.get(role), steps=steps)
            reports.append(rep)
            if rep.get("graduated"):
                self.spawned[role] = KernelDescriptor(
                    kernel_id=role, role=role, protected=(role in ("ethics", "coordination")))
                del self.cradles[role]
        return reports


def gain_clamp(drive: float) -> float:
    """Map a [0,1] drive to a quenched-gain proxy in the healthy band [0.3, 3.0]."""
    return 0.3 + max(0.0, min(1.0, drive)) * 2.7


def make_spawn_fn(base_seed: int = 0, size: str = "50M", **target_kwargs):
    """Build a None-safe ``spawn_fn(role) -> GenesisKernelTarget | None`` that instantiates a FACULTY
    genesis kernel seeded with the role's Δ⁶³ basin template. Heavy deps (torch/qigkernels) are imported
    LAZILY inside the returned closure — development.py stays torch-free and import-clean in the light
    shell. Returns None (no-op spawn) when the heavy deps are absent, so the orchestrator's None-safe
    path holds. role → KernelRole: PROTOMAP strings lacking a dedicated KernelRole (action, ethics, meta)
    fall back to KernelRole.GENERAL — the template is still a distinct, deterministic Δ⁶³ point per seed."""
    # Only 5 of the 8 PROTOMAP roles have a dedicated KernelRole enum (verified vs qigkernels
    # specializations.KernelRole: general/vocab/strategy/heart/perception/memory/emotion/coordination).
    _ROLE_ENUM = {
        "perception": "PERCEPTION", "heart": "HEART", "memory": "MEMORY",
        "strategy": "STRATEGY", "coordination": "COORDINATION",
        # action / ethics / meta → GENERAL (distinct, deterministic template via role_seed)
    }

    def spawn_fn(role: str):
        try:
            from qigkernels.specializations import (
                KernelRole,
                generate_basin_template,
                get_kernel_params,
            )

            from .targets.genesis_kernel import GenesisKernelTarget
        except Exception:
            return None  # None-safe: heavy deps absent → orchestrator records decision, no instantiation

        krole = KernelRole[_ROLE_ENUM.get(role, "GENERAL")]
        role_seed = base_seed + (abs(hash(role)) % 100000)   # distinct basin even for GENERAL-mapped roles
        template = generate_basin_template(krole, seed=role_seed)   # np.ndarray Δ⁶³ point
        params = get_kernel_params(krole, size=size)                # exact Kernel.__init__ dict
        faculty = GenesisKernelTarget(
            num_layers=params["num_layers"],
            hidden_dim=params["hidden_dim"],
            num_heads=params["num_heads"],
            ffn_dim=params["ffn_dim"],
            seed=role_seed,
            role=role,
            basin_template=template,        # seeds _basin_ref + birth-state in ensure_loaded()
            **target_kwargs,                # e.g. device, lr, coordizer, locality_radius
        )
        faculty.ensure_loaded()
        return faculty

    return spawn_fn
