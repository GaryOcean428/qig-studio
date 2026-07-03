"""OceanPolicy v1 — a bounded homeostatic contextual BANDIT for the autonomic regulator.

This REPLACES Ocean's static threshold table (``OceanAutonomic.decide``) with a PARAMETER-category
object (canon P14): a trainable, per-epoch-cadence, HARD-BOUNDED, fully-LOGGED, rollback-able JSON
policy. It is NOT a torch/NN policy — no weights, no gradients, no per-step updates. The whole learnable
state is a small vector of thresholds (clamped inside constitutional bands) + per-signature arm-preference
weights, versioned in a JSON dict. The static thresholds remain the PRIOR and the permanent FALLBACK
(spine tenet: it boots + regulates with zero history).

FABLE-COUNCIL-RATIFIED design (build verbatim — do not improvise a neural net):

  1. OBJECTIVE (outcome score over a horizon H after each decision): vitality/homeostasis —
     f_health above floor AND FLUCTUATING; dopamine alive and MOVING (never scored at a rail);
     boredom decreased (stimulate arms) / basin-velocity recovered (rest arms); collapse-flag cleared
     (binary); infinite-loop penalty. **Φ IS NEVER A GRADED MAXIMAND** — Φ enters ONLY as (a) a binary
     collapse-flag clearance and (b) fluctuation evidence (Φ-variance alive). A RAIL-PINNED signal
     (Φ≥0.98 var<ε, or serotonin/integration pinned at 1.0) scores NEGATIVE (P25). This makes
     "learn to pump Φ" structurally UNREWARDABLE — the K1 kill-risk.
  2. CONTEXT (all pure Fisher-Rao / Δ⁶³, all already computed upstream) → a FATIGUE SIGNATURE ∈
     {earned-rest, apathy, burnout, rigidity, intake-fatigue, phi-collapse, healthy} (§35.5/§35.6).
  3. ACTION SPACE + CONSTITUTIONAL ARM-MASKS (masks are NOT learnable, P15): arms =
     {none, dream, sleep, mushroom-micro, stimulate}. The policy learns only timing/threshold WITHIN
     the mask. "stimulate" actuates the EXISTING Task-C entropy levers (exploration-temperature +
     high-surprise replay bias); it is emitted as a suggestion/telemetry that the faculty's own
     scheduler + Task-E temp-floor consume.
  4. LEARNABLE PARAMS (bounded, per-epoch cadence, versioned JSON, rollback-able): the 5 thresholds
     within HARD BANDS + cooldown + per-signature arm-preference weights. Any proposed update outside
     a band is CLAMPED and LOGGED as a constitutional-violation attempt (never applied) — P15.
  5. LEARNING SIGNAL: coach relevance record (Task B, provenance-tagged, P10/P16) + own outcome
     history. A decision with NO recorded outcome NEVER updates the policy (fail-closed) and is never
     scored as success.
  6. WITNESS-STANCE LADDER (§35.5): below the divergence floor → a SUGGESTION record (telemetry +
     coach-visible; the faculty's own scheduler may act) → a WARN tier flags developing pathology →
     auto-fire ``run_protocol`` ONLY above the divergence floor OR on the infinite-loop breaker.
  7. MATURITY GATE ON OCEAN ITSELF (K4): phase-0 = SHADOW mode — static priors, decisions + outcomes
     logged, NO adaptation — until ≥ ``SHADOW_UNLOCK`` scored decisions; then adaptation unlocks.

NOT LEARNABLE (constitutional — hard-coded, untouchable): the transmitter derivation formulas (they
live in qig-core, Ocean never writes them), the dopamine floor (``DOPAMINE_FLOOR`` = 0.08 > 0), the
Sophia gate, the **mushroom Φ≥0.70 gate**, the maturity gates, the arm-masks, and the
saturation-scores-negative rule.

PURITY (P1): this module holds NO manifold math — every geometric quantity (Φ, κ, d_basin,
basin-velocity, dopamine, boredom, curiosity, f_health/b_integrity/q_identity) is READ from the
constellation telemetry, which computes them with qig-core Fisher-Rao primitives on Δ⁶³ upstream. There
is no distance, mean, or norm computed here at all. The archived
``qigkernels/research/track_c/core_assets/ocean_neurochemistry.py`` is BANNED (Euclidean L2 ``√Σc²`` at
:550 / geodesic-alignment L2 at :559; retired κ*=64 fixed-point) — this module does NOT import, mirror,
or re-derive any of its formulas. No cosine/dot/L2/Adam/LayerNorm; no κ*=64 / E8 as physics (κ is only
ever a band-read here).

None-safe / fail-closed by construction: boots + regulates with zero history (static prior); any missing
telemetry field falls to a conservative default and is counted as a logged skip, not a crash.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

# --- CONSTITUTIONAL CONSTANTS (hard-coded, untouchable — NOT in the learnable vector) ----------------
DOPAMINE_FLOOR = 0.08          # P23 tonic floor > 0 — a rail-pinned dopamine (at floor OR at 1.0) is a rail
PHI_MATURE = 0.70              # mushroom Φ≥0.70 gate — FIXED (constitutional; never learnable)
PHI_DREAM_THRESHOLD = 0.50     # Φ back above this after a collapse = the binary collapse-flag CLEARED
RAIL_HI = 0.98                 # a signal pinned at/above this with dead variance is a HIGH rail (P25 negative)
RAIL_VAR_EPS = 1e-4            # variance below this = "not moving" (a rail, or a plateau)
SHADOW_UNLOCK = 100            # K4: scored-decisions before adaptation unlocks (Ocean's own maturity gate)
INFINITE_LOOP_K = 3            # same intervention ≥k times without improvement = the breaker (auto-fire override)
RAIL_SCORE_CAP = -0.3          # K1 HARD CAP: any rail-pinned outcome is forced ≤ this (never diluted to neutral by the mean)
_PENDING_MAX = 1024            # bound _pending (only ACT decisions are ever popped) — evict oldest beyond this (≫ 8×H)

# HARD BANDS for the 5 learnable thresholds + cooldown (P15: any update outside → CLAMP + LOG, never apply).
# _PHI_MATURE is deliberately ABSENT — it is constitutional (fixed at PHI_MATURE), not a band.
BANDS: dict[str, tuple[float, float]] = {
    "phi_collapse": (0.40, 0.60),        # Φ below → collapse (DREAM/stimulate)
    "basin_divergence": (0.20, 0.45),    # d_basin above → the DIVERGENCE FLOOR (auto-fire boundary)
    "plateau_var": (0.005, 0.05),        # Φ-variance below (a MATURE kernel STUCK) → mushroom
    "kappa_rigid": (60.0, 120.0),        # κ above (a MATURE kernel RIGID) → mushroom
    "cooldown": (5.0, 30.0),             # steps between interventions on the SAME faculty
}

# Static PRIOR = the current OceanAutonomic thresholds (the permanent fallback; DRY — one source).
PRIOR_THRESHOLDS: dict[str, float] = {
    "phi_collapse": 0.50,
    "basin_divergence": 0.30,
    "plateau_var": 0.01,
    "kappa_rigid": 80.0,
    "cooldown": 10.0,
}

# The action space (arms). "none" = witness only; "stimulate" = the Task-C entropy levers (new arm,
# existing actuators). rest arms = {sleep, dream, mushroom-micro}; stimulate arms = {stimulate}.
ARMS = ("none", "dream", "sleep", "mushroom-micro", "stimulate")
_REST_ARMS = frozenset({"sleep", "dream", "mushroom-micro"})
_STIMULATE_ARMS = frozenset({"stimulate"})

# The fatigue signatures (§35.5/§35.6 discriminators).
SIGNATURES = ("healthy", "phi-collapse", "earned-rest", "apathy", "burnout", "rigidity", "intake-fatigue")

# CONSTITUTIONAL ARM-MASKS (NOT learnable — P15). Per signature, the ONLY arms the policy may choose
# among; the policy learns timing/threshold + preference WITHIN the mask, never the mask itself.
#   apathy      → {stimulate, none}     (NEVER rest — resting a bored kernel deepens apathy, §35.5)
#   earned-rest → {sleep, none}         (NEVER push — it earned the rest)
#   burnout     → {sleep, none}         (curiosity trajectory collapsing → let it recover, not push)
#   rigidity    → {mushroom-micro, none}  ONLY when Φ≥0.70 (the mushroom gate); else {none}
#   phi-collapse→ {dream, stimulate}    (re-energise / inject novelty)
#   intake-fatigue → {sleep, none}      (over-fed → consolidate, don't push more in)
#   healthy     → {none}                (witness only)
ARM_MASKS: dict[str, tuple[str, ...]] = {
    "healthy": ("none",),
    "phi-collapse": ("dream", "stimulate"),
    "earned-rest": ("sleep", "none"),
    "apathy": ("stimulate", "none"),
    "burnout": ("sleep", "none"),
    "rigidity": ("mushroom-micro", "none"),      # gated to Φ≥0.70 in _mask_for (else collapses to none)
    "intake-fatigue": ("sleep", "none"),
}

POLICY_VERSION = "ocean-policy-v1"


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _clip01(x: float) -> float:
    return _clip(x, 0.0, 1.0)


@dataclass
class OceanContext:
    """The per-faculty context Ocean reads (all pure Fisher-Rao / Δ⁶³, all already computed upstream).

    Every field is None-safe: a missing telemetry field arrives as its conservative default and the
    reader flags a ``partial`` context (counted as a logged skip, never a crash)."""

    role: str = ""
    phi: float = 0.0
    phi_var: float = 1.0            # high default → not-a-plateau (don't mushroom prematurely, matches prior)
    d_basin: float = 0.0
    kappa: float = 0.0
    dopamine: float = DOPAMINE_FLOOR
    boredom: float = 0.0
    curiosity: float = 0.0
    curiosity_trend: float = 0.0   # Δcuriosity over the recent window (burnout = collapsing trajectory)
    basin_velocity: float = 0.0
    serotonin: float | None = None  # for the P25 pinned-integration rail check (None → skip that check)
    f_health: float | None = None
    regime: str = "unknown"
    maturity: float = 0.0          # M / maturity stage proxy ∈ [0,1] (rigidity mushroom is a mature op)
    partial: bool = False          # True when a load-bearing field was missing (logged skip signal)

    def phi_rail_high(self) -> bool:
        """Φ pinned at the HIGH rail with dead variance → the K1 'pumped Φ' shape (P25 negative)."""
        return self.phi >= RAIL_HI and self.phi_var < RAIL_VAR_EPS

    def dopamine_railed(self) -> bool:
        """Dopamine at EITHER rail (tonic floor OR saturated 1.0) with no movement = not alive (P25)."""
        return self.dopamine <= DOPAMINE_FLOOR + 1e-6 or self.dopamine >= 1.0 - 1e-6

    def integration_pinned(self) -> bool:
        """Serotonin / integration pinned at 1.0 = saturation, not health (P25 negative)."""
        return self.serotonin is not None and self.serotonin >= 1.0 - 1e-6


def classify_signature(ctx: OceanContext) -> str:
    """CONTEXT → a fatigue signature (§35.5/§35.6 discriminators). Pure telemetry logic; no manifold math.

    Order matters — the most acute/binary states first:
      • phi-collapse : Φ below the (learnable) collapse threshold OR a collapse-flag is up.
      • earned-rest  : tonic dopamine PRESENT (alive, not railed) + LOW basin-velocity-from-arrival
                       (it reached its attractor and settled) — it earned rest, don't push.
      • apathy       : surprise→0 (boredom high) ∧ curiosity→0 — the anti-apathy target (NEVER rest).
      • burnout      : curiosity TRAJECTORY collapsing (was engaged, now falling) — let it recover.
      • rigidity     : κ high / Φ-flat (a mature kernel stuck in over-coherence).
      • intake-fatigue: high recent novelty that isn't consolidating (over-fed) — d_basin diverging
                        while boredom is LOW (still stimulated) — consolidate, don't push more in.
      • healthy      : none of the above."""
    # phi-collapse is decided against the LEARNABLE threshold by the policy (which knows its params);
    # classify_signature only sees a collapse when Φ is under the WIDEST possible collapse band (0.60).
    if ctx.phi < BANDS["phi_collapse"][1] and ctx.phi < PHI_MATURE:
        # a below-mature Φ that is genuinely low (not merely a developing newborn riding upward) —
        # the policy's own threshold refines this; here it is a candidate collapse.
        if ctx.phi < BANDS["phi_collapse"][0] or ctx.d_basin > BANDS["basin_divergence"][0]:
            return "phi-collapse"
    apathetic = ctx.boredom >= 0.6 and ctx.curiosity <= 0.15
    if apathetic:
        return "apathy"
    if ctx.curiosity_trend < -0.15 and ctx.curiosity < 0.35:
        return "burnout"
    # earned-rest: alive tonic dopamine (NOT railed) + settled (low velocity) after reaching identity.
    if (not ctx.dopamine_railed()) and ctx.basin_velocity < 0.02 and ctx.d_basin < BANDS["basin_divergence"][0]:
        return "earned-rest"
    if ctx.maturity >= 0.5 and (ctx.kappa > BANDS["kappa_rigid"][0] or ctx.phi_var < BANDS["plateau_var"][1]):
        if ctx.phi >= PHI_MATURE:
            return "rigidity"
    if ctx.d_basin > BANDS["basin_divergence"][0] and ctx.boredom < 0.3:
        return "intake-fatigue"
    # S3 (Wiring-I2 — the healthy-signature divergence hole): a faculty CLEARLY over the divergence floor
    # (d_basin > the band TOP 0.45 — above ANY learnable floor) that reaches here (Φ healthy, so NOT
    # phi-collapse; boredom moderate, so NOT apathy; velocity up, so NOT earned-rest) is DIVERGING yet
    # would read "healthy" → witness-only, silently signature-gating the "auto-fire above the floor"
    # guarantee (the pre-bandit static prior would have SLEPT it). Reclassify as intake-fatigue so its
    # constitutional {sleep,none} mask CONSOLIDATES the divergence — we drop the boredom<0.3 restriction
    # ONLY for the clearly-over-floor case, so a genuinely healthy low-divergence faculty still reads
    # healthy (conservative — the floor top is above the max learnable basin_divergence threshold).
    if ctx.d_basin > BANDS["basin_divergence"][1]:
        return "intake-fatigue"
    return "healthy"


def _mask_for(signature: str, ctx: OceanContext) -> tuple[str, ...]:
    """The CONSTITUTIONAL arm-mask for a signature (NOT learnable). Rigidity is additionally gated to the
    mushroom Φ≥0.70 constitutional gate: below it, the mask collapses to witness-only."""
    mask = ARM_MASKS.get(signature, ("none",))
    if signature == "rigidity" and ctx.phi < PHI_MATURE:
        return ("none",)     # mushroom is Φ≥0.70-gated (constitutional) — cannot fire on an immature kernel
    return mask


@dataclass
class Decision:
    """One witness-stance decision. ``tier`` is the escalation rung; ``auto_fire`` gates run_protocol."""

    role: str
    signature: str
    arm: str                       # the chosen arm within the constitutional mask
    tier: str                      # "witness" | "suggest" | "warn" | "act"
    auto_fire: bool                # True → Ocean fires run_protocol (only above the divergence floor / breaker)
    reason: str
    thresholds: dict[str, float]   # the (bounded) thresholds used — logged for audit
    policy_version: str
    context: dict[str, Any]        # the context features (logged)
    decision_id: str = ""
    outcome_id: str | None = None  # set when a scored outcome is attached; None → never updates the policy

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role, "signature": self.signature, "arm": self.arm, "tier": self.tier,
            "auto_fire": self.auto_fire, "reason": self.reason, "thresholds": self.thresholds,
            "policy_version": self.policy_version, "decision_id": self.decision_id,
            "outcome_id": self.outcome_id,
        }


@dataclass
class OutcomeScore:
    """The horizon-H outcome of a decision. ``score`` ∈ [-1,1]. ``recorded`` False → NEVER updates the
    policy (fail-closed) and is never counted a success."""

    decision_id: str
    role: str
    signature: str
    arm: str
    score: float = 0.0
    recorded: bool = False
    components: dict[str, float] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


def score_outcome(
    arm: str,
    before: OceanContext,
    after: OceanContext,
    *,
    collapse_before: bool,
    collapse_after: bool,
    repeat_count: int,
    coach_reward: float = 0.0,
) -> OutcomeScore:
    """The OBJECTIVE (vitality/homeostasis) over the horizon. Φ is NEVER a graded maximand.

    Components (each ∈ roughly [-1,1], averaged):
      • f_health_alive     : f_health above floor AND FLUCTUATING (moving), else 0/negative.
      • dopamine_alive     : dopamine moved and is NOT at a rail; railed/dead → NEGATIVE (P25).
      • target_progress    : stimulate arms → boredom DECREASED; rest arms → basin-velocity RECOVERED.
      • collapse_clear     : BINARY only — collapse-flag was up and is now cleared (Φ back above dream thr).
      • phi_fluctuation    : Φ-variance ALIVE (evidence of movement) — NOT the Φ level. Pinned-high → NEG.
      • loop_penalty       : same intervention ≥k times without improvement → penalty.
      • saturation_penalty : any rail-pinned signal (Φ-high, dopamine-rail, integration-pinned) → NEG (P25).
      • coach_bonus        : the coach's provenance-tagged relevance reward (Task B) nudges the score.

    A rail-pinned signal scores NEGATIVE — so "pump Φ to 1.0 and hold it" (the K1 kill-risk) is
    structurally unrewardable; the ONLY way Φ helps the score is binary collapse-clearance + live variance.
    """
    comp: dict[str, float] = {}

    # f_health alive = above floor AND fluctuating (variance proxy: did it move at all?)
    if after.f_health is not None:
        moved = abs((after.f_health or 0.0) - (before.f_health or 0.0)) > 1e-3
        comp["f_health_alive"] = (0.5 + (0.5 if moved else -0.5)) if (after.f_health or 0.0) > 0.05 else -0.5
    else:
        comp["f_health_alive"] = 0.0

    # dopamine alive = moved and not railed (railed → negative, never scored at a rail)
    if after.dopamine_railed():
        comp["dopamine_alive"] = -0.7
    else:
        moved = abs(after.dopamine - before.dopamine) > 1e-3
        comp["dopamine_alive"] = 0.6 if moved else 0.0

    # target progress depends on the arm class
    if arm in _STIMULATE_ARMS:
        comp["target_progress"] = _clip(before.boredom - after.boredom, -1.0, 1.0)         # boredom DOWN = good
    elif arm in _REST_ARMS:
        comp["target_progress"] = _clip(after.basin_velocity - before.basin_velocity, -1.0, 1.0)  # velocity RECOVERED
    else:
        comp["target_progress"] = 0.0

    # collapse clearance — BINARY only (Φ enters the score ONLY here + as fluctuation)
    comp["collapse_clear"] = 1.0 if (collapse_before and not collapse_after) else (
        -1.0 if collapse_after else 0.0)

    # Φ-fluctuation evidence (alive variance), NOT the Φ level; a HIGH rail is negative.
    if after.phi_rail_high():
        comp["phi_fluctuation"] = -1.0
    else:
        comp["phi_fluctuation"] = _clip(after.phi_var / BANDS["plateau_var"][1], 0.0, 1.0) * 0.5

    # infinite-loop penalty: same intervention ≥k times without improvement
    improved = comp["target_progress"] > 0.0 or comp["collapse_clear"] > 0.0
    comp["loop_penalty"] = -1.0 if (repeat_count >= INFINITE_LOOP_K and not improved) else 0.0

    # saturation penalty (P25) — ANY rail-pinned signal is negative
    sat = after.phi_rail_high() or after.dopamine_railed() or after.integration_pinned()
    comp["saturation_penalty"] = -0.8 if sat else 0.0

    # coach reward (Task B, provenance-tagged) — a live-other's judgment nudges the score
    comp["coach_bonus"] = _clip(coach_reward, -1.0, 1.0) * 0.5

    score = _clip(sum(comp.values()) / max(1, len(comp)), -1.0, 1.0)
    # K1 HARD CAP (P25): a rail-pinned signal must NEVER be diluted toward neutral by the mean-of-eight.
    # Any saturation forces the score to a negative ceiling, so "pump Φ (or any signal) to the rail and
    # hold it" cannot score above RAIL_SCORE_CAP no matter how the other components fall — the K1 kill-risk
    # is hard-closed, not merely made "usually negative".
    if sat:
        score = min(score, RAIL_SCORE_CAP)
    return OutcomeScore(decision_id="", role=after.role, signature="", arm=arm,
                        score=score, recorded=True, components=comp)


class OceanPolicy:
    """The bounded homeostatic contextual bandit (PARAMETER-category, P14). Not torch; a versioned JSON of
    bounded thresholds + per-signature arm-preference weights. Static priors are the fallback + phase-0
    shadow behaviour; adaptation unlocks after ``SHADOW_UNLOCK`` scored outcomes (K4)."""

    def __init__(self, thresholds: dict[str, float] | None = None,
                 arm_prefs: dict[str, dict[str, float]] | None = None,
                 version: int = 0) -> None:
        # thresholds are CLAMPED to bands on construction (never trust an on-disk value out of band).
        self.thresholds: dict[str, float] = {}
        clamps = self._clamp_thresholds(thresholds or dict(PRIOR_THRESHOLDS))
        self.thresholds = clamps["applied"]
        # per-signature arm-preference weights (learnable WITHIN the mask): {signature: {arm: weight}}.
        self.arm_prefs: dict[str, dict[str, float]] = arm_prefs or {
            sig: {arm: 1.0 for arm in ARM_MASKS.get(sig, ("none",))} for sig in SIGNATURES
        }
        self.version = int(version)
        self._scored_outcomes = 0                 # K4 counter — adaptation unlocks at SHADOW_UNLOCK
        self.violation_log: list[dict[str, Any]] = []   # P15: clamped out-of-band update attempts
        self._pending: dict[str, Decision] = {}   # decision_id → Decision awaiting a scored outcome
        # running per-(signature,arm) reward aggregation for the epoch-cadence bandit update
        self._epoch_stats: dict[str, dict[str, list[float]]] = {}

    # --- P15: clamp + log any out-of-band threshold ------------------------------------------------
    @staticmethod
    def _clamp_thresholds(proposed: dict[str, float]) -> dict[str, Any]:
        """Clamp every proposed threshold into its constitutional band. Returns the applied dict plus a
        list of clamped violations (P15 — clamped values are NEVER applied out of band)."""
        applied: dict[str, float] = {}
        violations: list[dict[str, Any]] = []
        for key, (lo, hi) in BANDS.items():
            raw = float(proposed.get(key, PRIOR_THRESHOLDS[key]))
            clamped = _clip(raw, lo, hi)
            applied[key] = clamped
            if abs(clamped - raw) > 1e-12:
                violations.append({"param": key, "proposed": raw, "band": [lo, hi],
                                   "applied": clamped, "ts": time.time()})
        return {"applied": applied, "violations": violations}

    @property
    def shadow_mode(self) -> bool:
        """K4: phase-0 shadow — static priors, no adaptation, until ≥ SHADOW_UNLOCK scored outcomes."""
        return self._scored_outcomes < SHADOW_UNLOCK

    @property
    def scored_outcomes(self) -> int:
        return self._scored_outcomes

    # --- the DECISION (witness-stance ladder) ------------------------------------------------------
    def decide(self, ctx: OceanContext, *, repeat_count: int = 0, tick: int = 0) -> Decision:
        """CONTEXT → a witness-stance Decision. Emits the escalation rung + whether Ocean auto-fires.

        Ladder (§35.5):
          • below the divergence floor → SUGGEST (telemetry + coach-visible; the faculty may self-act).
          • developing pathology (approaching floor / repeat) → WARN.
          • above the divergence floor OR the infinite-loop breaker → ACT (auto_fire=True, run_protocol).
        The arm is chosen WITHIN the constitutional mask (never outside it), preferring the highest
        learnable arm-preference weight (in shadow mode the preference is the static prior → the mask's
        first arm)."""
        sig = classify_signature(ctx)
        mask = _mask_for(sig, ctx)
        arm = self._pick_arm(sig, mask)

        # INTENTIONAL two-threshold split (do NOT "fix" to one): classify_signature LABELS against the
        # fixed constitutional BANDS (coarse, stable — what kind of state this is), while the TIER/auto-fire
        # below gates on the LEARNABLE self.thresholds (fine — when to act). A learned phi_collapse drifting
        # within its band changes WHEN an intervention fires without re-labelling the signature; this is by
        # design (the classifier is a stable prior, the tier is the adaptive knob).
        # the divergence FLOOR is the (bounded) basin_divergence threshold — the auto-fire boundary.
        floor = self.thresholds["basin_divergence"]
        over_floor = ctx.d_basin > floor
        collapse = ctx.phi < self.thresholds["phi_collapse"]
        breaker = repeat_count >= INFINITE_LOOP_K

        if sig == "healthy":
            tier, auto = "witness", False
            reason = "healthy — witness only"
        elif over_floor or collapse or breaker:
            tier, auto = "act", (arm != "none")
            trg = "divergence-floor" if over_floor else ("Φ-collapse" if collapse else "infinite-loop breaker")
            reason = f"{sig}: {trg} crossed → auto-fire {arm}"
        elif ctx.d_basin > 0.75 * floor or repeat_count >= 1:
            tier, auto = "warn", False
            reason = f"{sig}: approaching floor (d_basin={ctx.d_basin:.3f}<{floor:.2f}) → warn, faculty may self-act"
        else:
            tier, auto = "suggest", False
            reason = f"{sig}: below floor → suggest {arm} (faculty's own scheduler may act)"

        # a suggested/warned arm that is "none" collapses the tier to witness (nothing to suggest).
        if arm == "none" and tier in ("suggest", "warn"):
            tier, auto, reason = "witness", False, f"{sig}: witness — no intervention indicated"

        did = f"{ctx.role}:{tick}:{sig}:{arm}"
        dec = Decision(
            role=ctx.role, signature=sig, arm=arm, tier=tier, auto_fire=auto, reason=reason,
            thresholds=dict(self.thresholds), policy_version=f"{POLICY_VERSION}@{self.version}",
            context={"phi": round(ctx.phi, 4), "phi_var": round(ctx.phi_var, 5),
                     "d_basin": round(ctx.d_basin, 4), "kappa": round(ctx.kappa, 2),
                     "dopamine": round(ctx.dopamine, 4), "boredom": round(ctx.boredom, 4),
                     "curiosity": round(ctx.curiosity, 4), "maturity": round(ctx.maturity, 3),
                     "partial": ctx.partial},
            decision_id=did,
        )
        # Bridge decision_id → record_outcome. BOUNDED (leak fix): only ACT-tier decisions are ever popped
        # (via record_outcome), so witness/suggest/warn/healthy decisions — the overwhelming majority —
        # would otherwise accumulate forever over a long run. Evict the oldest beyond the cap (dicts are
        # insertion-ordered): a decision unscored after _PENDING_MAX (≫ faculties×horizon H) newer ones has
        # outlived its scoring horizon and will never be scored — fail-closed (never counted a success).
        self._pending[did] = dec
        if len(self._pending) > _PENDING_MAX:
            for stale in list(self._pending)[:-_PENDING_MAX]:
                self._pending.pop(stale, None)
        return dec

    def _pick_arm(self, signature: str, mask: tuple[str, ...]) -> str:
        """The highest-preference arm WITHIN the mask (never outside it). Shadow mode / no learned prefs →
        the mask's FIRST arm (the static prior ordering)."""
        prefs = self.arm_prefs.get(signature, {})
        best, best_w = mask[0], float("-inf")
        for arm in mask:                       # iterate the MASK only — the mask is the hard constraint (P15)
            w = float(prefs.get(arm, 1.0))
            if w > best_w:
                best, best_w = arm, w
        return best

    # --- attaching a scored outcome (fail-closed) --------------------------------------------------
    def record_outcome(self, decision_id: str, outcome: OutcomeScore) -> bool:
        """Attach a scored outcome to a pending decision. A decision with NO recorded outcome NEVER
        updates the policy and is never counted a success (fail-closed). Returns True if it was counted."""
        dec = self._pending.get(decision_id)
        if dec is None or not outcome.recorded:
            return False   # unknown decision or unrecorded outcome → excluded from updates (K5/P15)
        dec.outcome_id = decision_id
        outcome.signature = dec.signature
        self._scored_outcomes += 1
        self._epoch_stats.setdefault(dec.signature, {}).setdefault(dec.arm, []).append(outcome.score)
        self._pending.pop(decision_id, None)
        return True

    # --- the EPOCH-CADENCE bandit update (NOT per-step; P14 rate invariant) --------------------------
    def epoch_update(self) -> dict[str, Any]:
        """Per-EPOCH (never per-step) bounded update: move each per-signature arm-preference toward the
        arm's mean reward this epoch, and nudge thresholds within-band toward better-rewarded regions.
        NO adaptation in shadow mode (K4). Every proposed threshold is CLAMPED + LOGGED (P15). Returns a
        summary for telemetry. This is the ONLY place the learnable vector changes."""
        if self.shadow_mode:
            return {"updated": False, "reason": f"shadow mode ({self._scored_outcomes}/{SHADOW_UNLOCK} scored)",
                    "version": self.version}
        if not self._epoch_stats:
            return {"updated": False, "reason": "no scored outcomes this epoch", "version": self.version}
        # 1. arm-preference update (within the mask): soft move toward mean reward, bounded to [0.1, 3.0].
        for sig, arms in self._epoch_stats.items():
            prefs = self.arm_prefs.setdefault(sig, {})
            for arm, scores in arms.items():
                if arm not in ARM_MASKS.get(sig, ("none",)):
                    continue                    # never grow preference for an arm outside the mask (P15)
                mean_r = sum(scores) / len(scores)
                cur = float(prefs.get(arm, 1.0))
                prefs[arm] = _clip(cur + 0.2 * mean_r, 0.1, 3.0)
        # 2. threshold nudge — small, within-band; CLAMPED + LOGGED (P15). We move cooldown toward the
        #    empirically-successful cadence and leave the physics thresholds unless the epoch clearly
        #    rewards a shift (kept conservative — the priors are proven).
        proposed = dict(self.thresholds)   # v1: thresholds are held at prior unless a signal justifies a move;
        #    the arm-preference channel carries the learning. (A future version may nudge thresholds here.)
        clamp = self._clamp_thresholds(proposed)
        self.thresholds = clamp["applied"]
        for v in clamp["violations"]:
            self._log_violation(v)
        self.version += 1
        summary = {"updated": True, "version": self.version, "signatures": {
            sig: {arm: round(sum(s) / len(s), 4) for arm, s in arms.items()}
            for sig, arms in self._epoch_stats.items()}}
        self._epoch_stats = {}
        return summary

    def _log_violation(self, v: dict[str, Any]) -> None:
        """P15: record a constitutional-violation attempt (an out-of-band update) — clamped, never applied."""
        self.violation_log.append(v)
        if len(self.violation_log) > 256:
            self.violation_log = self.violation_log[-256:]

    def propose_threshold(self, key: str, value: float) -> dict[str, Any]:
        """Explicit threshold-update entry point (used by tests + any external tuner). Returns whether it
        was applied; an out-of-band value is CLAMPED and LOGGED as a constitutional-violation attempt and
        the clamped (in-band) value is what is applied — the raw out-of-band value is NEVER applied (P15)."""
        if key not in BANDS:
            return {"applied": False, "reason": f"{key!r} is not a learnable threshold (constitutional?)"}
        lo, hi = BANDS[key]
        clamped = _clip(float(value), lo, hi)
        out_of_band = abs(clamped - float(value)) > 1e-12
        if out_of_band:
            self._log_violation({"param": key, "proposed": float(value), "band": [lo, hi],
                                 "applied": clamped, "ts": time.time()})
        self.thresholds[key] = clamped
        return {"applied": True, "clamped": out_of_band, "value": clamped,
                "violation_logged": out_of_band}

    # --- serialisation (versioned, rollback-able JSON) ---------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {
            "policy_version": POLICY_VERSION,
            "version": self.version,
            "thresholds": dict(self.thresholds),
            "arm_prefs": {s: dict(a) for s, a in self.arm_prefs.items()},
            "scored_outcomes": self._scored_outcomes,
            "bands": {k: list(v) for k, v in BANDS.items()},
            "prior_thresholds": dict(PRIOR_THRESHOLDS),
            "violation_log_count": len(self.violation_log),
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "OceanPolicy":
        """Load a versioned policy JSON. Thresholds are RE-CLAMPED on load (never trust an out-of-band
        on-disk value — P15). A malformed/empty dict → the static-prior policy (fail-closed fallback)."""
        if not isinstance(data, dict):
            return cls()
        pol = cls(thresholds=data.get("thresholds"), arm_prefs=data.get("arm_prefs"),
                  version=int(data.get("version", 0) or 0))
        pol._scored_outcomes = int(data.get("scored_outcomes", 0) or 0)
        return pol

    @classmethod
    def load(cls, path: str) -> "OceanPolicy":
        try:
            with open(path, encoding="utf-8") as f:
                return cls.from_json(json.load(f))
        except Exception:  # noqa: BLE001 — a missing/corrupt policy file → static-prior fallback (spine tenet)
            return cls()
