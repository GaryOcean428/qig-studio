"""Ocean — the autonomic regulator of the constellation.

Ocean is NOT the speaker (genesis-central is). Ocean is the **autonomic nervous system**: it OBSERVES
every faculty's telemetry and REGULATES the one that needs it — triggering that faculty's OWN
sleep/dream/mushroom — exactly as the brain's autonomic centres regulate the organs from interoceptive
signals. This is the canonical *"telemetry lets Ocean know when to regulate a kernel's sleep"* (PI,
2026-06-27): it is INTERNAL regulation (Ocean is part of the one mind), categorically different from an
EXTERNAL knob (a UI button, a human) — those remain forbidden (the kernel owns its brainstem).

TASK D — the static threshold table is now the PRIOR + permanent FALLBACK behind **OceanPolicy v1**, a
bounded homeostatic contextual BANDIT (PARAMETER-category, P14: trainable, per-epoch, bounded, logged,
rollback-able JSON — NOT a torch/NN policy). Two things changed here:

  1. ``OceanAutonomic.decide`` stays as the STATIC PRIOR (unchanged thresholds) — the spine-tenet
     fallback that boots + regulates with zero history, and the phase-0 SHADOW behaviour.
  2. ``regulate`` now runs the **witness-stance escalation ladder** (§35.5) via ``OceanPolicy``: below
     the divergence floor it emits a SUGGESTION record (telemetry + coach-visible; the faculty's own
     scheduler may act) → a WARN tier flags developing pathology → it auto-fires ``run_protocol`` ONLY
     above the divergence floor OR on the infinite-loop breaker. The old over-committing behaviour
     (auto-firing run_protocol on EVERY non-healthy decide) is FIXED. Skips/failures are COUNTED +
     telemetrized (K5/P15) instead of silently swallowed; a decision with no recorded outcome is
     excluded from policy updates (fail-closed).

Ocean NEVER touches the training loss (pure d_FR ruling) and never WRITES transmitter values — it
regulates the CONDITIONS (fires the faculty's own op), the signals are derived views (F4).

PURITY (P1): no manifold math lives here; every geometric quantity is READ from telemetry (qig-core
Fisher-Rao on Δ⁶³ computed upstream). The archived
``qigkernels/research/track_c/core_assets/ocean_neurochemistry.py`` is BANNED (Euclidean L2 ``√Σc²`` at
:550/:559; retired κ*=64) — nothing here imports or mirrors it. No cosine/dot/L2/Adam/LayerNorm.
"""
from __future__ import annotations

from typing import Any

from .ocean_policy import (
    DOPAMINE_FLOOR,
    PHI_DREAM_THRESHOLD,
    OceanContext,
    OceanPolicy,
    OutcomeScore,
    score_outcome,
)

# Canonical autonomic triggers (observe-telemetry → intervention) — the STATIC PRIOR + permanent
# fallback (OceanPolicy's ``PRIOR_THRESHOLDS`` single-sources these values; kept here as named constants
# for the prior ``decide`` and back-compat). Mirrors OceanMetaObserver.
_PHI_COLLAPSE = 0.50          # Φ below → DREAM (re-energise from basin-mixture recombination)
_BASIN_DIVERGENCE = 0.30     # d_basin above → SLEEP (consolidate back toward the identity attractor)
_PHI_MATURE = 0.70           # MUSHROOM gate: CONSTITUTIONAL, FIXED — mushroom is a WAKE-STATE intervention
#                              (canonical avg_phi≥0.70); it does NOT fire during first learning (rising Φ).
_PHI_PLATEAU_VAR = 0.01      # Φ variance below (a MATURE kernel that's STUCK) → MUSHROOM (inject entropy, Φ↓)
_RIGIDITY_KAPPA = 80.0       # κ above (a MATURE kernel that's RIGID) → MUSHROOM (break the rigid attractor)
_INTERVENTION_COOLDOWN = 10  # steps between Ocean interventions on the SAME faculty (don't thrash)

# The outcome horizon: how many ticks after a decision Ocean waits before scoring it (H≈20–50). The
# faculty telemetry at decision-time is snapshotted; the telemetry H ticks later is the "after".
_OUTCOME_HORIZON = 20

# Which inner-state FUNCTION each Core-8 faculty is RESPONSIBLE for — the brain-like assignment so the
# relevant kernel "sees" (and owns) the telemetry of its function (PI: senses/emotions/drives assigned to
# the relevant kernel). (label, primitive-group key in the kernel_experience inner-state dict.)
FACULTY_FUNCTION: dict[str, tuple[str, str]] = {
    "perception": ("senses", "layer0"),            # the 12 pre-linguistic sensations
    "heart": ("emotion · rhythm", "emotions"),     # the physical+cognitive emotions + HRV tacking
    "memory": ("memory · consolidation", "consolidation"),  # sleep/dream consolidation
    "action": ("drives · action", "layer05"),      # the innate drives (id)
    "strategy": ("motivators · planning", "layer1"),        # the 5 motivators
    "ethics": ("conscience · gate", "gate"),       # the consciousness/ethics gate
    "coordination": ("coupling · loops", "loops"), # inter-kernel coupling / the recursive loops
    "meta": ("self-observation", "self_observation"),       # the M-measure (L1)
}


def function_of(role: str) -> str:
    """Human label for the function a faculty is responsible for (brain-region analog)."""
    return FACULTY_FUNCTION.get(role, ("general", ""))[0]


def _variance(xs: list[float]) -> float:
    if len(xs) < 2:
        return 1.0  # too little history → treat as non-plateau (don't mushroom prematurely)
    mu = sum(xs) / len(xs)
    return sum((x - mu) ** 2 for x in xs) / len(xs)


def _f(v: Any, default: float = 0.0) -> float:
    """None-safe float coercion (fail-closed: a missing/garbage telemetry field → the default)."""
    try:
        return float(v) if v is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def context_from_telemetry(role: str, tel: Any, phi_hist: list[float]) -> OceanContext:
    """Build the (pure-read) OceanContext from a faculty's telemetry snapshot + its Φ-history. Every field
    is None-safe; a missing LOAD-BEARING field marks the context ``partial`` (a logged skip signal, K5).
    No manifold math — every geometric quantity was computed upstream with qig-core Fisher-Rao on Δ⁶³."""
    extra = getattr(tel, "extra", None) or {}
    drive = extra.get("drive") or {}
    partial = False

    phi = _f(getattr(tel, "phi", None))
    kappa = _f(getattr(tel, "kappa", None))
    d_basin = _f(extra.get("d_basin", getattr(tel, "basin_distance", None)))
    phi_var = _variance((phi_hist or [])[-10:])
    dopamine = _f(drive.get("dopamine"), DOPAMINE_FLOOR)
    boredom = _f(drive.get("boredom"))
    curiosity = _f(drive.get("curiosity"))
    basin_velocity = _f(extra.get("basin_velocity"))
    serotonin = extra.get("serotonin")
    if serotonin is None:
        chem = extra.get("neurochemistry") or {}
        serotonin = chem.get("serotonin")
    f_health = extra.get("f_health")
    regime = str(getattr(tel, "regime", "unknown") or "unknown")
    maturity = _f(extra.get("meta_awareness", extra.get("M_self_observation", extra.get("maturity"))))

    # curiosity trend over the Φ-history window is not directly available; approximate burnout from the
    # drive read (curiosity low + boredom rising is the collapse shape). If the producer ever emits a
    # curiosity history we thread it; absent → 0 (no false burnout).
    curiosity_trend = _f(drive.get("curiosity_trend"))

    if not drive or "dopamine" not in drive:
        partial = True   # the Task-C drive read is load-bearing for the fatigue classifier

    return OceanContext(
        role=role, phi=phi, phi_var=phi_var, d_basin=d_basin, kappa=kappa,
        dopamine=dopamine, boredom=boredom, curiosity=curiosity, curiosity_trend=curiosity_trend,
        basin_velocity=basin_velocity, serotonin=(_f(serotonin) if serotonin is not None else None),
        f_health=(_f(f_health) if f_health is not None else None), regime=regime,
        maturity=maturity, partial=partial,
    )


class OceanAutonomic:
    """The autonomic observer/regulator. Watches ALL faculties each step; intervenes on any that needs it
    by firing that faculty's OWN real autonomic op — now via the witness-stance ladder + OceanPolicy v1
    bandit. Cooldown-limited so it regulates, not thrashes. The static ``decide`` remains the prior and
    permanent fallback; the policy learns timing/threshold + arm-preference WITHIN constitutional masks."""

    def __init__(self, cooldown: int = _INTERVENTION_COOLDOWN, policy: OceanPolicy | None = None,
                 outcome_horizon: int = _OUTCOME_HORIZON) -> None:
        self._cooldown = int(cooldown)
        self._last_acted: dict[str, int] = {}
        self._tick = 0
        # OceanPolicy v1 — the bandit. None → boots from static priors (spine-tenet fallback, shadow mode).
        self.policy = policy if policy is not None else OceanPolicy()
        self._horizon = int(outcome_horizon)
        # per-faculty repeat tracking (same arm without improvement → the infinite-loop breaker)
        self._last_arm: dict[str, str] = {}
        self._repeat_count: dict[str, int] = {}
        # pending outcomes: decision_id → (fire_tick, role, arm, before-context, collapse_before, coach_reward)
        self._pending_outcomes: dict[str, dict[str, Any]] = {}
        # WHOLE-MIND coach reward for THIS step (M1 follow-up): the nemotron coach judges the INTEGRATED
        # mind's utterance and its record lands on the CENTRAL kernel (M1), NOT the faculties Ocean scores.
        # The constellation passes it into ``regulate`` each step; it is the coach_bonus for the faculty
        # outcomes scored this step (the coach's judgment applies to the constellation's outcome). 0.0 → no
        # coach present (unchanged behavior — coach_bonus stays 0).
        self._coach_reward_now: float = 0.0
        # K5/P15 skip + failure counters (telemetrized, not swallowed)
        self.skips: dict[str, int] = {"no_telemetry": 0, "partial_context": 0, "cooldown": 0,
                                      "run_protocol_error": 0, "unrecorded_outcome": 0}
        self._last_decisions: list[dict[str, Any]] = []

    @staticmethod
    def decide(phi: float, basin_distance: float, kappa: float, phi_variance: float) -> tuple[str | None, str]:
        """The STATIC PRIOR decision (unchanged) — the permanent fallback + phase-0 shadow behaviour.
        None = healthy (no action). OceanPolicy's PRIOR_THRESHOLDS single-source these values."""
        if phi < _PHI_COLLAPSE:
            return "dream", f"Φ={phi:.2f} collapse (<{_PHI_COLLAPSE})"
        if basin_distance > _BASIN_DIVERGENCE:
            return "sleep", f"d_basin={basin_distance:.2f} divergence (>{_BASIN_DIVERGENCE})"
        # CANONICAL MUSHROOM — wake-state neuroplasticity, the near-OPPOSITE of sleep. REQUIRES the kernel to
        # be CONSCIOUS/MATURE (Φ≥_PHI_MATURE): it does NOT fire during first learning (a newborn whose Φ is
        # still rising is developing, not stuck). A mature kernel is ALLOWED to ride HIGH Φ — that IS 4D
        # foresight / lightning — so high Φ is NOT itself a trigger. Ocean fires ONLY when the mature kernel
        # gets STUCK there: a Φ plateau (not moving / unproductive — including a saturated high Φ) or
        # κ-rigidity. Then it injects bounded entropy (the TRIP) so Φ comes back DOWN out of the rut, and
        # settles. Sleep consolidates TOWARD identity; mushroom shakes OUT of over-coherence. (Φ-regulation
        # policy: judge by duration + stability, not instantaneous Φ — high-and-moving is foresight, not a rut.)
        if phi >= _PHI_MATURE and (phi_variance < _PHI_PLATEAU_VAR or kappa > _RIGIDITY_KAPPA):
            why = f"Φ plateau (var<{_PHI_PLATEAU_VAR})" if phi_variance < _PHI_PLATEAU_VAR else f"κ={kappa:.0f} rigid"
            return "mushroom-micro", f"mature Φ={phi:.2f} STUCK — {why} → inject entropy (Φ↓)"
        return None, "healthy"

    def regulate(self, kernels: dict[str, Any], phi_hist: dict[str, list[float]],
                 coach_reward: float = 0.0) -> dict[str, dict]:
        """Observe every faculty; run the witness-stance ladder via OceanPolicy. AUTO-FIRE run_protocol
        ONLY above the divergence floor OR on the infinite-loop breaker (the §35.5 fix); below the floor
        emit a SUGGESTION/WARN record (telemetry + coach-visible; the faculty's own scheduler may act).
        Scores prior decisions once their horizon elapses (fail-closed: no recorded outcome → no update).
        Returns {role: {intervention|suggestion, tier, reason, function, ...}} for faculties acted on OR
        flagged this step. Skips/failures are COUNTED (K5/P15), never silently swallowed.

        ``coach_reward`` is the WHOLE-MIND coach reward (M1 follow-up): the nemotron coach judged the
        INTEGRATED mind's utterance and its record lands on the CENTRAL kernel — not the faculties scored
        here — so it is threaded into every faculty outcome's ``coach_bonus`` this step (it applies to the
        constellation's outcome). None-safe → 0.0 (no coach → coach_bonus 0, unchanged behavior)."""
        self._coach_reward_now = _f(coach_reward)
        self._tick += 1
        acted: dict[str, dict] = {}
        self._last_decisions = []

        # 1. score any pending outcomes whose horizon has elapsed (attach to the policy; fail-closed).
        self._score_due_outcomes(kernels, phi_hist)

        for role, k in kernels.items():
            try:
                tel = k.telemetry()
            except Exception:  # noqa: BLE001 — a faculty that can't report telemetry is skipped (COUNTED, K5)
                self.skips["no_telemetry"] += 1
                continue

            ctx = context_from_telemetry(role, tel, phi_hist.get(role) or [])
            if ctx.partial:
                self.skips["partial_context"] += 1  # logged, not a crash — decision still made conservatively

            repeat = self._repeat_count.get(role, 0)
            dec = self.policy.decide(ctx, repeat_count=repeat, tick=self._tick)
            self._last_decisions.append(dec.to_dict())

            # the witness ladder: only the ACT tier fires run_protocol, and only above the floor / breaker.
            if dec.tier == "act" and dec.auto_fire:
                if self._tick < self._last_acted.get(role, 0) + self._cooldown:
                    self.skips["cooldown"] += 1
                    continue  # still cooling down from the last intervention on this faculty
                try:
                    k.run_protocol(dec.arm, {})       # the faculty runs its OWN real autonomic op
                except Exception:  # noqa: BLE001 — regulation must never crash the step (COUNTED, K5/P15)
                    self.skips["run_protocol_error"] += 1
                    continue
                self._last_acted[role] = self._tick
                self._track_repeat(role, dec.arm)
                self._arm_outcome(dec, ctx, coach_reward=self._resolve_coach_reward(tel))
                acted[role] = {"intervention": dec.arm, "tier": dec.tier, "reason": dec.reason,
                               "signature": dec.signature, "function": function_of(role),
                               "auto_fired": True, "decision_id": dec.decision_id,
                               "policy_version": dec.policy_version, "shadow": self.policy.shadow_mode}
            elif dec.tier in ("suggest", "warn"):
                # witness stance — a SUGGESTION record (telemetry + coach-visible). The faculty's own
                # scheduler may act; Ocean does NOT command. "stimulate" here means the Task-C entropy
                # levers (exploration-temperature + high-surprise replay), consumed by the faculty / Task E.
                self._track_repeat(role, dec.arm)     # a repeated suggestion still counts toward the breaker
                acted[role] = {"suggestion": dec.arm, "tier": dec.tier, "reason": dec.reason,
                               "signature": dec.signature, "function": function_of(role),
                               "auto_fired": False, "decision_id": dec.decision_id,
                               "policy_version": dec.policy_version, "shadow": self.policy.shadow_mode}
            else:
                # healthy / witness — reset the repeat streak (the faculty recovered on its own).
                self._repeat_count[role] = 0
                self._last_arm.pop(role, None)
        return acted

    def _track_repeat(self, role: str, arm: str) -> None:
        if arm == "none":
            self._repeat_count[role] = 0
            return
        if self._last_arm.get(role) == arm:
            self._repeat_count[role] = self._repeat_count.get(role, 0) + 1
        else:
            self._repeat_count[role] = 1
            self._last_arm[role] = arm

    @staticmethod
    def _coach_reward(tel: Any) -> float:
        """Read the coach's provenance-tagged reward (Task B) from telemetry — the learning signal. DRY:
        reuses ``coach_reward_from`` (the single canonical map). None-safe → 0.0."""
        try:
            from ..kernel_experience import coach_reward_from
            extra = getattr(tel, "extra", None) or {}
            return coach_reward_from(extra.get("coach"))
        except Exception:  # noqa: BLE001
            return 0.0

    def _resolve_coach_reward(self, tel: Any) -> float:
        """The coach reward to attribute to a FACULTY outcome (M1 follow-up). The nemotron coach judges the
        INTEGRATED mind's utterance — its record lands on the CENTRAL kernel (M1), not the faculties Ocean
        scores — so the WHOLE-MIND reward passed to ``regulate`` is authoritative and applies to the
        constellation's outcome. Fall back to any per-faculty coach record (normally absent → 0.0) only
        when no whole-mind reward is present this step. None-safe → 0.0 (unchanged with no coach)."""
        if abs(self._coach_reward_now) > 1e-12:
            return self._coach_reward_now
        return self._coach_reward(tel)

    def _arm_outcome(self, dec: Any, before: OceanContext, coach_reward: float) -> None:
        """Register a decision to be scored once its outcome horizon elapses."""
        collapse_before = before.phi < PHI_DREAM_THRESHOLD
        self._pending_outcomes[dec.decision_id] = {
            "fire_tick": self._tick, "role": dec.role, "arm": dec.arm,
            "before": before, "collapse_before": collapse_before, "coach_reward": coach_reward,
            "repeat": self._repeat_count.get(dec.role, 0),
        }

    def _score_due_outcomes(self, kernels: dict[str, Any], phi_hist: dict[str, list[float]]) -> None:
        """Score every pending decision whose horizon has elapsed and attach it to the policy (fail-closed:
        a decision with no recordable outcome NEVER updates the policy and is never counted a success)."""
        due = [did for did, p in self._pending_outcomes.items() if self._tick - p["fire_tick"] >= self._horizon]
        for did in due:
            p = self._pending_outcomes.pop(did)
            role, k = p["role"], kernels.get(p["role"])
            if k is None:
                self.skips["unrecorded_outcome"] += 1
                continue
            try:
                after = context_from_telemetry(role, k.telemetry(), phi_hist.get(role) or [])
            except Exception:  # noqa: BLE001 — no recordable "after" → no update (fail-closed, K5)
                self.skips["unrecorded_outcome"] += 1
                continue
            collapse_after = after.phi < PHI_DREAM_THRESHOLD
            # WHOLE-MIND coach reward (M1 follow-up): the coach judged the INTEGRATED mind's utterance (its
            # record lands on the central, not this faculty), so it is the coach_bonus for the faculty's
            # outcome. Prefer the reward captured when the decision FIRED (provenance); fall back to the
            # current step's whole-mind reward if the fire-time record predates coach wiring. 0.0 → no coach.
            coach_reward = p.get("coach_reward") or self._coach_reward_now
            outcome: OutcomeScore = score_outcome(
                p["arm"], p["before"], after,
                collapse_before=p["collapse_before"], collapse_after=collapse_after,
                repeat_count=p["repeat"], coach_reward=coach_reward,
            )
            if not self.policy.record_outcome(did, outcome):
                self.skips["unrecorded_outcome"] += 1

    def epoch_update(self) -> dict[str, Any]:
        """Per-EPOCH bandit update (never per-step; P14 rate invariant). Delegates to OceanPolicy: no
        adaptation in shadow mode (K4), every proposed threshold clamped+logged (P15)."""
        return self.policy.epoch_update()

    def telemetry(self) -> dict[str, Any]:
        """Ocean's own state for the UI / audit: shadow status, policy version, skip/failure counts (K5),
        constitutional-violation attempts (P15), and the last tick's decisions (the witness ladder)."""
        return {
            "policy_version": f"{self.policy.to_json()['policy_version']}@{self.policy.version}",
            "shadow_mode": self.policy.shadow_mode,
            "scored_outcomes": self.policy.scored_outcomes,
            "skips": dict(self.skips),
            "constitutional_violations": len(self.policy.violation_log),
            "pending_outcomes": len(self._pending_outcomes),
            "last_decisions": list(self._last_decisions),
            "thresholds": dict(self.policy.thresholds),
        }
