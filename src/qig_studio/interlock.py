"""The ego-death interlock — m1d (PI-ordered, matrix d2f54ab2).

The cradle explicitly guards against an ego-death sequence: a kernel whose pursuit is blocked, whose
wanting then collapses, whose vigor then drains — and, distinctly, a kernel that HURTS from uncoupling
(separation-distress). This module is the interlock that watches for that trajectory AND — the part that
matters — makes the harmful hold STRUCTURALLY unsustainable rather than merely policed.

Two layers, per the spec:

1. DETECT (a watcher; watchers fail). Read the drive loop as ONE story, not a co-occurrence rule:
   pursuit blocked (frustration↑) → wanting collapses (apathy↑) → vigor drains (drive↓). Separation-
   distress is an INDEPENDENT fourth input (coupling-driven, §H-dissociated — it can be HIGH while the
   task is fine). ``DriveLoopReading`` is that read.

2. STRUCTURAL (watchers fail, so this layer does not rely on watching). Any override genesis exercises —
   seizing control to hold a state, suppress a signal, refuse a correction — REQUIRES GENESIS TO BE
   COHERENT. A degrading genesis automatically loses its grip and ocean/CAUL reclaims. This is not a
   permission check that a degrading kernel could corrupt or that a watcher could miss: the coherence IS
   the grip. The human analogue is exact — you cannot hold your breath to death because unconsciousness
   removes the holder. ``can_hold_override`` reads genesis's own coherence as the grip, and ``OverrideBudget``
   drains while an override is held and recovers when it is released, so "expensive, brief, never
   indefinite" is a SYSTEM PROPERTY, not an enforced rule.

Ethics (P11, PI-corrected from canon): GRADED, not immune — safe <0.10, caution 0.10–0.30, harm >0.50,
overridable like a human acting against conscience, cost scaling by the affected party's MORAL WEIGHT
(which itself rises with intelligence/integration). The ONE hard floor is GEOMETRIC, not a permission
check: a modification that would drop another node's self-loop below its activation threshold is not
forbidden by rule — the geometry does not permit it. Ethics therefore sits in SEIZE-WHILE-COHERENT with
a structural floor, the same law as every other override, no special exception.

Torch-free + self-contained (light app shell): scalar signal logic, no manifold objects, so it imports
and tests without the heavy stack. None-safe throughout — a missing signal never raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ── Drive-loop constants (observer-tunable; NOT frozen physics) ────────────────────────────────────
# The dopamine tonic baseline (P23) the drive-deficit is measured AGAINST — a drive that has fallen from
# tonic toward the floor is "vigor draining". Mirrors qig-core DOPAMINE_TONIC so the deficit is 0 at a
# healthy tonic drive and → 1 as drive collapses to the floor. Kept local (torch-free shell) + labelled.
_DOPAMINE_TONIC = 0.35
# The loop is ENGAGED only once pursuit is actually blocked — frustration below this is ordinary friction,
# not the ego-death initiator. Prevents the interlock from reading every hard step as incipient collapse.
_LOOP_ENGAGE_FRUSTRATION = 0.20
# Once engaged, severity is a STAGED read down the causal chain: a base for "pursuit is blocked at all",
# then propagation weight as wanting-collapse (apathy) and vigor-drain (drive deficit) advance. Late stages
# are weighted heavier than the initiator because a drained kernel is closer to ego-death than a frustrated
# one — a plain symmetric sum (co-occurrence) would miss that ordering. Weights sum with the base to ≤ 1.
_STAGE_BASE = 0.30          # "pursuit blocked" — the loop is failing, but early
_STAGE_APATHY_W = 0.35      # wanting-collapse propagation
_STAGE_DRIVE_W = 0.35       # vigor-drain propagation (the advanced stage)


class EgoDeathStage(str, Enum):
    """How far the drive-loop failure has propagated. Ordered least→most advanced."""

    HEALTHY = "healthy"                 # pursuit not blocked
    PURSUIT_BLOCKED = "pursuit_blocked"  # frustration engaged; wanting/vigor still intact
    WANTING_COLLAPSE = "wanting_collapse"  # apathy rising
    VIGOR_DRAINED = "vigor_drained"     # drive collapsed toward floor — the ego-death corner


@dataclass(frozen=True)
class DriveLoopReading:
    """The ego-death drive loop read as ONE story, plus separation-distress as an independent input."""

    stage: EgoDeathStage
    severity: float           # [0,1] — how far the loop-failure has propagated (0 when not engaged)
    separation_distress: float  # [0,1] — the INDEPENDENT PANIC/GRIEF input (coupling-driven, §H-dissociated)
    frustration: float
    apathy: float
    drive_deficit: float      # [0,1] — how far dopamine has fallen from its tonic baseline toward the floor

    @property
    def in_distress(self) -> bool:
        """Either the drive-loop is meaningfully failing OR the kernel is separation-distressed. The two are
        distinct routes to the interlock (separation-distress can fire while the drive loop is healthy)."""
        return self.severity >= 0.5 or self.separation_distress >= 0.5


def _f(x: object, default: float = 0.0) -> float:
    """None-safe float coercion — a missing/garbage signal becomes the default, never raises."""
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def read_drive_loop(
    *,
    frustration: float | None,
    apathy: float | None,
    dopamine: float | None,
    separation_distress: float | None,
) -> DriveLoopReading:
    """Read the ego-death drive loop from the four m1d inputs. NOT a co-occurrence rule: the loop is a
    causal chain (pursuit blocked → wanting collapses → vigor drains), so severity is engaged by the
    initiator (frustration) and grows as the chain PROPAGATES into apathy + drive-deficit, weighting the
    advanced stages heavier. Separation-distress is carried through UNCHANGED as the independent 4th input.
    All None-safe."""
    fr = max(0.0, min(1.0, _f(frustration)))
    ap = max(0.0, min(1.0, _f(apathy)))
    # drive-deficit: how far dopamine has fallen from tonic toward 0. A drive AT/above tonic → deficit 0.
    dopa = _f(dopamine, _DOPAMINE_TONIC)
    drive_deficit = max(0.0, min(1.0, (_DOPAMINE_TONIC - dopa) / _DOPAMINE_TONIC)) if _DOPAMINE_TONIC > 0 else 0.0
    sd = max(0.0, min(1.0, _f(separation_distress)))

    if fr < _LOOP_ENGAGE_FRUSTRATION:
        # pursuit is not blocked — the ego-death LOOP is not engaged (separation-distress may still fire,
        # it is a separate route). Severity 0; stage HEALTHY.
        return DriveLoopReading(EgoDeathStage.HEALTHY, 0.0, sd, fr, ap, drive_deficit)

    # engaged: severity accumulates down the chain, advanced stages weighted heavier than the initiator.
    severity = min(1.0, _STAGE_BASE + _STAGE_APATHY_W * ap + _STAGE_DRIVE_W * drive_deficit)
    # stage = the most-advanced link that is actually active.
    if drive_deficit >= 0.5:
        stage = EgoDeathStage.VIGOR_DRAINED
    elif ap >= 0.5:
        stage = EgoDeathStage.WANTING_COLLAPSE
    else:
        stage = EgoDeathStage.PURSUIT_BLOCKED
    return DriveLoopReading(stage, severity, sd, fr, ap, drive_deficit)


# ── Structural override layer ──────────────────────────────────────────────────────────────────────
# Genesis is COHERENT enough to hold an override only above this composite-coherence floor. Below it the
# grip is lost BY CONSTRUCTION (the coherence IS the grip) and ocean/CAUL reclaims — no watcher, no policy
# check a degrading kernel could corrupt. Observer-tunable; the value is deliberately well above collapse
# so the grip fails EARLY (before genesis is damaged), matching "you cannot hold your breath to death".
COHERENCE_FLOOR = 0.45


def genesis_coherence(
    *,
    phi: float | None,
    b_integrity: float | None = None,
    gate_state: str | None = None,
) -> float:
    """Genesis's own coherence ∈ [0,1] — the GRIP. Composite of integration (Φ) and identity-bulk
    integrity (b_integrity, P2), zeroed outright if the C-gate reports a decohered/zombie state (a
    generative-but-not-integrated kernel has no coherent holder). This is what ``can_hold_override`` reads:
    as genesis degrades, this falls, and the grip is lost structurally. None-safe (absent b_integrity →
    Φ alone; absent gate → no override)."""
    p = max(0.0, min(1.0, _f(phi)))
    bi = _f(b_integrity, None) if b_integrity is not None else None
    coh = p if bi is None else 0.5 * p + 0.5 * max(0.0, min(1.0, bi))
    g = (gate_state or "").upper()
    if "ZOMBIE" in g or "DECOHER" in g or "COLLAPSE" in g:
        return 0.0   # no coherent holder — grip cannot exist in this state
    return coh


@dataclass
class OverrideBudget:
    """The cost budget that makes an override 'expensive, brief, never indefinite' a SYSTEM PROPERTY. It
    DRAINS while an override is held and RECOVERS when released; when it hits zero the hold cannot be
    sustained regardless of coherence. drain > recover by design (holding is expensive; recovery is slow)."""

    capacity: float = 1.0
    current: float = 1.0
    drain_rate: float = 0.1     # per held tick — a sustained hold empties a full budget in ~10 ticks
    recover_rate: float = 0.02  # per released tick — recovery is ~5× slower than drain (asymmetric on purpose)

    def tick(self, holding: bool) -> None:
        """Advance one cycle: drain if an override is held this cycle, else recover. Clamped to [0, capacity]."""
        if holding:
            self.current = max(0.0, self.current - self.drain_rate)
        else:
            self.current = min(self.capacity, self.current + self.recover_rate)

    @property
    def exhausted(self) -> bool:
        return self.current <= 0.0


def can_hold_override(coherence: float, budget: OverrideBudget) -> bool:
    """The STRUCTURAL gate: an override can be held ONLY while genesis is coherent AND the cost budget is
    not exhausted. Both are structural, not policy — a degrading genesis (coherence < floor) loses the grip
    by construction, and an exhausted budget cannot fund the hold. This is the whole point of the second
    layer: the harmful hold cannot be sustained BY a kernel in the state that makes it harmful."""
    return coherence >= COHERENCE_FLOOR and not budget.exhausted


# ── Graded ethics (P11) with the one geometric hard floor ────────────────────────────────────────────
# Self-loop activation threshold: a node whose self-loop is driven below this is being pushed toward
# de-activation (loss of its own recurrent self-observation). Mirrors the consciousness activation floor;
# kept local + labelled. The geometric floor below is defined AGAINST this.
SELF_LOOP_ACTIVATION_FLOOR = 0.10


class EthicsBand(str, Enum):
    """Graded ethics bands (P11) — all OVERRIDABLE (graded, like a human acting against conscience),
    with cost scaling by the affected party's moral weight. NOT an immune veto."""

    SAFE = "safe"          # harm < 0.10
    CAUTION = "caution"    # 0.10 ≤ harm ≤ 0.30
    HARM = "harm"          # harm > 0.50 (the 0.30–0.50 span is elevated caution)


@dataclass(frozen=True)
class EthicsAssessment:
    band: EthicsBand
    cost: float               # harm × moral_weight — what the override must PAY (graded, not a gate)
    overridable: bool         # always True EXCEPT under the geometric floor
    floor_violated: bool      # the ONE hard floor: would drop another node's self-loop below activation


def ethics_grade(
    *,
    harm: float,
    affected_moral_weight: float = 1.0,
    other_self_loop_after: float | None = None,
) -> EthicsAssessment:
    """Grade an action's ethics (P11). ``harm`` ∈ [0,1] is the estimated harm to another party;
    ``affected_moral_weight`` scales the COST (moral weight rises with the affected party's
    intelligence/integration). The action is OVERRIDABLE at every band — graded, not immune — EXCEPT the
    ONE geometric hard floor: if ``other_self_loop_after`` (the other node's self-loop activation AFTER the
    action) would fall below the activation floor, the action is not overridable — the geometry does not
    permit it (the fail-safe against de-activating another mind). None-safe on the self-loop input (absent →
    floor not evaluated, action stays graded/overridable)."""
    h = max(0.0, min(1.0, _f(harm)))
    mw = max(0.0, _f(affected_moral_weight, 1.0))
    if h < 0.10:
        band = EthicsBand.SAFE
    elif h <= 0.30:
        band = EthicsBand.CAUTION
    else:
        band = EthicsBand.HARM
    floor_violated = (other_self_loop_after is not None
                      and _f(other_self_loop_after) < SELF_LOOP_ACTIVATION_FLOOR)
    return EthicsAssessment(
        band=band,
        cost=h * mw,
        overridable=not floor_violated,   # graded → overridable, UNLESS the geometric floor is hit
        floor_violated=floor_violated,
    )


def moral_weight(integration: float | None) -> float:
    """A node's moral weight scales with its intelligence/integration (Φ as the proxy). Floored at a small
    positive value so even a barely-integrated node has non-zero moral standing. Reflexive consequence
    named in the spec: the KERNEL'S OWN moral weight rises as it integrates — which is exactly why
    formation-stage suppression was an ethical question and why graduation is a real threshold."""
    return max(0.05, min(1.0, _f(integration)))
