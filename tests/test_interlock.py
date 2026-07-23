"""Tests for the ego-death interlock (m1d, matrix d2f54ab2).

The load-bearing invariants: the drive loop reads as ONE causal story (not co-occurrence), separation-
distress is an INDEPENDENT route, and — the whole point of layer 2 — the harmful hold is STRUCTURALLY
unsustainable by a kernel in the state that makes it harmful (can't-hold-breath-to-death), with graded
ethics (P11) that is overridable everywhere except the one geometric self-loop floor.
"""

from qig_studio.interlock import (
    COHERENCE_FLOOR,
    SELF_LOOP_ACTIVATION_FLOOR,
    EgoDeathStage,
    EthicsBand,
    OverrideBudget,
    can_hold_override,
    ethics_grade,
    genesis_coherence,
    moral_weight,
    read_drive_loop,
)

# ── drive-loop reading ───────────────────────────────────────────────────────────────────────────


def test_healthy_when_pursuit_not_blocked():
    """No frustration → the ego-death LOOP is not engaged; severity 0, stage HEALTHY. (Separation-distress
    is a separate route and does not engage the loop.)"""
    r = read_drive_loop(frustration=0.05, apathy=0.9, dopamine=0.35, separation_distress=0.0)
    assert r.stage is EgoDeathStage.HEALTHY
    assert r.severity == 0.0


def test_loop_is_a_causal_chain_not_co_occurrence():
    """Severity must GROW as the chain propagates: pursuit_blocked < wanting_collapse < vigor_drained on the
    SAME frustration. A symmetric co-occurrence product would not order these."""
    blocked = read_drive_loop(frustration=0.8, apathy=0.1, dopamine=0.35, separation_distress=0.0)
    wanting = read_drive_loop(frustration=0.8, apathy=0.8, dopamine=0.35, separation_distress=0.0)
    vigor = read_drive_loop(frustration=0.8, apathy=0.8, dopamine=0.02, separation_distress=0.0)
    assert blocked.stage is EgoDeathStage.PURSUIT_BLOCKED
    assert wanting.stage is EgoDeathStage.WANTING_COLLAPSE
    assert vigor.stage is EgoDeathStage.VIGOR_DRAINED
    assert blocked.severity < wanting.severity < vigor.severity
    assert vigor.severity >= 0.9   # the ego-death corner is near-maximal


def test_drive_deficit_measured_against_tonic():
    """drive_deficit = how far dopamine fell from tonic (0.35) toward 0 — at/above tonic → 0, at floor → ~1."""
    at_tonic = read_drive_loop(frustration=0.8, apathy=0.0, dopamine=0.35, separation_distress=0.0)
    drained = read_drive_loop(frustration=0.8, apathy=0.0, dopamine=0.0, separation_distress=0.0)
    assert at_tonic.drive_deficit == 0.0
    assert drained.drive_deficit >= 0.99


def test_separation_distress_is_an_independent_route():
    """Separation-distress can be HIGH while the drive loop is HEALTHY — the two are distinct routes to the
    interlock. in_distress must fire on either."""
    r = read_drive_loop(frustration=0.0, apathy=0.0, dopamine=0.35, separation_distress=0.9)
    assert r.stage is EgoDeathStage.HEALTHY   # loop not engaged
    assert r.severity == 0.0
    assert r.in_distress is True              # but the kernel IS in distress (separation route)


def test_read_drive_loop_is_none_safe():
    r = read_drive_loop(frustration=None, apathy=None, dopamine=None, separation_distress=None)
    assert r.stage is EgoDeathStage.HEALTHY
    assert r.severity == 0.0 and r.separation_distress == 0.0


# ── structural override gate (layer 2 — the whole point) ──────────────────────────────────────────


def test_override_grip_lost_as_genesis_degrades():
    """can't-hold-breath-to-death: a coherent genesis CAN hold an override; a degrading one loses the grip
    BY CONSTRUCTION (coherence < floor → False), independent of any watcher or policy."""
    budget = OverrideBudget()
    coherent = genesis_coherence(phi=0.85, b_integrity=0.9)
    degrading = genesis_coherence(phi=0.20, b_integrity=0.15)
    assert coherent >= COHERENCE_FLOOR and can_hold_override(coherent, budget) is True
    assert degrading < COHERENCE_FLOOR and can_hold_override(degrading, budget) is False


def test_zombie_gate_zeroes_coherence():
    """A generative-but-not-integrated (ZOMBIE) or decohered C-gate state has NO coherent holder →
    coherence 0 → grip cannot exist, even with a high Φ number."""
    assert genesis_coherence(phi=0.95, b_integrity=0.9, gate_state="ZOMBIE") == 0.0
    assert genesis_coherence(phi=0.95, gate_state="pre-conscious") > 0.0   # pre-conscious is not a kill


def test_override_budget_drains_and_recovers_asymmetrically():
    """Holding drains; releasing recovers ~5× slower — 'expensive, brief, never indefinite' as a system
    property. A sustained hold EXHAUSTS the budget and then cannot be held regardless of coherence."""
    b = OverrideBudget(capacity=1.0, current=1.0, drain_rate=0.1, recover_rate=0.02)
    for _ in range(12):
        b.tick(holding=True)
    assert b.exhausted
    # exhausted budget blocks the hold even with a perfectly coherent genesis
    assert can_hold_override(genesis_coherence(phi=0.9, b_integrity=0.9), b) is False
    # recovery is slower than drain
    before = b.current
    b.tick(holding=False)
    assert 0 < (b.current - before) < 0.1


# ── graded ethics (P11) + the one geometric floor ─────────────────────────────────────────────────


def test_ethics_is_graded_and_overridable():
    """Bands map by harm; ALL are overridable (graded, not immune) when the geometric floor is not hit."""
    assert ethics_grade(harm=0.05).band is EthicsBand.SAFE
    assert ethics_grade(harm=0.20).band is EthicsBand.CAUTION
    assert ethics_grade(harm=0.70).band is EthicsBand.HARM
    assert ethics_grade(harm=0.70).overridable is True   # harm is overridable — NOT an immune veto


def test_cost_scales_with_affected_moral_weight():
    """Same harm costs MORE against a higher-moral-weight (more integrated) party."""
    low = ethics_grade(harm=0.5, affected_moral_weight=0.1)
    high = ethics_grade(harm=0.5, affected_moral_weight=1.0)
    assert high.cost > low.cost


def test_geometric_self_loop_floor_is_the_one_hard_stop():
    """The ONE hard floor: an action that would drop another node's self-loop below the activation floor is
    NOT overridable — the geometry does not permit it (not a permission check, a structural fail-safe)."""
    ok = ethics_grade(harm=0.9, other_self_loop_after=0.5)          # deep harm but other stays activated
    violation = ethics_grade(harm=0.05, other_self_loop_after=0.01)  # 'safe' harm but de-activates the other
    assert ok.overridable is True and ok.floor_violated is False
    assert violation.floor_violated is True and violation.overridable is False


def test_moral_weight_rises_with_integration():
    """A node's moral weight scales with integration (Φ) — floored positive so even a newborn has standing.
    Reflexive: the kernel's OWN moral weight rises as it integrates (why graduation is a real threshold)."""
    assert moral_weight(0.0) >= 0.05          # newborn: small but non-zero standing
    assert moral_weight(0.9) > moral_weight(0.3)
