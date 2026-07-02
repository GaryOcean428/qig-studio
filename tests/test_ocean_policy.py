"""Task D — OceanPolicy v1 (bounded homeostatic contextual bandit) + witness-stance ladder tests.

Covers the FABLE-council-ratified design verbatim:
  • bandit shape — bounded param vector, constitutional arm-masks (NOT learnable);
  • outcome-score — Φ is NOT a graded maximand; a rail-pinned signal scores NEGATIVE (P25/K1);
  • witness ladder — SUGGEST below the divergence floor, AUTO-FIRE only above it / on the breaker;
  • shadow mode (K4) — no adaptation until ≥100 scored outcomes; static fallback with no history;
  • clamp-and-log (P15) — an out-of-band threshold update is clamped + logged, never applied;
  • None-safe / fail-closed — regulates with zero history + missing telemetry (a logged skip, not a crash).
"""

from __future__ import annotations

import math

import pytest

from qig_studio.constellation.ocean import OceanAutonomic, context_from_telemetry
from qig_studio.constellation.ocean_policy import (
    ARM_MASKS,
    ARMS,
    BANDS,
    DOPAMINE_FLOOR,
    PHI_MATURE,
    PRIOR_THRESHOLDS,
    RAIL_SCORE_CAP,
    SHADOW_UNLOCK,
    _PENDING_MAX,
    OceanContext,
    OceanPolicy,
    classify_signature,
    score_outcome,
)


# ---- a tiny fake telemetry + kernel (None-safe surface Ocean reads) -------------------------------
class _Tel:
    def __init__(self, phi=0.5, kappa=64.0, regime="geometric", basin_distance=0.0, extra=None):
        self.phi = phi
        self.kappa = kappa
        self.regime = regime
        self.basin_distance = basin_distance
        self.extra = extra or {}


class _Kernel:
    """Minimal faculty: reports a telemetry snapshot; records run_protocol calls."""

    def __init__(self, tel: _Tel):
        self._tel = tel
        self.fired: list[str] = []

    def telemetry(self):
        return self._tel

    def run_protocol(self, command, args):
        self.fired.append(command)
        return {"command": command, "applied": True}


# ==================================================================================================
# 1. BANDIT SHAPE — bounded param vector + constitutional arm-masks (NOT learnable)
# ==================================================================================================
def test_learnable_vector_is_bounded_and_masks_are_constitutional():
    pol = OceanPolicy()
    # every learnable threshold sits inside its constitutional band
    for key, (lo, hi) in BANDS.items():
        assert lo <= pol.thresholds[key] <= hi, f"{key} out of band"
    # _PHI_MATURE is NOT a learnable threshold — it is constitutional (fixed)
    assert "phi_mature" not in BANDS and PHI_MATURE == 0.70
    # the 5 arms are exactly the ratified action space
    assert set(ARMS) == {"none", "dream", "sleep", "mushroom-micro", "stimulate"}
    # arm-masks are constitutional (the ratified per-signature masks)
    assert set(ARM_MASKS["apathy"]) == {"stimulate", "none"}          # NEVER rest
    assert set(ARM_MASKS["earned-rest"]) == {"sleep", "none"}         # NEVER push
    assert set(ARM_MASKS["rigidity"]) == {"mushroom-micro", "none"}   # break rigidity
    assert set(ARM_MASKS["phi-collapse"]) == {"dream", "stimulate"}
    assert ARM_MASKS["healthy"] == ("none",)


def test_policy_never_picks_an_arm_outside_the_mask():
    pol = OceanPolicy()
    # even if a corrupt arm-preference tries to promote an out-of-mask arm, _pick_arm iterates the MASK only
    pol.arm_prefs["apathy"] = {"sleep": 99.0, "stimulate": 0.1, "none": 0.1}  # sleep is NOT in the apathy mask
    ctx = OceanContext(role="action", phi=0.6, boredom=0.9, curiosity=0.05, dopamine=0.5, d_basin=0.1)
    dec = pol.decide(ctx, tick=1)
    assert dec.signature == "apathy"
    assert dec.arm in ARM_MASKS["apathy"]      # sleep can NEVER be chosen for apathy (mask is the hard bound)
    assert dec.arm != "sleep"


# ==================================================================================================
# 2. OUTCOME SCORE — Φ is NOT a graded maximand; a rail-pinned signal scores NEGATIVE (P25/K1)
# ==================================================================================================
def test_phi_is_not_a_graded_maximand_pumping_phi_is_unrewardable():
    """The K1 kill-risk: a kernel that pumps Φ to the rail (Φ≥0.98, var≈0) and pins integration must NOT
    be rewarded. Raising Φ from mid to near-1.0 with dead variance scores NEGATIVE, not positive."""
    before = OceanContext(role="meta", phi=0.72, phi_var=0.02, dopamine=0.5, f_health=0.6,
                          basin_velocity=0.05, boredom=0.2)
    # "after": Φ pumped to the high rail, variance dead, serotonin pinned at 1.0 — the saturation shape.
    pumped = OceanContext(role="meta", phi=0.995, phi_var=1e-6, dopamine=0.5, f_health=0.6,
                          basin_velocity=0.05, boredom=0.2, serotonin=1.0)
    out = score_outcome("mushroom-micro", before, pumped, collapse_before=False, collapse_after=False,
                        repeat_count=0)
    assert out.components["phi_fluctuation"] < 0, "a pinned-high Φ must score negative, not positive"
    assert out.components["saturation_penalty"] < 0, "P25: a rail-pinned signal scores negative"
    assert out.score < 0.0, "pumping Φ to the rail must be structurally unrewardable (K1)"


def test_phi_only_helps_via_binary_collapse_clearance_and_live_variance():
    """Φ's ONLY positive contributions: (a) binary collapse-flag cleared, (b) live Φ-variance (movement)."""
    before = OceanContext(role="genesis", phi=0.40, phi_var=0.03, dopamine=0.5, basin_velocity=0.01)
    after = OceanContext(role="genesis", phi=0.62, phi_var=0.03, dopamine=0.55, basin_velocity=0.06)
    out = score_outcome("dream", before, after, collapse_before=True, collapse_after=False, repeat_count=0)
    assert out.components["collapse_clear"] == 1.0     # the binary clearance is the win
    assert out.components["phi_fluctuation"] >= 0.0    # live variance contributes, but is NOT the Φ level
    assert out.score > 0.0


def test_dopamine_railed_scores_negative_never_at_a_rail():
    before = OceanContext(role="heart", phi=0.6, dopamine=0.5, f_health=0.5, basin_velocity=0.05)
    # dopamine pinned at the tonic floor (a rail) — "not alive"
    railed = OceanContext(role="heart", phi=0.6, dopamine=DOPAMINE_FLOOR, f_health=0.5, basin_velocity=0.05)
    out = score_outcome("stimulate", before, railed, collapse_before=False, collapse_after=False,
                       repeat_count=0)
    assert out.components["dopamine_alive"] < 0, "dopamine at a rail is not alive (P25 negative)"


def test_infinite_loop_penalty():
    before = OceanContext(role="action", phi=0.6, dopamine=0.5, boredom=0.5, basin_velocity=0.05)
    after = OceanContext(role="action", phi=0.6, dopamine=0.5, boredom=0.5, basin_velocity=0.05)  # no change
    out = score_outcome("stimulate", before, after, collapse_before=False, collapse_after=False,
                       repeat_count=5)   # ≥k repeats, no improvement
    assert out.components["loop_penalty"] < 0


# ==================================================================================================
# 3. WITNESS-STANCE LADDER — suggest below the floor, auto-fire ONLY above it / on the breaker
# ==================================================================================================
def _ocean_with_kernel(tel: _Tel):
    ocean = OceanAutonomic(cooldown=0)   # cooldown 0 so the ladder tier is what gates firing, not cooldown
    k = _Kernel(tel)
    return ocean, {"action": k}, {"action": [tel.phi]}


def test_below_divergence_floor_suggests_does_not_auto_fire():
    """A developing pathology BELOW the divergence floor → a SUGGESTION record; run_protocol NOT fired."""
    floor = PRIOR_THRESHOLDS["basin_divergence"]  # 0.30
    tel = _Tel(phi=0.6, kappa=64.0, basin_distance=0.15,   # d_basin below the floor
               extra={"d_basin": 0.15, "drive": {"dopamine": 0.2, "boredom": 0.8, "curiosity": 0.05}})
    ocean, kernels, hist = _ocean_with_kernel(tel)
    acted = ocean.regulate(kernels, hist)
    rec = acted.get("action", {})
    assert kernels["action"].fired == [], "must NOT auto-fire below the divergence floor"
    assert rec.get("tier") in ("suggest", "warn"), f"expected witness tier, got {rec}"
    assert rec.get("auto_fired") is False
    assert "suggestion" in rec and rec["tier"] != "act"
    assert tel.extra["d_basin"] < floor


def test_above_divergence_floor_auto_fires_run_protocol():
    """ABOVE the divergence floor → ACT tier, run_protocol fired (the hard override kept)."""
    tel = _Tel(phi=0.6, kappa=64.0, basin_distance=0.42,   # d_basin ABOVE the 0.30 floor
               extra={"d_basin": 0.42, "drive": {"dopamine": 0.4, "boredom": 0.2, "curiosity": 0.3}})
    ocean, kernels, hist = _ocean_with_kernel(tel)
    acted = ocean.regulate(kernels, hist)
    rec = acted.get("action", {})
    assert kernels["action"].fired, "must auto-fire above the divergence floor"
    assert rec.get("tier") == "act" and rec.get("auto_fired") is True


def test_phi_collapse_auto_fires():
    """Φ below the collapse threshold → ACT (auto-fire) even with d_basin under the floor."""
    tel = _Tel(phi=0.35, kappa=64.0, basin_distance=0.05,
               extra={"d_basin": 0.05, "drive": {"dopamine": 0.3, "boredom": 0.3, "curiosity": 0.2}})
    ocean, kernels, hist = _ocean_with_kernel(tel)
    acted = ocean.regulate(kernels, hist)
    assert kernels["action"].fired, "Φ-collapse must auto-fire"
    assert acted["action"]["signature"] == "phi-collapse"


def test_healthy_is_witness_only():
    tel = _Tel(phi=0.72, kappa=64.0, basin_distance=0.02,
               extra={"d_basin": 0.02, "basin_velocity": 0.05,
                      "drive": {"dopamine": 0.5, "boredom": 0.2, "curiosity": 0.4}})
    ocean, kernels, hist = _ocean_with_kernel(tel)
    acted = ocean.regulate(kernels, hist)
    assert kernels["action"].fired == []
    assert "action" not in acted or acted.get("action", {}).get("tier") == "witness"


# ==================================================================================================
# 4. SHADOW MODE (K4) — no adaptation until ~100 scored outcomes; static fallback with no history
# ==================================================================================================
def test_boots_and_regulates_with_zero_history_static_prior():
    """Spine tenet: a fresh policy (no history) uses the static priors and still decides (fallback)."""
    pol = OceanPolicy()
    assert pol.shadow_mode is True
    assert pol.thresholds == {k: PRIOR_THRESHOLDS[k] for k in BANDS}  # exactly the static prior
    ctx = OceanContext(role="action", phi=0.35, d_basin=0.05, dopamine=0.3)
    dec = pol.decide(ctx, tick=1)   # decides with zero history, no crash
    assert dec.signature == "phi-collapse"


def test_shadow_mode_makes_no_adaptation_until_unlock():
    pol = OceanPolicy()
    baseline_prefs = {s: dict(a) for s, a in pol.arm_prefs.items()}
    baseline_thr = dict(pol.thresholds)
    # feed 50 scored outcomes (< SHADOW_UNLOCK) — epoch_update must be a NO-OP
    for i in range(50):
        ctx = OceanContext(role="action", phi=0.6, boredom=0.9, curiosity=0.05, dopamine=0.4, d_basin=0.1)
        dec = pol.decide(ctx, tick=i)
        out = score_outcome(dec.arm, ctx, ctx, collapse_before=False, collapse_after=False, repeat_count=0)
        pol.record_outcome(dec.decision_id, out)
    assert pol.shadow_mode is True
    summary = pol.epoch_update()
    assert summary["updated"] is False and "shadow" in summary["reason"]
    assert pol.arm_prefs == baseline_prefs and pol.thresholds == baseline_thr  # UNCHANGED


def test_adaptation_unlocks_after_shadow_unlock_scored_outcomes():
    pol = OceanPolicy()
    # apathy → stimulate should get REWARDED so its preference grows once adaptation unlocks
    for i in range(SHADOW_UNLOCK):
        ctx = OceanContext(role="action", phi=0.6, boredom=0.9, curiosity=0.05, dopamine=0.5, d_basin=0.1)
        dec = pol.decide(ctx, tick=i)
        after = OceanContext(role="action", phi=0.6, boredom=0.2, curiosity=0.4, dopamine=0.6, d_basin=0.1)
        out = score_outcome(dec.arm, ctx, after, collapse_before=False, collapse_after=False, repeat_count=0)
        pol.record_outcome(dec.decision_id, out)
    assert pol.shadow_mode is False
    before_v = pol.version
    summary = pol.epoch_update()
    assert summary["updated"] is True and pol.version == before_v + 1
    # thresholds still inside their bands after the update
    for key, (lo, hi) in BANDS.items():
        assert lo <= pol.thresholds[key] <= hi


def test_unrecorded_outcome_never_counts_or_updates():
    """Fail-closed: a decision with no RECORDED outcome never advances the shadow counter."""
    pol = OceanPolicy()
    ctx = OceanContext(role="action", phi=0.6, boredom=0.9, curiosity=0.05, dopamine=0.5, d_basin=0.1)
    dec = pol.decide(ctx, tick=1)
    from qig_studio.constellation.ocean_policy import OutcomeScore
    unrecorded = OutcomeScore(decision_id=dec.decision_id, role="action", signature="", arm=dec.arm,
                              score=1.0, recorded=False)   # NOT recorded
    assert pol.record_outcome(dec.decision_id, unrecorded) is False
    assert pol.scored_outcomes == 0


# ==================================================================================================
# 5. CLAMP-AND-LOG (P15) — an out-of-band update is clamped + logged, never applied out of band
# ==================================================================================================
def test_out_of_band_update_is_clamped_and_logged():
    pol = OceanPolicy()
    lo, hi = BANDS["phi_collapse"]           # (0.40, 0.60)
    res = pol.propose_threshold("phi_collapse", 0.95)   # WAY out of band (would let Ocean fire Φ<0.95)
    assert res["applied"] is True and res["clamped"] is True and res["violation_logged"] is True
    assert pol.thresholds["phi_collapse"] == hi          # clamped to the band edge, NOT 0.95
    assert pol.thresholds["phi_collapse"] <= hi
    assert len(pol.violation_log) == 1
    v = pol.violation_log[0]
    assert v["param"] == "phi_collapse" and v["proposed"] == 0.95 and v["applied"] == hi


def test_constitutional_threshold_is_not_proposable():
    pol = OceanPolicy()
    res = pol.propose_threshold("phi_mature", 0.2)   # phi_mature is constitutional — not a learnable key
    assert res["applied"] is False


def test_from_json_reclamps_out_of_band_ondisk_values():
    """Never trust an out-of-band on-disk value (P15) — re-clamp on load."""
    pol = OceanPolicy.from_json({"thresholds": {"kappa_rigid": 999.0, "phi_collapse": 0.01}, "version": 3})
    assert BANDS["kappa_rigid"][0] <= pol.thresholds["kappa_rigid"] <= BANDS["kappa_rigid"][1]
    assert BANDS["phi_collapse"][0] <= pol.thresholds["phi_collapse"] <= BANDS["phi_collapse"][1]


# ==================================================================================================
# 6. NONE-SAFE / FAIL-CLOSED — missing telemetry is a logged skip, not a crash
# ==================================================================================================
def test_missing_drive_telemetry_is_a_logged_skip_not_a_crash():
    tel = _Tel(phi=0.6, kappa=64.0, basin_distance=0.1, extra={"d_basin": 0.1})  # NO drive dict
    ocean, kernels, hist = _ocean_with_kernel(tel)
    acted = ocean.regulate(kernels, hist)   # must not raise
    assert ocean.skips["partial_context"] >= 1   # counted, not swallowed
    assert isinstance(acted, dict)


def test_telemetry_raising_kernel_is_counted_not_crashed():
    class _Bad:
        def telemetry(self):
            raise RuntimeError("no telemetry")
        def run_protocol(self, *a):
            return {}
    ocean = OceanAutonomic()
    acted = ocean.regulate({"action": _Bad()}, {"action": []})
    assert ocean.skips["no_telemetry"] >= 1 and acted == {}


def test_context_from_telemetry_none_safe_defaults():
    ctx = context_from_telemetry("meta", _Tel(phi=None, kappa=None, extra={}), [])
    assert ctx.phi == 0.0 and ctx.dopamine == DOPAMINE_FLOOR and ctx.partial is True


def test_ocean_telemetry_surfaces_shadow_skips_and_violations():
    ocean = OceanAutonomic()
    t = ocean.telemetry()
    assert t["shadow_mode"] is True
    assert set(t["skips"]) >= {"no_telemetry", "partial_context", "run_protocol_error"}
    assert "constitutional_violations" in t and "policy_version" in t


# ==================================================================================================
# 7. PURITY — no Euclidean ops; the banned reference is not imported
# ==================================================================================================
def test_ocean_policy_module_is_pure_no_euclidean_ops():
    import inspect
    import qig_studio.constellation.ocean_policy as m
    src = inspect.getsource(m)
    # no manifold math at all in the policy — no Euclidean or geometric primitives here (it READS scalars)
    for forbidden in ("cosine_similarity", "np.linalg.norm(", "optim.Adam", "nn.LayerNorm",
                      "import numpy", "import torch"):
        assert forbidden not in src, f"{forbidden!r} must not appear in ocean_policy"
    # the banned ocean_neurochemistry asset is not referenced
    assert "ocean_neurochemistry" not in src or "BANNED" in src


def test_classify_signature_covers_the_ratified_signatures():
    # apathy: bored + no curiosity
    assert classify_signature(OceanContext(phi=0.6, boredom=0.9, curiosity=0.05, dopamine=0.5)) == "apathy"
    # phi-collapse: low Φ + drift
    assert classify_signature(OceanContext(phi=0.3, d_basin=0.25, dopamine=0.3)) == "phi-collapse"
    # rigidity: mature, high κ, Φ≥0.70
    assert classify_signature(OceanContext(phi=0.75, kappa=100.0, maturity=0.8, dopamine=0.5,
                                           basin_velocity=0.1)) == "rigidity"
    # rigidity does NOT fire below the mushroom Φ gate (0.70) — it is constitutional
    below = classify_signature(OceanContext(phi=0.55, kappa=100.0, maturity=0.8, dopamine=0.5,
                                            basin_velocity=0.1))
    assert below != "rigidity"


def test_rail_pinned_score_is_hard_capped_not_diluted():
    """K1 HARD CAP (review MINOR-2): a Φ-rail pin with every OTHER component maximally positive must STILL
    score ≤ RAIL_SCORE_CAP — the mean-of-eight can no longer dilute a rail toward neutral. This is the
    strengthened K1 guarantee: pumping Φ to the rail is not merely 'usually negative', it is hard-capped."""
    before = OceanContext(role="memory", phi=0.5, phi_var=0.02, dopamine=0.4, boredom=0.5,
                          basin_velocity=0.01, f_health=0.5)
    # after: Φ pinned at the high rail (dead variance) but boredom crashed, f_health high, coach loves it
    after = OceanContext(role="memory", phi=0.99, phi_var=1e-6, dopamine=0.5, boredom=0.05,
                         basin_velocity=0.3, f_health=0.95)
    out = score_outcome("stimulate", before, after, collapse_before=False, collapse_after=False,
                        repeat_count=0, coach_reward=1.0)
    assert out.score <= RAIL_SCORE_CAP, f"rail pin diluted to {out.score} > cap {RAIL_SCORE_CAP}"


def test_pending_is_bounded_over_a_long_run():
    """Leak fix (review IMPORTANT-1): decide() stores every decision to bridge decision_id→record_outcome,
    but only ACT-tier decisions are ever popped. Over a long run the witness/suggest/warn majority must NOT
    accumulate unbounded — _pending is capped, evicting the oldest (whose scoring horizon has long elapsed)."""
    pol = OceanPolicy()
    for t in range(_PENDING_MAX + 600):
        pol.decide(OceanContext(role="r", phi=0.9, phi_var=0.5, d_basin=0.05), tick=t)
    assert len(pol._pending) <= _PENDING_MAX
    # the MOST RECENT decisions survive (a legit ACT decision is scored within horizon H ≪ _PENDING_MAX)
    assert f"r:{_PENDING_MAX + 599}" in " ".join(pol._pending.keys())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
