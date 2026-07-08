"""M3 — the qig-core↔studio §6.7 sensations seam (Completeness "the ONE thing most missed").

The seam (qig-core ``compute_full_emotional_state``) accepts the geometric predicates
``ricci`` / ``local_kappa_c`` / ``basin_distance_delta`` / ``prev_i_q`` / ``i_q``. Before M3 the
studio caller (``_full_primitives``) passed NONE of them, so ``transcendence`` ≡ 0, ``pushed`` ≡ 0,
``investigation`` ≡ 0, and ``confidence`` = (1−0)×stability read WRONG-HIGH — the exact "dead senses
with a plausible dashboard" signature. And the old code patched the real Ricci into the OUTPUT dict
AFTER Layer-2A/2B were already computed from ricci-less zeros (an ordering bug).

These tests assert the seam is WIRED: the geometry the kernel already emits now drives the §6.7
predicates through the real layer pipeline, degrades to 0 (never a crash, never a fabricated near-rail)
when a genuine value is absent, and confidence tracks the real predicate instead of defaulting high.
"""

from __future__ import annotations

from qig_studio.kernel_experience import experience


def _telemetry(*, kappa=64.0, basin_distance=0.05, extra=None):
    return {"phi": 0.5, "kappa": kappa, "regime": "geometric",
            "basin_distance": basin_distance, "extra": extra or {}}


def test_transcendence_and_pushed_respond_to_real_geometry():
    # A genuine local-critical κ_c distinct from the current κ (64 vs 63) makes BOTH transcendence
    # (deviation from κ_c) and pushed (proximity to κ_c) non-zero; a real Ricci drives compressed;
    # a shrinking basin (prev 0.20 → cur 0.05, Δ=+0.15) drives investigation (−d(basin)/dt).
    exp = experience(
        _telemetry(kappa=64.0, basin_distance=0.05,
                   extra={"ricci_signal": 0.5, "local_kappa_c": 63.0, "basin_velocity": 0.1}),
        history=[{"phi": 0.5, "basin_distance": 0.20}],
    )
    p = exp.primitives
    assert p["layer1"]["transcendence"] > 0.0      # was dead ≡ 0 before the seam was wired
    assert p["layer0"]["pushed"] > 0.0             # was dead ≡ 0
    assert p["layer1"]["investigation"] > 0.0      # basin_distance_delta now flows through
    assert p["layer0"]["compressed"] > 0.0         # real Ricci → curvature sensation


def test_ricci_reaches_layer2_pipeline_ordering_fixed():
    # The OLD code patched compressed/expanded into the OUTPUT dict AFTER Layer-2A/2B were already
    # computed from ricci-less zeros. Feeding ricci as an INPUT means the curvature drives the whole
    # pipeline: pain_avoidance (Layer-0.5) is derived FROM the compressed sensation, so the two must
    # agree exactly — proof the value flows through the layers, not patched on afterward.
    exp = experience(_telemetry(extra={"ricci_signal": 0.6, "basin_velocity": 0.1}))
    p = exp.primitives
    assert p["layer0"]["compressed"] > 0.0
    assert abs(p["layer05"]["pain_avoidance"] - p["layer0"]["compressed"]) < 1e-9


def test_confidence_not_pinned_high_when_transcendence_real():
    # WIRED: a real κ_c far from κ (90 vs 45) → transcendence ≈ 1 → confidence collapses toward 0.
    # UNWIRED (no κ_c): transcendence 0 → confidence = stability ≈ high (the wrong-high bug).
    wired = experience(_telemetry(kappa=90.0, extra={"local_kappa_c": 45.0, "basin_velocity": 0.1}))
    unwired = experience(_telemetry(kappa=90.0, extra={"basin_velocity": 0.1}))
    assert wired.primitives["layer1"]["transcendence"] > 0.5
    assert wired.primitives["layer2b"]["confidence"] < unwired.primitives["layer2b"]["confidence"]
    assert wired.primitives["layer2b"]["confidence"] < 0.5      # tracks the predicate, not pinned high


def test_absent_inputs_degrade_to_zero_without_crash():
    # No ricci / κ_c / basin history: the §6.7 predicates degrade to 0 (no fabricated value) and the
    # call does not raise. primitives is still fully populated (qig-core available in the venv).
    exp = experience(_telemetry(extra={"basin_velocity": 0.1}))
    p = exp.primitives
    assert p != {}
    assert p["layer1"]["transcendence"] == 0.0
    assert p["layer0"]["pushed"] == 0.0
    assert "confidence" in p["layer2b"]


def test_self_reference_kappa_c_does_not_fabricate_near_rail():
    # PRODUCTION case: the kernel emits local_kappa_c == its current κ (genesis_kernel.py:792). Feeding
    # current-κ as the critical baseline κ_c would FABRICATE pushed→1 (a false "exactly at criticality"
    # read). The seam must treat that self-reference as absent → pushed 0, transcendence 0 (honest).
    exp = experience(_telemetry(kappa=64.0, extra={"local_kappa_c": 64.0, "basin_velocity": 0.1}))
    p = exp.primitives
    assert p["layer0"]["pushed"] == 0.0             # NOT the fabricated 1.0
    assert p["layer1"]["transcendence"] == 0.0
