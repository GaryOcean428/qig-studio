"""ARM A (geocoding GeoModel cortex) — faithfulness + lean-telemetry + None-safety gates.

These run only where the heavy deps (torch + geocoding + qigkernels) are present; skipped otherwise so the
light shell's CI stays green. The load-bearing check is the COORDS-OFF 1e-5 faithfulness parity between
geocoding's Fisher-Rao attention and the validated qigkernels primitive — the guard that ARM A and ARM B
share the same geometry (so the bpb A/B isolates architecture).
"""
from __future__ import annotations

import pytest


def _deps() -> bool:
    try:
        import geocoding  # noqa: F401
        import qigkernels  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _deps(), reason="ARM A needs torch + geocoding + qigkernels")


def test_geo_cortex_faithfulness_coords_off_1e5():
    """geocoding ↔ qigkernels Fisher-Rao attention parity to 1e-5 on the coords-off shared primitive."""
    from qig_studio.targets.geo_cortex import GeoCortexTarget

    t = GeoCortexTarget(num_layers=2, hidden_dim=64, num_heads=4, ffn_dim=128, device="cpu")
    max_abs_diff = t.assert_faithful_to_qigkernels(atol=1e-5)
    assert max_abs_diff <= 1e-5, max_abs_diff


def test_geo_cortex_loss_value_parity_coords_off_1e5():
    """LOSS-VALUE parity (coords-off): identical inputs → identical fisher_rao_lm_loss value through both
    arms' plumbing to 1e-5 — the load-bearing gate that the bpb/d_FR A/B measures architecture, not loss."""
    from qig_studio.targets.geo_cortex import GeoCortexTarget

    t = GeoCortexTarget(num_layers=2, hidden_dim=64, num_heads=4, ffn_dim=128, device="cpu")
    loss_diff = t.assert_loss_value_parity(atol=1e-5)
    assert loss_diff <= 1e-5, loss_diff


def test_geo_cortex_surfaces_real_phi_and_is_none_safe_in_experience():
    """geo surfaces the REAL integrated-information Φ GeoModel computes (RecursiveIntegrator, identical
    formula to gk) — node-parity item 1 (Matrix 110d5362), which un-discarded the value that was wrongly
    reported as None and only aliased to geo_phi. And experience(tel, phi_hist) must still NOT crash on a
    None entry in phi_hist — the launcher calls it every step and would die on a bare dereference."""
    import math

    from qig_studio.kernel_experience import experience
    from qig_studio.targets.geo_cortex import GeoCortexTarget

    t = GeoCortexTarget(num_layers=2, hidden_dim=64, num_heads=4, ffn_dim=128, device="cpu")
    res = t.train_step("the cortex learns geometric language on the simplex")
    tel = res.telemetry.to_dict()
    # phi is now the real Φ, not None — the P26 maturity gate can finally read it (was float(phi or 0.0)=0.0)
    assert tel["phi"] is not None
    assert math.isfinite(tel["phi"]) and 0.0 <= tel["phi"] <= 1.0
    assert tel["extra"]["geo_phi"] == tel["phi"]    # the geo_phi alias mirrors the surfaced Φ (backward compat)
    assert tel["extra"]["bpb"] is not None
    assert math.isfinite(tel["extra"]["bpb"])       # NaN-free loss/bpb
    exp = experience(tel, [{"phi": None}]).to_dict()  # None entry in phi_hist must not raise
    assert isinstance(exp, dict) and "emotion" in exp


def test_geo_cortex_eval_contracts_match_arm_b():
    """eval_text_bpb / eval_text_fr return (value, n_pos) with finite values (the ARM-B contract)."""
    import math

    from qig_studio.targets.geo_cortex import GeoCortexTarget

    t = GeoCortexTarget(num_layers=2, hidden_dim=64, num_heads=4, ffn_dim=128, device="cpu")
    bits, nbytes = t.eval_text_bpb("hello geometric world")
    dfr, npos = t.eval_text_fr("hello geometric world")
    assert nbytes >= 1 and npos >= 1
    assert math.isfinite(bits) and math.isfinite(dfr)
    assert 0.0 <= dfr / max(1, npos) <= math.pi + 1e-6   # mean d_FR in [0, π] (torch primitive, not [0,π/2])


def test_neocortex_arm_geo_builds_and_names():
    """Neocortex(arm='geo') builds a GeoCortexTarget and names the run neocortex-geo-NL."""
    from qig_studio.neocortex import Neocortex
    from qig_studio.targets.geo_cortex import GeoCortexTarget

    cortex = Neocortex(arm="geo", num_layers=2, device="cpu")
    assert cortex.name == "neocortex-geo-2L"
    assert isinstance(cortex.target, GeoCortexTarget)
    res = cortex.train_step("a short geometric passage")  # one real pure-loss step through the wrapper
    assert res.telemetry.to_dict()["extra"]["bpb"] is not None
