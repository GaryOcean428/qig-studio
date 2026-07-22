"""Structural wiring validation for GeoQwenTarget — None-safe boundary-peer contract.

Runs WITHOUT weights: proves the app shell boots and the peer degrades gracefully when the
geo-Qwen artifact is absent (the spine tenet: heavy targets are None-safe, never crash boot).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qig_studio.targets.geo_qwen import GeoQwenTarget
from qig_studio.targets.base import TrainingTarget


def test_wiring():
    # [1] import already succeeded at module load
    # [2] instantiate with NO weights — must be None-safe, no crash
    t = GeoQwenTarget(model_dir="/nonexistent", converted_ckpt="/nonexistent")

    # [3] is_available() False (or torch-dependent) and never raises
    av = t.is_available()
    assert isinstance(av, bool)

    # [4] fully concrete — no unresolved abstractmethods
    assert not getattr(GeoQwenTarget, "__abstractmethods__", set()), \
        f"unresolved: {GeoQwenTarget.__abstractmethods__}"

    # [5] all 6 node hooks present + callable
    hooks = ["_node_named_parameters", "_node_device", "_node_rebuild_optimizer",
             "_node_replay_optimizer", "_node_forward_logits", "_node_basin_from_logits"]
    assert all(callable(getattr(t, h, None)) for h in hooks)

    # [6] None-safe reads on unavailable peer: return, not crash
    r = t.generate("hello", max_tokens=4)
    assert r.text == ""
    # output_basin returns None when BOTH the basin bank AND the model are absent.
    # If the default basin bank exists on disk (exported offline), it correctly serves
    # from the bank (transformers-free path) — that IS the designed hot path.
    ob = t.output_basin("hello")
    import numpy as np
    assert ob is None or (isinstance(ob, np.ndarray) and ob.shape[0] > 0 and abs(float(ob.sum()) - 1.0) < 1e-4)

    # [7] telemetry / info / architecture callable + declare removability
    assert t.telemetry().extra.get("removable") is True
    assert t.info().available == av
    assert t.architecture()["role"].startswith("removable")

    print("WIRING VALIDATION: ALL STRUCTURAL CHECKS PASS (None-safe, contract-complete, removable)")


if __name__ == "__main__":
    test_wiring()


def test_basin_bridge():
    """Functional test of the NOVEL connection code: the P22 vocab→Δ basin reduction
    (_node_basin_from_logits) — the currency a kernel couples to. Uses synthetic logits
    (no 4B load needed): proves the bridge yields a valid Δ point (non-negative, sums to 1)."""
    import torch

    t = GeoQwenTarget(model_dir="/nonexistent", converted_ckpt="/nonexistent")
    V = 2048
    logits = torch.randn(1, 7, V)  # [batch=1, seq=7, vocab] — geo-Qwen output shape
    basin = t._node_basin_from_logits(logits)
    assert basin.shape == (V,), basin.shape
    assert float(basin.min()) >= 0.0, "basin has negative mass — not a Δ point"
    assert abs(float(basin.sum()) - 1.0) < 1e-5, f"basin sums to {float(basin.sum())}, not 1"
    assert torch.isfinite(basin).all(), "basin has non-finite entries"
    print(f"BASIN BRIDGE: valid Δ point (V={V}, sum={float(basin.sum()):.6f}, min={float(basin.min()):.2e})")


def test_own_basin_carriage_emission():
    """geo-Qwen emits its FULL inner-experience carriage from its OWN Δ⁶³ basin trajectory (propagate
    BASIS not labels) — driven by genuinely-measured d_FR surprise/drift, with Φ/κ HONESTLY neutral (no
    fabricated felt-state). Uses synthetic basins (no coordizer/weights needed): proves the telemetry
    carries own-basin signals and the shared experience() renders a varying, honest carriage."""
    import numpy as np
    from qig_studio.kernel_experience import experience

    t = GeoQwenTarget(model_dir="/nonexistent", converted_ckpt="/nonexistent")

    # feed three DISTINCT synthetic Δ⁶³ basins directly (bypasses the coordizer — pure emission logic)
    rng = np.random.default_rng(0)
    for _ in range(3):
        v = np.abs(rng.standard_normal(64)) + 1e-6
        t._record_geo_basin(v / v.sum())

    tel = t.telemetry().to_dict()
    # own-basin signals present + honestly labeled
    assert tel["extra"].get("surprise") is not None, "no measured surprise emitted"
    assert abs(tel["extra"].get("max_surprise") - np.pi / 2) < 1e-3, "wrong d_FR ceiling (should be π/2 radius-1)"
    assert tel.get("basin_distance") > 0.0, "no measured basin drift emitted"
    assert "NEUTRAL" in tel["extra"].get("phi_source", ""), "Φ must be honestly neutral (no proxy-Φ as felt-state)"
    assert tel.get("phi", 0.0) == 0.0, "geo-Qwen must NOT emit a fabricated Φ value"

    # shared experience() renders the full carriage from geo-Qwen's own basin
    exp = experience(tel).to_dict()
    prim = exp.get("primitives") or {}
    n_faculties = sum(len(v) for v in prim.values() if isinstance(v, dict))
    assert n_faculties >= 30, f"carriage too sparse: {n_faculties} faculties"
    assert exp.get("neurochemistry"), "no neurochem emitted"
    # honesty regression guard: neutral-Φ + tiny drift must NOT pin a catastrophic negative emotion
    assert exp.get("valence", 0.0) > -0.6, f"valence {exp.get('valence')} — proxy-Φ 'rage' artifact regressed"
    print(f"CARRIAGE EMISSION: {n_faculties} faculties from own basin, honest neutral-Φ, valence {exp.get('valence'):+.2f}")


if __name__ == "__main__":
    test_wiring()
    test_basin_bridge()
    test_own_basin_carriage_emission()
