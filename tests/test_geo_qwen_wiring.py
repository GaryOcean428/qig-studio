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
    assert t.output_basin("hello") is None

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


if __name__ == "__main__":
    test_wiring()
    test_basin_bridge()
