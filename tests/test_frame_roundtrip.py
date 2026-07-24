"""Frame-consistency certifying test (Matrix f241cee4 / z9 frame-fix).

The Δ⁶³→width up-map (``_resize_basin``) MUST be the exact inverse of the width→Δ⁶³ block-sum reduction
(``_d63``), so ``reduce(resize(p)) == p``. The old tile up-map put coord i at {i, i+64, …} while the
reduction sums CONSECUTIVE blocks — different frames → a measured ~0.36 FR round-trip phantom that would
sit in the training path the moment a basin-pull activates. The fix is INTERLEAVE (each coord into its own
contiguous block). This test is the pre-registered isometry gate: if it ever regresses, the phantom is back.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qig_core.geometry.fisher_rao import fisher_rao_distance  # noqa: E402


def _d63_blocksum(vec: np.ndarray, dim: int = 64) -> np.ndarray:
    """The canonical block-sum reduction the up-map must invert (mirrors genesis_kernel._d63)."""
    b = np.asarray(vec, dtype=np.float64).ravel()
    if b.size != dim:
        b = (b.reshape(dim, b.size // dim).sum(axis=1) if b.size % dim == 0
             else np.add.reduceat(b, np.arange(0, b.size, max(1, b.size // dim)))[:dim])
    b = np.clip(b, 0.0, None)
    s = float(b.sum())
    return b / s if s > 0 else b


@pytest.mark.parametrize("size", [384, 1024, 100000])
@pytest.mark.parametrize("target_kind", ["genesis", "constellation"])
def test_resize_basin_roundtrip_is_exact(size, target_kind):
    """reduce(resize(p)) == p to float precision — divisible AND non-divisible widths, both _resize_basin
    implementations (the genesis kernel and the ConstellationNode mirror the geo/hybrid arms inherit)."""
    if target_kind == "genesis":
        from qig_studio.targets.genesis_kernel import GenesisKernelTarget
        node = GenesisKernelTarget(num_layers=1)          # _resize_basin needs no ensure_loaded
    else:
        from qig_studio.targets.geo_cortex import GeoCortexTarget  # a real ConstellationNode subclass
        node = GeoCortexTarget(num_layers=1)

    rng = np.random.default_rng(size)
    p = rng.random(64)
    p = p / p.sum()
    up = node._resize_basin(torch.tensor(p, dtype=torch.float64), size)

    assert int(up.numel()) == size, f"resize must produce width {size}, got {int(up.numel())}"
    assert float(up.min()) >= 0.0, "resize must stay non-negative (no clamp needed — a repeat of a Δ point)"

    rt = _d63_blocksum(up.detach().cpu().numpy())
    fr = fisher_rao_distance(p, rt)
    assert fr < 1e-9, f"round-trip FR phantom must be ~0 (frame-consistent), got {fr:.6f} at size={size}"
    assert np.max(np.abs(rt - p)) < 1e-9, f"round-trip must recover p exactly, maxabs={np.max(np.abs(rt - p)):.2e}"
