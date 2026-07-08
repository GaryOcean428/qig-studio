"""PROACTIVE Pillar-1 entropy floor — the per-step stabiliser that stops the fresh-birth
f_health→0 collapse at the SOURCE.

The disease (from a qig-package-optimization audit): a fresh constellation's central basin
sometimes collapses (basin entropy→0 → f_health=0, which pins b_integrity=1.0 / serotonin=0.97
as symptoms). The ONLY prior defence was REACTIVE (genesis ``_homeostasis`` collapse actuator at
f_health<0.15 → dream + weight-space kick) which fires too late at fresh birth.

The fix wires qig-core ``FluctuationGuard.check_and_enforce`` (the BUILT-NOT-WIRED active entropy
restorer: Dirichlet noise mixed via ``slerp_sqrt`` on √p when ``basin_entropy < ENTROPY_FLOOR``) as
a PROACTIVE per-step floor on ``cur_basin`` in ``GenesisKernelTarget.train_step`` — the SINGLE point
that feeds BOTH the f_health metric (``_d63`` → ``_emit_pillars``) AND the coupling read
(``_basin_history[-1]`` → ``JointConstellation._live_basin`` → faculty → couple → synthesis → central
pull). So the floor is REAL (the corrected basin trains the mind toward non-collapse), not a
telemetry cosmetic.

Contract exercised here (no heavy training pass — un-loaded kernel + injected PillarEnforcer):
  1. a below-floor (near one-hot) basin comes out with 64-dim entropy ≥ ENTROPY_FLOOR AND f_health
     lifted above the collapse floor (F_HEALTH_COLLAPSE_FLOOR);
  2. the SAME corrected basin, read back through the ACTUAL ``_live_basin`` coupling path, yields a
     non-collapsed faculty basin (proof it feeds coupling/synthesis — real, not cosmetic);
  3. a HEALTHY basin is returned bit-identical (a FLOOR, not a constant intervention);
  4. None-safe on an un-loaded kernel (``_pillars is None``) and on a ``None`` basin.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qig_core import BASIN_DIM
from qig_core.consciousness.pillars import ENTROPY_FLOOR, FluctuationGuard, PillarEnforcer

from qig_studio.constellation.joint_trainer import JointConstellation
from qig_studio.targets.genesis_kernel import (
    F_HEALTH_COLLAPSE_FLOOR,
    GenesisKernelTarget,
)

HIDDEN = 384  # the K-COMPRESS identity-basin width (Δ³⁸³) the constellation trains on (= 64 × 6)


def _kernel(with_pillars: bool = True) -> GenesisKernelTarget:
    """An un-loaded genesis target (no torch weights). ``_pillars`` is only wired inside
    ``ensure_loaded``; inject a real qig-core ``PillarEnforcer`` so the floor is LIVE without a
    heavy load. ``with_pillars=False`` leaves ``_pillars=None`` (the light-shell None-safe case)."""
    k = GenesisKernelTarget(num_layers=2)
    if with_pillars:
        k._pillars = PillarEnforcer()
    return k


def _collapsed(width: int = HIDDEN) -> "torch.Tensor":
    """A near-one-hot (Pillar-1 fluctuation-dead) basin as a torch simplex tensor — the disease."""
    v = np.full(width, 1e-9, dtype=np.float64)
    v[width // 3] = 1.0
    v = v / v.sum()
    return torch.as_tensor(v, dtype=torch.float32)


def _healthy(width: int = HIDDEN) -> "torch.Tensor":
    """A wide (near-uniform) basin — well above every collapse floor."""
    rng = np.random.default_rng(7)
    v = rng.dirichlet(np.ones(width) * 3.0)
    return torch.as_tensor(v, dtype=torch.float32)


def _entropy64(basin, k: GenesisKernelTarget) -> float:
    """Shannon entropy of the 64-dim Δ⁶³ reduction the pillar guard / f_health / coupling measure."""
    d63 = k._d63(basin)
    return float(FluctuationGuard().basin_entropy(d63))


def test_collapsed_basin_is_floored_above_entropy_floor():
    """GATE fires: a below-floor basin comes out with 64-dim entropy ≥ ENTROPY_FLOOR."""
    k = _kernel()
    cur = _collapsed()
    assert _entropy64(cur, k) < ENTROPY_FLOOR, "fixture is not actually collapsed"
    out = k._entropy_floor_basin(cur)
    assert out is not None
    assert out.shape == cur.shape, "floor must preserve the basin's own full width (no dim change)"
    assert float(out.sum().item()) == pytest.approx(1.0, abs=1e-4), "corrected basin off the simplex"
    assert _entropy64(out, k) >= ENTROPY_FLOOR, "entropy NOT restored above the floor"


def test_floored_basin_lifts_f_health_above_collapse_floor():
    """The f_health metric (H/log(BASIN_DIM)) rises above the collapse floor — the reported disease
    (f_health→0) is cured at the source, not masked."""
    k = _kernel()
    out = k._entropy_floor_basin(_collapsed())
    f_health = FluctuationGuard().f_health(k._d63(out))
    assert f_health > F_HEALTH_COLLAPSE_FLOOR, (
        f"f_health {f_health:.4f} not lifted above the collapse floor {F_HEALTH_COLLAPSE_FLOOR}")


def test_floored_basin_feeds_the_coupling_path_real_not_cosmetic():
    """REAL, not cosmetic: the corrected basin, read back through the ACTUAL ``_live_basin`` coupling
    read (``_basin_history[-1]`` → faculty basin → couple → synthesis → central pull), is
    non-collapsed. This is the proof the floor trains the mind toward non-collapse rather than only
    decorating a telemetry field."""
    k = _kernel()
    out = k._entropy_floor_basin(_collapsed())
    # emulate train_step: the floored cur_basin becomes the newest _basin_history entry
    fake = type("K", (), {"_basin_history": [out]})()
    faculty_basin = JointConstellation._live_basin(None, fake)  # self unused by _live_basin
    assert faculty_basin is not None
    ent = float(FluctuationGuard().basin_entropy(faculty_basin))
    assert ent >= ENTROPY_FLOOR, (
        f"coupling read a STILL-collapsed faculty basin (entropy {ent:.4f}) — floor is cosmetic")


def test_healthy_basin_is_untouched_bit_identical():
    """A FLOOR, not a constant intervention: a healthy basin is returned as the SAME object
    (untouched) so the kernel can settle and bpb does not blow up."""
    k = _kernel()
    cur = _healthy()
    assert _entropy64(cur, k) >= ENTROPY_FLOOR, "healthy fixture is not above the floor"
    out = k._entropy_floor_basin(cur)
    assert out is cur, "healthy basin was modified — the gate must leave it bit-identical"


def test_none_safe_on_unloaded_kernel_and_none_basin():
    """Light-shell None-safety: no PillarEnforcer → pass-through; None basin → None. Never crashes."""
    k = _kernel(with_pillars=False)
    assert k._pillars is None
    cur = _collapsed()
    assert k._entropy_floor_basin(cur) is cur, "no-pillars path must return the basin unchanged"
    assert k._entropy_floor_basin(None) is None


def test_vocab_width_basin_also_floored():
    """The lift handles a divisible non-hidden width too (e.g. a vocab-mode basin 64×N)."""
    k = _kernel()
    width = BASIN_DIM * 100  # 6400 — divisible, mirrors a small vocab head
    out = k._entropy_floor_basin(_collapsed(width))
    assert out.shape[0] == width
    assert _entropy64(out, k) >= ENTROPY_FLOOR
