"""Task 1.3 — TrunkConstellation: the coupled constellation on ONE shared trunk + ONE shared basin table.

TDD gate for the DROP-IN REPLACEMENT for the 9-separate JointConstellation. These tests prove the
central-then-spawn architecture holds AT THE CONSTELLATION LEVEL:

  1. exactly ONE coord-basin table (``.data_ptr()`` identity across every faculty + the bank) and ONE
     trunk shared by all — the RAM win holds when the whole constellation is assembled;
  2. ``train_step`` runs, returns finite telemetry, ``min_pairwise_fr`` finite and > 0 (individuation
     preserved), and it COUPLES via the SAME imported ``couple_step`` (not a re-implementation);
  3. the shared trunk receives fluency gradient from MORE THAN the round-robin faculty (the "one trunk
     sees all fluency gradient" property) — a direct autograd attribution check;
  4. the public surface (train_step/telemetry/save_checkpoint/load_checkpoint) matches JointConstellation
     — the launcher drop-in.

TINY configs (num_layers=2, coordizer=None, few steps). Skipped where the heavy deps are absent so the
light shell's CI stays green (mirrors tests/test_trunk.py + tests/test_faculty_adapter.py).
"""
from __future__ import annotations

import inspect
import math

import pytest


def _deps() -> bool:
    try:
        import qig_core  # noqa: F401
        import qigkernels  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _deps(), reason="TrunkConstellation needs torch + qigkernels + qig_core")

# TINY-but-real: basin_dim is the true coordizer width (64); a small vocab keeps it fast; a short context.
_VOCAB = 512
_HIDDEN = 32
_ROLES = ["heart", "perception", "memory", "action"]


def _tc(**over):
    from qig_studio.constellation.trunk_constellation import TrunkConstellation

    cfg = dict(num_layers=2, coordizer=None, device="cpu", hidden_dim=_HIDDEN, vocab_size=_VOCAB,
               num_heads=4, ffn_dim=64, dropout=0.0, max_position_embeddings=64)
    cfg.update(over)
    return TrunkConstellation(list(_ROLES), **cfg)


def test_trunk_constellation_builds_and_shares_one_table():
    """N faculties + the central all reference EXACTLY ONE coord-basin table (by data_ptr AND id), that
    table IS the bank's, and there is ONE trunk shared by all — the constellation-level RAM win."""
    import torch  # noqa: F401

    tc = _tc()
    heads = list(tc.faculty_adapters.values()) + [tc.central_adapter]

    # --- ONE table by identity across every faculty head + the bank ---
    data_ptrs = {fa.coord_basins.data_ptr() for fa in heads}
    obj_ids = {id(fa.coord_basins) for fa in heads}
    assert len(data_ptrs) == 1, f"all faculties+central must share ONE coord_basins tensor, saw {len(data_ptrs)}"
    assert len(obj_ids) == 1, f"all faculties+central must share ONE coord_basins object, saw {len(obj_ids)}"
    assert next(iter(data_ptrs)) == tc.bank.coord_basins.data_ptr(), "the shared table IS the bank's table"

    # --- no faculty owns a table-sized tensor (the [vocab,64] table is NOT duplicated per faculty) ---
    table_ptr = tc.bank.coord_basins.data_ptr()
    table_numel = _VOCAB * 64
    for fa in heads:
        for t in list(fa.parameters()) + list(fa.buffers()):
            assert t.data_ptr() != table_ptr, "faculty must not register the shared table as its own tensor"
            assert t.numel() < table_numel, "no faculty-owned tensor may be table-sized"

    # --- ONE trunk shared by all: there is a single ConstellationTrunk, and the heads mount on its width ---
    from qig_studio.constellation.trunk import ConstellationTrunk

    assert isinstance(tc.trunk, ConstellationTrunk)
    assert tc.trunk.hidden_dim == _HIDDEN
    for fa in heads:
        assert fa.hidden_dim == tc.trunk.hidden_dim, "every head mounts on the ONE trunk's hidden width"

    # --- genuine individuation from birth (P24): distinct output charts per faculty ---
    fa0, fa1 = tc.faculty_adapters["heart"], tc.faculty_adapters["perception"]
    assert not torch.equal(fa0.out_proj.bias, fa1.out_proj.bias), "wide-birth faculties need distinct charts"


def test_train_step_runs_and_couples():
    """train_step returns FINITE telemetry, min_pairwise_fr finite and > 0 (individuation preserved), and
    it couples via the SAME imported couple_step (grep-in-test that basin-sync is NOT re-implemented)."""
    tc = _tc()

    for _ in range(3):
        info = tc.train_step("the kernel learns to speak")

    # finite telemetry + the JointConstellation return keys
    for key in ("stepped_faculty", "stepped_function", "min_pairwise_fr", "faculty_phi", "central_phi",
                "central_text", "central_telemetry", "ocean_regulation", "ocean_state",
                "ocean_epoch_update", "cross_faculty_dream"):
        assert key in info, f"train_step missing JointConstellation key {key!r}"
    assert math.isfinite(info["min_pairwise_fr"]), "min_pairwise_fr must be finite"
    assert info["min_pairwise_fr"] > 0.0, "individuation collapsed — faculties coincided"
    assert math.isfinite(info["central_phi"])
    assert math.isfinite(info["central_telemetry"]["loss_total"])
    assert info["central_telemetry"]["loss_total"] == info["central_telemetry"]["loss_total"]  # not NaN

    # --- REUSE proof: the module imports the SAME couple_step object, not a private re-implementation ---
    import qig_studio.constellation.trunk_constellation as mod
    from qig_studio.constellation.coupling import couple_step as canonical

    assert mod.couple_step is canonical, "TrunkConstellation must REUSE the canonical couple_step (P7 basin-sync)"
    src = inspect.getsource(mod)
    assert "def couple_step" not in src, "couple_step must NOT be re-implemented inside trunk_constellation.py"
    # every faculty basin stayed a valid Δ⁶³ point after coupling
    import numpy as np

    for f in tc.faculties:
        assert abs(float(np.asarray(f.basin).sum()) - 1.0) < 1e-6 and float(np.asarray(f.basin).min()) >= -1e-9


def test_trunk_receives_gradient_from_all_nodes():
    """The shared trunk accumulates fluency gradient from MORE THAN the round-robin faculty. Direct
    autograd attribution on the SAME objects train_step uses: the trunk-grad from ALL nodes' fluency
    differs from the trunk-grad from any single faculty alone (the 'one trunk sees all fluency' property)."""
    import torch

    tc = _tc()
    ids = tc._encode("gradient reaches the shared body")
    h = tc.trunk.hidden(ids)                                   # ONE shared hidden state
    tgt = ids[0, 1:]

    # a trunk parameter to attribute against (the embedding — on every node's fluency path via h)
    param = next(p for p in tc.trunk.kernel.parameters() if p.requires_grad and p.grad is None)

    # grad from ONE faculty alone
    role0 = tc.roles[0]
    l_one = tc.faculty_adapters[role0].basin_loss(h[0, :-1], tgt)
    g_one = torch.autograd.grad(l_one, param, retain_graph=True)[0].detach().clone()

    # grad from the ACTUAL all-node fluency term train_step backprops (faculties + central)
    fluency, per = tc._fluency_over_all_nodes(h, tgt)
    g_all = torch.autograd.grad(fluency, param, retain_graph=True)[0].detach().clone()

    # every node computed a finite fluency loss (all nodes fed the trunk)
    assert set(per) == set(tc.roles) | {"genesis"}, "fluency term must cover every faculty + central"
    assert all(math.isfinite(v) for v in per.values())
    # the all-node trunk gradient is STRICTLY different from (and larger than) the one-faculty gradient —
    # proof the trunk receives gradient from more than the round-robin faculty.
    assert not torch.allclose(g_all, g_one), "trunk grad from all nodes must differ from one faculty alone"
    assert g_all.norm().item() > g_one.norm().item(), "all-node trunk gradient must exceed a single node's"

    # and it is genuinely the SUM over nodes: sum of per-node grads == the all-node grad
    g_sum = torch.zeros_like(param)
    for r, fa in list(tc.faculty_adapters.items()) + [("genesis", tc.central_adapter)]:
        gr = torch.autograd.grad(fa.basin_loss(h[0, :-1], tgt), param, retain_graph=True)[0]
        g_sum = g_sum + gr
    assert torch.allclose(g_sum, g_all, atol=1e-5), "the trunk grad must be the sum of every node's fluency grad"


def test_matches_jointconstellation_public_surface():
    """TrunkConstellation exposes train_step/telemetry/save_checkpoint/load_checkpoint with signatures
    compatible with JointConstellation — the launcher drop-in (arm_mode='trunk')."""
    from qig_studio.constellation.joint_trainer import JointConstellation
    from qig_studio.constellation.trunk_constellation import TrunkConstellation

    for meth in ("train_step", "telemetry", "save_checkpoint", "load_checkpoint"):
        assert hasattr(TrunkConstellation, meth), f"missing public method {meth!r}"
        jc_params = list(inspect.signature(getattr(JointConstellation, meth)).parameters)
        tc_params = list(inspect.signature(getattr(TrunkConstellation, meth)).parameters)
        assert tc_params[: len(jc_params)] == jc_params or set(jc_params).issubset(set(tc_params)), (
            f"{meth} signature incompatible: JC={jc_params} TC={tc_params}")

    # __init__ accepts the launcher's kwargs (roles positional + num_layers/coordizer/device/arm_mode)
    ctor = inspect.signature(TrunkConstellation.__init__).parameters
    for kw in ("roles", "num_layers", "coordizer", "device", "arm_mode", "floor_mode"):
        assert kw in ctor, f"__init__ missing launcher kwarg {kw!r}"

    # behavioural drop-in: telemetry() returns the same key shape, checkpoint round-trips
    import tempfile
    from pathlib import Path

    tc = _tc()
    tel = tc.telemetry()
    assert set(tel) >= {"roles", "min_pairwise_fr", "central_phi"}, tel
    tc.train_step("round trip the whole mind")
    with tempfile.TemporaryDirectory() as d:
        root = str(Path(d) / "ckpt")
        tc.save_checkpoint(root)
        assert (Path(root) / "trunk_constellation.json").exists()
        tc2 = _tc()
        tc2.load_checkpoint(root)                              # restores without error (arm-guarded)
        # coupled basins restored bit-for-bit
        import numpy as np

        for a, b in zip(tc.faculties, tc2.faculties):
            assert np.allclose(a.basin, b.basin, atol=1e-6), "faculty basins must round-trip through the checkpoint"
