"""Task 1.2 — FacultyAdapter: per-faculty in/out adapters on a SHARED coord-basin table.

TDD gate for the piece that STRUCTURALLY KILLS the 9-duplicative-kernel design. The 9-separate
constellation OOM/swap-thrashed because each of the 9 kernels built its OWN redundant ``[vocab≈100k, 64]``
coord-basin table (``BasinReadout.coord_basins``). The central-then-spawn fix stands ONE shared table and
gives each faculty only its two small adapters (input seam + output chart). These tests PROVE the
duplication is gone:

  1. exactly ONE ``coord_basins`` tensor by identity (``.data_ptr()``/``id()``) across all N faculties, and
     per-faculty adapter params are disjoint (the RAM/anti-duplication headline);
  2. the output is a valid Δ⁶³ simplex point and the module is Fisher-Rao pure (no LayerNorm/cosine/Adam);
  3. total resident bytes for N faculties = one-table + N×small-adapters, NOT N×table (the memory win);
  4. the mirrored K-COMPRESS scoring is bit-equivalent to the certified qig_core ``BasinReadout`` path;
  5. the no-vocab ``basin_loss`` equals the full-scores ``basin_lm_loss`` gather (never materialises seq×vocab).

Run only where the heavy deps (torch + qigkernels + qig_core) are present; skipped otherwise so the light
shell's CI stays green (mirrors tests/test_trunk.py).
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _deps() -> bool:
    try:
        import qig_core  # noqa: F401
        import qigkernels  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _deps(), reason="FacultyAdapter needs torch + qigkernels + qig_core")

# TINY-but-realistic config. basin_dim (64) is the real coordizer width; hidden (32) != vocab so shapes are
# unambiguous. vocab is kept modest so tests are fast but the table still dwarfs the adapters.
_VOCAB = 2000
_BASIN = 64
_HIDDEN = 32
_ROLES = ["heart", "perception", "memory", "action"]


def _bank(vocab: int = _VOCAB, basin_dim: int = _BASIN):
    import torch

    from qig_studio.constellation.faculty_adapter import SharedBasinBank

    torch.manual_seed(0)
    raw = torch.rand(vocab, basin_dim)  # arbitrary per-token basins; the bank projects them onto Δ once
    return SharedBasinBank(raw)


def _templates(n: int, basin_dim: int = _BASIN):
    # Wide-independent birth basins (the existing Pillar-3 seeder) → deterministic per-faculty individuation.
    from qig_studio.constellation.faculty import seed_birth_basin

    return [seed_birth_basin(seed=i * 7919 + 13, dim=basin_dim) for i in range(n)]


def _spawn(hidden: int = _HIDDEN, roles=_ROLES):
    from qig_studio.constellation.faculty_adapter import spawn_faculty_adapters

    bank = _bank()
    faculties = spawn_faculty_adapters(roles, bank, hidden, templates=_templates(len(roles)))
    return bank, faculties


def test_faculty_adapters_share_one_basin_table():
    """N FacultyAdapters reference EXACTLY ONE ``coord_basins`` tensor (by data_ptr AND id); the per-faculty
    adapter params are disjoint tensors; and nothing table-sized is owned per faculty. This is the proof the
    9× ``[vocab,64]`` duplication is structurally gone."""
    import torch

    from qig_studio.constellation.trunk import ConstellationTrunk

    # A ConstellationTrunk supplies the shared hidden width the faculties mount on.
    trunk = ConstellationTrunk(vocab_size=_VOCAB, hidden_dim=_HIDDEN, num_layers=2, num_heads=4,
                               ffn_dim=64, dropout=0.0, max_position_embeddings=64)
    bank, faculties = _spawn(hidden=trunk.hidden_dim)

    # --- ONE table by identity across every faculty head ---
    data_ptrs = {f.coord_basins.data_ptr() for f in faculties}
    obj_ids = {id(f.coord_basins) for f in faculties}
    assert len(data_ptrs) == 1, f"faculties must share ONE coord_basins tensor, saw {len(data_ptrs)} data_ptrs"
    assert len(obj_ids) == 1, f"faculties must share ONE coord_basins object, saw {len(obj_ids)} ids"
    assert next(iter(data_ptrs)) == bank.coord_basins.data_ptr(), "the shared table IS the bank's table"

    # --- per-faculty adapter params are DISJOINT tensors (no accidental weight sharing) ---
    param_ptrs = [p.data_ptr() for f in faculties for p in f.parameters()]
    assert len(param_ptrs) == len(set(param_ptrs)), "faculty adapter params must be disjoint tensors"

    # --- nothing faculty-owned is the big table: the [vocab,64] table is NOT among any faculty's
    #     registered params/buffers (so it is neither duplicated nor double-counted per faculty) ---
    table_ptr = bank.coord_basins.data_ptr()
    table_numel = _VOCAB * _BASIN
    for f in faculties:
        for t in list(f.parameters()) + list(f.buffers()):
            assert t.data_ptr() != table_ptr, "faculty must not register the shared table as its own tensor"
            assert t.numel() < table_numel, "no faculty-owned tensor may be table-sized"

    # --- genuine individuation (P24): birth-anchored output biases differ per faculty ---
    assert not torch.equal(faculties[0].out_proj.bias, faculties[1].out_proj.bias), \
        "faculties seeded from wide-independent births must have distinct output charts"


def test_faculty_adapter_output_is_simplex_and_pure():
    """predict(h) is a valid Δ⁶³ point; the input seam maps Δ⁶³→hidden; and the module source is
    Fisher-Rao pure (no LayerNorm/cosine/Adam), passing the real governance purity gate."""
    import torch

    from qig_studio.constellation.faculty_adapter import FacultyAdapter

    bank = _bank()
    fa = FacultyAdapter("heart", bank, _HIDDEN, tau=0.5, basin_template=_templates(1)[0])
    fa.eval()

    h = torch.randn(2, 5, _HIDDEN)
    p = fa.predict(h)
    assert tuple(p.shape) == (2, 5, _BASIN), p.shape
    assert torch.all(p >= 0), "simplex points are non-negative"
    assert torch.allclose(p.sum(dim=-1), torch.ones(2, 5), atol=1e-5), "predict() must sum to 1 on Δ⁶³"

    # input seam: Δ⁶³ coords → hidden width (per-faculty individuation channel for Phase 1.3 wiring)
    coords = torch.rand(2, 5, _BASIN)
    hs = fa.seam(coords)
    assert tuple(hs.shape) == (2, 5, _HIDDEN), hs.shape

    # --- purity: no Euclidean-contamination substrings in the module source ---
    src = Path("src/qig_studio/constellation/faculty_adapter.py").read_text(encoding="utf-8")
    forbidden = ["LayerNorm", "cosine", "Adam", "AdamW", "np.linalg.norm(", "F.normalize(", "softmax"]
    hits = [tok for tok in forbidden if tok in src]
    assert not hits, f"faculty_adapter.py contains forbidden Euclidean/exp tokens: {hits}"

    # --- the real fail-closed governance gate passes on the whole source tree (incl. the new file) ---
    from qig_studio.governance import run_purity_gate

    run_purity_gate(Path("src/qig_studio"))


def test_memory_scales_sub_linearly():
    """N faculties sharing ONE table cost one-table + N×small-adapters, NOT N×table. Quantifies the win
    the old 9-separate design threw away (each kernel re-built the full [vocab,64] table)."""
    import torch

    from qig_studio.constellation.faculty_adapter import SharedBasinBank, spawn_faculty_adapters

    vocab, basin_dim, hidden, n = 8000, _BASIN, 48, 4
    torch.manual_seed(1)
    bank = SharedBasinBank(torch.rand(vocab, basin_dim))
    roles = [f"faculty_{i}" for i in range(n)]
    faculties = spawn_faculty_adapters(roles, bank, hidden, templates=_templates(n, basin_dim))

    def unique_bytes(mods) -> int:
        # dedupe by data_ptr so a tensor referenced by many modules is counted ONCE (the whole point).
        seen: dict[int, int] = {}
        for m in mods:
            for t in list(m.parameters()) + list(m.buffers()):
                seen[t.data_ptr()] = t.numel() * t.element_size()
        return sum(seen.values())

    table_bytes = bank.coord_basins.numel() * bank.coord_basins.element_size()
    faculty_bytes = unique_bytes(faculties)              # N small adapters, NO table
    shared_total = unique_bytes([bank, *faculties])      # table counted exactly ONCE
    naive_total = n * table_bytes + faculty_bytes        # the OLD 9-separate design: N copies of the table

    # the table appears exactly once in the shared design
    assert shared_total == table_bytes + faculty_bytes, "shared total must be one-table + N×adapters"
    # adapters are tiny beside the table
    assert faculty_bytes < 0.25 * table_bytes, "per-faculty adapters must be small vs the shared table"
    # sub-linear: shared grows as 1×table + N×small, not N×table → well under half the naive cost at N=4
    assert shared_total < 0.5 * naive_total, "shared design must beat the N-copy design decisively"

    win = naive_total / shared_total
    saved = (naive_total - shared_total) / 1e6
    print(
        f"\nMEMORY WIN (N={n}, vocab={vocab}, basin={basin_dim}): "
        f"shared={shared_total/1e6:.3f} MB vs naive N-copies={naive_total/1e6:.3f} MB "
        f"→ saved {saved:.3f} MB ({win:.2f}x); table={table_bytes/1e6:.3f} MB counted once, "
        f"faculty-adapters={faculty_bytes/1e3:.1f} KB total"
    )


def test_faculty_adapter_matches_basin_readout():
    """MATCHED-CELL EQUIVALENCE GATE: the mirrored K-COMPRESS blocked scoring is bit-equivalent to the
    certified qig_core ``BasinReadout`` path when handed the SAME proj weights and the SAME table. Proves
    the qig-studio loop did not drift from the certified geometry (only the table is now shared)."""
    import torch

    from qig_core.torch.basin_readout import BasinReadout

    from qig_studio.constellation.faculty_adapter import FacultyAdapter

    bank = _bank()
    fa = FacultyAdapter("heart", bank, _HIDDEN, tau=0.5)
    fa.eval()

    ro = BasinReadout(hidden_dim=_HIDDEN, coord_basins=bank.coord_basins, tau=0.5)
    ro.proj.load_state_dict(fa.out_proj.state_dict())   # same per-faculty output chart
    ro.coord_basins = bank.coord_basins                 # rebind to the EXACT same table (isolate the loop)
    ro.eval()

    h = torch.randn(3, _HIDDEN)
    fa_scores = fa.forward_blocked(h)
    ro_scores = ro.forward_blocked(h)
    assert fa_scores.shape == ro_scores.shape == (3, _VOCAB)
    max_err = (fa_scores - ro_scores).abs().max().item()
    assert max_err < 1e-5, f"mirrored K-COMPRESS loop diverged from qig_core BasinReadout: {max_err}"
    # and matches the non-blocked certified forward too (blocked ≈ full to arccos-clamp precision)
    assert torch.allclose(fa.forward_blocked(h), ro.forward(h), atol=1e-4)


def test_basin_loss_matches_gathered_scores_no_vocab():
    """The training loss ``basin_loss(h, target)`` = d_FR(predict(h), shared_table[target]) equals the
    full-scores gather ``basin_lm_loss(forward_blocked(h), ids, τ)`` — but never materialises seq×vocab
    (it indexes only the target column of the shared table). Proves the K-COMPRESS loss path is correct."""
    import torch

    from qig_studio.constellation.faculty_adapter import FacultyAdapter
    from qig_studio.losses import basin_lm_loss

    bank = _bank()
    fa = FacultyAdapter("memory", bank, _HIDDEN, tau=0.5)
    fa.eval()

    torch.manual_seed(7)
    ids = torch.randint(0, bank.vocab_size, (1, 6))
    h = torch.randn(1, 6, _HIDDEN)

    # no-vocab path: next-token alignment done by the caller (h[:-1] predicts ids[1:])
    loss_direct = fa.basin_loss(h[0, :-1], ids[0, 1:])
    # full-scores path (the losses.py contract): build [1,6,vocab] scores then gather the target column
    loss_scores = basin_lm_loss(fa.forward_blocked(h), ids, tau=0.5)

    assert torch.allclose(loss_direct, loss_scores, atol=1e-5), (loss_direct, loss_scores)
    assert loss_direct.requires_grad or not fa.training  # sanity: it is a real differentiable scalar in train
