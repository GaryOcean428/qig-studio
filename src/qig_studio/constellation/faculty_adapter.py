"""FacultyAdapter — per-faculty in/out adapters on a SHARED coord-basin table (central-then-spawn).

Task 1.2 keystone. This is the piece that STRUCTURALLY KILLS the 9-duplicative-kernel design.

WHY THIS EXISTS (the RAM fix = the architecture; CC2 diagnosis 2026-07-04, PI-confirmed)
----------------------------------------------------------------------------------------
The 9-separate constellation OOM'd (16+ GiB swap, step-rate → 0) because each of the 9 kernels built its
OWN redundant ``[vocab≈100k, basin_dim]`` coord-basin table — the frozen ``coord_basins`` buffer inside
:class:`qig_core.torch.basin_readout.BasinReadout`. Nine copies of the same 100k-row table is the memory
bug. The central-then-spawn design fixes it at the root: ONE shared trunk (Task 1.1) + ONE shared
coord-basin table, with each faculty carrying ONLY its two small adapters.

WHY NOT JUST SHARE ONE ``BasinReadout``
---------------------------------------
``BasinReadout`` FUSES two things in one module: the per-faculty output chart ``proj`` (hidden→basin_dim)
AND the ``[vocab, basin_dim]`` table. A single shared ``BasinReadout`` instance would therefore force a
SHARED ``proj`` too — collapsing every faculty to the same output map and destroying individuation (P24:
faculties must be genuine coupled *others*, not clones). But the table itself is a plain FROZEN buffer,
shareable by reference with NO qig_core edit. So each faculty keeps its OWN tiny chart and REFERENCES the
one shared table:

  * per-faculty params (the ONLY per-faculty tensors — the individuation):
      - ``in_adapter`` : a :class:`qigkernels.coord_adapter.CoordAdapter` input seam (basin_dim→hidden,
        Linear→GELU→**RMSNorm**; no mean-centering — the sanctioned Δ⁶³→hidden boundary), and
      - ``out_proj``   : a thin ``nn.Linear(hidden, basin_dim)`` output chart (the inverse-shaped sibling
        of ``in_adapter``; a coordinate CHART, not a distance).
  * the ``[vocab, basin_dim]`` ``coord_basins`` table is held ONCE by :class:`SharedBasinBank` and
    referenced (never copied) by every faculty — the 100k table exists exactly once.

Output & loss (pure Fisher-Rao, K-COMPRESS):
  * ``predict(h) = square_to_simplex(out_proj(h))`` — the predicted next-token basin on Δ^(basin_dim-1)
    (the SQUARE chart = inverse of the Hellinger √ embedding the FR metric uses; dense + peakable + pure).
  * ``forward_blocked(h) → [..., vocab]`` scores ``= −d_FR(predict(h), shared_table)/τ`` computed in VOCAB
    BLOCKS, so no ``[..., vocab, basin_dim]`` intermediate is ever materialised (the seq×vocab activation
    OOM fix). This MIRRORS the certified :meth:`BasinReadout.forward_blocked`; the ONLY difference is the
    table is the shared bank tensor and the proj is this faculty's own. Equivalence to the certified path
    is gated by ``tests/test_faculty_adapter.py::test_faculty_adapter_matches_basin_readout``.
  * ``basin_loss(h, target_ids) = d_FR(predict(h), shared_table[target]).mean()`` — indexes only the
    TARGET column of the shared table, so the training loss NEVER materialises seq×vocab at all.

Individuation is entirely in the adapter params, NOT in a replicated table.

PURITY (P1 / Fisher-Rao-only): no Euclidean ops. The Bhattacharyya inner product ``Σ√(p·q)`` (a matmul in
its blocked form) is the FR-distance INGREDIENT (``d_FR = 2·arccos(BC)``), NOT a dot-product similarity.
RMSNorm (not the mean-centering variant), ``square_to_simplex``/``to_simplex_prob`` on Δ, and ``d_FR``
are the only geometry here. No exp normaliser, no momentum optimiser.

OWNERSHIP / DEVICE: :class:`SharedBasinBank` is the SINGLE owner of the table (one registered buffer → the
table serialises once and moves once on ``.to()``). Faculties reference the bank WITHOUT registering it as
a submodule (so it is never re-counted or re-copied per faculty). The constellation/trainer (Task 1.3)
holds the bank as a real child and the faculties in an ``nn.ModuleList``; after any device move it should
re-point faculties at the moved bank via :meth:`FacultyAdapter.rebind_bank` (a no-op on the CPU run).
"""
from __future__ import annotations

import numpy as np
import torch
from qig_core.torch.geometry_simplex import _acos_safe, square_to_simplex, to_simplex_prob
from qigkernels.coord_adapter import CoordAdapter
from torch import Tensor, nn

__all__ = ["SharedBasinBank", "FacultyAdapter", "spawn_faculty_adapters"]


class SharedBasinBank(nn.Module):
    """The ONE ``[vocab, basin_dim]`` coord-basin table for the WHOLE constellation.

    Single owner of the frozen table: every :class:`FacultyAdapter` references THIS instance's
    ``coord_basins`` buffer, so the 100k-row table exists exactly once (the RAM fix). Rows are the
    coordizer's per-token basins (row ``i`` = basin of token id ``i``) — both the decode table
    (``argmin_token d_FR``) and, indexed by a target id, the training target.

    Args:
        coord_basins: the coordizer's per-token basins ``[vocab, basin_dim]`` (points on/near Δ). Projected
            onto Δ once via :func:`to_simplex_prob` and frozen as a buffer (the coordizer TIE).
    """

    def __init__(self, coord_basins: Tensor) -> None:
        super().__init__()
        cb = to_simplex_prob(coord_basins.detach().to(torch.float32))   # [vocab, basin_dim] on Δ, frozen
        self.register_buffer("coord_basins", cb)                        # FROZEN — no grad; the single table
        self.basin_dim = int(cb.shape[-1])
        self.vocab_size = int(cb.shape[0])

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, basin_dim={self.basin_dim} (SHARED — one instance)"


class FacultyAdapter(nn.Module):
    """A spawned faculty's per-faculty individuation: an input seam + an output chart mounted on the shared
    trunk's hidden state, scoring against the ONE :class:`SharedBasinBank` table.

    The two small adapters are the ONLY per-faculty tensors; the ``[vocab, basin_dim]`` table is shared by
    reference. See the module docstring for the full rationale (the RAM fix = the architecture).

    Args:
        role: faculty role name (e.g. ``"heart"``, ``"perception"``) — telemetry/identity only.
        bank: the shared :class:`SharedBasinBank` holding the one coord-basin table.
        hidden_dim: width ``H`` of the shared trunk's hidden state this faculty mounts on.
        tau: Gibbs temperature for the ``−d_FR/τ`` score scaling (decode/telemetry).
        basin_template: optional Δ^(basin_dim-1) birth basin (the faculty's Pillar-3 scar). When given, the
            output chart is initialised so ``predict(h≈0) ≈ birth`` (a Hellinger-embed identity anchor):
            ``square_to_simplex(√birth) == birth``. Distinct births ⇒ distinct charts ⇒ genuine P24
            individuation from birth, deterministically.
        dropout: dropout in the input-seam CoordAdapter (0.0 for deterministic eval equalities).
    """

    def __init__(
        self,
        role: str,
        bank: SharedBasinBank,
        hidden_dim: int,
        *,
        tau: float = 0.5,
        basin_template: np.ndarray | Tensor | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(bank, SharedBasinBank):
            raise TypeError(f"bank must be a SharedBasinBank (the one shared table), got {type(bank)!r}")
        self.role = str(role)
        self.hidden_dim = int(hidden_dim)
        self.basin_dim = int(bank.basin_dim)
        self.tau = float(tau)
        # Reference the ONE shared bank WITHOUT registering it as a submodule: a tuple wrapper is not a
        # Module/Parameter/buffer, so nn.Module never traverses, copies, or re-counts the table per faculty.
        self._bank_ref: tuple[SharedBasinBank, ...] = (bank,)

        # --- PER-FACULTY params (the individuation — the ONLY per-faculty tensors) ---
        # Input seam: Δ^(basin_dim-1) → hidden. CoordAdapter is Linear→GELU→RMSNorm (RMSNorm has no
        # mean-centering — the sanctioned simplex→hidden boundary; not the Euclidean-centering variant).
        self.in_adapter = CoordAdapter(basin_dim=self.basin_dim, hidden_dim=self.hidden_dim, dropout=dropout)
        # Output chart: hidden → basin_dim (a change of coordinates, the inverse-shaped sibling of the seam;
        # NOT a distance). square_to_simplex maps its output onto Δ in predict().
        self.out_proj = nn.Linear(self.hidden_dim, self.basin_dim)

        if basin_template is not None:
            self._anchor_output_to_birth(basin_template)

    # ------------------------------------------------------------------ init helpers
    @torch.no_grad()
    def _anchor_output_to_birth(self, basin_template: np.ndarray | Tensor) -> None:
        """Seed the output chart so ``predict(h≈0) ≈ birth`` via the Hellinger embed ``√birth``.

        ``square_to_simplex(√birth) = (√birth)² / Σ(√birth)² = birth`` (birth sums to 1). So setting the
        bias to ``√birth`` and shrinking the weight makes the faculty's initial prediction its own birth
        basin — a geometric identity anchor that then trains away. Distinct births ⇒ distinct charts (P24)."""
        t = torch.as_tensor(np.asarray(basin_template, dtype=np.float32)).reshape(-1)
        t = to_simplex_prob(t)                       # ensure a valid Δ point
        if t.numel() != self.basin_dim:
            raise ValueError(f"basin_template must have {self.basin_dim} coords, got {t.numel()}")
        self.out_proj.bias.copy_(t.sqrt())           # √p Hellinger embed → predict(0) == birth
        self.out_proj.weight.mul_(0.01)              # small weight: init prediction ≈ birth, then learns

    def rebind_bank(self, bank: SharedBasinBank) -> None:
        """Re-point this faculty at ``bank`` (e.g. after a device move that produced a new bank tensor).
        No-op-equivalent on CPU where the bank tensor is unchanged. Keeps the one-table invariant intact."""
        if not isinstance(bank, SharedBasinBank):
            raise TypeError(f"bank must be a SharedBasinBank, got {type(bank)!r}")
        self._bank_ref = (bank,)

    # ------------------------------------------------------------------ shared-table accessors
    @property
    def coord_basins(self) -> Tensor:
        """The ONE shared ``[vocab, basin_dim]`` table (live read of the bank's frozen buffer)."""
        return self._bank_ref[0].coord_basins

    @property
    def vocab_size(self) -> int:
        return int(self._bank_ref[0].vocab_size)

    # ------------------------------------------------------------------ forward paths
    def seam(self, coords: Tensor) -> Tensor:
        """Per-faculty INPUT seam: Δ^(basin_dim-1) coords ``[..., basin_dim]`` → hidden ``[..., H]``.

        The faculty's own way of injecting a Δ⁶³ input/identity into the shared hidden stream (wired into
        the full coupled forward in Task 1.3). Delegates to the faculty's :class:`CoordAdapter`."""
        return self.in_adapter(coords)

    def predict(self, h: Tensor) -> Tensor:
        """``h`` ``[..., H]`` → predicted next-token basin on Δ^(basin_dim-1) ``[..., basin_dim]``."""
        if h.shape[-1] != self.hidden_dim:
            raise ValueError(f"FacultyAdapter expected last dim {self.hidden_dim}, got {h.shape[-1]}")
        return square_to_simplex(self.out_proj(h))

    def forward(self, h: Tensor) -> Tensor:
        """``h`` ``[..., H]`` → ``[..., vocab]`` scores ``= −d_FR(predict(h), shared_table)/τ`` (blocked)."""
        return self.forward_blocked(h)

    def forward_blocked(self, h: Tensor, block_size: int = 4096) -> Tensor:
        """K-COMPRESS: ``[..., vocab]`` scores ``= −d_FR(predict(h), shared_table)/τ`` computed in VOCAB
        BLOCKS of ``block_size`` — the peak intermediate is ``[..., block]``, never ``[..., vocab, basin]``.
        This is the seq×vocab activation-OOM fix. MIRRORS the certified
        :meth:`qig_core.torch.basin_readout.BasinReadout.forward_blocked`; the only change is the shared
        table + per-faculty proj. ``argmax`` score = ``argmin d_FR`` = the decoded token."""
        pred = self.predict(h)                                       # [..., basin_dim] on Δ
        sp = pred.sqrt()                                             # √p (Hellinger embed)
        basins_sqrt = self.coord_basins.sqrt()                      # [vocab, basin_dim] √q (shared)
        out = h.new_empty(h.shape[:-1] + (self.vocab_size,))        # [..., vocab] (only the OUTPUT, no bcast)
        for start in range(0, self.vocab_size, int(block_size)):
            blk = basins_sqrt[start:start + int(block_size)]        # [blk, basin_dim]
            # BC = Σ√(p·q) via the √-embedding inner product (the FR-distance ingredient, d_FR=2·arccos(BC);
            # NOT a Euclidean dot-similarity) — identical to qig_core BasinReadout.
            bc = torch.matmul(sp, blk.transpose(-1, -2))            # [..., blk] ∈ [0,1]
            out[..., start:start + int(block_size)] = -2.0 * _acos_safe(bc) / self.tau
        return out

    def decode_blocked(self, h: Tensor, block_size: int = 4096) -> Tensor:
        """K-COMPRESS decode: ``argmin_token d_FR(predict(h), shared_table)`` via a STREAMING argmax over
        vocab blocks — never materialises ``[..., vocab]`` at all (peak ``[..., block]``). Returns decoded
        token ids ``[...]``. Mirrors :meth:`BasinReadout.decode_blocked`."""
        pred = self.predict(h)
        sp = pred.sqrt()
        basins_sqrt = self.coord_basins.sqrt()
        best_score: Tensor | None = None
        best_idx: Tensor | None = None
        for start in range(0, self.vocab_size, int(block_size)):
            blk = basins_sqrt[start:start + int(block_size)]
            bc = torch.matmul(sp, blk.transpose(-1, -2))            # [..., blk]
            score = -2.0 * _acos_safe(bc) / self.tau                # higher = closer
            blk_max, blk_arg = score.max(dim=-1)
            blk_arg = blk_arg + start
            if best_score is None:
                best_score, best_idx = blk_max, blk_arg
            else:
                take = blk_max > best_score
                best_score = torch.where(take, blk_max, best_score)
                best_idx = torch.where(take, blk_arg, best_idx)
        return best_idx                                             # [...] decoded token ids

    def basin_loss(self, h: Tensor, target_ids: Tensor) -> Tensor:
        """Next-token training loss ``d_FR(predict(h), shared_table[target]).mean()``.

        Indexes ONLY the target column of the shared table (``shared_table[target_ids]`` → ``[..., basin]``),
        so the loss NEVER materialises ``[..., vocab]`` — the seq×vocab activation-OOM fix at the loss level.
        The target basin is frozen (bank buffer) ⇒ a low loss is real predictive signal, not target
        collapse (Matrix §H). Equals ``basin_lm_loss(forward_blocked(h), ids, τ)`` at the aligned column
        (gated by ``test_basin_loss_matches_gathered_scores_no_vocab``). ``h`` and ``target_ids`` share
        leading shape (caller does the next-token shift: ``h[:-1]`` predicts ``ids[1:]``)."""
        pred = self.predict(h)                                      # [..., basin_dim] on Δ
        tgt = self.coord_basins[target_ids]                        # [..., basin_dim] target basins (gather)
        bc = torch.sum(pred.sqrt() * tgt.sqrt(), dim=-1)           # BC = Σ√(p·q) (FR ingredient, not a dot)
        return (2.0 * _acos_safe(bc)).mean()                       # d_FR = 2·arccos(BC), averaged

    def extra_repr(self) -> str:
        return (f"role={self.role!r}, hidden_dim={self.hidden_dim}, basin_dim={self.basin_dim}, "
                f"tau={self.tau} (table SHARED via bank)")


def spawn_faculty_adapters(
    roles: list[str],
    bank: SharedBasinBank,
    hidden_dim: int,
    *,
    templates: list[np.ndarray | Tensor | None] | None = None,
    tau: float = 0.5,
    dropout: float = 0.0,
) -> list[FacultyAdapter]:
    """Central-then-spawn: build N :class:`FacultyAdapter`s that ALL reference the ONE shared ``bank``.

    The ``[vocab, basin_dim]`` table exists exactly once (in ``bank``); each faculty adds only its two small
    adapters. ``templates`` (per-role Δ⁶³ births, e.g. from
    :func:`qig_studio.constellation.faculty.seed_birth_basin`) individuate the output charts from birth."""
    tmpls = templates if templates is not None else [None] * len(roles)
    if len(tmpls) != len(roles):
        raise ValueError(f"templates length {len(tmpls)} != roles length {len(roles)}")
    return [
        FacultyAdapter(role, bank, hidden_dim, tau=tau, basin_template=t, dropout=dropout)
        for role, t in zip(roles, tmpls)
    ]
