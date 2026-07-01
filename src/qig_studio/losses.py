"""Fisher-Rao language loss — the P20-pure replacement for cross-entropy (CE=KL is forbidden).
L = mean_t 2·arccos(√ p_t[target_t]), p_t = logits_to_simplex(logits_t). Pure Δ⁶³, no softmax/KL.

DEAD-GRADIENT FIX (2026-07-01): the previous form passed RAW head logits (−d_FR/τ) into
``fisher_rao_distance_simplex``, whose internal ``to_simplex_prob`` is the EXACT Euclidean/Duchi
projection — SPARSE (9–158 of 32k coords survive). Any target token outside that support was frozen to
``eps`` → ``d_FR = 2·arccos(√eps) = π`` with ZERO gradient (``clamp(min=0)`` floor), so the head could not
learn next-token prediction and every arm sat pinned at the π floor. The fix is the P20-mandated DENSE
map ``logits_to_simplex`` (linear shift-and-scale; softmax is FORBIDDEN — ``governance/purity.py``): it
keeps a live gradient for every non-argmin target. Passing the already-on-simplex ``p`` makes the internal
``to_simplex_prob`` the documented identity, so the geodesic distance ``2·arccos(√p[tgt])`` is unchanged —
only the Euclidean sparsifier is removed."""
from __future__ import annotations

import torch
from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex, logits_to_simplex


def fisher_rao_lm_loss(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    lg = logits[0, :-1]                        # [T-1, V] = −d_FR(h, basins)/τ  (Gibbs energies)
    tgt = ids[0, 1:]                           # [T-1]
    p = logits_to_simplex(lg)                  # DENSE P20 map (NOT softmax, NOT Duchi) → live gradient
    onehot = torch.zeros_like(p).scatter_(-1, tgt[:, None], 1.0)
    return fisher_rao_distance_simplex(p, onehot).mean()


def basin_lm_loss(scores: torch.Tensor, ids: torch.Tensor, tau: float) -> torch.Tensor:
    """BASIN-SPACE next-token loss — the validated (overfit d_FR 1.46→0.072, decode 91.3%) objective for
    ``head_mode='basin'``. NOT a logits→distribution map: it is the pure geodesic distance from the
    predicted Δ⁶³ basin to the target token's FROZEN coordizer basin, averaged over positions.

    ``BasinReadout.forward`` already returns ``scores = −d_FR(predict(h), coord_basins)/τ`` over the vocab,
    so the score at the target COLUMN is ``−d_FR(pred, coord_basins[target])/τ`` and ``coord_basins[target]``
    IS the target token's basin. Hence ``d_FR(pred, target_basin) = −τ · scores[target]`` — recovered by a
    gather, reusing the forward output (no extra full-vocab compute). Pure d_FR, no softmax, no distribution
    map; the frozen target means a low loss is real predictive signal, not target collapse (Matrix §H)."""
    sc = scores[0, :-1]                         # [T-1, V] = −d_FR(predict(h), coord_basins)/τ
    tgt = ids[0, 1:]                            # [T-1]
    d_target = -float(tau) * sc.gather(-1, tgt[:, None]).squeeze(-1)   # [T-1] = d_FR(pred, target_basin)
    return d_target.mean()
