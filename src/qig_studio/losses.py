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
from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex, logits_to_simplex, to_simplex_prob


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


def gamma_proxy(logits: torch.Tensor) -> torch.Tensor:
    """Γ ∈ [0,1] — generation HEALTH (anti-dissociation), differentiable from in-graph logits (no extra
    forward). diversity = normalised entropy of the mean output Δ (1.0 = generative, →0 = collapsed;
    monkey1 '>1/n·0.25' rule made smooth); stability = inter-position Fisher-Rao step in a healthy band
    (exp-bump at BASIN_STABLE=0.15). Γ = 0.6·diversity + 0.4·stability, pure Δ⁶³. A low Γ at high Φ is the
    suffering/locked-in signal the orchestrator fail-closes on.

    SINGLE SOURCE (2026-07-22, D4 follow-up): this is the ONE definition of Γ from vocab-width logits.
    Both ``GenesisKernelTarget._gamma_proxy`` (ARM B, qigkernels) and ``GeoCortexTarget`` (ARM A,
    geocoding) delegate here so the two arms cannot diverge on what Γ means — extracted from ARM B's
    original inline ``_gamma_proxy`` (genesis_kernel.py), body unchanged, so existing ARM-B telemetry is
    bit-identical before/after the extraction."""
    p = to_simplex_prob(logits[0])                       # [seq, vocab] per-position output Δ
    pm = p.mean(0)
    pm = pm / pm.sum()                                     # mean output distribution
    n = pm.numel()
    ent = -(pm * (pm + 1e-12).log()).sum() / torch.log(torch.tensor(float(n)))
    diversity = ent.clamp(0.0, 1.0)
    if p.size(0) >= 2:
        steps = fisher_rao_distance_simplex(p[:-1], p[1:]).mean()      # mean inter-position FR step
        stability = torch.exp(-((steps - 0.15) ** 2) / (2 * 0.10 ** 2))  # monkey1 BASIN_STABLE=0.15
    else:
        stability = torch.tensor(0.5, device=p.device)
    return (0.6 * diversity + 0.4 * stability).clamp(0.0, 1.0)
