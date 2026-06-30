"""Fisher-Rao language loss — the P20-pure replacement for cross-entropy (CE=KL is forbidden).
L = mean_t 2·arccos(√ p_t[target_t]), p_t = to_simplex_prob(logits_t). Pure Δ⁶³, no softmax/KL."""
from __future__ import annotations

import torch
from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex


def fisher_rao_lm_loss(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    lg = logits[0, :-1]                        # [T-1, V] predict next token
    tgt = ids[0, 1:]                           # [T-1]
    onehot = torch.zeros_like(lg).scatter_(-1, tgt[:, None], 1.0)
    return fisher_rao_distance_simplex(lg, onehot).mean()
