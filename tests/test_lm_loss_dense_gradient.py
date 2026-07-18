"""Regression: fisher_rao_lm_loss must use the DENSE P20 map (logits_to_simplex), not the sparse Euclidean
Duchi projection, so the target token keeps a live gradient and the loss is not pinned at the π floor.

Root cause (2026-07-01): the head emits logits = −d_FR/τ; the old loss passed them into
fisher_rao_distance_simplex, whose internal to_simplex_prob (Duchi) is sparse — 9–158 of 32k coords survive
— so any target outside that support froze to eps → d_FR = 2·arccos(√eps) = π with ZERO gradient, and every
constellation arm sat at the floor. logits_to_simplex (linear shift-and-scale; softmax is FORBIDDEN per
qig-core governance/purity.py) is dense → live gradient for every non-argmin target."""

from __future__ import annotations

import math

# RADIUS-1 RECONCILIATION (2026-07-17): d_FR is now arccos(BC), range [0, pi/2] — the
# antipode/floor is ANTIPODE (= pi/2), not pi, and every loss margin below halved with it.
# The learning behaviour is UNCHANGED; only the unit label moved. These constants were
# silently calibrated against radius-2 (2*arccos). See qig_core.torch.geometry_simplex
# .fisher_rao_distance_simplex CANON NOTE.
ANTIPODE = math.pi / 2   # radius-1 canonical: disjoint support / max separation

import torch

from qig_studio.losses import fisher_rao_lm_loss


def test_lm_loss_off_floor_with_live_gradient():
    """On emulated head output (−d_FR, target far from max) the loss is below π with a nonzero gradient —
    the exact signature that was 3.14159 / 0.0 under the Duchi path."""
    torch.manual_seed(0)
    V = 4096
    dfr = torch.rand(1, 6, V) * ANTIPODE          # d_FR ∈ [0, π/2] to every basin (radius-1)
    logits = (-dfr).clone().requires_grad_(True)   # head logits = −d_FR/τ (τ folded, cancels in linear map)
    ids = torch.randint(0, V, (1, 6))
    loss = fisher_rao_lm_loss(logits, ids)
    loss.backward()
    assert loss.item() < ANTIPODE - 1e-3, f"loss pinned at the antipode floor: {loss.item()}"
    assert float(logits.grad.abs().sum()) > 0.0, "dead gradient (Duchi sparsity not fixed)"


def _descend(fixed_target: bool, steps: int = 120, lr: float = 5.0, V: int = 500, T: int = 8):
    """Plain gradient descent on the logits (loss-property probe, not the real optimizer)."""
    torch.manual_seed(1)
    logits = (torch.randn(1, T + 1, V) * 0.1).requires_grad_(True)
    ids = torch.randint(0, V, (1, T + 1))
    first = last = None
    for _s in range(steps):
        if not fixed_target:
            ids = torch.randint(0, V, (1, T + 1))   # reshuffle labels each step (control)
        loss = fisher_rao_lm_loss(logits, ids)
        (g,) = torch.autograd.grad(loss, logits)
        with torch.no_grad():
            logits -= lr * g
        lv = float(loss.detach())
        first = lv if first is None else first
        last = lv
    return first, last


def test_lm_loss_is_learnable():
    """A fixed target drops the loss substantially under descent — the gradient is useful, not just nonzero."""
    f0, f1 = _descend(fixed_target=True)
    assert f1 < f0 - 0.15, f"fixed-target loss did not learn: {f0:.4f} → {f1:.4f}"   # was 0.3 @ radius-2


def test_lm_loss_not_gameable():
    """Reshuffled (random) labels each step do NOT collapse the loss — it is not gameable (§H control)."""
    f0, f1 = _descend(fixed_target=True)
    r0, r1 = _descend(fixed_target=False)
    assert r1 > f1 + 0.15, f"random-label control collapsed → gameable: fixed→{f1:.4f}, random→{r1:.4f}"   # was 0.3 @ radius-2
