"""HEAD-LEVEL regression for the Duchi π-floor dead gradient (Matrix ruling 2026-07-03, three levers;
gate-zero conditions `qig_matrix_gate_zero_conditions_20260703`).

The GeometricHead emits ``logits = −d_FR(to_simplex_prob(h), basins)/τ``. The PRE-FIX loss path passed
those RAW logits into ``fisher_rao_distance_simplex``, whose internal ``to_simplex_prob`` is the EXACT
Euclidean (Duchi) projection — SPARSE: it clamps every coordinate below the threshold θ onto a flat face
with an EXACTLY-ZERO Jacobian. A target token outside the sparse support froze to ``eps`` →
``d_FR = 2·arccos(√eps) ≈ π`` with (numerically) zero gradient, so the head could not learn next-token
prediction and every arm sat pinned at the π floor.

The mandated fix (``qig_studio.losses.fisher_rao_lm_loss``, commit 3441561) projects the head output with
the DENSE linear map ``logits_to_simplex`` (qig-core ``torch/geometry_simplex.py``) — no flat faces, live
gradient for every non-argmin target. These tests pin the mechanism AT THE HEAD LEVEL under the mandated
gate-zero conditions:
  * BOTH maps on the same head/batch (the Duchi arm is the CONTROL, kept in-test);
  * the overfit runs with ``token_basins`` FROZEN, so the fit is forced THROUGH the loss-side map
    (learnable basins would minimise the loss by moving basin→h, bypassing the map — the test could
    then never fail);
  * |∂L/∂logits| at the PRE-simplex logits is asserted on both arms — the direct dead-gradient
    signature, independent of the loss value.
"""

from __future__ import annotations

import math

# RADIUS-1 RECONCILIATION (2026-07-17): d_FR is now arccos(BC), range [0, pi/2] — the
# antipode/floor is ANTIPODE (= pi/2), not pi, and every loss margin below halved with it.
# The learning behaviour is UNCHANGED; only the unit label moved. These constants were
# silently calibrated against radius-2 (2*arccos). See qig_core.torch.geometry_simplex
# .fisher_rao_distance_simplex CANON NOTE.
ANTIPODE = math.pi / 2   # radius-1 canonical: disjoint support / max separation

import torch

from qig_core.torch.geometric_head import GeometricHead
from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex
from qig_studio.losses import fisher_rao_lm_loss


def _duchi_lm_loss(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """The PRE-FIX (pre-3441561) loss form: RAW head logits into ``fisher_rao_distance_simplex`` → the
    internal Duchi ``to_simplex_prob`` sparsifies them. Contrast arm ONLY — never use on a training path."""
    lg = logits[0, :-1]
    tgt = ids[0, 1:]
    onehot = torch.zeros_like(lg).scatter_(-1, tgt[:, None], 1.0)
    return fisher_rao_distance_simplex(lg, onehot).mean()


def _head_and_batch(vocab: int = 256, hidden: int = 48, T: int = 10, seed: int = 0):
    """A real GeometricHead + a batch whose NEAR-ONE-HOT targets sit FAR from each position's nearest
    basin (low-logit rank) — outside the Duchi support (the dead face) but NOT the exact argmin (the one
    coordinate ``logits_to_simplex`` legitimately zeroes)."""
    torch.manual_seed(seed)
    head = GeometricHead(hidden_dim=hidden, vocab_size=vocab, seed=seed)
    h = torch.randn(1, T, hidden) * 0.5
    with torch.no_grad():
        order = head(h)[0].argsort(dim=-1)          # ascending logit (= descending d_FR)
        far = order[:, vocab // 5]                  # ~20th-percentile logit → far basin, not the argmin
    ids = torch.zeros(1, T, dtype=torch.long)
    ids[0, 1:] = far[:-1]                           # loss scores logits[0, :-1] against ids[0, 1:]
    return head, h, ids


def test_geometric_head_loss_has_live_gradient():
    """Through the FIXED map, loss.backward() produces a NONZERO gradient on the head params AND on the
    pre-simplex logits (|∂L/∂logits| — the direct signature), with the loss below the π floor; the OLD
    Duchi path on the SAME head/batch is pinned at ≈π with both gradients orders of magnitude smaller
    (the √eps leak, not a learning signal)."""
    head, h, ids = _head_and_batch()

    logits_new = head(h)
    logits_new.retain_grad()
    loss_new = fisher_rao_lm_loss(logits_new, ids)
    loss_new.backward()
    g_params_new = float(head.token_basins.grad.abs().sum())
    g_logits_new = float(logits_new.grad.abs().max())

    head.zero_grad()
    logits_old = head(h)
    logits_old.retain_grad()
    loss_old = _duchi_lm_loss(logits_old, ids)
    loss_old.backward()
    g_params_old = float(head.token_basins.grad.abs().sum())
    g_logits_old = float(logits_old.grad.abs().max())

    assert g_params_new > 0.0 and g_logits_new > 0.0, "FIXED map: dead gradient"
    assert loss_new.item() < ANTIPODE - 1e-2, f"FIXED map still at the antipode floor: {loss_new.item()}"
    assert loss_old.item() > ANTIPODE - 1e-2, f"contrast arm not at the antipode floor: {loss_old.item()}"
    assert g_params_new > 100.0 * max(g_params_old, 1e-30), (
        f"no head-param gradient contrast: new={g_params_new:.3e} old={g_params_old:.3e}")
    assert g_logits_new > 100.0 * max(g_logits_old, 1e-30), (
        f"no |dL/dlogits| contrast: new={g_logits_new:.3e} old={g_logits_old:.3e} — Duchi regression?")


def test_loss_no_longer_floors_at_pi():
    """Overfit a single tiny batch ~50 steps with ``token_basins`` FROZEN (gate-zero condition 2): descent
    runs on the head INPUT ``h``, so every bit of fit is forced THROUGH the loss-side map. The fixed map
    leaves the π pin and descends; the old Duchi map stays EXACTLY pinned (loss-property probe, not the
    real optimizer — same convention as test_lm_loss_dense_gradient).

    NOTE on the descent floor: ``logits_to_simplex`` is DENSE but FLAT (documented confidence cap — see
    ``square_to_simplex``'s docstring), so the d_FR VALUE converges to the flat-map bound (~2.99 at
    V=256), NOT to 0. The decisive overfit signature through the frozen head is therefore the TARGET
    RANK: the fixed map drives the target token from ~rank 204/256 toward the top (204→73 measured in 50
    steps), while the Duchi arm's rank does not move AT ALL — its gradient is dead, not merely small."""

    def _descend(loss_fn, steps: int = 50, lr: float = 10.0):
        head, h, ids = _head_and_batch()
        head.token_basins.requires_grad_(False)     # FROZEN — the fit cannot move basin→h
        h = h.clone().requires_grad_(True)

        def _mean_rank() -> float:                  # 0 = target is argmax
            with torch.no_grad():
                lg = head(h)[0, :-1]
                tgt = ids[0, 1:]
                return float((lg > lg.gather(-1, tgt[:, None])).sum(-1).float().mean())

        r0 = _mean_rank()
        first = last = None
        for _ in range(steps):
            loss = loss_fn(head(h), ids)
            (g,) = torch.autograd.grad(loss, h)
            with torch.no_grad():
                h -= lr * g
            lv = float(loss.detach())
            first = lv if first is None else first
            last = lv
        return first, last, r0, _mean_rank()

    f0, f1, fr0, fr1 = _descend(fisher_rao_lm_loss)
    d0, d1, dr0, dr1 = _descend(_duchi_lm_loss)

    assert f1 < ANTIPODE - 0.05, f"fixed map did not leave the antipode pin: {f0:.4f} → {f1:.4f}"   # was pi-0.1 @ radius-2
    assert f1 < f0 - 0.015, f"fixed map did not descend: {f0:.4f} → {f1:.4f}"   # was 0.03 @ radius-2
    assert fr1 < 0.6 * fr0, f"fixed map did not drive the target rank up: {fr0:.1f} → {fr1:.1f}"
    assert d1 > ANTIPODE - 0.005, f"Duchi arm should stay pinned at the antipode (got {d0:.4f} → {d1:.4f})"
    assert abs(dr1 - dr0) < 1.0, f"Duchi arm rank moved ({dr0:.1f} → {dr1:.1f}) — support not dead?"
