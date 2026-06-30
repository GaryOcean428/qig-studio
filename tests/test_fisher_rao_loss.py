"""Unit tests for the P20-pure Fisher-Rao language loss (replaces CE/KL).

CONTRACT (matches the wired kernel path, genesis_kernel.train_step): ``logits`` and ``ids`` are
EQUAL-length [1, T, V] / [1, T]; ``fisher_rao_lm_loss`` scores the NEXT-TOKEN prediction, i.e.
``logits[0, :-1]`` (positions 0..T-2) against the one-hot of ``ids[0, 1:]`` (tokens 1..T-1) — the
exact alignment of the ``F.cross_entropy(logits[0,:-1], ids[0,1:])`` it replaces. The fixtures below
build logits peaked at the NEXT token (position k peaked at ids[k+1]); the original draft used
mismatched lengths (logits length T, ids length T+1) which cannot satisfy the documented
``mean_t 2·arccos(√ p_t[target_t])`` next-token contract (scatter_ shape error). Loss body unchanged
from spec; only the fixtures were made shape-consistent with the next-token contract (spec license:
"adjust the TEST ... only if the loss is mathematically correct" — it is). See task report.
"""
import torch
from qig_studio.losses import fisher_rao_lm_loss


def test_fr_loss_zero_when_perfect():
    # ids=[3,1,0]; next-token targets for positions 0,1 are ids[1]=1, ids[2]=0.
    logits = torch.full((1, 3, 4), -10.0)
    logits[0, 0, 1] = 10.0  # position 0 predicts ids[1] == 1
    logits[0, 1, 0] = 10.0  # position 1 predicts ids[2] == 0
    ids = torch.tensor([[3, 1, 0]])
    assert fisher_rao_lm_loss(logits, ids).item() < 0.05


def test_fr_loss_large_when_wrong():
    logits = torch.zeros((1, 3, 4))  # uniform → mass 1/4 on target → d_FR = 2·arccos(1/2) ≈ 2.09
    ids = torch.tensor([[0, 1, 2]])
    assert fisher_rao_lm_loss(logits, ids).item() > 0.5


def test_fr_loss_differentiable():
    logits = torch.randn((1, 4, 5), requires_grad=True)
    ids = torch.tensor([[0, 1, 2, 3]])
    fisher_rao_lm_loss(logits, ids).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
