"""Task 1.1 — ConstellationTrunk: ONE shared geometric core (coordizer-agnostic hidden()).

TDD gate for the shared-trunk keystone. The trunk wraps ONE ``qigkernels.Kernel`` and exposes
``hidden(input_ids, coords=None) -> h[B,T,H]`` — the forward-minus-``lm_head`` path (the coordizer-
agnostic byte/Fourier core) that faculties will later mount adapters on (Task 1.2). It owns ONE
natural-gradient optimizer over its params (NOT a Euclidean momentum optimiser).

These run only where the heavy deps (torch + qigkernels + qig_core) are present; skipped otherwise so
the light shell's CI stays green (mirrors tests/test_geo_cortex.py).
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


pytestmark = pytest.mark.skipif(not _deps(), reason="ConstellationTrunk needs torch + qigkernels + qig_core")

# TINY test config — byte vocab, 2 layers. hidden_dim (32) is deliberately != vocab_size (256) so the
# [B,T,H] hidden shape is unambiguously the hidden dim, never the vocab dim.
_VOCAB = 256
_HIDDEN = 32
_LAYERS = 2
_HEADS = 4
_FFN = 64
_MAXPOS = 64


def _build_trunk():
    import torch

    from qig_studio.constellation.trunk import ConstellationTrunk

    torch.manual_seed(0)
    trunk = ConstellationTrunk(
        vocab_size=_VOCAB,
        hidden_dim=_HIDDEN,
        num_layers=_LAYERS,
        num_heads=_HEADS,
        ffn_dim=_FFN,
        dropout=0.0,
        max_position_embeddings=_MAXPOS,
    )
    trunk.eval()  # deterministic (dropout off, no decoherence) so the shared-core equalities hold bit-for-bit
    return trunk


def _fixed_input():
    import torch

    torch.manual_seed(123)
    return torch.randint(0, _VOCAB, (2, 5))


def test_trunk_hidden_is_shared():
    """Two ``trunk.hidden(same_x)`` calls return the SAME tensor; shape is [B,T,H] (hidden dim, NOT vocab)."""
    import torch

    trunk = _build_trunk()
    x = _fixed_input()

    h1 = trunk.hidden(x)
    h2 = trunk.hidden(x)

    assert torch.equal(h1, h2), "shared trunk must be deterministic in eval mode (same core, same tensor)"
    assert tuple(h1.shape) == (2, 5, _HIDDEN), h1.shape
    assert h1.shape[-1] == _HIDDEN != _VOCAB, "hidden() must return the HIDDEN dim, not vocab logits"


def test_trunk_hidden_matches_kernel_prehead():
    """``trunk.hidden(x)`` equals the wrapped Kernel's forward-minus-``lm_head`` (the pre-head hidden state)."""
    import torch

    trunk = _build_trunk()
    x = _fixed_input()

    h = trunk.hidden(x)

    # Reference pre-head hidden state straight from the wrapped kernel's own documented skip_head path.
    _, tel = trunk.kernel.forward(x, return_telemetry=True, skip_head=True)
    ref_prehead = tel.hidden_state
    assert torch.equal(h, ref_prehead), "hidden() must be the kernel's pre-head hidden state"

    # And applying the head to hidden(x) must reproduce the FULL forward logits exactly — i.e. hidden() is
    # genuinely forward MINUS lm_head.
    assert torch.equal(trunk.kernel.lm_head(h), trunk.kernel.forward(x))


def test_trunk_is_pure():
    """No Euclidean-contamination tokens in trunk.py; the optimizer is a natural-gradient type."""
    from qig_core.torch.natural_gradient import DiagonalNaturalGradient

    src = Path("src/qig_studio/constellation/trunk.py").read_text(encoding="utf-8")
    forbidden = ["LayerNorm", "cosine", "Adam", "AdamW", "np.linalg.norm(", "F.normalize("]
    hits = [tok for tok in forbidden if tok in src]
    assert not hits, f"trunk.py contains forbidden Euclidean tokens: {hits}"

    trunk = _build_trunk()
    opt = trunk.optimizer
    assert isinstance(opt, DiagonalNaturalGradient), type(opt)
    assert "NaturalGradient" in type(opt).__name__
    assert "Adam" not in type(opt).__name__


def test_forward_unchanged():
    """Adding ``Kernel.hidden()`` left ``Kernel.forward`` bit-identical.

    ``hidden()`` is a pure additive wrapper over forward's OWN ``skip_head`` path (forward's body is
    untouched), so forward stays deterministic AND ``forward(x) == lm_head(hidden(x))``.
    """
    import torch

    trunk = _build_trunk()
    kernel = trunk.kernel
    x = _fixed_input()

    logits_a = kernel.forward(x)
    logits_b = kernel.forward(x)
    assert torch.equal(logits_a, logits_b), "forward must be deterministic in eval mode"

    assert torch.equal(kernel.lm_head(kernel.hidden(x)), logits_a), "forward must equal hidden() + lm_head"
