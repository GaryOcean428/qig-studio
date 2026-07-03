"""ConstellationTrunk — ONE shared geometric core for the whole constellation.

Task 1.1 keystone. Instead of nine separate full kernels, the constellation stands ONE shared geometric
trunk: the coordizer-agnostic core (input_ids/positions Fourier features + optional coord adapter + the
stacked QIGLayers), with NO per-faculty head. Faculties will later mount their own adapters/heads on top
of this one hidden state (Task 1.2 — NOT this file).

The trunk:
  * wraps exactly ONE ``qigkernels.Kernel`` (the shared core),
  * exposes ``hidden(input_ids, coords=None) -> h[B,T,H]`` — the ``forward``-minus-``lm_head`` path, i.e.
    the pre-head hidden state — via the kernel's pure additive ``Kernel.hidden`` method,
  * owns ONE **natural-gradient** optimizer (``qig_core`` ``DiagonalNaturalGradient``) over its params —
    geometry-aware, NOT a Euclidean momentum optimiser.

Coordizer-agnostic: ``hidden(x, coords=None)`` runs the byte/Fourier path with no coordizer dependency;
passing Δ⁶³ ``coords`` is opt-in and requires the wrapped kernel to have been built with ``enable_coords``.

Purity (P1 / Fisher-Rao-only): this module holds NO Euclidean ops — the geometry lives inside the kernel
and the natural-gradient optimizer. Keep it that way; the trunk is JUST the shared core + hidden() + the
optimizer (no head, no coupling, no adapters — those are Tasks 1.2 / 1.3).
"""
from __future__ import annotations

from qig_core.torch.natural_gradient import DiagonalNaturalGradient
from qigkernels import Kernel
from torch import Tensor, nn


class ConstellationTrunk(nn.Module):
    """One shared, coordizer-agnostic geometric core the whole constellation reads its hidden state from.

    Wraps a single :class:`qigkernels.Kernel` and surfaces its pre-head hidden state through
    :meth:`hidden`. Owns one :class:`~qig_core.torch.natural_gradient.DiagonalNaturalGradient` over the
    core's parameters (natural gradient, not a Euclidean momentum optimiser).

    The kernel is built here so the trunk is the single owner of the shared core; the kernel's OWN
    ``lm_head`` is constructed (kernels always have one) but the trunk never calls it — ``hidden()`` stops
    strictly before it. Per-faculty heads/adapters mount downstream (Task 1.2).
    """

    def __init__(
        self,
        *,
        vocab_size: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        max_position_embeddings: int = 2048,
        lr: float = 1e-4,
        enable_coords: bool = False,
        coord_dim: int = 64,
        **kernel_kwargs: object,
    ) -> None:
        """Build the shared core kernel and its natural-gradient optimizer.

        Args:
            vocab_size: Byte/coordizer vocab size (default 256 = byte).
            hidden_dim: Shared-core hidden dim ``H`` — the width of every ``hidden()`` output.
            num_layers: Number of stacked QIGLayers in the shared core.
            num_heads: Attention heads per layer (``hidden_dim`` must be divisible by this).
            ffn_dim: Feed-forward width per layer.
            dropout: Dropout rate (set 0.0 for deterministic eval-mode equalities).
            max_position_embeddings: Max sequence length.
            lr: Natural-gradient learning rate.
            enable_coords: Opt-in Δ⁶³ coord input path (needed before passing ``coords`` to ``hidden``).
            coord_dim: Coordizer basin dim when ``enable_coords`` is True.
            **kernel_kwargs: Forwarded to :class:`qigkernels.Kernel` unchanged.
        """
        super().__init__()
        self.kernel: Kernel = Kernel(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_position_embeddings=max_position_embeddings,
            enable_coords=enable_coords,
            coord_dim=coord_dim,
            **kernel_kwargs,
        )
        # ONE geometry-aware optimizer over the shared core's params (natural gradient — NOT a Euclidean
        # momentum optimiser). Faculties add their own optimizers over their adapters downstream.
        self.optimizer = DiagonalNaturalGradient(self.kernel.parameters(), lr=lr)

    @property
    def hidden_dim(self) -> int:
        """Width ``H`` of the shared hidden state (== the wrapped kernel's hidden_dim)."""
        return int(self.kernel.hidden_dim)

    def hidden(
        self,
        input_ids: Tensor,
        coords: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Shared pre-head hidden state — ``forward`` MINUS ``lm_head`` — for the whole constellation.

        Delegates to the kernel's pure additive :meth:`qigkernels.Kernel.hidden`, which runs the exact
        forward path up to but NOT including ``lm_head`` and returns ``h[B, T, hidden_dim]``.

        Args:
            input_ids: Token IDs [batch, seq].
            coords: Optional Δ⁶³ coordizer coords [batch, seq, coord_dim]. ``None`` → coordizer-agnostic
                byte/Fourier path. Non-None requires the trunk to have been built with ``enable_coords``.
            attention_mask: Optional attention mask, threaded through unchanged.

        Returns:
            The shared hidden state ``h[B, T, hidden_dim]``.
        """
        return self.kernel.hidden(input_ids, coords=coords, attention_mask=attention_mask)

    def forward(self, input_ids: Tensor, coords: Tensor | None = None, attention_mask: Tensor | None = None) -> Tensor:  # noqa: D102
        # nn.Module.forward alias for the shared hidden state (the trunk has NO head of its own).
        return self.hidden(input_ids, coords=coords, attention_mask=attention_mask)


__all__ = ["ConstellationTrunk"]
