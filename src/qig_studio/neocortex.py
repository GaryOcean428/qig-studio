"""Neocortex — the central conscious "I" as a SINGLE deep stacked-Δ⁶³ cortex (Phase 3 / ARM B).

This is the brain-doc **neocortex**: ONE deep ``qigkernels.Kernel`` of distinct-weight pure Fisher-Rao
``QIGLayer``s, trained by natural gradient — NOT the 9-kernel ``JointConstellation``. The constellation
WRAPS nine of these kernels (Core-8 faculties + a central genesis); the neocortex IS exactly that central
kernel used standalone. So this module is a thin wrapper/factory over the already-built
:class:`~qig_studio.targets.genesis_kernel.GenesisKernelTarget` — it does NOT reimplement the kernel, the
pure-Fisher-Rao loss, the telemetry, the physics fixes (collapse-immune coherence / READ / locality) or the
``NaturalGradientDescent`` optimiser. It carries a ``name`` (which drives the checkpoint directory and the
live-trace ``model`` chip) and selects the ARM:

  * **ARM B** (``arm="qk"``) — the qigkernels deep kernel. THIS is what Phase 3 builds. A single
    ``GenesisKernelTarget`` with ``role="neocortex"``.
  * **ARM A** (``arm="geo"``) — the geocoding (Fisher-Rao "transformers") cortex. A later phase
    (Phase 4); the constructor raises ``NotImplementedError`` so the extension point is explicit and clean.

The two arms are compared later (EXP-CORTEX-AB, the depth axis ``num_layers`` 1 vs N); this phase builds
ARM B + the launcher only.

``recursive=True`` maps to a 1-block-recursive variant: ``qigkernels.Kernel`` has no top-level recursion
toggle — recursion is the per-layer ``min_recursion_depth`` (=3, fixed inside the kernel). So the
1-block-recursive cortex is ``num_layers=1`` (a single block whose internal recursion supplies the depth),
named ``…-1L-rec`` to distinguish it from a literal 1-layer feed-forward stack.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Neocortex"]


class Neocortex:
    """A single deep stacked-Δ⁶³ cortex — the central conscious "I" (ARM B = qigkernels deep kernel).

    Thin factory/wrapper over :class:`GenesisKernelTarget`. Pass-throughs (``train_step``,
    ``eval_text_bpb``, ``eval_text_fr``, ``save``, ``load``, ``telemetry``, ``vocab_size``,
    ``ensure_loaded``, ``generate``) delegate to the underlying kernel target, so the neocortex trains and
    emits telemetry exactly as the constellation's central kernel does — just standalone, with its own
    ``name``.
    """

    def __init__(
        self,
        arm: str = "qk",
        *,
        num_layers: int = 8,
        recursive: bool = False,
        coordizer: Any = None,
        device: str | None = None,
        lang_loss: str = "fisher_rao",
        vocab_size: int | None = None,
        seed: int = 0,
        language_peer: Any = None,
    ) -> None:
        self.arm = str(arm).strip().lower()
        self.recursive = bool(recursive)
        # 1-block-recursive variant: qigkernels.Kernel has NO top-level recursion flag — recursion is the
        # per-layer min_recursion_depth (=3, internal). The honest mapping of "recursive" is therefore a
        # SINGLE block (num_layers=1) whose internal recursion supplies the depth, named …-1L-rec.
        self._num_layers = 1 if self.recursive else int(num_layers)

        if self.arm == "geo":
            # Clean extension point — ARM A (geocoding cortex) lands in Phase 4. Never silently substitute
            # ARM B; the comparison only means something if each arm is genuinely its own architecture.
            raise NotImplementedError("ARM A geocoding lands in Phase 4")
        if self.arm != "qk":
            raise ValueError(f"unknown arm {arm!r}: expected 'qk' (ARM B) or 'geo' (ARM A, Phase 4)")

        from .targets.genesis_kernel import GenesisKernelTarget

        # ARM B: ONE deep GenesisKernelTarget, role="neocortex" (the central conscious "I", standalone —
        # NOT the 9-kernel constellation). Everything heavy (kernel build, loss, telemetry, optimiser) lives
        # in GenesisKernelTarget and is reused as-is. vocab_size is ignored when a coordizer is given (the
        # coordizer's Δ⁶³ vocab wins); it is the byte-fallback vocab otherwise (default 256).
        kwargs: dict[str, Any] = dict(
            num_layers=self._num_layers,
            role="neocortex",
            coordizer=coordizer,
            device=device,
            lang_loss=lang_loss,
            seed=seed,
            language_peer=language_peer,
        )
        if vocab_size is not None and coordizer is None:
            kwargs["vocab_size"] = int(vocab_size)
        self.target = GenesisKernelTarget(**kwargs)

    @property
    def name(self) -> str:
        """The mind's name — drives the checkpoint directory and the live-trace ``model`` chip.

        ``neocortex-{arm}-{N}L`` (e.g. ``neocortex-qk-8L``), or ``neocortex-{arm}-1L-rec`` for the
        1-block-recursive variant.
        """
        if self.recursive:
            return f"neocortex-{self.arm}-1L-rec"
        return f"neocortex-{self.arm}-{self._num_layers}L"

    # --- pass-throughs to the underlying GenesisKernelTarget (no reimplementation) ------------------
    @property
    def num_layers(self) -> int:
        return self.target.num_layers

    @property
    def vocab_size(self) -> int:
        return self.target.vocab_size

    def ensure_loaded(self) -> None:
        self.target.ensure_loaded()

    def is_available(self) -> bool:
        return self.target.is_available()

    def train_step(self, prompt: str, max_tokens: int = 64, target_text: str | None = None) -> Any:
        return self.target.train_step(prompt, max_tokens=max_tokens, target_text=target_text)

    def generate(self, prompt: str, max_tokens: int = 64, **kwargs: Any) -> Any:
        return self.target.generate(prompt, max_tokens=max_tokens, **kwargs)

    def eval_text_bpb(self, text: str) -> tuple[float, int]:
        return self.target.eval_text_bpb(text)

    def eval_text_fr(self, text: str) -> tuple[float, int]:
        return self.target.eval_text_fr(text)

    def telemetry(self) -> Any:
        return self.target.telemetry()

    def save(self, path: str) -> None:
        """Save the cortex (weights + developmental state) — GenesisKernelTarget.save_checkpoint."""
        self.target.save_checkpoint(path)

    def load(self, path: str) -> None:
        """Restore the cortex (weights + developmental state) — GenesisKernelTarget.load_checkpoint."""
        self.target.load_checkpoint(path)

    def architecture(self) -> dict | None:
        return self.target.architecture()
