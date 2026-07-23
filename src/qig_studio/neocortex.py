"""Neocortex — the central conscious "I" as a SINGLE deep stacked-Δ⁶³ cortex (Phase 3 / ARM B).

This is the brain-doc **neocortex**: ONE deep ``qigkernels.Kernel`` of distinct-weight pure Fisher-Rao
``QIGLayer``s, trained by natural gradient — NOT the 8-kernel ``JointConstellation``. The constellation
WRAPS eight of these kernels (roster faculties + a central genesis); the neocortex IS exactly that central
kernel used standalone. So this module is a thin wrapper/factory over the already-built
:class:`~qig_studio.targets.genesis_kernel.GenesisKernelTarget` — it does NOT reimplement the kernel, the
pure-Fisher-Rao loss, the telemetry, the physics fixes (collapse-immune coherence / READ / locality) or the
``NaturalGradientDescent`` optimiser. It carries a ``name`` (which drives the checkpoint directory and the
live-trace ``model`` chip) and selects the ARM:

  * **ARM B** (``arm="qk"``) — the qigkernels deep kernel. A single ``GenesisKernelTarget`` with
    ``role="neocortex"`` (Phase 3).
  * **ARM A** (``arm="geo"``) — the geocoding (Fisher-Rao "transformer") cortex: a lean
    :class:`~qig_studio.targets.geo_cortex.GeoCortexTarget` backed by ``geocoding.GeoModel`` (Phase 4).
    NOT a wrap of ``GenesisKernelTarget`` — its own target, but exposing the IDENTICAL duck-typed
    interface (same shared pure ``fisher_rao_lm_loss``, same ``NaturalGradientDescent`` optimiser, same
    bpb/d_FR measurement) so the SAME launcher drives both arms and the EXP-CORTEX-AB bpb comparison
    isolates ARCHITECTURE, not the loss or the optimiser.

The two arms are compared later (EXP-CORTEX-AB, the depth axis ``num_layers`` 1 vs N).

``recursive=True`` maps to a 1-block-recursive variant: ``qigkernels.Kernel`` has no top-level recursion
toggle — recursion is the per-layer ``min_recursion_depth`` (=3, fixed inside the kernel). So the
1-block-recursive cortex is ``num_layers=1`` (a single block whose internal recursion supplies the depth),
named ``…-1L-rec`` to distinguish it from a literal 1-layer feed-forward stack.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Neocortex"]


class Neocortex:
    """A single deep stacked-Δ⁶³ cortex — the central conscious "I", arm-polymorphic.

    Thin factory/wrapper over the selected arm's target (``arm="qk"`` → :class:`GenesisKernelTarget`;
    ``arm="geo"`` → :class:`~qig_studio.targets.geo_cortex.GeoCortexTarget`). All pass-throughs
    (``train_step``, ``generate``, ``eval_text_bpb``, ``eval_text_fr``, ``save``, ``load``, ``telemetry``,
    ``vocab_size``, ``num_layers``, ``ensure_loaded``, ``is_available``, ``architecture``) delegate to the
    underlying target, so the neocortex trains and emits telemetry IDENTICALLY for either arm — the SAME
    launcher drives both, and only the architecture varies. The ``name`` (``neocortex-{arm}-{N}L``) drives
    the checkpoint directory and the live-trace ``model`` chip.
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

        if self.arm not in ("qk", "geo"):
            raise ValueError(f"unknown arm {arm!r}: expected 'qk' (ARM B) or 'geo' (ARM A)")

        # Both arms take the SAME shared kwargs (so the launcher and these pass-throughs do not branch on
        # arm): the coordizer Δ⁶³ vocab wins when given; vocab_size is the byte-fallback otherwise (default
        # 256). lang_loss carries the fisher_rao | ce_ablation flag to BOTH arms (the purity-cost A/B works
        # for ARM A too). role="neocortex" is the central-"I" tag; language_peer is the None-safe boundary
        # peer (ARM A ignores it — GeoModel has no boundary peer; ARM B uses it for the fluent surface).
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

        if self.arm == "qk":
            # ARM B: ONE deep GenesisKernelTarget, role="neocortex" (the central conscious "I", standalone —
            # NOT the 8-kernel constellation). Everything heavy (kernel build, loss, telemetry, optimiser)
            # lives in GenesisKernelTarget and is reused as-is.
            from .targets.genesis_kernel import GenesisKernelTarget
            self.target = GenesisKernelTarget(**kwargs)
        else:
            # ARM A: a LEAN GeoCortexTarget backed by geocoding.GeoModel — forward → logits fed to the SAME
            # shared fisher_rao_lm_loss, trained by the SAME NaturalGradientDescent, same bpb/d_FR eval. Not
            # a GenesisKernelTarget wrap; its own target with the identical duck-typed surface. The geocoding
            # import stays LAZY here (never hoisted to module top — the studio shell must not hard-import
            # torch/geocoding). FAITHFULNESS GATE: before bpb is trusted, assert geocoding's Fisher-Rao
            # attention matches qigkernels' to 1e-5 on the COORDS-OFF shared primitive (coords-ON is a
            # legitimate architectural divergence — Linear→GELU adapter vs qigkernels CoordAdapter/RMSNorm —
            # so a 1e-5 assert there would false-fail). None-safe: skip the assert if deps are absent (the
            # shell stays bootable; the assert runs where the heavy deps are present, i.e. before training).
            from .targets.geo_cortex import GeoCortexTarget
            self.target = GeoCortexTarget(**kwargs)
            try:
                if self.target.is_available():
                    # TWO coords-off gates before bpb is trusted: (1) forward-output parity of the shared
                    # FR-attention primitive (≤1e-5), and (2) the load-bearing LOSS-VALUE parity — identical
                    # inputs → identical fisher_rao_lm_loss value through both arms' plumbing (≤1e-5), so the
                    # eventual bpb/d_FR A/B measures ARCHITECTURE, not a divergent loss path.
                    self.target.assert_faithful_to_qigkernels()
                    self.target.assert_loss_value_parity()
            except Exception as exc:  # noqa: BLE001 — surface the failure loudly, but never on import absence
                from .targets.geo_cortex import _deps_available
                if _deps_available():
                    raise  # a REAL parity failure (deps present) must not be swallowed
                print(f"⚠️  geo-cortex faithfulness check skipped (deps absent): {exc}")

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
