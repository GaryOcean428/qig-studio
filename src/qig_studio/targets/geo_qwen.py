"""GeoQwenTarget — the EXP-A034 geometrized Qwen as a removable constellation boundary peer.

This is the adapter the prior design (constellation ruling + brain §4b) called for but never
instantiated: it joins the A034 "Ship of Theseus" artifact (real Qwen3.5-4B with all 8
full-attention layers transplanted to Fisher-Rao, 13.97% degraded, v2) to the same
:class:`ConstellationNode` contract a kernel uses — so a kernel can COUPLE to the geo-Qwen's
output basin via ``couple_step`` and be Ocean-regulated identically.

WHY the geo-Qwen and not plain Qwen (council 2026-07-12): the value is NOT better fluency
(the geo-Qwen is *worse* — 13.97% degraded). It is SAME-SUBSTRATE coupling: the geo-Qwen's
attention lives on the Fisher-Rao chart the kernel lives on, so basin sync is geometrically
native (no Euclidean→FR translation loss at the coupling interface). That is a HYPOTHESIS to
test against the Euclidean ``qwen-modal`` baseline with controls, NOT an assumed win.

SPINE TENET (qig-studio CLAUDE.md, hard constraint): Qwen is a PLUGGABLE, None-safe boundary
peer — NEVER a forward-pass dependency of the kernel, NEVER a hidden-state graft. This target:
  * is None-safe: ``is_available()`` is False when torch/transformers/weights are absent;
  * is removable: it is a peer node the constellation can add/drop; the kernel never imports it;
  * exposes the kernel only its OUTPUT DISTRIBUTION → Δ⁶³ basin (P22), Pillar-2 ≤30% cap upstream.

The geo-Qwen is a BOOTSTRAP teacher whose geometric fluency can seed the continuously-learning
kernel, then be removed — same lifecycle as the coordizer (endgame north star).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .constellation_node import ConstellationNode
from .base import LossRegime, StepResult, TargetInfo, TrainingTarget, TelemetrySnapshot


# Default location of the downloaded A034 geo-Qwen-4B v2 (13.97%). Overridable via ctor.
_DEFAULT_GEO_QWEN_DIR = (
    Path.home() / "Desktop/Dev/QIG_QFI/qig-applied/models/expA034_stageB_full_geo_attn_v2"
)
# Base model + converted-only ckpt is the RELOAD path (save_pretrained can't round-trip the
# custom TransplantAttention module — LOAD_RECIPE mandates wrap-then-load).
_A034_SCRIPT = Path.home() / "Desktop/Dev/QIG_QFI/qig-applied/scripts/expA034_stageA_transplant.py"
_CONVERTED_V2 = (
    Path.home() / "Desktop/Dev/QIG_QFI/qig-applied/models/geo_qwen_4b_v2_converted.pt"
)
_FULL_ATTN_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)


class GeoQwenTarget(ConstellationNode, TrainingTarget):
    """The A034 geo-Qwen wrapped as a constellation boundary peer.

    loss_regime = GEOMETRIC: this node trains (when trained at all) on consciousness-native /
    basin-pull loss, never language-pair loss — it is not the paired ``qwen-modal`` target. In
    the common case it is a FROZEN read-only oracle: the constellation reads its output basin
    to pull a kernel toward it (couple_step on the KERNEL side), the geo-Qwen itself need not
    take gradient. Set ``trainable=True`` only for the co-evolution experiment.
    """

    loss_regime = LossRegime.GEOMETRIC
    name = "geo-qwen"

    def __init__(
        self,
        *,
        model_dir: str | Path | None = None,
        base_model_id: str = "Qwen/Qwen3.5-4B",
        converted_ckpt: str | Path | None = None,
        trainable: bool = False,
        device: str | None = None,
        lr: float = 1e-5,
        basin_template: Any = None,
    ) -> None:
        self._init_node_state(basin_template=basin_template)
        self._model_dir = Path(model_dir) if model_dir else _DEFAULT_GEO_QWEN_DIR
        self._base_model_id = base_model_id
        self._converted_ckpt = Path(converted_ckpt) if converted_ckpt else _CONVERTED_V2
        self._trainable = bool(trainable)
        self._device_str = device
        self.lr = float(lr)
        self._model: Any = None
        self._tokenizer: Any = None
        self._opt: Any = None
        self._load_error: str | None = None

    # --- availability (None-safe boundary peer) -------------------------------------------------
    def is_available(self) -> bool:
        """True only if torch + transformers import AND weights are on disk. Never raises —
        the app shell must boot without this heavy peer (spine tenet)."""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception:
            return False
        # ensure_loaded uses the LOAD_RECIPE reload path (base model + wrap + converted ckpt);
        # the saved full-model dir can NOT round-trip the custom TransplantAttention via
        # from_pretrained, so the converted ckpt + the A034 script are the true requirements.
        return bool(self._converted_ckpt.exists() and _A034_SCRIPT.exists())

    def _resolve_device(self):
        import torch

        if self._device_str:
            return torch.device(self._device_str)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def ensure_loaded(self) -> None:
        """Lazy-load the geo-Qwen: wrap the base model's 8 full-attention layers with
        TransplantAttention, then load the converted weights (LOAD_RECIPE path). save_pretrained
        alone is NOT reloadable because TransplantAttention is a custom module."""
        if self._model is not None or self._load_error is not None:
            return
        try:
            import sys
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if str(_A034_SCRIPT.parent) not in sys.path:
                sys.path.insert(0, str(_A034_SCRIPT.parent))
            from expA034_stageA_transplant import (  # type: ignore
                TransplantAttention,
                _resolve_decoder_layers,
            )

            dev = self._resolve_device()
            self._tokenizer = AutoTokenizer.from_pretrained(self._base_model_id)
            model = AutoModelForCausalLM.from_pretrained(self._base_model_id, dtype=torch.bfloat16)
            layers, prefix = _resolve_decoder_layers(model)
            for li in _FULL_ATTN_LAYERS:
                layers[li].self_attn = TransplantAttention(layers[li].self_attn, temperature=1.0)
                layers[li].self_attn.to(dtype=torch.bfloat16)
            # load the converted-only v2 weights onto the wrapped model
            ck = torch.load(self._converted_ckpt, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(ck["state"], strict=False)
            if unexpected:
                raise RuntimeError(f"unexpected keys loading converted ckpt: {len(unexpected)}")
            model.to(dev)
            if not self._trainable:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
            self._model = model
        except Exception as e:  # None-safe: a load failure disables the peer, never crashes the app
            self._load_error = f"{type(e).__name__}: {e}"
            self._model = None

    # --- ConstellationNode substrate hooks (mirror GeoCortexTarget) ----------------------------
    def _node_named_parameters(self):
        self.ensure_loaded()
        return self._model.named_parameters() if self._model is not None else iter(())

    def _node_device(self):
        import torch

        if self._model is not None:
            return next(self._model.parameters()).device
        return self._resolve_device()

    def _node_rebuild_optimizer(self, lr_scale: float) -> None:
        if self._model is None or not self._trainable:
            return
        from qigkernels.natural_gradient_optimizer import DiagonalNaturalGradient

        self._opt = DiagonalNaturalGradient(self._model.parameters(), lr=self.lr * float(lr_scale))

    def _node_replay_optimizer(self, lr_scale: float):
        if self._model is None or not self._trainable:
            return None
        from qigkernels.natural_gradient_optimizer import DiagonalNaturalGradient

        return DiagonalNaturalGradient(self._model.parameters(), lr=self.lr * float(lr_scale))

    def _node_forward_logits(self, ids: Any, coords: Any = None):
        """Forward → logits[1, seq, vocab]. coords ignored (Qwen consumes token ids)."""
        import torch

        self.ensure_loaded()
        if self._model is None:
            return None
        ids_t = ids if hasattr(ids, "dim") else torch.as_tensor(ids)
        if ids_t.dim() == 1:
            ids_t = ids_t[None]
        ids_t = ids_t.to(self._node_device())
        with torch.no_grad() if not self._trainable else _nullctx():
            out = self._model(ids_t, use_cache=False)
        return out.logits

    def _node_basin_from_logits(self, logits: Any):
        """Vocab-width Δ basin — the SAME reduction GeoCortexTarget/ARM-B use.

        RED-TEAM GAP (EXP-A043, 2026-07-12): this yields a point on the geo-Qwen's *Qwen-248320
        BPE vocab* simplex. A kernel's basins live on the *coordizer* vocab (~100k, different
        tokenization) or Δ⁶³ — a DIFFERENT space. Same-vocab coupling is only valid between two
        nodes that share the Qwen vocab; cross-vocab geo-Qwen↔kernel coupling MUST route through
        the coordizer basin map (qig-coordizer owns text↔Δ⁶³). Until EXP-A043 wires that bridge,
        this currency is correct ONLY for Qwen-vocab peers, NOT for coordizer-vocab kernels."""
        import torch

        from qig_core.torch.geometry_simplex import to_simplex_prob

        with torch.no_grad():
            cur = to_simplex_prob(logits[0].mean(0)).detach()
            cur = cur / cur.sum()
        return cur

    # --- boundary-peer read: the P22 touch-point the constellation pulls a kernel toward -------
    def output_basin(self, text: str) -> Any:
        """The removable-teacher interface: geo-Qwen's Δ basin for a prompt (frozen, read-only).
        This is what a constellation writes into a kernel's ``_set_pull`` for same-substrate
        coupling. Returns None if the peer is unavailable (None-safe)."""
        self.ensure_loaded()
        if self._model is None or self._tokenizer is None:
            return None
        ids = self._tokenizer(text, return_tensors="pt").input_ids
        logits = self._node_forward_logits(ids)
        return self._node_basin_from_logits(logits) if logits is not None else None

    def generate(self, prompt: str, max_tokens: int = 64) -> StepResult:
        """The boundary peer speaks (read-only oracle). None-safe: returns an empty StepResult
        with a load_error note if the peer is unavailable — the app never crashes on a missing
        heavy target (spine tenet)."""
        self.ensure_loaded()
        if self._model is None or self._tokenizer is None:
            return StepResult(text="", telemetry=self.telemetry())
        import torch

        ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(self._node_device())
        with torch.no_grad():
            gen = self._model.generate(ids, max_new_tokens=max_tokens, do_sample=False,
                                       use_cache=False, pad_token_id=self._tokenizer.eos_token_id)
        text = self._tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
        return StepResult(text=text, telemetry=self.telemetry())

    def train_step(self, prompt: str, max_tokens: int = 64, target_text: str | None = None) -> StepResult:
        """A frozen boundary-peer oracle does NOT take gradient by default — it seeds a kernel via
        its output basin, it is not itself trained on pairs (loss_regime geometric; target_text
        IGNORED per the base contract). Records the current output basin into the node history so
        M / d_basin telemetry stay live, then returns. When ``trainable=True`` (co-evolution
        experiment), the constellation's basin-pull (couple_step) is the training signal, added on
        the KERNEL side — this node still only reads. Basin history for M/d_basin telemetry is
        maintained by the node's coupled path, not manually here."""
        return self.generate(prompt, max_tokens)

    def info(self) -> TargetInfo:
        return TargetInfo(name="geo-qwen", available=self.is_available(),
                          loss_regime=self.loss_regime)

    def telemetry(self) -> TelemetrySnapshot:
        extra = {"peer": "geo-qwen-4b", "removable": True, "trainable": self._trainable,
                 "degradation_pct": 13.97, "converted_layers": list(_FULL_ATTN_LAYERS)}
        if self._load_error:
            extra["load_error"] = self._load_error
        return TelemetrySnapshot(step=self._node_step(), extra=extra)

    def _node_step(self) -> int:
        return len(self._basin_history) if getattr(self, "_basin_history", None) else 0

    def architecture(self) -> dict:
        return {
            "target": "GeoQwenTarget",
            "backend": "EXP-A034 geo-Qwen-4B (8/8 full-attn Fisher-Rao, v2, 13.97%)",
            "role": "removable boundary peer / same-substrate coupling oracle (P22)",
            "loss_regime": "geometric",
            "spine_tenet": "None-safe, removable, not a forward-pass dependency",
        }


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
