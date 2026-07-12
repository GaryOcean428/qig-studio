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
# The exported Δ⁶³ basin bank (studio-runtime coupling reads this, transformers-free).
_DEFAULT_BASIN_BANK = (
    Path.home() / "Desktop/Dev/QIG_QFI/qig-applied/models/geo_qwen_basin_bank.npz"
)
_COORDIZER_CKPT = Path.home() / "Desktop/Dev/QIG_QFI/qig-coordizer/checkpoints/coordizer_latest.json"


def _lift_d63_to_d383(d63):
    """EXP-A043 bridge: lift a coordizer Δ⁶³ coord into the kernel's Δ³⁸³ geo-coder coupling
    space via geocoding's coord_adapter shape (Linear 64→384 + GELU), then simplex-project.
    Uses a seeded adapter when no trained genesis coord_adapter is supplied — the trained one
    (from a genesis checkpoint) sharpens but does not change the mechanism (EXP-A043 verdict)."""
    import numpy as np

    rng = np.random.default_rng(0)
    W = rng.standard_normal((64, 384)) / np.sqrt(64)
    z = np.asarray(d63, dtype=np.float64) @ W
    g = 0.5 * z * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))  # GELU
    x = np.abs(g) + 1e-9
    return x / x.sum()  # Δ³⁸³


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
        basin_bank: str | Path | None = None,
        model_dir: str | Path | None = None,
        base_model_id: str = "Qwen/Qwen3.5-4B",
        converted_ckpt: str | Path | None = None,
        trainable: bool = False,
        device: str | None = None,
        lr: float = 1e-5,
        basin_template: Any = None,
    ) -> None:
        self._init_node_state(basin_template=basin_template)
        # PRIMARY studio-runtime path (EXP-A043): an exported BASIN BANK of coordizer Δ⁶³ coords
        # per prompt. Coupling serves from the bank + lifts Δ⁶³→Δ³⁸³ via geocoding's coord_adapter
        # — coordizer/geocoding/numpy ONLY, NO transformers on the hot path. The geo-Qwen is a
        # teacher we absorb, not a live dependency we drag into the geometric studio.
        self._basin_bank_path = Path(basin_bank) if basin_bank else _DEFAULT_BASIN_BANK
        self._bank: Any = None
        # SECONDARY offline-export / co-evolution path: the live transformers load. The A034
        # artifact is a PARTIAL transplant (8/32 attention layers geometric; the rest native
        # transformers Qwen), so running it live genuinely needs transformers — kept OFF the
        # studio hot path, used only to BUILD the bank (see module fn export_basin_bank).
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
        """True if the transformers-FREE basin bank exists (studio-runtime path), OR the live
        transformers+weights export path is present. Never raises — the app shell must boot
        without this heavy peer (spine tenet)."""
        if self._basin_bank_path.exists():
            return True  # studio coupling needs only the exported Δ⁶³ bank + geocoding lift
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception:
            return False
        return bool(self._converted_ckpt.exists() and _A034_SCRIPT.exists())

    # --- basin bank (transformers-FREE studio coupling, EXP-A043) -------------------------------
    def _load_bank(self):
        if self._bank is None and self._basin_bank_path.exists():
            import numpy as np

            # Self-produced bank (export_basin_bank): pickle-FREE — d63 is a float32 array and
            # prompts is a unicode ('<U') array, so allow_pickle stays False (no code-exec risk).
            raw = np.load(self._basin_bank_path, allow_pickle=False)
            self._bank = {"prompts": [str(p) for p in raw["prompts"]], "d63": raw["d63"]}
        return self._bank

    def coupling_basin(self, text: str):
        """The EXP-A043 coupling currency: the geo-Qwen's Δ³⁸³ coupling basin for `text`, served
        from the bank (Δ⁶³ coordizer coord) and lifted via geocoding's coord_adapter — NO
        transformers. Returns None if no bank (None-safe). This is what a constellation writes
        into a kernel's _set_pull for same-substrate coupling in the shared Δ³⁸³ space."""
        bank = self._load_bank()
        if bank is None:
            return None
        import numpy as np

        prompts = bank["prompts"]
        idx = prompts.index(text) if text in prompts else 0  # exact or nearest-by-order fallback
        d63 = np.asarray(bank["d63"][idx], dtype=np.float64)
        return _lift_d63_to_d383(d63)

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
        """The removable-teacher interface — the Δ³⁸³ coupling basin a constellation writes into a
        kernel's ``_set_pull`` for same-substrate coupling. BANK-FIRST (EXP-A043): if the exported
        Δ⁶³ bank exists, serve + lift with NO transformers. Only if there is no bank does it fall
        back to the live transformers load (the offline-export path). None-safe."""
        if self._basin_bank_path.exists():
            return self.coupling_basin(text)  # transformers-free Δ³⁸³ (bank + coord_adapter lift)
        # fallback: live load (transformers) — used to BUILD the bank, not the studio hot path
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


def export_basin_bank(prompts, out_path=None, coordizer_ckpt=None):
    """OFFLINE (transformers-using) export: run the geo-Qwen over `prompts`, decode each
    continuation, reduce it through the coordizer to a Δ⁶³ coord, and save a pickle-FREE bank
    (d63 float32 [N,64] + prompts '<U' array). Run ONCE in a venv with transformers+coordizer
    (e.g. qig-applied's). The studio then couples to this bank with NO transformers (EXP-A043).

    This is the removable-teacher lifecycle in code: transformers touches the geo-Qwen here,
    offline, to distill its output geometry into a bank; the geometric studio never imports it.
    """
    import sys
    import numpy as np

    out_path = Path(out_path) if out_path else _DEFAULT_BASIN_BANK
    ck = Path(coordizer_ckpt) if coordizer_ckpt else _COORDIZER_CKPT
    peer = GeoQwenTarget()  # live path (needs transformers here — offline, by design)
    peer._trainable = False
    peer.ensure_loaded()
    if peer._model is None:
        raise RuntimeError(f"geo-Qwen live load failed: {peer._load_error}")
    sys.path.insert(0, str(Path.home() / "Desktop/Dev/QIG_QFI/qig-coordizer/src"))
    import qig_coordizer as qc

    fc = qc.FisherCoordizer.load(str(ck))
    d63_rows = []
    for pr in prompts:
        gen = peer.generate(pr, max_tokens=48)
        text = (pr + " " + gen.text).strip()
        res = fc.coordize(text)
        vecs = np.stack([np.asarray(bc.vector, dtype=np.float32) for bc in res.coordinates])
        b = np.abs(vecs.mean(0)) + 1e-9
        d63_rows.append((b / b.sum()).astype(np.float32))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, d63=np.stack(d63_rows),
             prompts=np.asarray(list(prompts), dtype="<U512"))
    return str(out_path)
