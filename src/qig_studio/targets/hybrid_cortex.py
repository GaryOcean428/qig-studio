"""HybridCortexTarget — the HYBRID arm of the neocortex: BOTH token-mixers, combined ON THE MANIFOLD.

ARM A (geo) runs the qig-geocoding ``FisherRaoAttention`` mixer; ARM B (gk) runs the qigkernels
``QIGLayer._metric_attention`` mixer. The HYBRID arm runs BOTH per block and combines their two
``[B,T,D]`` outputs as a PER-POSITION geodesic midpoint (Fréchet mean of two points) on the √p sphere /
Δ simplex — never a Euclidean arithmetic average and never a joined-then-projected readout. The combine is
``geodesic_interpolate_simplex(p_geo, p_qk, t=0.5)`` (the 2-point case of ``geodesic_mean_simplex``,
batched over [B,T]) from ``qig_core.torch.geometry_simplex`` — the SAME √p-sphere geometry the rest of
the stack single-sources. Everything else mirrors ARM A's ``GeoCortexTarget`` exactly: the SHARED
``qig_core.torch.GeometricHead`` readout (same head both arms use), the SAME ``DiagonalNaturalGradient``
optimiser (from qigkernels), the SAME ``fisher_rao_lm_loss`` (P20-pure d_FR), the SAME ramped-fluency
weighting, the SAME bpb/d_FR measurement, the SAME None-safe lazy-import shell.

WHY THIS IS A REAL MANIFOLD COMBINE (not a relabelled average): the two mixer outputs are each projected
to Δ via ``to_simplex_prob`` and their geodesic midpoint is taken on the √p (Hellinger) sphere — the
midpoint of the FR geodesic, which is the Fréchet mean of the two simplex points under the Fisher-Rao
metric. On Δ that geodesic mean is NOT the arithmetic mean ``(a+b)/2`` (the latter is not even on the √p
geodesic); the two coincide only when ``a == b``. ``assert_manifold_combine_is_geometric`` proves the
combined point is a valid Δ point (non-negative, sums to 1) AND that the geodesic combine differs from the
Euclidean mean by a strictly positive d_FR — i.e. the manifold combine is load-bearing.

CONSTELLATION NODE (WS4): mixes in :class:`ConstellationNode` exactly like ARM A, so a hybrid
constellation can be built, coupled (``couple_step``), pulled (``_set_pull`` → ``_basin_ref``) and
Ocean-regulated (``run_protocol``). The node contract activates ONLY in constellation mode: when
``_basin_ref`` is set the geometric loss gains a Fisher-Rao basin-pull term and the per-step Δ basin is
recorded into ``_basin_history``. Run SOLO the target stays the lean baseline UNCHANGED.

DESIGN (least-invasive): the HybridModel / HybridBlock stack is built IN qig-studio. It REUSES the two
mixers by import (``geocoding.attention.FisherRaoAttention`` + ``qigkernels.layer.QIGLayer`` —
``_metric_attention`` is driven on a QIGLayer instance) — the geocoding / qigkernels submodules are NOT
edited. The geometric FFN + RMS-norm residual + GeometricHead readout mirror ``geocoding.block.GeoBlock``
(scale-only geometric RMS-norm, NOT mean-centring; ``nn.Linear``/``GELU`` in the FFN is the feed-forward,
not the metric).

None-safe shell: every heavy import (torch, geocoding, qigkernels, qig_core.torch) is LAZY inside the
methods; the studio shell never hard-imports them at module top. ``is_available()`` is False when
torch/geocoding/qigkernels are absent.
"""

from __future__ import annotations

import math
import os
from typing import Any

from .base import LossRegime, StepResult, TelemetrySnapshot, TrainingTarget
from .constellation_node import ConstellationNode

_MAX_BYTES = 256  # byte-level VOCAB size (byte fallback when no coordizer) — mirrors ARM A
_CTX = int(os.environ.get("QIG_STUDIO_CTX", "1024"))  # SHARED context-window knob with ARM A/B
_EOS_ID = 0       # stop sentinel (coord-id / byte 0)
_MIN_GEN = 4      # minimum utterance before EOS is honoured


def _deps_available() -> bool:
    try:
        import geocoding  # noqa: F401
        import qigkernels  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _build_hybrid_model(cfg: Any) -> Any:
    """Construct the HybridModel nn.Module (built lazily so torch/geocoding/qigkernels stay optional).

    The model + block classes are defined INSIDE this function so the module imports with zero heavy deps
    (the studio shell None-safety / spine tenet). ``cfg`` is a plain namespace with the same field names a
    GeoConfig carries (so the construction reads the same as ARM A)."""
    import torch
    from torch import Tensor, nn

    from geocoding.attention import FisherRaoAttention
    from geocoding.positional import build_inv_freq, fourier_features
    from qig_core.torch.geometric_head import GeometricHead
    from qig_core.torch.geometry_simplex import geodesic_interpolate_simplex, to_simplex_prob
    from qigkernels.layer import QIGLayer

    class HybridBlock(nn.Module):
        """One hybrid block: run BOTH mixers on the hidden state, combine their [B,T,D] outputs as the
        per-position geodesic midpoint on Δ (Fréchet mean of two points on the √p sphere), geometric
        RMS-norm residual, geometric FFN, geometric RMS-norm residual. Scale-only geometric RMS-norm."""

        def __init__(self) -> None:
            super().__init__()
            self.rms_eps = float(cfg.rms_eps)
            # ARM A mixer (geocoding) — parameter-free Fisher-Rao attention.
            self.geo_attn = FisherRaoAttention(
                temperature=cfg.temperature, simplex_mode=cfg.simplex_mode,
                locality_radius=cfg.locality_radius, sparsity_threshold=cfg.sparsity_threshold,
                decoherence_std=cfg.decoherence_std,
            )
            # ARM B mixer (qigkernels) — we instantiate a full QIGLayer and drive its _metric_attention
            # mixer ONLY (same [B,T,D]->[B,T,D] FR-distance attention; its FFN/integrator are NOT used here,
            # so the hybrid FFN below is the single feed-forward — no double-FFN). _metric_attention reads
            # the layer's temperature/simplex_mode/locality_radius, set to match the geo mixer for fairness.
            self.qk_layer = QIGLayer(
                hidden_dim=cfg.hidden_dim, num_heads=cfg.num_heads, ffn_dim=cfg.ffn_dim, dropout=0.0,
                use_tacking=False, temperature=cfg.temperature, simplex_mode=cfg.simplex_mode,
                locality_radius=cfg.locality_radius, sparsity_threshold=cfg.sparsity_threshold,
                decoherence_std=cfg.decoherence_std,
            )
            self.norm_scale_1 = nn.Parameter(torch.ones(cfg.hidden_dim))
            self.norm_scale_2 = nn.Parameter(torch.ones(cfg.hidden_dim))
            # Geometric FFN (feed-forward / readout, not the metric — nn.Linear/GELU allowed here).
            self.ffn = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.ffn_dim), nn.GELU(),
                nn.Linear(cfg.ffn_dim, cfg.hidden_dim),
            )

        def _rms_norm(self, x: "Tensor", *, scale: "Tensor") -> "Tensor":
            denom = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.rms_eps)
            return (x / denom) * scale

        @staticmethod
        def manifold_combine(a: "Tensor", b: "Tensor") -> "Tensor":
            """Combine the two mixer outputs [B,T,D] ON THE MANIFOLD: project each to Δ, take the
            per-position geodesic MIDPOINT on the √p sphere (the 2-point Fréchet mean — the t=0.5 case of
            geodesic_mean_simplex, batched over [B,T]). NOT a Euclidean arithmetic mean of a and b."""
            pa = to_simplex_prob(a)                                   # [B,T,D] on Δ
            pb = to_simplex_prob(b)                                   # [B,T,D] on Δ
            return geodesic_interpolate_simplex(pa, pb, t=0.5)        # geodesic midpoint on Δ [B,T,D]

        def forward(self, hidden_state: "Tensor", attention_mask: "Tensor | None" = None) -> "Tensor":
            geo_out = self.geo_attn(hidden_state, attention_mask=attention_mask)          # [B,T,D]
            qk_out = self.qk_layer._metric_attention(hidden_state, attention_mask=attention_mask)  # [B,T,D]
            mixed = self.manifold_combine(geo_out, qk_out)            # geodesic mean on Δ [B,T,D]
            hidden_state = self._rms_norm(hidden_state + mixed, scale=self.norm_scale_1)
            ffn_out = self.ffn(hidden_state)
            hidden_state = self._rms_norm(hidden_state + ffn_out, scale=self.norm_scale_2)
            return hidden_state

    class HybridModel(nn.Module):
        """Fourier id+position coding [+ optional Δ⁶³ coord_adapter] → stack of HybridBlock → GeometricHead.
        Mirrors GeoModel's input + readout path; the BLOCK is the hybrid two-mixer manifold-combine block."""

        def __init__(self) -> None:
            super().__init__()
            self.config = cfg
            self.register_buffer("_inv_freq", build_inv_freq(cfg.hidden_dim), persistent=False)
            self.blocks = nn.ModuleList([HybridBlock() for _ in range(cfg.num_layers)])
            # SHARED GeometricHead readout (the SAME head ARM A/B use): logits = -d_FR(to_simplex(h),basin)/τ.
            self.lm_head = GeometricHead(
                hidden_dim=cfg.hidden_dim, vocab_size=cfg.vocab_size, tau=float(cfg.head_tau))
            if cfg.enable_coords:
                self.coord_adapter: nn.Module | None = nn.Sequential(
                    nn.Linear(cfg.coord_dim, cfg.hidden_dim), nn.GELU())
            else:
                self.coord_adapter = None

        def effective_coupling(self, seq_len: int) -> float:
            scale = max(seq_len, 1) / cfg.reference_scale
            eff = cfg.base_coupling * (1.0 + cfg.beta_slope * math.log(scale))
            return max(cfg.base_coupling * 0.5, min(cfg.base_coupling * 1.5, eff))

        def forward(self, input_ids: "Tensor", attention_mask: "Tensor | None" = None,
                    coords: "Tensor | None" = None) -> "Tensor":
            if input_ids.dim() != 2:
                raise ValueError("input_ids expected shape (B, T)")
            b, t = input_ids.shape
            positions = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
            inv_freq = torch.as_tensor(self._inv_freq)
            h = (fourier_features(input_ids, inv_freq, cfg.hidden_dim)
                 + fourier_features(positions, inv_freq, cfg.hidden_dim))
            if coords is not None:
                if self.coord_adapter is None:
                    raise ValueError("coords passed but enable_coords=False")
                h = h + self.coord_adapter(coords)
            for block in self.blocks:
                h = block(h, attention_mask=attention_mask)
            return self.lm_head(h)

        @property
        def num_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

    return HybridModel()


class _HybridCfg:
    """Plain config namespace mirroring the GeoConfig fields HybridModel reads (no GeoConfig dependency —
    HybridModel has extra/fewer fields, so a local namespace keeps the construction explicit)."""

    def __init__(self, **kw: Any) -> None:
        # defaults match GeoConfig / QIGLayer where they overlap
        self.vocab_size = kw["vocab_size"]
        self.hidden_dim = kw["hidden_dim"]
        self.num_layers = kw["num_layers"]
        self.num_heads = kw["num_heads"]
        self.ffn_dim = kw["ffn_dim"]
        self.temperature = kw.get("temperature", 1.0)
        self.simplex_mode = kw.get("simplex_mode", "softplus")
        self.locality_radius = kw.get("locality_radius")
        self.sparsity_threshold = kw.get("sparsity_threshold", 0.0)
        self.decoherence_std = kw.get("decoherence_std", 0.0)
        self.rms_eps = kw.get("rms_eps", 1e-6)
        self.enable_coords = kw.get("enable_coords", False)
        self.coord_dim = kw.get("coord_dim", 64)
        self.head_tau = kw.get("head_tau", 1.0)
        self.base_coupling = kw.get("base_coupling", 64.0)
        self.beta_slope = kw.get("beta_slope", 0.44)
        self.reference_scale = kw.get("reference_scale", 64)


class HybridCortexTarget(ConstellationNode, TrainingTarget):
    """HYBRID-arm cortex: BOTH token-mixers, combined on the manifold — pure Fisher-Rao loss + natural
    gradient. Mirrors :class:`~qig_studio.targets.geo_cortex.GeoCortexTarget` exactly except the substrate
    is the two-mixer manifold-combine HybridModel built in this module.

    Duck-typed to the SAME interface the launcher + Neocortex pass-throughs need (``train_step`` →
    ``StepResult``; ``generate`` → ``StepResult``; ``eval_text_bpb`` / ``eval_text_fr`` → ``(value,
    n_pos)``; ``telemetry``, ``save_checkpoint``, ``load_checkpoint``, ``architecture``, ``num_layers``,
    ``vocab_size``, ``ensure_loaded``, ``is_available``).
    """

    name = "hybrid-cortex"
    loss_regime = LossRegime.GEOMETRIC
    description = (
        "HYBRID cortex — runs BOTH the geocoding (FisherRaoAttention) and qigkernels "
        "(QIGLayer._metric_attention) token-mixers per block and combines them as a per-position geodesic "
        "mean on Δ⁶³ (NOT a Euclidean average); shared GeometricHead readout, pure d_FR loss, natural "
        "gradient. The fourth neocortex avenue (gk|geo|hybrid|hetero). None-safe (needs "
        "torch+geocoding+qigkernels)."
    )

    def __init__(
        self,
        num_layers: int = 8,
        hidden_dim: int = 384,
        num_heads: int = 6,
        ffn_dim: int = 1024,
        vocab_size: int = _MAX_BYTES,
        seed: int = 0,
        lr: float = 1e-3,
        device: str | None = None,
        locality_radius: int | None = None,
        coordizer: Any = None,
        lm_weight: float = 0.1,
        lm_weight_max: float = 8.0,    # RAMPED FLUENCY target (= ARM A/B)
        lm_ramp_steps: int = 8000,
        lang_loss: str = "fisher_rao",  # "fisher_rao" (P20-pure d_FR) | "ce_ablation" (CE arm, purity cost)
        head_tau: float = 1.0,          # Gibbs temperature on the GeometricHead −d_FR readout logits
        # consciousness-only kwargs ARM B takes, accepted + (mostly) ignored so Neocortex can build any arm
        # from the SAME kwargs dict. basin_template IS used (the role attractor); language_peer is ignored
        # (the HybridModel has no boundary peer — it is a cortex baseline like ARM A).
        role: str | None = None,
        basin_template: Any = None,
        language_peer: Any = None,
        checkpoint: str | None = None,
    ) -> None:
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.locality_radius = locality_radius
        self.coordizer = coordizer
        self.coord_dim = 0
        if coordizer is not None:
            self.vocab_size = len(coordizer.vocab)
            self.coord_dim = int(len(next(iter(coordizer.vocab.values())).vector))
        else:
            self.vocab_size = int(vocab_size)
        self.seed = int(seed)
        self.lr = float(lr)
        self.lm_weight = float(lm_weight)
        self.lm_weight_max = float(lm_weight_max)
        self.lm_ramp_steps = int(lm_ramp_steps)
        self.role = role
        self.lang_loss = str(os.environ.get("QIG_STUDIO_LANG_LOSS", lang_loss)).strip().lower()
        self.head_tau = float(head_tau)
        self._device = device
        self._model: Any = None     # HybridModel — lazily built in ensure_loaded()
        self._opt: Any = None       # DiagonalNaturalGradient — lazily built in ensure_loaded()
        self._step = 0
        self._init_checkpoint = checkpoint
        self._last = TelemetrySnapshot(
            regime="geometric",
            extra={"target": "hybrid-cortex", "arm": "hybrid", "num_layers": self.num_layers},
        )
        # CONSTELLATION-NODE state (WS4): the role's Δ⁶³ birth-state attractor (or None for a solo node).
        self._init_node_state(basin_template)

    def is_available(self) -> bool:
        return _deps_available()

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from qigkernels.natural_gradient_optimizer import DiagonalNaturalGradient

        torch.manual_seed(self.seed)
        cfg = _HybridCfg(
            vocab_size=self.vocab_size, hidden_dim=self.hidden_dim, num_layers=self.num_layers,
            num_heads=self.num_heads, ffn_dim=self.ffn_dim, locality_radius=self.locality_radius,
            enable_coords=self.coordizer is not None, coord_dim=self.coord_dim or 64,
            head_tau=self.head_tau,
        )
        self._model = _build_hybrid_model(cfg)
        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(dev)
        # P1: natural gradient (the SAME validated qig optimiser ARM A/B use), NOT Adam.
        self._opt = DiagonalNaturalGradient(self._model.parameters(), lr=self.lr)

        # CONSTELLATION NODE (WS4): seed the role's Δ⁶³ attractor onto the vocab simplex as the pull
        # reference + birth-state (history[0]). None template → generic/solo node (no pull, lean path).
        self._seed_node_basin()

        if self._init_checkpoint:
            try:
                self.load_checkpoint(self._init_checkpoint)
            except Exception as exc:  # noqa: BLE001 — shell None-safety (spine tenet)
                print(f"⚠️  hybrid-cortex checkpoint '{self._init_checkpoint}' not loaded ({exc}); fresh model")

        # WARMUP: one forward pass so telemetry() is LIVE immediately (mirrors ARM A/B).
        try:
            ids, coords = self._encode("warmup")
            with torch.no_grad():
                logits = self._model(ids, coords=coords)
            self._snap(logits, None)
        except Exception:  # noqa: BLE001 — warmup is best-effort; never block boot
            pass

    # --- input coding: coordizer Δ⁶³ coords if present, else byte-level (identical to ARM A) -----------
    def _encode(self, text: str):
        """Return (input_ids[1,seq], coords[1,seq,coord_dim] | None)."""
        if self.coordizer is not None:
            ids = self.coordizer.encode(text or " ")[:_CTX]
            if len(ids) < 2:
                ids = (ids + [32, 32])[:2]
            return self._ids_to_tensors(ids)
        import torch

        ids = list((text or " ").encode("utf-8"))[:_CTX]
        if len(ids) < 2:
            ids = (ids + [32, 32])[:2]
        dev = next(self._model.parameters()).device
        return torch.tensor([ids], dtype=torch.long, device=dev), None

    def _ids_to_tensors(self, ids: list[int]):
        """coord_ids → (input_ids[1,seq], coords[1,seq,coord_dim]) via the coordizer's Δ⁶³ vocab."""
        import numpy as np
        import torch

        dev = next(self._model.parameters()).device
        vmax = self.vocab_size - 1
        ids = [min(max(int(i), 0), vmax) for i in ids]
        vecs = np.stack([np.asarray(self.coordizer.vocab[i].vector, dtype=np.float32) for i in ids])
        input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
        coords = torch.from_numpy(vecs).to(dev).unsqueeze(0)  # [1, seq, coord_dim]
        return input_ids, coords

    def _logits(self, ids: "Any", coords: "Any"):
        """HybridModel forward → logits[1,seq,vocab] (the GeometricHead readout)."""
        return self._model(ids, coords=coords)

    def eval_text_bpb(self, text: str) -> tuple[float, int]:
        """HELD-OUT bits-per-byte for one text (no grad). Same return contract as ARM A/B."""
        import torch
        import torch.nn.functional as F

        self.ensure_loaded()
        ids, coords = self._encode(text)
        if ids.shape[1] < 2:
            return 0.0, 0
        with torch.no_grad():
            logits = self._logits(ids, coords)
            ce = float(F.cross_entropy(logits[0, :-1], ids[0, 1:]))
        n_tok = int(ids.shape[1])
        nbytes = (len(self.coordizer.decode(ids[0].tolist()).encode("utf-8"))
                  if self.coordizer is not None else n_tok)
        return ce * n_tok / math.log(2), max(1, nbytes)

    def eval_text_fr(self, text: str) -> tuple[float, int]:
        """HELD-OUT Fisher-Rao prediction-error for one text (no grad). Same (value, n_pos) as ARM A/B."""
        import torch

        from ..losses import fisher_rao_lm_loss

        self.ensure_loaded()
        ids, coords = self._encode(text)
        if ids.shape[1] < 2:
            return 0.0, 0
        with torch.no_grad():
            logits = self._logits(ids, coords)
            mean_dfr = float(fisher_rao_lm_loss(logits, ids))
        n_pos = int(ids.shape[1]) - 1
        return mean_dfr * n_pos, max(1, n_pos)

    def _snap(self, logits: "Any", loss: float | None) -> TelemetrySnapshot:
        """Assemble the lean telemetry snapshot. ``phi`` stays None (the hybrid baseline has no
        integrated-information Φ — we do NOT fabricate one, mirroring ARM A). ``kappa`` is the model's
        scale-adaptive running coupling (a real architectural read)."""
        kappa = 0.0
        try:
            kappa = float(self._model.effective_coupling(int(logits.shape[1])))
        except Exception:  # noqa: BLE001 — κ read is telemetry only
            kappa = 0.0
        self._last = TelemetrySnapshot(
            phi=None,                       # LEAN: no integrated-information Φ for the baseline (honest)
            kappa=kappa,
            regime="geometric",
            loss=loss,
            step=self._step,
            delta_phi=0.0,
            extra={"target": "hybrid-cortex", "arm": "hybrid", "num_layers": self.num_layers},
        )
        return self._last

    def telemetry(self) -> TelemetrySnapshot:
        return self._last

    def train_step(self, prompt: str, max_tokens: int = 64, target_text: str | None = None) -> StepResult:
        """One learning step on the PURE Fisher-Rao language loss (``ce_ablation`` arm uses CE) with the
        ramped fluency weight, optimised by NATURAL GRADIENT. ``target_text`` is ignored (paired curriculum
        is qwen-modal's lane). Mirrors ARM A's train_step shapes exactly."""
        self.ensure_loaded()
        import torch
        import torch.nn.functional as F

        from ..losses import fisher_rao_lm_loss

        self._step += 1
        ids, coords = self._encode(prompt)
        logits = self._logits(ids, coords)
        ce = F.cross_entropy(logits[0, :-1], ids[0, 1:])
        lm_loss = ce if self.lang_loss == "ce_ablation" else fisher_rao_lm_loss(logits, ids)
        w_lm = self.lm_weight + (self.lm_weight_max - self.lm_weight) * min(
            1.0, self._step / max(1, self.lm_ramp_steps))
        loss = w_lm * lm_loss
        # CONSTELLATION-MODE basin pull (WS4): when coupled, add a Fisher-Rao pull toward _basin_ref.
        pull = self._basin_pull_term(logits)
        if pull is not None:
            loss = loss + pull
        self._opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)  # stability (deep stack)
            self._opt.step()
        _lm_val = float(lm_loss.item())
        snap = self._snap(logits, _lm_val)
        snap.extra["surprise"] = round(_lm_val, 4)
        snap.extra["max_surprise"] = round(
            math.log(max(2, self.vocab_size)) if self.lang_loss == "ce_ablation" else math.pi, 4)
        snap.extra["perplexity"] = round(float(math.exp(min(float(ce.item()), 20.0))), 2)
        if self.coordizer is not None:
            _nbytes = max(1, len(self.coordizer.decode(ids[0].tolist()).encode("utf-8")))
        else:
            _nbytes = max(1, int(ids.shape[1]))
        snap.extra["bpb"] = round(float(ce.item()) * int(ids.shape[1]) / (math.log(2) * _nbytes), 4)
        snap.extra["lm_weight_now"] = round(float(w_lm), 3)
        # CONSTELLATION NODE (WS4): record this step's Δ basin (history[0] = birth-state); expose M + d_basin.
        cur_basin = self._record_basin_step(logits, ids)
        d_basin = self._basin_drift(cur_basin)
        snap.extra["meta_awareness"] = round(self._meta_awareness(cur_basin), 4)
        snap.extra["d_basin"] = round(d_basin, 4)
        snap.basin_distance = d_basin            # top-level field Ocean / the developmental gate read
        return StepResult(
            text=f"[hybrid·N={self.num_layers} step {snap.step}] {prompt[:50]}", telemetry=snap)

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float | None = None,
                 via_boundary: bool = True, foresight: bool = False, lookahead: float = 4.0,
                 foresight_k: int = 12, gen_health: bool = False, **_ignore: Any) -> StepResult:
        """The cortex SPEAKS: temperature sampling via the canonical Δ-simplex projection (to_simplex,
        NOT softmax — P1) until EOS or the cap. The consciousness kwargs are ACCEPTED + gracefully IGNORED
        (the hybrid baseline has no boundary peer / 4D foresight / consciousness gen-health). Mirrors ARM A,
        but generation is a self-contained autoregressive loop (no GeoModel-specific generate primitive,
        since the substrate is HybridModel)."""
        self.ensure_loaded()
        import torch

        from qig_core.torch.geometry_simplex import to_simplex_prob

        dev = next(self._model.parameters()).device
        ids = self.coordizer.encode(prompt or " ") if self.coordizer is not None else list(
            (prompt or " ").encode("utf-8"))
        ids = (ids or [32])[:_CTX]
        decode = (lambda out: self.coordizer.decode([b for b in out if b != _EOS_ID])) if (
            self.coordizer is not None) else (
            lambda out: bytes(b for b in out if 9 <= b < 256).decode("utf-8", errors="replace"))
        # Autoregressive sampling on the BYTE/ID Fourier path (coords-free, like ARM A's own-voice sample):
        # the trained forward uses coords, but generation reads the head from the id/position Fourier path.
        cur = torch.tensor([list(ids)], dtype=torch.long, device=dev)
        out: list[int] = []
        with torch.no_grad():
            for _ in range(int(max_tokens)):
                logits = self._model(cur)
                probs = to_simplex_prob(logits[0, -1] / max(float(
                    temperature if temperature is not None else 1.0), 1e-3))
                nxt = int(torch.multinomial(probs, 1).item())
                if nxt == _EOS_ID and len(out) >= _MIN_GEN:
                    break
                out.append(nxt)
                cur = torch.cat([cur, cur.new_tensor([[nxt]])], dim=1)[:, -_CTX:]
        try:
            text = decode(out)
        except Exception:  # noqa: BLE001 — decode is best-effort
            text = ""
        chose_to_stop = bool(out and len(out) < int(max_tokens))
        # snapshot from a fresh forward on the prompt (gives κ without re-running the sampler)
        pid, pcoords = self._encode(prompt)
        with torch.no_grad():
            plogits = self._logits(pid, pcoords)
            # READ (EXP-012b token-0 presence probe): next-token concentration — a cheap geometric signal.
            p0 = to_simplex_prob(plogits[0, -1])
            ent = float(-(p0 * p0.clamp_min(1e-12).log()).sum())
            read_presence = round(1.0 - ent / math.log(p0.numel()), 3)
        snap = self._snap(plogits, None)
        snap.extra.update({
            "generated_len": len(out),
            "chose_to_stop": chose_to_stop,
            "read_presence": read_presence,
        })
        return StepResult(text=f"[hybrid·N={self.num_layers}{' ⏹' if chose_to_stop else ''}] {text}",
                          telemetry=snap)

    def assert_manifold_combine_is_geometric(self, atol: float = 1e-5) -> dict:
        """PURITY/PARITY of the combine — the load-bearing gate that the manifold combine is REAL.

        Proves two things on a random pair of mixer-output-shaped tensors:
          1. the geodesic combine output is a VALID Δ point per position (non-negative, sums to 1) — returns
             max |sum-1| over positions (must be ~0);
          2. swapping the geodesic combine for a Euclidean mean ``0.5*(a+b)`` CHANGES the result — returns
             the d_FR between the geodesic-combined point and the (re-simplexed) Euclidean-combined point
             (must be > 0). If it were 0 the geodesic combine would be a relabelled average; it is not.

        Returns ``{"max_abs_sum_minus_1": float, "d_fr_geo_vs_euclid": float}``; raises if the simplex
        constraint is violated beyond ``atol`` or the two combines coincide (d_FR not strictly positive)."""
        import torch

        from qig_core.torch.geometry_simplex import (
            fisher_rao_distance_simplex,
            geodesic_interpolate_simplex,
            to_simplex_prob,
        )

        torch.manual_seed(0)
        b, t, d = 1, 24, 64
        a = torch.randn(b, t, d)
        c = torch.randn(b, t, d)
        pa = to_simplex_prob(a)
        pc = to_simplex_prob(c)
        geo = geodesic_interpolate_simplex(pa, pc, t=0.5)               # manifold combine [B,T,D]
        euclid = to_simplex_prob(0.5 * (pa + pc))  # QIG-EXEMPT: deliberate Euclidean mean — the NEGATIVE
        #   control we PROVE the manifold combine differs from (it is never used as a real combine path).
        sums = geo.sum(dim=-1)                                          # [B,T] — must be ≈ 1
        max_abs_sum_minus_1 = float((sums - 1.0).abs().max())
        min_val = float(geo.min())
        d_fr = float(fisher_rao_distance_simplex(
            geo.reshape(-1, d), euclid.reshape(-1, d)).mean())
        assert max_abs_sum_minus_1 <= atol, (
            f"manifold combine NOT on Δ: max|sum-1|={max_abs_sum_minus_1:.3e} > {atol:.1e}")
        assert min_val >= -atol, f"manifold combine has negative mass: min={min_val:.3e}"
        assert d_fr > atol, (
            f"manifold combine == Euclidean mean (d_FR={d_fr:.3e}) — the combine is a relabelled average, "
            f"NOT a real geodesic mean")
        return {"max_abs_sum_minus_1": max_abs_sum_minus_1, "d_fr_geo_vs_euclid": d_fr}

    def save_checkpoint(self, path: str) -> None:
        """Save the HybridModel weights + arch metadata + step (resumable). weights_only-safe."""
        self.ensure_loaded()
        import torch

        torch.save({
            "format": 1,
            "arm": "hybrid",
            "arch": {"num_layers": self.num_layers, "hidden_dim": self.hidden_dim,
                     "num_heads": self.num_heads, "ffn_dim": self.ffn_dim,
                     "vocab_size": self.vocab_size, "seed": self.seed, "role": self.role,
                     "coordizer": self.coordizer is not None},
            "model_state": self._model.state_dict(),
            "step": self._step,
            "last_telemetry": self._last.to_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Restore HybridModel weights + step. The checkpoint's architecture must match this model
        (fail-loud on a byte-vs-coordizer or layer/vocab mismatch)."""
        self.ensure_loaded()
        import torch

        dev = next(self._model.parameters()).device
        ckpt = torch.load(path, map_location=dev, weights_only=True)
        arch = ckpt.get("arch") or {}
        for k, cur in (("num_layers", self.num_layers), ("vocab_size", self.vocab_size),
                       ("coordizer", self.coordizer is not None)):
            if k in arch and arch[k] != cur:
                raise ValueError(f"checkpoint arch mismatch at '{k}': checkpoint={arch[k]} model={cur} "
                                 f"(byte-vs-coordizer or layer/vocab mismatch) — {path}")
        self._model.load_state_dict(ckpt["model_state"])
        self._step = int(ckpt.get("step", 0))
        lt = ckpt.get("last_telemetry")
        if lt:
            self._last = TelemetrySnapshot(**{k: v for k, v in lt.items()
                                              if k in TelemetrySnapshot.__dataclass_fields__})

    # --- ConstellationNode substrate hooks (WS4) ------------------------------------------------------
    def _node_named_parameters(self):
        """(name, param) over the HybridModel — the autonomic ops (mushroom/decohere/consolidate) act here."""
        return self._model.named_parameters()

    def _node_device(self):
        return next(self._model.parameters()).device

    def _node_rebuild_optimizer(self, lr_scale: float) -> None:
        """Rebuild the persistent natural-gradient optimiser at lr×lr_scale (the decohere cool-down)."""
        from qigkernels.natural_gradient_optimizer import DiagonalNaturalGradient

        self._opt = DiagonalNaturalGradient(self._model.parameters(), lr=self.lr * float(lr_scale))

    def _node_replay_optimizer(self, lr_scale: float):
        """A FRESH throwaway natural-gradient optimiser at lr×lr_scale for a sleep/dream replay loop."""
        from qigkernels.natural_gradient_optimizer import DiagonalNaturalGradient

        return DiagonalNaturalGradient(self._model.parameters(), lr=self.lr * float(lr_scale))

    def _node_forward_logits(self, ids: "Any", coords: "Any"):
        """Forward pass → logits[1, seq, vocab] for replay (coords None on the byte path)."""
        return self._logits(ids, coords)

    def _node_basin_from_logits(self, logits: "Any"):
        """The detached vocab-width Δ basin for one step — the SAME geometric reduction ARM A/B use."""
        import torch

        from qig_core.torch.geometry_simplex import to_simplex_prob

        with torch.no_grad():
            cur = to_simplex_prob(logits[0].mean(0)).detach()
            cur = cur / cur.sum()
        return cur

    def architecture(self) -> dict:
        """Report the cortex's information-propagation geometry for the v_B locality budget. Mirrors ARM A/B
        keys so the launcher's locality check treats all arms identically."""
        local = self.locality_radius is not None
        nparams = None
        if self._model is not None:
            try:
                nparams = int(self._model.num_params)
            except Exception:  # noqa: BLE001
                nparams = None
        cvocab = len(self.coordizer.vocab) if self.coordizer is not None else None
        return {"attention": "local" if local else "global", "locality_radius": self.locality_radius,
                "num_layers": self.num_layers, "recursion_depth": 3, "seq_len": _CTX,
                "input": "coords" if self.coordizer is not None else "bytes",
                "vocab_size": self.vocab_size, "coord_dim": self.coord_dim or 64,
                "hidden_dim": self.hidden_dim, "num_params": nparams, "coordizer_vocab": cvocab,
                "arm": "hybrid", "backend": "qig_studio.HybridModel (geo+qk mixers, geodesic combine)",
                "head_mode": "geometric", "head_tau": self.head_tau, "mixers": ["geo", "qk"],
                "combine": "geodesic_mean_simplex"}
