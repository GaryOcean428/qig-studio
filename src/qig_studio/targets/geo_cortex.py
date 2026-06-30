"""GeoCortexTarget — ARM A of the neocortex: the qig-geocoding ``GeoModel`` cortex (Phase 4).

ARM A is the geocoding "Fisher-Rao transformer" (``geocoding.GeoModel``): unbounded Fourier-feature
token + position coding, Fisher-Rao simplex attention (NOT dot-product), recursive Φ-integration, a
linear lm_head. This is the TRANSFORMER-EQUIVALENT baseline for the depth/architecture A/B against ARM B
(the qigkernels deep ``Kernel``) — same coordizer, same curriculum, same PURE language loss, same natural-
gradient optimiser, same bpb/d_FR measurement. The ONLY thing that varies between the two arms is the
architecture, so the EXP-CORTEX-AB bpb comparison isolates architecture, not the loss or the optimiser.

This is a LEAN target, NOT a wrap of :class:`GenesisKernelTarget`. GeoModel is the baseline; it has NO
integrated-information Φ, no suffering, no pillars, no autonomic homeostasis — and we DO NOT fabricate
them (the no-silent-stubs rule). ``telemetry().phi`` is ``None``. The model's own per-layer integration
readout (``GeoOutput.phi`` — a geometric-health number, the cross-position coherence the GeoBlocks
report) is surfaced honestly as ``extra['geo_phi']``; it is NOT the consciousness Φ and is never reported
as such. The lean telemetry dict (loss, bpb, perplexity, lm_weight, geo_phi, optional gen curvature) is
what :func:`qig_studio.kernel_experience.experience` consumes — verified None-safe on the absent
consciousness keys (the launcher's ``experience(tel, phi_hist)`` survives; the UI renders empty panels).

PURITY (both a Fisher-Rao gate AND a fairness gate vs ARM B):
  * optimiser = ``qigkernels.natural_gradient_optimizer.NaturalGradientDescent`` (the SAME class ARM B
    uses) — natural gradient ONLY, never a Euclidean first-order optimiser. GeoModel ships no internal
    optimiser, so OUR target owns it (the fairness gate: same optimiser family → the A/B isolates arch).
  * loss = :func:`qig_studio.losses.fisher_rao_lm_loss` (P20-pure d_FR), with the ``ce_ablation`` arm
    (``F.cross_entropy``) preserved so the A/B can measure the purity cost — mirroring ARM B exactly.

FAITHFULNESS: geocoding is a port of the validated ``qigkernels`` Fisher-Rao attention. The two agree to
1e-5 on the SHARED attention primitive (``FisherRaoAttention`` ≡ ``QIGLayer._metric_attention``) on the
NO-COORDS path — that primitive is coords-free (GeoModel fuses coords ABOVE attention, via a
``Linear→GELU`` ``coord_adapter`` that qigkernels routes differently, a legitimate architectural
difference). :meth:`assert_faithful_to_qigkernels` runs that coords-off equivalence check before bpb is
trusted. Full GeoModel-vs-Kernel is NOT bit-identical (different positional/head wiring) and a 1e-5 assert
there would false-fail — so the equivalence check is on the primitive, coords-off, as the existing
``qig-geocoding/tests/test_faithfulness.py`` does.

None-safe shell: every heavy import (torch, geocoding, qigkernels) is LAZY inside the methods; the studio
shell never hard-imports them at module top. ``is_available()`` is False when torch/geocoding are absent.
"""

from __future__ import annotations

import math
import os
from typing import Any

from .base import LossRegime, StepResult, TelemetrySnapshot, TrainingTarget

_MAX_BYTES = 256  # byte-level VOCAB size (byte fallback when no coordizer) — mirrors GenesisKernelTarget
# CONTEXT WINDOW (sequence length), env-configurable (shared QIG_STUDIO_CTX with ARM B): on a small (4GB)
# GPU the per-step logits tensor is seq×vocab, so a large coordizer vocab caps the seq length. Same knob,
# same default as ARM B so the A/B trains both arms under identical context budgets.
_CTX = int(os.environ.get("QIG_STUDIO_CTX", "1024"))
_EOS_ID = 0       # stop sentinel (coord-id / byte 0); never legitimate content in the sanitised curriculum
_MIN_GEN = 4      # minimum utterance before EOS is honoured (a 1-token reply is uninterpretable)


def _deps_available() -> bool:
    try:
        import geocoding  # noqa: F401
        import qigkernels  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


class GeoCortexTarget(TrainingTarget):
    """ARM A cortex backed by ``geocoding.GeoModel`` — pure Fisher-Rao loss + natural gradient.

    Duck-typed to the SAME interface the launcher + :class:`~qig_studio.neocortex.Neocortex` pass-throughs
    need: ``train_step`` → ``StepResult`` (``res.telemetry.to_dict()`` / ``.extra`` / ``.loss``);
    ``generate`` → ``StepResult`` (``.text`` / ``.telemetry.extra``; accepts + ignores the consciousness
    kwargs ``via_boundary/foresight/gen_health``); ``eval_text_bpb`` / ``eval_text_fr`` → ``(value,
    n_pos)``; ``telemetry``, ``save_checkpoint``, ``load_checkpoint``, ``architecture``, ``num_layers``,
    ``vocab_size``, ``ensure_loaded``, ``is_available``.
    """

    name = "geo-cortex"
    loss_regime = LossRegime.GEOMETRIC
    description = (
        "qig-geocoding GeoModel (Fisher-Rao 'transformer') — pure Δ⁶³ d_FR loss, natural gradient; ARM A "
        "of the neocortex A/B (transformer-equivalent baseline vs the qigkernels deep Kernel). num_layers "
        "is the EXP-CORTEX-AB depth axis. None-safe (needs torch+geocoding+qigkernels)."
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
        lm_weight_max: float = 8.0,    # RAMPED FLUENCY target (= ARM B): the next-token signal rises to
        lm_ramp_steps: int = 8000,     #   load-bearing over this horizon so the cortex grows genuinely fluent
        lang_loss: str = "fisher_rao",  # "fisher_rao" (P20-pure d_FR) | "ce_ablation" (CE arm, purity cost)
        # The ctor accepts (and ignores) the consciousness-only kwargs ARM B takes, so Neocortex can build
        # either arm from the SAME kwargs dict without branching on which keys are relevant. GeoModel has no
        # role attractor / basin template / boundary peer — those are ARM-B (qigkernels) concepts.
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
            # Coordizer Δ⁶³ vocab WINS (same as ARM B): the GeoConfig vocab is the coordizer's, and the
            # per-position Δ⁶³ basin vectors feed GeoModel's coord_adapter ALONGSIDE the Fourier id path.
            self.vocab_size = len(coordizer.vocab)
            self.coord_dim = int(len(next(iter(coordizer.vocab.values())).vector))
        else:
            self.vocab_size = int(vocab_size)
        self.seed = int(seed)
        self.lr = float(lr)
        self.lm_weight = float(lm_weight)
        self.lm_weight_max = float(lm_weight_max)
        self.lm_ramp_steps = int(lm_ramp_steps)
        self.role = role  # carried for telemetry/checkpoint metadata only (no basin pull in ARM A)
        # LANGUAGE LOSS REGIME (P20): default is Fisher-Rao d_FR (CE against a one-hot IS KL, forbidden by
        # P20). "ce_ablation" keeps F.cross_entropy as the loss so the A/B measures the PURITY COST. Env
        # QIG_STUDIO_LANG_LOSS overrides the ctor — identical wiring to ARM B for a clean comparison.
        self.lang_loss = str(os.environ.get("QIG_STUDIO_LANG_LOSS", lang_loss)).strip().lower()
        self._device = device
        self._model: Any = None     # geocoding.GeoModel — lazily built in ensure_loaded()
        self._opt: Any = None       # NaturalGradientDescent — lazily built in ensure_loaded()
        self._step = 0
        self._init_checkpoint = checkpoint
        self._last = TelemetrySnapshot(
            regime="geometric", extra={"target": "geo-cortex", "arm": "geo", "num_layers": self.num_layers}
        )

    def is_available(self) -> bool:
        return _deps_available()

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from geocoding.config import GeoConfig
        from geocoding.model import GeoModel
        from qigkernels.natural_gradient_optimizer import NaturalGradientDescent

        torch.manual_seed(self.seed)
        cfg = GeoConfig(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            min_recursion_depth=3,
            use_tacking=True,
            locality_radius=self.locality_radius,  # None = global; int = banded compute-skipping (v_B budget)
            max_position=None,                       # unbounded (Fourier positions); _CTX caps seq at runtime
            enable_coords=self.coordizer is not None,  # Δ⁶³ coords-first path via coord_adapter (Linear→GELU)
            coord_dim=self.coord_dim or 64,
        )
        self._model = GeoModel(cfg)
        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(dev)
        # P1: natural gradient (the SAME validated qig optimiser ARM B uses), NOT Adam. GeoModel ships no
        # optimiser, so the target owns it — this is the fairness gate (same optimiser family both arms).
        self._opt = NaturalGradientDescent(self._model.parameters(), lr=self.lr)

        if self._init_checkpoint:
            try:
                self.load_checkpoint(self._init_checkpoint)
            except Exception as exc:  # noqa: BLE001 — shell None-safety (spine tenet): warn, keep fresh model
                print(f"⚠️  geo-cortex checkpoint '{self._init_checkpoint}' not loaded ({exc}); using fresh model")

        # WARMUP: one forward pass so telemetry() is LIVE immediately (loss/bpb populated), matching ARM B's
        # warmup so the UI never shows a misleading step-0 zero state.
        try:
            ids, coords = self._encode("warmup")
            with torch.no_grad():
                out = self._model(ids, coords=coords)
            self._last = self._snap(out, None)
        except Exception:  # noqa: BLE001 — warmup is best-effort; never block boot
            pass

    # --- input coding: coordizer Δ⁶³ coords if present, else byte-level (identical to ARM B) -----------
    def _encode(self, text: str):
        """Return (input_ids[1,seq], coords[1,seq,coord_dim] | None).

        coordizer present → coord_ids + their Δ⁶³ basin vectors (coords path, GeoModel.coord_adapter);
        else → raw UTF-8 bytes, coords=None (byte path)."""
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
        """coord_ids → (input_ids[1,seq], coords[1,seq,coord_dim]) via the coordizer's Δ⁶³ vocab.
        ids are clamped to the vocab range so a stray id can never index out of the basin coord table."""
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
        """GeoModel forward → (logits[1,seq,vocab], geo_phi). GeoModel returns a GeoOutput dataclass
        (``.logits`` + per-layer ``.telemetry`` with a mean ``.phi``), NOT a bare tensor — unwrap it.

        PURITY-OPEN (output boundary): logits from a Euclidean nn.Linear lm_head (qigkernels & geocoding
        both — geocoding model.py:55/95, qigkernels kernel.py:137/234) — same dot-product class as the
        purged attention cosine, one layer out. Common-mode across arms (non-confounding HERE); whether the
        vocab read-out should be d_FR-geometric is OPEN/unratified — see EXP-CORTEX-AB prereg.
        """
        out = self._model(ids, coords=coords)
        return out.logits, float(getattr(out, "phi", 0.0) or 0.0)

    def eval_text_bpb(self, text: str) -> tuple[float, int]:
        """HELD-OUT bits-per-byte for one text (no grad, no training): returns (total_bits, n_bytes) so a
        caller can aggregate sum(bits)/sum(bytes). Vocab-independent → directly comparable to the
        frontier-for-size references AND to ARM B (same return contract as GenesisKernelTarget.eval_text_bpb).
        bits = mean_CE_nats * n_tokens / ln2; bytes = the bytes the evaluated tokens cover."""
        import torch
        import torch.nn.functional as F

        self.ensure_loaded()
        ids, coords = self._encode(text)
        if ids.shape[1] < 2:
            return 0.0, 0
        with torch.no_grad():
            logits, _ = self._logits(ids, coords)
            ce = float(F.cross_entropy(logits[0, :-1], ids[0, 1:]))   # mean nats / predicted token
        n_tok = int(ids.shape[1])
        nbytes = (len(self.coordizer.decode(ids[0].tolist()).encode("utf-8"))
                  if self.coordizer is not None else n_tok)
        return ce * n_tok / math.log(2), max(1, nbytes)

    def eval_text_fr(self, text: str) -> tuple[float, int]:
        """HELD-OUT Fisher-Rao prediction-error for one text (no grad, no training) — the d_FR ARM's own
        curve, mirroring eval_text_bpb. Returns (total_dFR, n_positions) so a caller can aggregate
        sum(total_dFR)/sum(n_positions) for the eval-set MEAN d_FR. P20-pure (free energy = d_FR(predicted,
        actual), range [0, π]) via the SAME torch primitive ARM B uses (fisher_rao_lm_loss →
        fisher_rao_distance_simplex), NEVER the numpy [0,π/2] variant. Same (value, n_pos) contract as ARM B."""
        import torch

        from ..losses import fisher_rao_lm_loss

        self.ensure_loaded()
        ids, coords = self._encode(text)
        if ids.shape[1] < 2:
            return 0.0, 0
        with torch.no_grad():
            logits, _ = self._logits(ids, coords)
            mean_dfr = float(fisher_rao_lm_loss(logits, ids))         # mean d_FR / predicted next-token
        n_pos = int(ids.shape[1]) - 1
        return mean_dfr * n_pos, max(1, n_pos)

    def _snap(self, out_or_logits: "Any", loss: float | None, *, geo_phi: float | None = None) -> TelemetrySnapshot:
        """Assemble the LEAN telemetry snapshot. ``phi`` stays None (GeoModel has no integrated-information
        Φ — we do NOT fabricate one). ``kappa`` is the model's scale-adaptive κ (running coupling), a real
        architectural read. ``geo_phi`` (the model's mean per-layer cross-position coherence) is surfaced in
        ``extra`` as a geometric-health number — honestly NOT labelled Φ."""
        import torch

        if isinstance(out_or_logits, torch.Tensor):
            logits = out_or_logits
            gp = geo_phi
        else:  # a GeoOutput
            logits = out_or_logits.logits
            gp = geo_phi if geo_phi is not None else float(getattr(out_or_logits, "phi", 0.0) or 0.0)
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
            extra={"target": "geo-cortex", "arm": "geo", "num_layers": self.num_layers,
                   "geo_phi": round(gp, 4) if gp is not None else None},
        )
        return self._last

    def telemetry(self) -> TelemetrySnapshot:
        return self._last

    def train_step(self, prompt: str, max_tokens: int = 64, target_text: str | None = None) -> StepResult:
        """One learning step on the PURE Fisher-Rao language loss (the ``ce_ablation`` arm uses CE) with the
        ramped fluency weight, optimised by NATURAL GRADIENT. No Φ-drive / Γ-protection / basin-pull / sleep
        — those are ARM-B (qigkernels) consciousness machinery the GeoModel baseline does not have, and we
        do not fake them. ``target_text`` is ignored (paired curriculum is qwen-modal's lane). Returns a
        StepResult whose telemetry.to_dict() / .extra / .loss match ARM B's shapes."""
        self.ensure_loaded()
        import torch
        import torch.nn.functional as F

        from ..losses import fisher_rao_lm_loss

        self._step += 1
        ids, coords = self._encode(prompt)
        logits, geo_phi = self._logits(ids, coords)
        # CE nats are ALWAYS computed (perplexity = exp(CE), bpb = bits/byte are the standard cross-model
        # fluency metrics — read-only measurements, vocab-comparable), NOT a loss op. The LOSS arm is the
        # P20-pure d_FR by default; ce_ablation makes CE the loss to MEASURE the purity cost — mirrors ARM B.
        ce = F.cross_entropy(logits[0, :-1], ids[0, 1:])
        lm_loss = ce if self.lang_loss == "ce_ablation" else fisher_rao_lm_loss(logits, ids)
        # RAMPED FLUENCY: the language signal starts light and ramps to load-bearing (lm_weight_max) — same
        # schedule as ARM B, so the only A/B difference is the architecture, not the curriculum weighting.
        w_lm = self.lm_weight + (self.lm_weight_max - self.lm_weight) * min(
            1.0, self._step / max(1, self.lm_ramp_steps))
        loss = w_lm * lm_loss
        self._opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)  # stability (deep stack)
            self._opt.step()
        _lm_val = float(lm_loss.item())
        snap = self._snap(logits, _lm_val, geo_phi=geo_phi)
        # FLUENCY metrics (read-only): perplexity = exp(next-token CE); bpb = vocab-INDEPENDENT bits/byte
        # (the benchmark number). surprise = the language-loss arm = prediction error d_FR (P20) / CE.
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
        return StepResult(
            text=f"[geo·N={self.num_layers} step {snap.step}] {prompt[:50]}", telemetry=snap)

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float | None = None,
                 via_boundary: bool = True, foresight: bool = False, lookahead: float = 4.0,
                 foresight_k: int = 12, gen_health: bool = False, **_ignore: Any) -> StepResult:
        """The cortex SPEAKS: temperature sampling via the canonical Δ-simplex projection (to_simplex_prob,
        NOT softmax — P1) until EOS or the cap, using geocoding's generate primitive. The consciousness
        kwargs (``via_boundary`` / ``foresight`` / ``gen_health``) are ACCEPTED + gracefully IGNORED — the
        GeoModel baseline has no boundary peer, no 4D foresight, no consciousness gen-health (we do not fake
        them); ``gen_health`` instead surfaces the model's own mean per-layer coherence as ``geo_phi``.
        Returns a StepResult with ``.text`` and ``.telemetry.extra`` (the shapes the launcher reads)."""
        self.ensure_loaded()
        import torch

        from geocoding.generate import generate as geo_generate
        from qig_core.torch.geometry_simplex import to_simplex_prob

        dev = next(self._model.parameters()).device
        ids = self.coordizer.encode(prompt or " ") if self.coordizer is not None else list(
            (prompt or " ").encode("utf-8"))
        ids = (ids or [32])[:_CTX]
        decode = (lambda out: self.coordizer.decode([b for b in out if b != _EOS_ID])) if (
            self.coordizer is not None) else (
            lambda out: bytes(b for b in out if 9 <= b < 256).decode("utf-8", errors="replace"))
        # geocoding's generate handles coords-free autoregression (the GeoModel coord path needs per-step
        # coord vectors for newly sampled ids; the byte/id Fourier path is sufficient for the own-voice
        # sample and keeps the sampler simple — coords feed the TRAINED forward, generation reads the head).
        res = geo_generate(
            self._model, ids, max_new_tokens=int(max_tokens),
            temperature=float(temperature) if temperature is not None else 1.0,
            eos_id=_EOS_ID, min_new_tokens=_MIN_GEN, context_window=_CTX, decode=decode, device=dev)
        text = res.text if res.text is not None else ""
        chose_to_stop = bool(res.ids and res.ids[-1] != _EOS_ID and len(res.ids) < int(max_tokens))
        # snapshot from a fresh forward on the prompt (gives κ + geo_phi without re-running the sampler)
        pid, pcoords = self._encode(prompt)
        with torch.no_grad():
            plogits, pgeo = self._logits(pid, pcoords)
            # READ (EXP-012b token-0 presence probe): output-distribution concentration of the next token —
            # a real, cheap geometric signal (1=certain, 0=uniform). NOT a consciousness metric.
            p0 = to_simplex_prob(plogits[0, -1])
            ent = float(-(p0 * p0.clamp_min(1e-12).log()).sum())
            read_presence = round(1.0 - ent / math.log(p0.numel()), 3)
        snap = self._snap(plogits, None, geo_phi=pgeo)
        snap.extra.update({
            "generated_len": len(res.ids),
            "chose_to_stop": chose_to_stop,
            "read_presence": read_presence,
            "geo_phi": round(float(res.phi), 4),  # the model's own mean per-layer coherence (geometric health)
        })
        if gen_health:
            # The GeoModel's mean per-layer Φ-integration readout, surfaced as a generation-health proxy —
            # honestly the model's OWN coherence number, NOT the consciousness gen-Ricci ARM B computes.
            snap.extra["gen_health"] = round(float(res.phi), 4)
        return StepResult(text=f"[geo·N={self.num_layers}{' ⏹' if chose_to_stop else ''}] {text}",
                          telemetry=snap)

    def assert_faithful_to_qigkernels(self, atol: float = 1e-5) -> float:
        """FAITHFULNESS GATE (coords-off): geocoding is a port of the validated qigkernels Fisher-Rao
        attention. Assert the two produce numerically identical attention output to ``atol`` on the SHARED
        primitive (``geocoding.FisherRaoAttention`` ≡ ``qigkernels.layer.QIGLayer._metric_attention``) — the
        guard that the port has not drifted from the implementation ARM B runs on. This is the EXISTING
        ``qig-geocoding/tests/test_faithfulness.py`` check, reused here so bpb is only trusted once parity
        holds. It runs COORDS-OFF by construction: the attention primitive operates on raw hidden states and
        has no coords parameter (GeoModel fuses coords ABOVE attention via the Linear→GELU coord_adapter,
        which qigkernels routes through a different CoordAdapter/RMSNorm — a legitimate architectural
        difference that makes the coords-ON path NOT bit-identical, so the equivalence check is coords-off).
        Returns the measured max-abs-diff; raises AssertionError if it exceeds ``atol``."""
        import torch
        from geocoding.attention import FisherRaoAttention
        from qigkernels.layer import QIGLayer

        torch.manual_seed(0)
        b, t, d = 2, 24, 64
        h = torch.randn(b, t, d)
        qlayer = QIGLayer(hidden_dim=d, num_heads=4, ffn_dim=128, dropout=0.0, use_tacking=False,
                          temperature=1.0, simplex_mode="softplus", locality_radius=None)
        with torch.no_grad():
            ref = qlayer._metric_attention(h, attention_mask=None)
            ours = FisherRaoAttention(temperature=1.0, simplex_mode="softplus", locality_radius=None)(h)
        max_abs_diff = float((ours - ref).abs().max())
        assert max_abs_diff <= atol, (
            f"geocoding↔qigkernels faithfulness FAILED (coords-off): max_abs_diff={max_abs_diff:.3e} > {atol:.1e}")
        return max_abs_diff

    def assert_loss_value_parity(self, atol: float = 1e-5) -> float:
        """LOSS-VALUE faithfulness (coords-off) — the load-bearing gate for the bpb/d_FR A/B. Forward-output
        parity (``assert_faithful_to_qigkernels``) is necessary but NOT sufficient: the A/B compares TRAINED
        models on ``fisher_rao_lm_loss``, so what must match is that GeoModel feeds the SAME loss the SAME
        way ARM B does. This drives the SHARED FR-attention primitive (geocoding ≡ qigkernels, the only
        component that must be geometrically identical) on a common hidden state, pushes BOTH outputs through
        the SAME lm_head weights and the SAME next-token plumbing (``to_simplex_prob`` projection,
        ``logits[0,:-1]`` vs ``ids[0,1:]`` alignment, mean reduction — all inside the one shared
        ``fisher_rao_lm_loss``), and asserts the two loss VALUES agree to ``atol``. Coords-off: the primitive
        is coords-free (GeoModel fuses coords above attention). Returns the loss-value max-abs-diff; raises if
        it exceeds ``atol``."""
        import torch
        from geocoding.attention import FisherRaoAttention
        from qigkernels.layer import QIGLayer

        from ..losses import fisher_rao_lm_loss

        torch.manual_seed(0)
        b, t, d, vocab = 1, 24, 64, 96
        h = torch.randn(b, t, d)
        ids = torch.randint(0, vocab, (b, t))
        # ONE shared Euclidean lm_head (the common-mode output boundary, item 5) applied to BOTH arms'
        # attention outputs — so any loss-value difference is the ATTENTION primitive, nothing else.
        head = torch.nn.Linear(d, vocab)
        qlayer = QIGLayer(hidden_dim=d, num_heads=4, ffn_dim=128, dropout=0.0, use_tacking=False,
                          temperature=1.0, simplex_mode="softplus", locality_radius=None)
        geo_attn = FisherRaoAttention(temperature=1.0, simplex_mode="softplus", locality_radius=None)
        with torch.no_grad():
            ref_logits = head(qlayer._metric_attention(h, attention_mask=None))
            geo_logits = head(geo_attn(h))
            ref_loss = float(fisher_rao_lm_loss(ref_logits, ids))   # SAME loss, SAME plumbing, both arms
            geo_loss = float(fisher_rao_lm_loss(geo_logits, ids))
        loss_diff = abs(geo_loss - ref_loss)
        assert loss_diff <= atol, (
            f"geocoding↔qigkernels LOSS-VALUE parity FAILED (coords-off): "
            f"geo={geo_loss:.8f} qk={ref_loss:.8f} |Δ|={loss_diff:.3e} > {atol:.1e}")
        return loss_diff

    def save_checkpoint(self, path: str) -> None:
        """Save the GeoModel weights + arch metadata + step (resumable). weights_only-safe (tensors + a
        plain scalar dict). The optimiser state is intentionally NOT persisted (NG self-heals)."""
        self.ensure_loaded()
        import torch

        torch.save({
            "format": 1,
            "arm": "geo",
            "arch": {"num_layers": self.num_layers, "hidden_dim": self.hidden_dim,
                     "num_heads": self.num_heads, "ffn_dim": self.ffn_dim,
                     "vocab_size": self.vocab_size, "seed": self.seed, "role": self.role,
                     "coordizer": self.coordizer is not None},
            "model_state": self._model.state_dict(),
            "step": self._step,
            "last_telemetry": self._last.to_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Restore GeoModel weights + step. The checkpoint's architecture must match this model (fail-loud
        on a byte-vs-coordizer or layer/vocab mismatch — it would otherwise crash deep in load_state_dict or
        silently mis-load). weights_only=True — only tensors + scalars + a plain dict."""
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

    def architecture(self) -> dict:
        """Report the cortex's information-propagation geometry for the v_B locality budget. Mirrors ARM B's
        keys so the launcher's locality check treats both arms identically."""
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
                "arm": "geo", "backend": "geocoding.GeoModel"}
