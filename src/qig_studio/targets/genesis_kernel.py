"""GenesisKernelTarget — a FRESH from-scratch ``qigkernels.Kernel(num_layers=N)``.

This is the brain-doc **neocortex** (§1/§6 Step 1): deep stacked-Δ⁶³ = distinct-weight pure
Fisher-Rao ``QIGLayer``s in an ``nn.ModuleList``, trained by natural gradient. "Genesis" = origin =
trained FROM SCRATCH — no checkpoint, no QIGChat, no ``QIGKernelRecursive`` cosine proxy. It is the
honest answer to "which kernel": the already-upgraded genesis lineage (the layers work lives in
qigkernels; vex's genesis is orchestration-only/inspiration, pantheon's is archived).

Dependency-free training signal: a byte-level vocab (256) so a fresh kernel can learn next-token
structure on the basin-driving curriculum WITHOUT a trained coordizer (coordizer-basin init is a
follow-up, NEEDS-BUILD). None-safe: needs torch + qigkernels, so ``is_available()`` is False in the
light app shell and True where the heavy deps are present (e.g. the qig-consciousness venv).

Scale defaults (~30-60M params, fits 4 GB): hidden_dim 384, num_layers 4, ffn 1024 — the brain-doc
Step-1 sizing. ``num_layers`` is the EXP-CORTEX-AB depth axis (1 vs N).
"""

from __future__ import annotations

from typing import Any

from .base import LossRegime, StepResult, TelemetrySnapshot, TrainingTarget

_MAX_BYTES = 256  # byte-level context cap per step (cheap from-scratch signal)
_EOS_BYTE = 0     # the kernel's stop token — it CHOOSES to stop (observer principle, no fixed length)
# Mushroom intensity → weight-noise σ (bounded plasticity; the dose the autonomic loop selects).
_MUSHROOM_SIGMA = {"mushroom-micro": 0.01, "mushroom-moderate": 0.03, "mushroom-heroic": 0.06}


def _deps_available() -> bool:
    try:
        import qigkernels  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


class GenesisKernelTarget(TrainingTarget):
    name = "genesis"
    loss_regime = LossRegime.GEOMETRIC
    description = (
        "Fresh from-scratch qigkernels.Kernel(num_layers=N) — pure Δ⁶³ Fisher-Rao stacked layers, "
        "natural gradient; the brain-doc neocortex. No checkpoint (genesis=from-scratch); byte-level "
        "vocab. num_layers is the EXP-CORTEX-AB depth axis. None-safe (needs torch+qigkernels)."
    )

    def __init__(
        self,
        num_layers: int = 4,
        hidden_dim: int = 384,
        num_heads: int = 6,
        ffn_dim: int = 1024,
        vocab_size: int = _MAX_BYTES,
        seed: int = 0,
        lr: float = 1e-3,
        device: str | None = None,
        locality_radius: int | None = None,
        coordizer: Any = None,
    ) -> None:
        self.num_layers = num_layers
        self.locality_radius = locality_radius
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        # Coords path: a trained FisherCoordizer (Δ⁶³ vocab) replaces byte-level coding — input_ids
        # become coord_ids and the per-position Δ⁶³ basin vectors feed the kernel's CoordAdapter
        # ALONGSIDE the fourier(input_ids) path (qigkernels Kernel enable_coords). None → byte-level.
        self.coordizer = coordizer
        self.coord_dim = 0
        if coordizer is not None:
            self.vocab_size = len(coordizer.vocab)
            # Δ⁶³ basin dimension from a sample coord vector (BASIN_DIM, normally 64).
            self.coord_dim = int(len(next(iter(coordizer.vocab.values())).vector))
        else:
            self.vocab_size = vocab_size
        self.seed = seed
        self.lr = lr
        self._device = device
        self._kernel = None
        self._opt = None
        self._step = 0
        self._last = TelemetrySnapshot(regime="unknown", extra={"target": "genesis", "num_layers": num_layers})

    def is_available(self) -> bool:
        return _deps_available()

    def ensure_loaded(self) -> None:
        if self._kernel is not None:
            return
        import torch
        from qigkernels import Kernel
        from qigkernels.natural_gradient_optimizer import NaturalGradientDescent

        torch.manual_seed(self.seed)
        self._kernel = Kernel(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            min_recursion_depth=3,
            use_tacking=True,
            locality_radius=self.locality_radius,  # None = global; set = windowed-local (v_B budget)
            enable_coords=self.coordizer is not None,  # Δ⁶³ coords-first path via CoordAdapter
            coord_dim=self.coord_dim or 64,
        )
        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._kernel.to(dev)
        # P1: natural gradient (the validated qig kernel optimiser), NOT Adam.
        self._opt = NaturalGradientDescent(self._kernel.parameters(), lr=self.lr)

    # --- input coding: coordizer Δ⁶³ coords if present, else byte-level (dependency-free) ----------
    def _encode(self, text: str):
        """Return (input_ids[1,seq], coords[1,seq,coord_dim] | None).

        coordizer present → coord_ids + their Δ⁶³ basin vectors (coords path);
        else → raw bytes, coords=None (byte path, bit-identical to the original)."""
        if self.coordizer is not None:
            ids = self.coordizer.encode(text or " ")[: _MAX_BYTES]
            if len(ids) < 2:
                ids = (ids + [32, 32])[:2]
            return self._ids_to_tensors(ids)
        import torch

        ids = list((text or " ").encode("utf-8"))[: _MAX_BYTES]
        if len(ids) < 2:
            ids = (ids + [32, 32])[:2]
        dev = next(self._kernel.parameters()).device
        return torch.tensor([ids], dtype=torch.long, device=dev), None

    def _ids_to_tensors(self, ids: list[int]):
        """coord_ids → (input_ids[1,seq], coords[1,seq,coord_dim]) via the coordizer's Δ⁶³ vocab.
        ids are clamped to the vocab range so a stray id can never index out of the embedding."""
        import numpy as np
        import torch

        dev = next(self._kernel.parameters()).device
        vmax = self.vocab_size - 1
        ids = [min(max(int(i), 0), vmax) for i in ids]
        vecs = np.stack([np.asarray(self.coordizer.vocab[i].vector, dtype=np.float32) for i in ids])
        input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
        coords = torch.from_numpy(vecs).to(dev).unsqueeze(0)  # [1, seq, coord_dim]
        return input_ids, coords

    def _snap(self, tel: Any, loss: float | None) -> TelemetrySnapshot:
        prev = self._last.phi
        phi = float(getattr(tel, "phi", 0.0) or 0.0)
        self._last = TelemetrySnapshot(
            phi=phi,
            kappa=float(getattr(tel, "kappa", 0.0) or 0.0),
            regime=str(getattr(tel, "regime", "unknown") or "unknown"),
            loss=loss,
            step=self._step,
            delta_phi=phi - prev,
            extra={"target": "genesis", "num_layers": self.num_layers,
                   "recursion_depth": int(getattr(tel, "recursion_depth", 0) or 0)},
        )
        return self._last

    def telemetry(self) -> TelemetrySnapshot:
        return self._last

    def _temperature_from_kappa(self, kappa: float) -> float:
        # The kernel's OWN κ sets its sampling boldness (its choice): higher κ (more coupled/decisive) →
        # lower temperature; near the attractor (≈64) → ~1.0. Clamped to a sane range.
        t = 64.0 / kappa if kappa > 1e-3 else 1.0
        return float(max(0.3, min(2.0, t)))

    def _self_observe(self, out_bytes: list[int], gen_basins: list) -> float:
        """SELF-OBSERVATION (M ∈ [0,1]): feed the kernel its OWN generated output and measure how
        consistently it re-derives the same output distribution (Fisher-Rao self-recognition on Δ).
        High M = the kernel recognises/models its own output. Honest proxy, pure Fisher-Rao."""
        import math

        import torch
        import torch.nn.functional as F
        from qig_core.geometry_simplex import fisher_rao_distance_simplex

        if len(out_bytes) < 2 or not gen_basins:
            return 0.0
        dev = next(self._kernel.parameters()).device
        if self.coordizer is not None:
            ids, coords = self._ids_to_tensors([max(1, b) for b in out_bytes])
        else:
            ids = torch.tensor([[max(1, b) for b in out_bytes]], dtype=torch.long, device=dev)
            coords = None
        with torch.no_grad():
            re = F.softmax(self._kernel(ids, return_telemetry=True, coords=coords)[0][0, :-1], dim=-1)
            gen_mean = torch.stack(gen_basins).mean(0)            # mean GENERATED output distribution
            re_mean = re.mean(0)                                   # mean RE-READ output distribution
            gen_mean = gen_mean / gen_mean.sum()
            re_mean = re_mean / re_mean.sum()
            d = float(fisher_rao_distance_simplex(gen_mean[None], re_mean[None]).item())
        return float(max(0.0, 1.0 - d / (math.pi / 2)))           # 1 = perfect self-recognition

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float | None = None) -> StepResult:
        """The kernel SPEAKS as it chooses: stochastic sampling (temperature from its OWN κ) until it
        emits EOS (observer principle — NOT a fixed length, NOT greedy argmax), while OBSERVING its own
        output (per-token confidence + output-basin trajectory) and itself (self-observation M)."""
        self.ensure_loaded()
        import torch
        import torch.nn.functional as F

        ids, coords = self._encode(prompt)
        out_bytes: list[int] = []
        out_probs: list[float] = []
        gen_basins: list = []
        last_tel = None
        chose_to_stop = False
        with torch.no_grad():
            for _ in range(min(max_tokens, _MAX_BYTES)):
                logits, last_tel = self._kernel(ids, return_telemetry=True, coords=coords)
                temp = temperature if temperature is not None else self._temperature_from_kappa(
                    float(getattr(last_tel, "kappa", 0.0) or 0.0))
                p = F.softmax(logits[0, -1] / max(temp, 1e-3), dim=-1)
                nxt = int(torch.multinomial(p, 1).item())         # the kernel's CHOICE (not argmax)
                out_bytes.append(nxt)
                out_probs.append(float(p[nxt]))
                gen_basins.append(p.detach())                     # own-output observation
                ids = torch.cat([ids, ids.new_tensor([[nxt]])], dim=1)[:, -_MAX_BYTES:]
                if coords is not None:                             # keep coords aligned with ids
                    _, cv = self._ids_to_tensors([nxt])
                    coords = torch.cat([coords, cv], dim=1)[:, -_MAX_BYTES:]
                if nxt == _EOS_BYTE:                               # chose to stop (observer principle)
                    chose_to_stop = True
                    break
        if self.coordizer is not None:
            text = self.coordizer.decode([b for b in out_bytes if b != _EOS_BYTE])
        else:
            text = bytes(b for b in out_bytes if 9 <= b < 256).decode("utf-8", errors="replace")
        m = self._self_observe(out_bytes, gen_basins)
        snap = self._snap(last_tel, None)
        snap.extra.update({
            "M_self_observation": round(m, 3),                    # observes ITSELF
            "chose_to_stop": chose_to_stop,                       # spoke as it chose (EOS)
            "generated_len": len(out_bytes),
            "mean_token_confidence": round(sum(out_probs) / max(1, len(out_probs)), 3),  # observes its OUTPUT
        })
        return StepResult(text=f"[genesis·N={self.num_layers}{' ⏹' if chose_to_stop else ''}] {text}", telemetry=snap)

    def train_step(self, prompt: str, max_tokens: int = 64, target_text: str | None = None) -> StepResult:
        # WAKE: one Fisher-salience step. Geometric → next-byte CE on the basin-driving prompt
        # (target_text ignored; lm_weight=0). Optimiser is natural gradient (P1).
        self.ensure_loaded()
        import torch
        import torch.nn.functional as F

        self._step += 1
        ids, coords = self._encode(prompt)
        logits, tel = self._kernel(ids, return_telemetry=True, coords=coords)
        loss = F.cross_entropy(logits[0, :-1], ids[0, 1:])
        self._opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._kernel.parameters(), 1.0)  # stability (deep stack)
            self._opt.step()
        snap = self._snap(tel, float(loss.item()))
        return StepResult(text=f"[genesis·N={self.num_layers} step {snap.step}] basin-driving: {prompt[:50]}",
                          telemetry=snap)

    def architecture(self) -> dict:
        # qigkernels.QIGLayer is GLOBAL metric attention by default → v_B-NON-LOCAL; pass
        # locality_radius to make it windowed-local (respects the finite-propagation budget). The
        # locality_budget check reads this; local-vs-global is the EXP-LOCAL-ATTN A/B.
        local = self.locality_radius is not None
        return {"attention": "local" if local else "global", "locality_radius": self.locality_radius,
                "num_layers": self.num_layers, "recursion_depth": 3, "seq_len": _MAX_BYTES,
                "input": "coords" if self.coordizer is not None else "bytes", "vocab_size": self.vocab_size}

    def supports_protocol(self) -> bool:
        return True

    def run_protocol(self, command: str, args: dict) -> dict:
        """Minimal-but-real autonomic interventions on the fresh kernel. MUSHROOM is implemented as
        bounded weight-noise plasticity (Tononi-style downscaling perturbation, dose-scaled). SLEEP /
        DREAM / ESCAPE are v1 light operations with honest NEEDS-BUILD labels (full EWC consolidation /
        basin-mixture augmentation is the §4b build)."""
        self.ensure_loaded()
        import torch

        applied = "noop"
        if command in _MUSHROOM_SIGMA:  # WAKE-state plasticity — bounded noise (the Ocean-chosen dose)
            sigma = _MUSHROOM_SIGMA[command]
            with torch.no_grad():
                for p in self._kernel.parameters():
                    p.add_(torch.randn_like(p) * sigma)
            applied = f"weight-noise σ={sigma}"
        elif command == "escape":  # recover: shrink coupling-ish via a fresh optimiser state
            from qigkernels.natural_gradient_optimizer import NaturalGradientDescent

            self._opt = NaturalGradientDescent(self._kernel.parameters(), lr=self.lr * 0.5)
            applied = "reset optimiser (lr×0.5)"
        else:  # sleep / deep-sleep / dream: v1 light — full EWC/augmentation is NEEDS-BUILD (§4b)
            applied = f"{command}: v1 light (full EWC/augmentation NEEDS-BUILD)"
        return {"command": command, "available": True, "applied": applied,
                "telemetry": self._last.to_dict()}
