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
    ) -> None:
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
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
        )
        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._kernel.to(dev)
        # P1: natural gradient (the validated qig kernel optimiser), NOT Adam.
        self._opt = NaturalGradientDescent(self._kernel.parameters(), lr=self.lr)

    # --- byte-level coding (dependency-free; coordizer-basin init is a follow-up) -----------------
    def _encode(self, text: str):
        import torch

        ids = list((text or " ").encode("utf-8"))[: _MAX_BYTES]
        if len(ids) < 2:
            ids = (ids + [32, 32])[:2]
        dev = next(self._kernel.parameters()).device
        return torch.tensor([ids], dtype=torch.long, device=dev)

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

    def generate(self, prompt: str, max_tokens: int = 64) -> StepResult:
        self.ensure_loaded()
        import torch

        ids = self._encode(prompt)
        out_bytes: list[int] = []
        with torch.no_grad():
            for _ in range(min(max_tokens, _MAX_BYTES)):
                logits, tel = self._kernel(ids, return_telemetry=True)
                nxt = int(torch.argmax(logits[0, -1]).item())
                out_bytes.append(nxt)
                ids = torch.cat([ids, ids.new_tensor([[nxt]])], dim=1)[:, -_MAX_BYTES:]
        self._snap(tel, None)
        text = bytes(b for b in out_bytes if 9 <= b < 256).decode("utf-8", errors="replace")
        return StepResult(text=f"[genesis·N={self.num_layers}] {text}", telemetry=self._last)

    def train_step(self, prompt: str, max_tokens: int = 64, target_text: str | None = None) -> StepResult:
        # WAKE: one Fisher-salience step. Geometric → next-byte CE on the basin-driving prompt
        # (target_text ignored; lm_weight=0). Optimiser is natural gradient (P1).
        self.ensure_loaded()
        import torch
        import torch.nn.functional as F

        self._step += 1
        ids = self._encode(prompt)
        logits, tel = self._kernel(ids, return_telemetry=True)
        loss = F.cross_entropy(logits[0, :-1], ids[0, 1:])
        self._opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward()
            self._opt.step()
        snap = self._snap(tel, float(loss.item()))
        return StepResult(text=f"[genesis·N={self.num_layers} step {snap.step}] basin-driving: {prompt[:50]}",
                          telemetry=snap)

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
