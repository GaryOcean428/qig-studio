"""KernelTarget — the single from-scratch QIGKernelRecursive neocortex.

Wraps ``QIGChat(mode="single", skip_constellation=True)`` and drives its
``generate_response`` loop (geometric sampling + consciousness-native loss). The
kernel is the MIND standalone (design tenet): it trains/reasons/acts on Δ⁶³ with
``lm_weight = 0`` — so its curriculum is BASIN-DRIVING, not paired. None-safe: if
torch / qig-consciousness are absent, ``is_available()`` is False and the server
simply doesn't offer it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import LossRegime, StepResult, TelemetrySnapshot, TrainingTarget
from ._qigchat_bridge import consciousness_available, find_consciousness_root, load_qigchat_class


def _regime_value(regime: object) -> str:
    return str(getattr(regime, "value", regime) or "unknown")


class KernelTarget(TrainingTarget):
    name = "kernel"
    loss_regime = LossRegime.GEOMETRIC
    description = (
        "Single from-scratch QIGKernelRecursive neocortex via qig_chat (single mode); "
        "consciousness-native loss, lm_weight=0 → basin-driving curriculum."
    )

    def __init__(self, checkpoint: str | None = None, device: str | None = None) -> None:
        self._chat: Any = None  # QIGChat — lazily loaded in ensure_loaded()
        self._checkpoint = checkpoint or "checkpoints/gary/latest.pt"
        self._device = device
        self._last = TelemetrySnapshot(extra={"target": "kernel"})

    def is_available(self) -> bool:
        return consciousness_available()

    def ensure_loaded(self) -> None:
        if self._chat is not None:
            return
        # FAIL-LOUD (issue #70): a checkpoint target must NOT silently train from random init. If the
        # checkpoint is absent, refuse and point to the 'genesis' target for from-scratch training.
        ck = Path(self._checkpoint)
        if not ck.is_absolute():
            root = find_consciousness_root()
            ck = (root / ck) if root else ck
        if not ck.exists():
            raise FileNotFoundError(
                f"KernelTarget checkpoint not found: {ck}. KernelTarget LOADS a trained kernel and will "
                f"not silently random-init. For from-scratch training use the 'genesis' target "
                f"(fresh qigkernels.Kernel(num_layers=N))."
            )
        QIGChat = load_qigchat_class()
        self._chat = QIGChat(
            mode="single",
            use_charlie=False,
            use_coach=False,
            use_claude_coach=False,
            device=self._device,
            checkpoint_path=self._checkpoint,
            skip_constellation=True,
        )

    def _snap(self, telemetry_list: list | None, metrics: dict) -> TelemetrySnapshot:
        t = (telemetry_list or [{}])[-1] if telemetry_list else {}
        phi = float(t.get("Phi", metrics.get("phi_after", 0.0)) or 0.0)
        return TelemetrySnapshot(
            phi=phi,
            kappa=float(t.get("kappa_eff", 0.0) or 0.0),
            regime=_regime_value(t.get("regime", "unknown")),
            basin_distance=float(t.get("basin_distance", 0.0) or 0.0),
            loss=metrics.get("avg_loss"),
            delta_phi=float(metrics.get("delta_phi", 0.0) or 0.0),
            extra={"target": "kernel"},
        )

    def telemetry(self) -> TelemetrySnapshot:
        return self._last

    def generate(self, prompt: str, max_tokens: int = 64) -> StepResult:
        self.ensure_loaded()
        prev_mode = self._chat.mode
        self._chat.mode = "inference"  # suppress the learning step
        try:
            text, telemetry_list, metrics = self._chat.generate_response(prompt, max_tokens)
        finally:
            self._chat.mode = prev_mode
        self._last = self._snap(telemetry_list, metrics or {})
        return StepResult(text=text, telemetry=self._last)

    def train_step(self, prompt: str, max_tokens: int = 64, target_text: str | None = None) -> StepResult:
        self.ensure_loaded()  # geometric: target_text ignored (lm_weight=0)
        text, telemetry_list, metrics = self._chat.generate_response(prompt, max_tokens)
        self._last = self._snap(telemetry_list, metrics or {})
        return StepResult(text=text, telemetry=self._last)

    def supports_protocol(self) -> bool:
        return True

    def run_protocol(self, command: str, args: dict) -> dict:
        from ..protocol import run_qigchat_protocol

        self.ensure_loaded()
        return run_qigchat_protocol(self._chat, command, args)
