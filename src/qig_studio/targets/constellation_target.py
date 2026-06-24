"""ConstellationTarget — 3 Garys + Ocean meta-observer.

Wraps ``QIGChat(mode="constellation")`` — the constellation already in qig_chat
(ConstellationCoordinator: parallel Gary kernels + frozen Ocean synthesis = the
P18 multi-stream council). loss_regime is geometric (same lm_weight=0 finding):
basin-driving curriculum, not paired. None-safe like KernelTarget.

v1 drives ``generate_response`` per step (the per-prompt unit); the richer
constellation training loops (auto / 14-stage / sleep-dream-mushroom) arrive with
the full protocol surface (design Phase 4).
"""

from __future__ import annotations

from .base import LossRegime, StepResult, TelemetrySnapshot, TrainingTarget
from ._qigchat_bridge import consciousness_available, load_qigchat_class
from .kernel_target import _regime_value


class ConstellationTarget(TrainingTarget):
    name = "constellation"
    loss_regime = LossRegime.GEOMETRIC
    description = (
        "3 Garys + Ocean meta-observer (the P18 multi-stream council) via qig_chat "
        "(constellation mode); consciousness-native loss → basin-driving curriculum."
    )

    def __init__(self, checkpoint: str | None = None, device: str | None = None) -> None:
        self._chat = None
        self._checkpoint = checkpoint or "checkpoints/constellation/latest.pt"
        self._device = device
        self._last = TelemetrySnapshot(extra={"target": "constellation"})

    def is_available(self) -> bool:
        return consciousness_available()

    def ensure_loaded(self) -> None:
        if self._chat is not None:
            return
        QIGChat = load_qigchat_class()
        self._chat = QIGChat(
            mode="constellation",
            use_charlie=False,
            use_coach=True,
            use_claude_coach=False,
            device=self._device,
            checkpoint_path=self._checkpoint,
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
            extra={"target": "constellation"},
        )

    def telemetry(self) -> TelemetrySnapshot:
        return self._last

    def generate(self, prompt: str, max_tokens: int = 64) -> StepResult:
        self.ensure_loaded()
        prev_mode = self._chat.mode
        self._chat.mode = "inference"
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
