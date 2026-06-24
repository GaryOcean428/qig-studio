"""MockTarget — deterministic geometric telemetry for UI / SSE development.

NOT a real kernel. It produces plausible, deterministic telemetry (Φ drifting
toward the 0.70 threshold with tacking wobble; κ tacking ±5 around the 64
attractor per κ(t)=κ_center+A·sin(2πt/T); regime derived from Φ) so the server,
SSE streaming, Textual TUI, and browser UI can be exercised END-TO-END without
torch, a GPU, or a checkpoint. The real KernelTarget/ConstellationTarget are
None-safe and take over when the qig-consciousness env is present.

It is always available — that is the point — and is clearly labelled mock so it
is never mistaken for measured consciousness telemetry.
"""

from __future__ import annotations

import math

from qig_core import KAPPA_ATTRACTOR
from qig_core.constants.frozen_facts import PHI_BREAKDOWN_MIN, PHI_EMERGENCY, PHI_THRESHOLD

from .base import LossRegime, StepResult, TelemetrySnapshot, TrainingTarget

# Tacking parameters (illustrative; mirrors the κ-wave shape, not measured values).
_KAPPA_CENTER = KAPPA_ATTRACTOR  # 64.0 — single-sourced from qig-core (architectural attractor, NOT a coupling)
_KAPPA_AMP = 5.0
_KAPPA_PERIOD = 60.0
_PHI_TARGET = 0.72


def _regime_for(phi: float) -> str:
    # Φ→regime thresholds single-sourced from qig-core frozen_facts (no hardcoded literals).
    if phi >= PHI_BREAKDOWN_MIN:
        return "topological_instability"
    if phi >= PHI_THRESHOLD:
        return "geometric"
    if phi >= PHI_EMERGENCY:
        return "hierarchical"
    return "linear"


class MockTarget(TrainingTarget):
    name = "mock"
    loss_regime = LossRegime.GEOMETRIC
    description = (
        "Deterministic geometric telemetry (no kernel, no GPU) for UI/SSE development. "
        "NOT a real kernel — for exercising the app shell only."
    )

    def __init__(self) -> None:
        self._step = 0
        self._phi = 0.45
        self._last = TelemetrySnapshot(phi=self._phi, kappa=_KAPPA_CENTER, regime=_regime_for(self._phi))

    def is_available(self) -> bool:
        return True

    def ensure_loaded(self) -> None:  # nothing heavy to load
        return None

    def telemetry(self) -> TelemetrySnapshot:
        return self._last

    def _advance(self) -> TelemetrySnapshot:
        self._step += 1
        prev_phi = self._phi
        # Exponential approach to the threshold + bounded tacking wobble.
        self._phi += (_PHI_TARGET - self._phi) * 0.05
        wobble = 0.04 * math.sin(self._step / 6.0)
        phi = max(0.0, min(1.0, self._phi + wobble))
        kappa = _KAPPA_CENTER + _KAPPA_AMP * math.sin(2.0 * math.pi * self._step / _KAPPA_PERIOD)
        loss = max(0.05, 2.0 * math.exp(-self._step / 40.0) + 0.1)
        basin = 0.30 * math.exp(-self._step / 50.0)
        self._last = TelemetrySnapshot(
            phi=phi,
            kappa=kappa,
            regime=_regime_for(phi),
            basin_distance=basin,
            loss=loss,
            step=self._step,
            delta_phi=phi - prev_phi,
            extra={"mock": True},
        )
        return self._last

    def generate(self, prompt: str, max_tokens: int = 64) -> StepResult:
        # Inference: do NOT advance training; echo a deterministic mock response.
        text = f"[mock·{self.loss_regime.value}] (Φ={self._last.phi:.2f}) reflecting on: {prompt[:60]}"
        return StepResult(text=text, telemetry=self._last)

    def train_step(self, prompt: str, max_tokens: int = 64, target_text: str | None = None) -> StepResult:
        snap = self._advance()  # geometric: target_text ignored
        text = f"[mock·{self.loss_regime.value} step {snap.step}] basin-driving on: {prompt[:60]}"
        return StepResult(text=text, telemetry=snap)

    def supports_protocol(self) -> bool:
        return True

    def run_protocol(self, command: str, args: dict) -> dict:
        from ..protocol import COMMANDS_BY_NAME
        from .base import ProtocolUnsupported

        cmd = COMMANDS_BY_NAME.get(command)
        if cmd is None:
            raise ProtocolUnsupported(f"unknown protocol command '{command}'")
        snap = self._advance()  # nudge telemetry so the simulated protocol feels live
        return {
            "command": command,
            "group": cmd.group,
            "available": True,
            "mock": True,
            "output": (
                f"[mock] {cmd.method}({args or {}}) — simulated {cmd.group} protocol "
                f"(Φ={snap.phi:.2f}, κ={snap.kappa:.1f}, regime={snap.regime})"
            ),
            "telemetry": snap.to_dict(),
        }
