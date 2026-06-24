"""Training targets — the loss-regime-declaring units the app trains/chats with."""

from __future__ import annotations

from .base import LossRegime, StepResult, TargetInfo, TelemetrySnapshot, TrainingTarget
from .constellation_target import ConstellationTarget
from .kernel_target import KernelTarget
from .mock_target import MockTarget
from .registry import TargetRegistry, default_registry

__all__ = [
    "LossRegime",
    "TelemetrySnapshot",
    "StepResult",
    "TargetInfo",
    "TrainingTarget",
    "MockTarget",
    "KernelTarget",
    "ConstellationTarget",
    "TargetRegistry",
    "default_registry",
]
