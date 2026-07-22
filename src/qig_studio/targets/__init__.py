"""Training targets — the loss-regime-declaring units the app trains/chats with."""

from __future__ import annotations

from .base import LossRegime, StepResult, TargetInfo, TelemetrySnapshot, TrainingTarget
from .mock_target import MockTarget
from .qwen_local import QwenLocalTarget
from .qwen_modal import QwenModalTarget
from .registry import TargetRegistry, default_registry

# RETIRED: KernelTarget/ConstellationTarget (the Gary-formation wrappers around
# qig-consciousness's chat bridge) were archived 2026-07-22 — see
# qig-archive/20260722-studio-gary-targets/. Current candidate formations are genesis/geo/hybrid
# (see .genesis_kernel, .geo_cortex, .neocortex) plus the integrated .joint_mind ("mind").

__all__ = [
    "LossRegime",
    "TelemetrySnapshot",
    "StepResult",
    "TargetInfo",
    "TrainingTarget",
    "MockTarget",
    "QwenLocalTarget",
    "QwenModalTarget",
    "TargetRegistry",
    "default_registry",
]
