"""TargetRegistry — holds the available training targets and the active selection."""

from __future__ import annotations

from .base import TargetInfo, TrainingTarget
from .constellation_target import ConstellationTarget
from .kernel_target import KernelTarget
from .mock_target import MockTarget
from .qwen_local import QwenLocalTarget
from .qwen_modal import QwenModalTarget


class TargetRegistry:
    def __init__(self) -> None:
        self._targets: dict[str, TrainingTarget] = {}
        self._active: str | None = None

    def register(self, target: TrainingTarget) -> None:
        self._targets[target.name] = target

    def names(self) -> list[str]:
        return list(self._targets)

    def get(self, name: str) -> TrainingTarget | None:
        return self._targets.get(name)

    def list_info(self) -> list[TargetInfo]:
        return [t.info() for t in self._targets.values()]

    @property
    def active(self) -> TrainingTarget | None:
        return self._targets.get(self._active) if self._active else None

    def select(self, name: str) -> TrainingTarget:
        if name not in self._targets:
            raise KeyError(name)
        self._active = name
        return self._targets[name]


def default_registry(
    *,
    default_target: str = "mock",
    kernel_checkpoint: str | None = None,
    constellation_checkpoint: str | None = None,
    device: str | None = None,
) -> TargetRegistry:
    """Build the registry: mock (always) + geometric kernel/constellation +
    language qwen-local/qwen-modal (all None-safe on their backends)."""
    r = TargetRegistry()
    r.register(MockTarget())
    r.register(KernelTarget(checkpoint=kernel_checkpoint, device=device))
    r.register(ConstellationTarget(checkpoint=constellation_checkpoint, device=device))
    r.register(QwenLocalTarget())
    r.register(QwenModalTarget())
    chosen = default_target if default_target in r.names() else "mock"
    r.select(chosen)
    return r
