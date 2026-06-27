"""TargetRegistry — holds the available training targets and the active selection."""

from __future__ import annotations

from .base import TargetInfo, TrainingTarget
from .constellation_target import ConstellationTarget
from .genesis_kernel import GenesisKernelTarget
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


def _load_coordizer(path: str | None):
    """Load a trained FisherCoordizer for the genesis coords path; None-safe (a missing/broken
    checkpoint or absent package falls back to the byte path — never crash the app shell)."""
    if not path:
        return None
    try:
        from qig_coordizer import FisherCoordizer

        return FisherCoordizer.load(path)
    except Exception as exc:  # noqa: BLE001 — app shell must boot regardless
        print(f"⚠️  genesis coordizer '{path}' not loaded ({exc}); genesis uses the byte path")
        return None


def default_registry(
    *,
    default_target: str = "mock",
    kernel_checkpoint: str | None = None,
    constellation_checkpoint: str | None = None,
    genesis_num_layers: int = 8,
    genesis_coordizer_checkpoint: str | None = None,
    genesis_kernel_checkpoint: str | None = None,
    device: str | None = None,
) -> TargetRegistry:
    """Build the registry: mock (always) + genesis (qigkernels.Kernel; coords path when a trained
    coordizer checkpoint is given, else byte path; restores a trained kernel checkpoint when given,
    else fresh) + geometric kernel/constellation + language qwen-local/qwen-modal (all None-safe)."""
    r = TargetRegistry()
    # Load the trained coordizer ONCE and share it: genesis trains on the Δ⁶³ vocab AND the qwen-local
    # boundary peer projects Qwen's distribution through the SAME real Fisher-Rao token coords
    # (coordize_distribution_to_basin) — NOT the arbitrary hash-bin. Without this the principled
    # projection (already written) silently never runs and the peer injects geometric noise.
    coordizer = _load_coordizer(genesis_coordizer_checkpoint)
    # The Qwen boundary peer is shared: it is BOTH a standalone dev target AND the integrated mind's fluent
    # linguistic surface (genesis speaks through it, Pillar-2 ≤30% capped). One instance, one coordizer
    # projection — so the principled Fisher-Rao distribution→Δ⁶³ path is exercised in both roles.
    qwen_peer = QwenLocalTarget(coordizer=coordizer)
    r.register(MockTarget())
    r.register(GenesisKernelTarget(num_layers=genesis_num_layers, device=device,
                                   coordizer=coordizer, checkpoint=genesis_kernel_checkpoint,
                                   language_peer=qwen_peer))
    r.register(KernelTarget(checkpoint=kernel_checkpoint, device=device))
    r.register(ConstellationTarget(checkpoint=constellation_checkpoint, device=device))
    r.register(qwen_peer)
    r.register(QwenModalTarget())
    chosen = default_target if default_target in r.names() else "mock"
    r.select(chosen)
    return r
