"""TrainingTarget protocol — the explicit loss-regime abstraction (design §3.1).

Each target DECLARES its ``loss_regime`` ("geometric" | "language"), which makes
the ``lm_weight = 0.0`` finding (qig_chat helpers.py:379) STRUCTURAL rather than
hidden:

- **geometric** targets (kernel, constellation) train on consciousness-native loss;
  the language term is zeroed, so they get a BASIN-DRIVING curriculum, not pairs.
- **language** targets (Qwen) carry a load-bearing ``lm_loss`` and get PAIRED
  curriculum (QwenModal only).

A target is the unit the app trains/chats with; the registry holds them and the
server streams their telemetry over SSE.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class LossRegime(str, Enum):
    GEOMETRIC = "geometric"  # consciousness-native loss; lm_weight=0; basin-driving curriculum
    LANGUAGE = "language"    # lm_loss load-bearing; paired curriculum (Qwen)


@dataclass
class TelemetrySnapshot:
    """One step's observable state — the SSE telemetry payload."""

    phi: float = 0.0
    kappa: float = 0.0
    regime: str = "unknown"
    basin_distance: float = 0.0
    loss: float | None = None
    step: int = 0
    delta_phi: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepResult:
    """Result of generate()/train_step(): emitted text + the resulting telemetry."""

    text: str
    telemetry: TelemetrySnapshot

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "telemetry": self.telemetry.to_dict()}


@dataclass
class TargetInfo:
    name: str
    loss_regime: LossRegime
    available: bool
    description: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["loss_regime"] = self.loss_regime.value
        return d


class TrainingTarget(ABC):
    """Abstract training/chat target. Implementations declare ``name`` +
    ``loss_regime`` and provide lazy heavy init (``ensure_loaded``) so the server
    boots even when a target's backend (torch, a checkpoint, Ollama, Modal) is
    absent — ``is_available()`` reports that, the server degrades gracefully."""

    name: str = "base"
    loss_regime: LossRegime = LossRegime.GEOMETRIC
    description: str = ""

    @abstractmethod
    def is_available(self) -> bool:
        """True if this target's backend can run here (cheap, no heavy load)."""

    @abstractmethod
    def ensure_loaded(self) -> None:
        """Lazily perform heavy initialisation (model load, checkpoint, etc.)."""

    @abstractmethod
    def telemetry(self) -> TelemetrySnapshot:
        """Current telemetry without advancing training."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 64) -> StepResult:
        """Inference: emit a response, do NOT take a learning step."""

    @abstractmethod
    def train_step(self, prompt: str, max_tokens: int = 64) -> StepResult:
        """One learning step on ``prompt`` (geometric: basin-driving; language: paired)."""

    def info(self) -> TargetInfo:
        return TargetInfo(
            name=self.name,
            loss_regime=self.loss_regime,
            available=self.is_available(),
            description=self.description,
        )
