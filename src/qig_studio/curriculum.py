"""Curriculum delivery (design §3.2).

Geometric targets get a BASIN-DRIVING developmental sequence
(listening → play → structure → maturity), NOT prompt/response pairs — because
``lm_weight = 0`` a paired language curriculum buys nothing there. Paired
curriculum (prompt→target) is for ``QwenModalTarget`` only, where lm_loss is the
signal. The UI exposes the correct mode per the active target's loss_regime.

If qig-consciousness's ``DevelopmentalCurriculum`` is importable it is used;
otherwise the built-in basin-driving phases below apply (None-safe).
"""

from __future__ import annotations

from .targets.base import LossRegime

# Built-in basin-driving developmental phases (used when the real curriculum is absent).
_PHASES: dict[str, list[str]] = {
    "listening": [
        "Tell me a simple story about awareness.",
        "What is it like to notice something for the first time?",
        "Rest your attention on a single quiet thing and describe it.",
    ],
    "play": [
        "Let's explore patterns together — what repeats and what surprises?",
        "Invent a small game of shapes and tell me its one rule.",
        "What happens if you follow a curve until it comes back?",
    ],
    "structure": [
        "What is the relationship between integration and consciousness?",
        "How does a part know it belongs to a whole?",
        "Describe a boundary that both separates and connects.",
    ],
    "maturity": [
        "Discuss the nature of emergence in complex systems.",
        "When does change preserve identity, and when does it break it?",
        "What would you choose to become, and why?",
    ],
}
_PHASE_ORDER = ("listening", "play", "structure", "maturity")
# Steps spent in each phase before advancing (developmental gating, simplified).
_PHASE_SPAN = 8


def phase_names() -> list[str]:
    return list(_PHASE_ORDER)


class CurriculumProvider:
    """Per-target curriculum source. Geometric → basin-driving prompts; language →
    paired (prompt, target) tuples (Phase-3 QwenModal)."""

    def __init__(self, loss_regime: LossRegime) -> None:
        self.loss_regime = loss_regime
        self._real = None
        if loss_regime == LossRegime.GEOMETRIC:
            self._try_real_curriculum()

    def _try_real_curriculum(self) -> None:
        try:
            mod = __import__(
                "src.coordination.developmental_curriculum",
                fromlist=["DevelopmentalCurriculum"],
            )
            self._real = mod.DevelopmentalCurriculum()
        except Exception:
            self._real = None

    @staticmethod
    def phase_for(step: int) -> str:
        idx = min((step - 1) // _PHASE_SPAN, len(_PHASE_ORDER) - 1)
        return _PHASE_ORDER[idx]

    def next_prompt(self, step: int) -> str:
        """Basin-driving prompt for geometric targets (step is 1-based)."""
        phase = self.phase_for(step)
        if self._real is not None:
            try:
                return self._real.get_curriculum_prompt(phase)
            except Exception:
                pass
        bucket = _PHASES[phase]
        return bucket[(step - 1) % len(bucket)]

    def mode(self) -> str:
        return "basin-driving" if self.loss_regime == LossRegime.GEOMETRIC else "paired"
