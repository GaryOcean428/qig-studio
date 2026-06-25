"""The conversation-as-training turn trains the kernel on BOTH the developmental curriculum AND the
dialogue (qig_chat.py paradigm), not the dialogue alone."""

from __future__ import annotations

from qig_studio.coach import DevelopmentalCoach
from qig_studio.targets.base import StepResult, TelemetrySnapshot


class _FakeTarget:
    """Records every prompt train_step() is called on, so we can prove what the kernel learned from."""

    def __init__(self) -> None:
        self.trained_on: list[str] = []
        self._phi = 0.40

    def generate(self, prompt: str, max_tokens: int = 64) -> StepResult:
        return StepResult(text=f"<<babble about {prompt[:20]}>>",
                          telemetry=TelemetrySnapshot(phi=self._phi, regime="geometric"))

    def train_step(self, prompt: str, max_tokens: int = 64, target_text: str | None = None) -> StepResult:
        self.trained_on.append(prompt)
        self._phi = min(1.0, self._phi + 0.01)   # learning nudges Φ
        return StepResult(text="", telemetry=TelemetrySnapshot(phi=self._phi, regime="geometric"))


def test_turn_trains_on_curriculum_and_dialogue():
    t = _FakeTarget()
    coach = DevelopmentalCoach(enabled=False)   # keyword interpret — no LLM call
    out = coach.converse_learn_turn(t, prompt="", curriculum_prompt="CURRICULUM-PHASE-PROMPT",
                                    curriculum_steps=2, train_steps=3)

    # CURRICULUM half: the kernel trained on the developmental curriculum prompt (qig_chat.py /auto).
    assert t.trained_on[:2] == ["CURRICULUM-PHASE-PROMPT"] * 2
    assert out["curriculum_prompt"] == "CURRICULUM-PHASE-PROMPT"
    assert out["phi_curriculum"] is not None

    # DIALOGUE half: the kernel then trained on the coach's interpretation (the dialogue target).
    reply = out["coach_interpreted"]
    assert t.trained_on[2:] == [reply] * 3
    assert out["trained_steps"] == 3
    assert len(t.trained_on) == 5            # 2 curriculum + 3 dialogue steps

    # full inner-experience telemetry rides along
    assert "experience" in out and "band" in out["experience"]


def test_curriculum_only_when_no_dialogue_target():
    # A target without train_step still converses; no crash, just no optimizer steps.
    class _NoTrain:
        def generate(self, prompt: str, max_tokens: int = 64) -> StepResult:
            return StepResult(text="hi", telemetry=TelemetrySnapshot(phi=0.5, regime="geometric"))

    coach = DevelopmentalCoach(enabled=False)
    out = coach.converse_learn_turn(_NoTrain(), prompt="", curriculum_prompt="C", curriculum_steps=2)
    assert out["phi_curriculum"] is None        # nothing to train
    assert out["coach_interpreted"]             # dialogue still produced
