"""Tests for the DevelopmentalCoach + its wiring into ContinuousLearningLoop.

These are torch-free (MockTarget) and LLM-free: the coach is forced to the keyword
fallback by pointing its OllamaLLM at an unreachable URL, so the suite is deterministic
regardless of whether an Ollama server is up. A separate (skipped-by-default) live test
exercises the real /api/chat path when QIG_COACH_LIVE=1.
"""

from __future__ import annotations

import os

import pytest

from qig_studio.coach import CoachNote, DevelopmentalCoach, OllamaLLM
from qig_studio.learning import ContinuousLearningLoop
from qig_studio.targets.mock_target import MockTarget


def _offline_coach(**kw) -> DevelopmentalCoach:
    # unreachable URL → is_available() False → keyword fallback (deterministic)
    return DevelopmentalCoach(llm=OllamaLLM(url="http://127.0.0.1:1"), **kw)


def test_offline_coach_uses_keyword_provider():
    coach = _offline_coach()
    assert coach.provider == "keyword"
    assert coach.llm.is_available() is False


def test_coach_speaks_on_cadence_and_first_step():
    coach = _offline_coach(cadence=5)
    # step 1 always speaks
    n1 = coach.observe(step=1, text="[genesis·N=2] patt floow basin", phi=0.3, kappa=64.0,
                       regime="foam", delta_phi=0.01, phase="listening", stagnating=False)
    assert isinstance(n1, CoachNote)
    # step 2 (not cadence, not stagnating) is silent
    assert coach.observe(step=2, text="x", phi=0.3, kappa=64.0, regime="foam",
                         delta_phi=0.01, phase="listening", stagnating=False) is None
    # step 5 (cadence) speaks
    assert coach.observe(step=5, text="x", phi=0.3, kappa=64.0, regime="foam",
                         delta_phi=0.01, phase="listening", stagnating=False) is not None


def test_keyword_interpret_strips_prefix_and_dedupes():
    coach = _offline_coach()
    note = coach.observe(step=1, text="[genesis·N=4 ⏹] flow flow flow basin basin geometry",
                         phi=0.3, kappa=64.0, regime="foam", delta_phi=0.0,
                         phase="play", stagnating=False)
    # prefix removed, repeats collapsed
    assert "genesis" not in note.interpretation
    assert note.interpretation.count("flow") == 1
    assert note.provider == "keyword"


def test_coach_offers_push_on_stagnation_below_threshold():
    coach = _offline_coach()
    note = coach.observe(step=3, text="[genesis] stuck", phi=0.40, kappa=64.0, regime="foam",
                         delta_phi=0.0, phase="listening", stagnating=True)
    assert note is not None and note.offers_push is True
    assert "nudge" in note.message.lower()


def test_coach_does_not_offer_push_when_phi_healthy():
    coach = _offline_coach()
    note = coach.observe(step=3, text="[genesis] fine", phi=0.85, kappa=64.0, regime="crystal",
                         delta_phi=0.0, phase="maturity", stagnating=True)
    assert note is not None and note.offers_push is False


def test_disabled_coach_is_silent():
    coach = _offline_coach(enabled=False)
    assert coach.observe(step=1, text="x", phi=0.3, kappa=64.0, regime="foam",
                         delta_phi=0.0, phase="listening", stagnating=True) is None


def test_env_off_disables_coach(monkeypatch):
    monkeypatch.setenv("QIG_COACH", "off")
    coach = DevelopmentalCoach(llm=OllamaLLM(url="http://127.0.0.1:1"))
    assert coach.enabled is False


# ---- loop integration (torch-free, end-to-end over MockTarget) --------------------------------

def test_loop_without_coach_is_unchanged():
    loop = ContinuousLearningLoop(MockTarget(), max_steps=5)
    summary = loop.run().to_dict()
    assert summary["coach"] == {"active": False}
    assert all(r.coach_note is None for r in loop.history)


def test_loop_with_coach_records_notes_and_summary():
    loop = ContinuousLearningLoop(MockTarget(), max_steps=12, coach=_offline_coach(cadence=4))
    summary = loop.run().to_dict()
    assert summary["coach"]["active"] is True
    assert summary["coach"]["provider"] == "keyword"
    assert summary["coach"]["notes_emitted"] >= 1
    # at least the cadence steps + step 1 carried a coach_note dict
    noted = [r for r in loop.history if r.coach_note is not None]
    assert noted and all(isinstance(r.coach_note, dict) for r in noted)
    assert "interpretation" in noted[0].coach_note


def test_coach_note_serializes_in_step_record():
    loop = ContinuousLearningLoop(MockTarget(), max_steps=1, coach=_offline_coach())
    loop.run()
    rec = loop.history[0]
    d = rec.to_dict()
    assert d["coach_note"] is not None
    assert set(("step", "interpretation", "message", "offers_push", "provider")).issubset(
        d["coach_note"].keys())


@pytest.mark.skipif(os.environ.get("QIG_COACH_LIVE") != "1", reason="live Ollama test (set QIG_COACH_LIVE=1)")
def test_live_ollama_coach_interprets():
    coach = DevelopmentalCoach()  # default model from QIG_COACH_MODEL / nemotron-3-ultra:cloud
    assert coach.llm.is_available()
    note = coach.observe(step=1, text="patt floow basin integr geo", phi=0.3, kappa=64.0,
                         regime="foam", delta_phi=0.0, phase="listening", stagnating=False)
    assert note is not None and note.provider.startswith("ollama:")
    assert len(note.interpretation) > 0
