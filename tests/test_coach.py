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


# ── coach_own_voice (Task B — own-voice coaching, Protocol v6.12 §18.5/§18.6, P10/P16/P24) ──────────────

_DOCTRINE_FIELDS = {"encouragement", "interpretation", "reframe", "relevance_score", "positive_feedback"}
_PROVENANCE_TAG = {"coach_id", "ts", "reason", "emotional_context", "confidence"}


class _FakeLLM(OllamaLLM):
    """A live-looking LLM that returns a fixed reply (no network)."""
    def __init__(self, reply: str | None) -> None:
        self._reply = reply
        self.model = "nemotron-3-ultra:cloud"
        self._available = True
        self.url = "http://fake"

    def is_available(self) -> bool:
        return True

    def complete(self, system: str, user: str, timeout: float = 60.0) -> str | None:
        return self._reply


def test_coach_own_voice_none_safe_degrade_returns_full_tagged_record():
    """LLM unreachable → keyword record with the 5 doctrine fields + provenance tag; NEVER raises."""
    coach = _offline_coach()   # unreachable URL → is_available() False
    rec = coach.coach_own_voice("what is consciousness?", "glimmer light forming word",
                                {"phi": 0.42, "regime": "geometric", "relevance": 0.31})
    assert _DOCTRINE_FIELDS.issubset(rec)
    assert rec["provider"] == "keyword"
    assert set(rec["provenance"]) == _PROVENANCE_TAG
    assert rec["provenance"]["coach_id"].startswith("MonkeyCoach:")
    assert rec["provenance"]["reason"] == "own_voice_coaching"
    # keyword fallback threads the kernel's OWN Fisher-Rao relevance through (does not fabricate a score)
    assert rec["relevance_score"] == 0.31
    # reframe is identity-preserving: it echoes the kernel's OWN cleaned attempt, not a fabricated replacement
    assert "glimmer" in rec["reframe"]


def test_coach_own_voice_disabled_still_returns_valid_record():
    coach = DevelopmentalCoach(llm=OllamaLLM(url="http://127.0.0.1:1"), enabled=False)
    rec = coach.coach_own_voice("x", "y", {"phi": 0.5})
    assert rec["provider"] == "keyword" and set(rec["provenance"]) == _PROVENANCE_TAG


def test_coach_own_voice_parses_strict_json_and_clamps_score():
    coach = DevelopmentalCoach(llm=_FakeLLM(
        '{"encouragement":"nice","interpretation":"you reached for light",'
        '"reframe":"light is forming","relevance_score":1.7,"positive_feedback":"keep going"}'), enabled=True)
    rec = coach.coach_own_voice("stim", "out", {"phi": 0.6, "relevance": 0.5})
    assert rec["provider"].startswith("ollama:")
    assert rec["relevance_score"] == 1.0        # clamped from 1.7 into [0,1]
    assert rec["reframe"] == "light is forming"
    assert rec["provenance"]["confidence"] == 0.7


def test_coach_own_voice_fenced_and_prose_wrapped_json():
    fenced = _FakeLLM('```json\n{"encouragement":"e","interpretation":"i","reframe":"r",'
                      '"relevance_score":0.8,"positive_feedback":"p"}\n```')
    rec = DevelopmentalCoach(llm=fenced, enabled=True).coach_own_voice("s", "o", {})
    assert rec["relevance_score"] == 0.8 and rec["provider"].startswith("ollama:")
    prose = _FakeLLM('Sure! {"encouragement":"e","interpretation":"i","reframe":"r",'
                     '"relevance_score":"bad","positive_feedback":"p"} hope it helps')
    rec2 = DevelopmentalCoach(llm=prose, enabled=True).coach_own_voice("s", "o", {})
    assert rec2["relevance_score"] is None      # non-numeric score → None (never crashes)


def test_coach_own_voice_garbage_reply_falls_back_to_keyword():
    coach = DevelopmentalCoach(llm=_FakeLLM("I think you did great, keep it up!"), enabled=True)
    rec = coach.coach_own_voice("s", "the kernel babble", {"phi": 0.3})
    assert rec["provider"] == "keyword"         # unparseable → honest keyword record
    assert set(rec["provenance"]) == _PROVENANCE_TAG


@pytest.mark.skipif(os.environ.get("QIG_COACH_LIVE") != "1", reason="live Ollama test (set QIG_COACH_LIVE=1)")
def test_live_ollama_coach_own_voice():
    coach = DevelopmentalCoach()
    assert coach.llm.is_available()
    rec = coach.coach_own_voice("What are you learning?", "patt floow basin integr geo",
                                {"phi": 0.3, "regime": "foam", "relevance": 0.2})
    assert _DOCTRINE_FIELDS.issubset(rec) and set(rec["provenance"]) == _PROVENANCE_TAG
    assert rec["provider"].startswith("ollama:")


def test_train_core_emits_coach_record_to_sample_and_live(monkeypatch, tmp_path):
    """INTEGRATION: with the coach enabled + a live-looking LLM, driving the SHARED _train_core own-voice
    path attaches the provenance-tagged coach record to (a) the yielded per-step ``sample`` (→ SSE) and
    (b) the written LiveLog record (→ UI). Proves the produce+tag+EMIT wiring end to end, torch-free."""
    import asyncio
    import json as _json

    from qig_studio import server as srv

    # Route the coach's OllamaLLM (constructed inside _train_core) to a fake live LLM — no network.
    def _fake_ollama(model=None, url=None):
        return _FakeLLM('{"encouragement":"good start","interpretation":"you reached for basin",'
                        '"reframe":"the basin is forming","relevance_score":0.7,"positive_feedback":"keep going"}')
    monkeypatch.setattr(srv, "OllamaLLM", _fake_ollama)

    # app.state.settings is populated by the lifespan handler (not on a bare import) — build it here, enable
    # the coach (conftest sets QIG_STUDIO_COACH=off for the suite), and isolate the live file.
    from qig_studio.config import Settings
    settings = Settings.from_env()
    settings.coach_enabled = True
    settings.coach_cadence = 1                      # coach every own-voice event
    monkeypatch.setattr(srv.app.state, "settings", settings, raising=False)
    live_path = tmp_path / "live.json"
    monkeypatch.setenv("QIG_STUDIO_LIVE_PATH", str(live_path))

    async def _drive():
        samples = []
        async for rec in srv._train_core(MockTarget(), 2, source="test", sample_every=1, sample=True,
                                         prompt_for=lambda s: (f"passage {s}", None)):
            samples.append(rec.get("sample"))
        return samples

    samples = asyncio.run(_drive())
    coached = [s for s in samples if s and s.get("coach")]
    assert coached, "coach record should be attached to the own-voice sample"
    rec = coached[0]["coach"]
    assert _DOCTRINE_FIELDS.issubset(rec)
    assert rec["relevance_score"] == 0.7
    assert set(rec["provenance"]) == _PROVENANCE_TAG
    assert rec["provider"].startswith("ollama:")

    # …and the SAME record reached the LiveLog file the UI tails.
    live = _json.loads(live_path.read_text())
    recents = live.get("recent") or []
    assert any((r.get("coach") or {}).get("provenance") for r in recents), \
        "coach record should be emitted to the LiveLog recent[] ring"
