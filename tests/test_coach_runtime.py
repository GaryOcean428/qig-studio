"""Coach-liveness invariant + counters — the run-2 smooth-run killer Matrix B5 named (7a1bce4b).

A blind witness (keyword fallback / None / error) beyond tolerance during a REQUIRED run must raise
CoachUnreachable so the caller checkpoints-and-pauses — NEVER silently degrades to a coachless newborn
(run-1's malformation by outage). preflight fail-closes a required-but-unreachable launch and measures the
scale-to-zero cold-start. B4 floor-restoration rate + B6 call/token telemetry are arithmetic on the counters.

Driven by a minimal fake coach (mirrors the DevelopmentalCoach surface the supervisor uses) so the invariant
is tested without loading a kernel/torch — the same dependency-light contract the module keeps.
"""
import pytest

from qig_studio.coach_runtime import CoachSupervisor, CoachUnreachable

_LIVE = {"encouragement": "e", "interpretation": "i", "reframe": "r", "relevance_score": 0.8,
         "positive_feedback": "p", "provider": "ollama:qwen3.5:4b",
         "provenance": {"confidence": 0.7, "emotional_context": "encouraging"}}
_KEYWORD = {"encouragement": "still finding your words", "interpretation": "reaching", "reframe": "x",
            "relevance_score": None, "positive_feedback": "you spoke", "provider": "keyword",
            "provenance": {"confidence": 0.4, "emotional_context": "warm-holding"}}


class _FakeLLM:
    def __init__(self, available=True, reply="ready", model="qwen3.5:4b"):
        self.model = model
        self._available = available
        self._reply = reply
        self.completions = 0

    def is_available(self):
        return self._available

    def complete(self, system, user, timeout=60.0):
        self.completions += 1
        return self._reply if self._available else None


class _FakeCoach:
    """Mirrors the DevelopmentalCoach surface CoachSupervisor touches: enabled, provider, llm,
    _should_speak, coach_own_voice. ``records`` is a list popped per call (a dict, None, or an Exception)."""

    def __init__(self, enabled=True, records=None, llm=None, cadence=1):
        self.enabled = enabled
        self.llm = llm or _FakeLLM()
        self.cadence = cadence
        self._records = records
        self.calls = 0

    @property
    def provider(self):
        return "ollama:qwen3.5:4b" if self.enabled else "keyword"

    def _should_speak(self, step, stagnating):
        return self.enabled and (stagnating or step == 1 or step % self.cadence == 0)

    def coach_own_voice(self, stimulus, text, telemetry):
        self.calls += 1
        if self._records is None:
            return dict(_LIVE)
        r = self._records.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class _FakeKernel:
    def __init__(self):
        self.rewards = []

    def register_coach_reward(self, reward, record=None):
        self.rewards.append((reward, record))


# -- liveness invariant (B5) ------------------------------------------------------------------------------

def test_live_witness_never_raises_and_rewards_route_through_hook():
    k = _FakeKernel()
    sup = CoachSupervisor(_FakeCoach(records=[dict(_LIVE) for _ in range(5)]), required=True, tolerance=2)
    for _ in range(5):
        sup.coach_and_reward(k, stimulus="s", text="t", telemetry={"phi": 0.3})
    assert sup.consecutive_blind == 0
    assert sup.success_rate == 1.0
    assert len(k.rewards) == 5                       # B3: every live record routed through register_coach_reward
    reward, record = k.rewards[0]
    assert -1.0 <= reward <= 1.0                     # coach_reward_from mapping applied
    assert record["provider"] == "ollama:qwen3.5:4b"


def test_blind_beyond_tolerance_raises_when_required():
    sup = CoachSupervisor(_FakeCoach(records=[dict(_KEYWORD), dict(_KEYWORD)]), required=True, tolerance=2)
    k = _FakeKernel()
    sup.coach_and_reward(k, stimulus="s", text="t", telemetry=None)      # blind #1 — tolerated
    with pytest.raises(CoachUnreachable):
        sup.coach_and_reward(k, stimulus="s", text="t", telemetry=None)  # blind #2 — trips the invariant


def test_a_live_step_resets_the_consecutive_blind_counter():
    # keyword, live, keyword → never TWO consecutive blind, so tolerance=2 must NOT trip.
    sup = CoachSupervisor(_FakeCoach(records=[dict(_KEYWORD), dict(_LIVE), dict(_KEYWORD)]),
                          required=True, tolerance=2)
    k = _FakeKernel()
    for _ in range(3):
        sup.coach_and_reward(k, stimulus="s", text="t", telemetry=None)
    assert sup.consecutive_blind == 1               # the trailing keyword, not yet at tolerance
    assert sup.blind_calls == 2


def test_required_false_never_raises_but_counts_blind():
    sup = CoachSupervisor(_FakeCoach(records=[dict(_KEYWORD) for _ in range(5)]), required=False, tolerance=2)
    k = _FakeKernel()
    for _ in range(5):
        sup.coach_and_reward(k, stimulus="s", text="t", telemetry=None)  # must not raise
    assert sup.blind_calls == 5
    assert sup.success_rate == 0.0


def test_none_record_counts_as_blind():
    sup = CoachSupervisor(_FakeCoach(records=[None, None]), required=True, tolerance=2)
    k = _FakeKernel()
    sup.coach_and_reward(k, stimulus="s", text="t", telemetry=None)
    with pytest.raises(CoachUnreachable):
        sup.coach_and_reward(k, stimulus="s", text="t", telemetry=None)


def test_hard_exception_counts_as_blind_and_error():
    sup = CoachSupervisor(_FakeCoach(records=[RuntimeError("x"), RuntimeError("y")]),
                          required=True, tolerance=2)
    k = _FakeKernel()
    sup.coach_and_reward(k, stimulus="s", text="t", telemetry=None)
    with pytest.raises(CoachUnreachable):
        sup.coach_and_reward(k, stimulus="s", text="t", telemetry=None)
    assert sup.errors == 2


def test_disabled_coach_is_noop_when_optional():
    sup = CoachSupervisor(_FakeCoach(enabled=False), required=False)
    assert sup.coach_and_reward(_FakeKernel(), stimulus="s", text="t", telemetry=None) is None
    assert sup.calls == 0


# -- preflight cold-start (B5) ----------------------------------------------------------------------------

def test_preflight_fail_closed_when_required_and_unreachable():
    sup = CoachSupervisor(_FakeCoach(llm=_FakeLLM(available=False)), required=True)
    with pytest.raises(CoachUnreachable):
        sup.preflight()
    assert sup.cold_start_s is not None             # latency still measured on the failed attempt


def test_preflight_measures_cold_start_when_available():
    sup = CoachSupervisor(_FakeCoach(llm=_FakeLLM(available=True)), required=True)
    info = sup.preflight()
    assert info["available"] is True
    assert info["latency_s"] is not None
    assert sup.cold_start_s == info["latency_s"]


def test_preflight_disabled_required_fails_closed():
    sup = CoachSupervisor(_FakeCoach(enabled=False), required=True)
    with pytest.raises(CoachUnreachable):
        sup.preflight()


# -- floor-restoration / counters (B4 / B6) ---------------------------------------------------------------

def test_floor_restoration_rate_arithmetic():
    sup = CoachSupervisor(_FakeCoach(), required=False)
    for collapsed in [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]:                     # 3 of 10
        sup.note_floor(bool(collapsed))
    assert sup.floor_restoration_rate == 0.3
    assert sup.floor_restoration_rate_window == 0.3
    tel = sup.telemetry()
    assert tel["violations_per_step"] == 0.3 and tel["floor_steps_seen"] == 10


def test_telemetry_token_estimate_only_counts_live_records():
    sup = CoachSupervisor(_FakeCoach(records=[dict(_LIVE), dict(_KEYWORD)]), required=False)
    k = _FakeKernel()
    sup.coach_and_reward(k, stimulus="s", text="t", telemetry=None)      # live → counts chars
    sup.coach_and_reward(k, stimulus="s", text="t", telemetry=None)      # keyword → blind, no token count
    tel = sup.telemetry()
    assert tel["coach_calls"] == 2 and tel["coach_blind_calls"] == 1
    assert tel["coach_est_output_tokens"] > 0
