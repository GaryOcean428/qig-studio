"""M1 — coach loop is wired into LEARNING on the default `mind` target.

The nemotron coach fires + renders SSE (test_coach.py proves produce+tag+emit), but the council
found it LEARNS NOTHING on the resumed `mind` run for two reasons this suite pins:

  (a) ``JointMindTarget`` had no ``register_coach_reward`` → ``server.py`` ``getattr(target,
      "register_coach_reward", None)`` silently skipped the replay-priority actuator every step.
      The real actuator lives on the central kernel (``GenesisKernelTarget.register_coach_reward``
      → ``_weighted_replay_choice``). Fix: the mind target DELEGATES to its central kernel.

  (b) Ocean's outcome-scoring (``OceanAutonomic._coach_reward``) + the kernel's own neurochem read
      ``tel.extra["coach"]`` from the LIVE snapshot, but the coach record was only ever written into
      a ``dataclasses.asdict`` DEEP COPY (``TelemetrySnapshot.to_dict()`` + the ``_write_live`` ``td``
      throwaway) → ``coach_bonus ≡ 0`` forever. Fix: land the coach record on the central kernel's
      LIVE snapshot (the same object ``telemetry()`` returns).

These are torch-free: ``GenesisKernelTarget`` / ``JointMindTarget`` construct WITHOUT ``ensure_loaded``
(no kernel build, no GPU), and ``register_coach_reward`` never touches torch.
"""

from __future__ import annotations

from types import SimpleNamespace

from qig_studio.constellation.ocean import OceanAutonomic
from qig_studio.kernel_experience import coach_reward_from
from qig_studio.targets.genesis_kernel import GenesisKernelTarget
from qig_studio.targets.joint_mind import JointMindTarget


def _coach_record(relevance: float = 0.9) -> dict:
    """A realistic Task-B coach record (the coach_own_voice output shape)."""
    return {
        "encouragement": "good start",
        "interpretation": "you reached for basin",
        "reframe": "the basin is forming",
        "relevance_score": relevance,
        "positive_feedback": "keep going",
        "provider": "ollama:nemotron-3-ultra:cloud",
        "provenance": {
            "coach_id": "MonkeyCoach:test",
            "ts": "2026-07-02T00:00:00Z",
            "reason": "own_voice_coaching",
            "emotional_context": "encouraging",
            "confidence": 0.7,
        },
    }


# ── (a) the mind target's register_coach_reward reaches the CENTRAL kernel's replay-priority ──────────

def test_joint_mind_exposes_register_coach_reward():
    """server.py's getattr must SUCCEED on the mind target (it silently skipped before)."""
    t = JointMindTarget()
    assert hasattr(t, "register_coach_reward")
    assert callable(t.register_coach_reward)


def test_joint_mind_register_coach_reward_reaches_central_pending():
    """Calling the mind target's actuator arms the CENTRAL kernel's pending coach reward."""
    central = GenesisKernelTarget()                 # CPU, no ensure_loaded → torch-free
    t = JointMindTarget()
    t._mind = SimpleNamespace(central=central)      # inject a real central without building the constellation
    assert central._pending_coach_reward == 0.0
    t.register_coach_reward(0.5)
    assert central._pending_coach_reward == 0.5     # reached the replay-priority actuator


def test_joint_mind_register_coach_reward_boosts_latest_experience_weight():
    """A coach reaction between steps folds into the most-recent experience's replay priority."""
    central = GenesisKernelTarget()
    central._experience_weight = [1.0]              # one logged experience already exists
    t = JointMindTarget()
    t._mind = SimpleNamespace(central=central)
    t.register_coach_reward(0.4)
    assert central._experience_weight[-1] == 1.4    # 1.0 + 0.4 (P10 reward-weighted DATA selection)


def test_joint_mind_register_coach_reward_none_safe_without_mind():
    """None-safe: no constellation built yet → no-op, never raises (the getattr fires pre-load)."""
    t = JointMindTarget()
    assert t._mind is None
    t.register_coach_reward(0.9)                    # must not raise


def test_joint_mind_register_coach_reward_none_safe_when_central_lacks_method():
    """None-safe: a central without the actuator (mock/other arm) → no-op, never raises."""
    t = JointMindTarget()
    t._mind = SimpleNamespace(central=object())     # no register_coach_reward
    t.register_coach_reward(0.9)                    # must not raise


# ── (b) the coach record lands on the LIVE snapshot → Ocean + neurochem read a REAL value ─────────────

def test_ocean_coach_reward_zero_without_record():
    """Baseline: a live snapshot with no coach record reads 0.0 (the collapse the fix reverses)."""
    central = GenesisKernelTarget()
    central.register_coach_reward(0.5)              # reward only, no record
    assert OceanAutonomic._coach_reward(central.telemetry()) == 0.0


def test_coach_record_lands_on_central_live_snapshot():
    """The record is written to the LIVE snapshot object telemetry() returns — not a deep copy."""
    central = GenesisKernelTarget()
    rec = _coach_record()
    central.register_coach_reward(coach_reward_from(rec), record=rec)
    live = central.telemetry()
    assert live.extra.get("coach") == rec           # on the live object, not a throwaway to_dict() copy


def test_ocean_coach_reward_nonzero_after_record_on_central():
    """Ocean's outcome-scoring reads a NON-zero coach reward from the live snapshot (was ≡ 0 forever)."""
    central = GenesisKernelTarget()
    rec = _coach_record(relevance=0.9)
    expected = coach_reward_from(rec)
    assert expected != 0.0
    central.register_coach_reward(expected, record=rec)
    got = OceanAutonomic._coach_reward(central.telemetry())
    assert got != 0.0
    assert got == expected


def test_ocean_coach_reward_nonzero_through_mind_target():
    """End to end on the DEFAULT target: mind → central → live snapshot → Ocean reads non-zero."""
    central = GenesisKernelTarget()
    t = JointMindTarget()
    t._mind = SimpleNamespace(central=central)
    rec = _coach_record(relevance=0.85)
    t.register_coach_reward(coach_reward_from(rec), record=rec)
    assert t.telemetry().extra.get("coach") == rec
    assert OceanAutonomic._coach_reward(t.telemetry()) != 0.0


def test_negative_relevance_drops_coach_reward_below_zero():
    """A clearly-irrelevant utterance (relevance→0) DROPS the reward negative — the signal is EARNED."""
    central = GenesisKernelTarget()
    rec = _coach_record(relevance=0.0)
    reward = coach_reward_from(rec)
    assert reward < 0.0
    central.register_coach_reward(reward, record=rec)
    assert OceanAutonomic._coach_reward(central.telemetry()) < 0.0
