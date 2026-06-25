"""Tests for the continuous-learning loop (brain §4b) — the autonomic P12 scheduler.

The decision-branch tests force the pure-Python canon fallback (``use_real=False``) so they are
deterministic regardless of whether qig-consciousness/torch is installed. The integration test runs
the real loop over MockTarget (always available, no torch) end-to-end.
"""

from __future__ import annotations

from qig_studio.learning import (
    AutonomicScheduler,
    ContinuousLearningLoop,
    Intervention,
)
from qig_studio.protocol import COMMANDS_BY_NAME
from qig_studio.targets.base import LossRegime
from qig_studio.targets.mock_target import MockTarget


# ---- P12 decision branches (forced fallback, deterministic) -----------------------------------

def test_breakdown_triggers_escape():
    sched = AutonomicScheduler(use_real=False)
    assert sched.decide({"phi": 0.85, "basin_distance": 0.0}) is Intervention.ESCAPE


def test_basin_drift_triggers_sleep():
    sched = AutonomicScheduler(use_real=False)
    assert sched.decide({"phi": 0.65, "basin_distance": 0.5}) is Intervention.SLEEP


def test_low_phi_triggers_dream():
    sched = AutonomicScheduler(use_real=False)
    assert sched.decide({"phi": 0.40, "basin_distance": 0.0}) is Intervention.DREAM


def test_healthy_state_continues_wake():
    sched = AutonomicScheduler(use_real=False)
    assert sched.decide({"phi": 0.60, "basin_distance": 0.0}) is Intervention.WAKE


def test_plateau_at_high_phi_triggers_mushroom():
    # Window of constant high Φ → variance 0 → plateau; Φ ≥ 0.70 → MUSHROOM (WAKE-state gate).
    sched = AutonomicScheduler(phi_window=5, use_real=False)
    tel = {"phi": 0.72, "basin_distance": 0.0}
    decisions = [sched.decide(tel) for _ in range(5)]
    assert decisions[-1] is Intervention.MUSHROOM  # plateau detected once the window fills
    assert all(d is Intervention.WAKE for d in decisions[:4])  # before the window fills


def test_plateau_below_threshold_triggers_dream_not_mushroom():
    # Plateau but Φ in [0.50, 0.70): the mushroom Φ≥0.70 gate forbids mushroom → DREAM instead.
    sched = AutonomicScheduler(phi_window=5, use_real=False)
    tel = {"phi": 0.60, "basin_distance": 0.0}
    decisions = [sched.decide(tel) for _ in range(5)]
    assert decisions[-1] is Intervention.DREAM


def test_intervention_values_route_to_real_protocol_commands():
    # Every non-WAKE intervention's value must be a real protocol command name (so run_protocol works).
    for itv in Intervention:
        if itv is Intervention.WAKE:
            continue
        assert itv.value in COMMANDS_BY_NAME, f"{itv} not a protocol command"


# ---- full loop over MockTarget (torch-free, end-to-end) ----------------------------------------

def test_loop_runs_to_completion_over_mock():
    loop = ContinuousLearningLoop(MockTarget(), max_steps=30)
    summary = loop.run()
    assert summary.steps == 30
    assert len(loop.history) == 30
    assert sum(loop.intervention_counts.values()) == 30
    # Mock Φ starts at ~0.47 (< PHI_EMERGENCY) then climbs → early DREAMs, later WAKEs.
    assert summary.interventions.get("dream", 0) >= 1
    assert summary.interventions.get("wake", 0) >= 1
    assert 0.0 <= summary.kernel_autonomy <= 1.0
    assert summary.final_phi > 0.0


def test_loop_geometric_uses_basin_driving_curriculum():
    target = MockTarget()
    loop = ContinuousLearningLoop(target, max_steps=3)
    assert target.loss_regime is LossRegime.GEOMETRIC
    assert loop.curriculum.mode() == "basin-driving"
    loop.run()
    # geometric records carry the basin-driving prompt marker from MockTarget.train_step
    assert any("basin-driving" in r.text for r in loop.history)


def test_loop_step_record_shape():
    loop = ContinuousLearningLoop(MockTarget(), max_steps=1)
    rec = loop.step()
    d = rec.to_dict()
    for key in ("step", "intervention", "phi", "kappa", "regime", "basin_distance", "delta_phi", "text"):
        assert key in d
    assert d["intervention"] in {i.value for i in Intervention}


def test_summary_reports_manager_mode():
    # In the light qig-studio env (no torch/qig-consciousness) the fallback policy is used.
    loop = ContinuousLearningLoop(MockTarget(), scheduler=AutonomicScheduler(use_real=False), max_steps=2)
    summary = loop.run()
    assert summary.using_real_manager is False
    assert "PROXY" in summary.notes  # kernel_autonomy honesty label preserved
