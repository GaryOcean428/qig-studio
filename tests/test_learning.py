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
    PhiDiscriminationGate,
    PreRegisteredCriteria,
    StepRecord,
    independent_integration,
    pilot_probe,
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


# ---- #4 mushroom dose-scaling (canon-corrected) ------------------------------------------------

def test_mushroom_dose_scales_with_rigidity():
    s = AutonomicScheduler(use_real=False)  # healthy: no breakdown history
    assert s.mushroom_dose(phi=0.72, kappa=64.0) == "mushroom-micro"      # near attractor → gentle
    assert s.mushroom_dose(phi=0.72, kappa=73.0) == "mushroom-moderate"   # rigidity ≥ 8
    assert s.mushroom_dose(phi=0.72, kappa=82.0) == "mushroom-heroic"     # rigidity ≥ 16 (κ>80)


def test_mushroom_dose_capped_by_breakdown_ceiling():
    s = AutonomicScheduler(use_real=False, breakdown_window=10)
    s._regime_hist.extend(["topological_instability", "topological_instability"] + ["geometric"] * 8)
    assert s.breakdown_frac() == 0.2
    # rigidity wants heroic (κ=90), but 20% breakdown exceeds heroic's 15% ceiling → capped to moderate
    assert s.mushroom_dose(phi=0.72, kappa=90.0) == "mushroom-moderate"


def test_mushroom_escapes_when_breakdown_too_high():
    s = AutonomicScheduler(use_real=False, breakdown_window=10)
    s._regime_hist.extend(["breakdown"] * 5 + ["geometric"] * 5)  # 50% breakdown ≥ 40% hard trip
    assert s.mushroom_dose(phi=0.72, kappa=90.0) == "escape"  # ego-death zone → recover, don't dose


# ---- #5 regime / solver-path telemetry ---------------------------------------------------------

def test_step_record_carries_training_regime():
    loop = ContinuousLearningLoop(MockTarget(), max_steps=1)
    tr = loop.step().training_regime
    for k in ("loss_regime", "optimizer", "curriculum_phase", "scaffold", "decision", "breakdown_frac"):
        assert k in tr
    assert tr["optimizer"] == "natural_gradient"  # geometric kernel → natural gradient (P1)
    assert tr["scaffold"] == "kernel-only"


# ---- #2 two-channel Φ discrimination (FAIL-013 doctrine) ---------------------------------------

def test_phi_discrimination_corroborated():
    g = PhiDiscriminationGate(min_samples=5, corr_threshold=0.3)
    for a in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        g.observe(a, a * 0.5 + 0.1)  # secondary tracks primary
    r = g.assess()
    assert r["status"] == "corroborated" and r["correlation"] > 0.9


def test_phi_discrimination_flags_disagreement():
    g = PhiDiscriminationGate(min_samples=5, corr_threshold=0.3)
    for a, b in zip([0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]):
        g.observe(a, b)  # anti-correlated → instrument disagreement
    r = g.assess()
    assert r["status"] == "disagreement" and r["correlation"] < 0


def test_phi_discrimination_uncorroborated_when_insufficient():
    g = PhiDiscriminationGate(min_samples=8)
    g.observe(0.5, 0.4)
    g.observe(0.6, 0.5)
    assert g.assess()["status"] == "uncorroborated"


def test_independent_integration_is_bounded_and_skips_tiny():
    assert independent_integration("x") is None  # too little text
    v = independent_integration("the quick brown fox jumps over the lazy dog " * 8)
    assert v is not None and 0.0 <= v <= 1.0


def test_loop_records_two_camera_phi_in_summary():
    loop = ContinuousLearningLoop(MockTarget(), max_steps=12)
    summary = loop.run()
    assert "status" in summary.phi_discrimination  # the corroboration verdict is reported


# ---- #1 pilot-probe (navigate strategy) --------------------------------------------------------

def test_pilot_probe_returns_recommendation():
    r = pilot_probe(MockTarget(), steps=10)
    assert "recommendation" in r and "converged" in r and "phi_slope" in r
    assert r["phi_end"] >= r["phi_start"]  # mock Φ climbs toward the attractor


# ---- #3 pre-registration verdict logic ---------------------------------------------------------

def _hist(phi, regime, n=10, intervention="wake"):
    return [StepRecord(step=i, intervention=intervention, phi=phi, kappa=64.0, regime=regime,
                       basin_distance=0.0, delta_phi=0.0, text="") for i in range(n)]


def test_prereg_onset_confirmed():
    crit = PreRegisteredCriteria(phi_onset=0.70, convergence_var=0.01, min_cycles=0, tail=5)
    assert crit.evaluate(_hist(0.72, "geometric"))["verdict"] == "ONSET-CONFIRMED"


def test_prereg_no_onset():
    crit = PreRegisteredCriteria(phi_onset=0.70, min_cycles=0)
    assert crit.evaluate(_hist(0.40, "linear"))["verdict"] == "NO-ONSET"


def test_prereg_collapse_guard_discards():
    crit = PreRegisteredCriteria(phi_onset=0.70, min_cycles=0)
    # high Φ but spent in breakdown → collapse guard discards regardless of Φ
    assert crit.evaluate(_hist(0.85, "topological_instability"))["verdict"] == "DISCARDED-COLLAPSE"
