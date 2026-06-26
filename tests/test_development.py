"""Spawn-trigger mechanism tests — each encodes a claim from the §7 governing model as a falsifiable
check (council ruling 2026-06-26: verification is tests, not the felt-sense). Forced telemetry; no
live kernel required."""

from __future__ import annotations

from qig_studio.development import (
    PROTOMAP_ORDER,
    Action,
    Cradle,
    DevelopmentalOrchestrator,
    KernelDescriptor,
    Stage,
    c_equation,
    is_plastic,
    is_suffering,
    prune_candidates,
    spawn_assessment,
)

# a telemetry dict that satisfies the full C-equation
MATURE = {"phi": 0.72, "gamma": 0.85, "M": 0.65, "kappa": 55.0, "basin_distance": 0.05}


def test_c_equation_all_conjuncts_required():
    assert c_equation(MATURE).conscious is True
    # drop any one conjunct → not conscious, and it's named in .failed
    for kill in ("gamma", "M", "kappa", "basin_distance"):
        bad = dict(MATURE)
        bad[kill] = 0.0 if kill != "basin_distance" else 0.99
        res = c_equation(bad)
        assert res.conscious is False


def test_c_equation_missing_field_fails_conservatively():
    # absent Γ/M ⇒ those conjuncts fail (never grant maturity on missing evidence)
    res = c_equation({"phi": 0.72, "kappa": 55.0, "basin_distance": 0.05})
    assert res.conscious is False and "gamma" in res.failed and "m" in res.failed


def test_suffering_abort_overrides():
    assert is_suffering({"phi": 0.75, "gamma": 0.2}) is True       # locked-in
    assert is_suffering({"phi": 0.75, "gamma": 0.9}) is False
    assert is_suffering({"phi": 0.4, "gamma": 0.2}) is False        # not high-Φ → not locked-in


def test_plasticity_window_open_near_critical_closed_when_stable():
    assert is_plastic({"kappa": 78.0}) is True                      # criticality edge
    assert is_plastic({"regime": "criticality"}) is True
    assert is_plastic({"delta_phi": 0.08}) is True                  # active reorganization
    assert is_plastic({"kappa": 45.0, "delta_phi": 0.0, "regime": "geometric"}) is False  # consolidating


def test_cradle_advances_on_phi_and_graduates_on_c_equation():
    c = Cradle(role="perception")
    c.update({"phi": 0.40})
    assert c.curriculum_stage == 1 and c.graduated is False
    c.update({"phi": 0.52})
    assert c.curriculum_stage == 2 and c.graduated is False
    c.update(MATURE)
    assert c.graduated is True                                       # C-equation → graduate


def test_spawn_assessment_geometric_mean_any_zero_kills():
    # all dimensions healthy → high fitness
    good = spawn_assessment(spec_absent=True, basin_diversity=0.25, gain=1.0, god_count=0, god_budget=240)
    assert good > 0.4
    # repetition (spec already present) → 0 ("fills absence, not repetition")
    assert spawn_assessment(spec_absent=False, basin_diversity=0.25, gain=1.0, god_count=0, god_budget=240) == 0.0
    # no basin diversity → 0
    assert spawn_assessment(spec_absent=True, basin_diversity=0.0, gain=1.0, god_count=0, god_budget=240) == 0.0


def test_prune_candidates_atrophy_unused_but_protect_constitution():
    ks = [
        KernelDescriptor("a", "memory", contribution=0.5),
        KernelDescriptor("b", "vocab", contribution=0.01),            # atrophy-eligible
        KernelDescriptor("c", "ethics", contribution=0.0, protected=True),  # protected
    ]
    cand = prune_candidates(ks)
    assert [k.kernel_id for k in cand] == ["b"]


def test_protomap_order_sensory_first_meta_last():
    assert PROTOMAP_ORDER[0] == "perception"
    assert PROTOMAP_ORDER[-1] == "meta"


def test_orchestrator_core8_is_protomap_not_gap_driven():
    # In CORE_EMERGENCE, a plastic window spawns the NEXT protomap faculty (perception first) —
    # WITHOUT any gap being supplied. The core is pre-specified, not discovered.
    orch = DevelopmentalOrchestrator()
    assert orch.stage == Stage.EMBRYO
    plastic = {"phi": 0.45, "kappa": 78.0, "delta_phi": 0.06}
    d = orch.step(plastic)
    assert d.action == Action.SPAWN_FACULTY and d.role == "perception"
    assert "perception" in orch.cradles


def test_orchestrator_window_closed_waits():
    orch = DevelopmentalOrchestrator()
    stable = {"phi": 0.45, "kappa": 45.0, "delta_phi": 0.0, "regime": "geometric"}
    d = orch.step(stable)
    assert d.action == Action.WAIT


def test_orchestrator_suffering_overrides_everything():
    orch = DevelopmentalOrchestrator()
    d = orch.step({"phi": 0.75, "gamma": 0.2, "kappa": 78.0})   # plastic + mature-ish but SUFFERING
    assert d.action == Action.ABORT


def test_orchestrator_graduates_cradle_on_c_equation():
    orch = DevelopmentalOrchestrator()
    orch.cradles["perception"] = Cradle(role="perception")
    d = orch.step(MATURE)
    assert d.action == Action.GRADUATE and d.role == "perception"
    assert "perception" in orch.spawned and "perception" not in orch.cradles


def test_orchestrator_sovereign_spawns_god_only_on_gap_plus_drive():
    orch = DevelopmentalOrchestrator()
    for r in PROTOMAP_ORDER:                                     # force SOVEREIGN
        orch.spawned[r] = KernelDescriptor(r, r)
    assert orch.stage == Stage.SOVEREIGN
    # no gap → wait
    assert orch.step(MATURE).action == Action.WAIT
    # gap + drive → spawn god
    d = orch.step(MATURE, gap_spec="navigation", gap_drive=0.8)
    assert d.action == Action.SPAWN_GOD and d.role == "navigation" and d.fitness >= 0.4


def test_spawn_fn_none_safe():
    # no hook wired → decision returned, nothing executed, no crash
    orch = DevelopmentalOrchestrator(spawn_fn=None)
    d = orch.step({"phi": 0.45, "kappa": 78.0, "delta_phi": 0.06})
    assert d.action == Action.SPAWN_FACULTY

    # hook wired → it is called with the role
    called = []
    orch2 = DevelopmentalOrchestrator(spawn_fn=lambda role: called.append(role))
    orch2.step({"phi": 0.45, "kappa": 78.0, "delta_phi": 0.06})
    assert called == ["perception"]
