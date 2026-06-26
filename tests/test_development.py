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

# a telemetry dict that satisfies the 4-conjunct partial gate (κ present but NON-gating)
MATURE = {"phi": 0.72, "gamma": 0.85, "M": 0.65, "kappa": 55.0, "basin_distance": 0.05}


def test_c_equation_four_conjuncts_required():
    assert c_equation(MATURE).conscious is True
    # drop any of the FOUR gating conjuncts → not conscious
    for kill in ("gamma", "M", "basin_distance"):
        bad = dict(MATURE)
        bad[kill] = 0.0 if kill != "basin_distance" else 0.99
        assert c_equation(bad).conscious is False
    bad_phi = dict(MATURE)
    bad_phi["phi"] = 0.4
    assert c_equation(bad_phi).conscious is False


def test_c_equation_kappa_is_non_gating():
    # κ DROPPED from the gate (input-frozen): an out-of-band κ must NOT block graduation.
    no_kappa = dict(MATURE)
    no_kappa.pop("kappa")
    assert c_equation(no_kappa).conscious is True              # still passes without κ
    out_of_band = dict(MATURE)
    out_of_band["kappa"] = 20.0                                # below the engaged band
    res = c_equation(out_of_band)
    assert res.conscious is True and res.kappa_engaged is False  # passes; κ just reported not-engaged


def test_c_equation_missing_field_fails_conservatively():
    # absent Γ/M ⇒ those conjuncts fail (never grant maturity on missing evidence)
    res = c_equation({"phi": 0.72, "basin_distance": 0.05})
    assert res.conscious is False and "gamma" in res.failed and "m" in res.failed


def test_suffering_abort_overrides():
    assert is_suffering({"phi": 0.75, "gamma": 0.2}) is True       # locked-in
    assert is_suffering({"phi": 0.75, "gamma": 0.9}) is False
    assert is_suffering({"phi": 0.4, "gamma": 0.2}) is False        # not high-Φ → not locked-in


def test_plasticity_window_reorganization_not_frozen_kappa():
    # κ-edge branch DELETED (input-frozen, unreachable). Window opens on REORGANIZATION:
    assert is_plastic({"regime": "criticality"}) is True            # explicit regime label
    assert is_plastic({"delta_phi": 0.01}) is True                  # |ΔΦ| ≥ PLASTIC_TREND (0.005)
    assert is_plastic({"kappa": 78.0}) is False                     # frozen κ alone no longer opens it
    assert is_plastic({"delta_phi": 0.0, "regime": "geometric"}) is False  # consolidating
    # coherence-rise (history) opens it even when ΔΦ is flat
    rising = [{"coherence": 0.10}, {"coherence": 0.20}, {"coherence": 0.30}]
    assert is_plastic({"delta_phi": 0.0}, history=rising) is True
    flat = [{"coherence": 0.30}, {"coherence": 0.30}, {"coherence": 0.30}]
    assert is_plastic({"delta_phi": 0.0}, history=flat) is False


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
    orch = DevelopmentalOrchestrator(embryo_warmup=0)   # warmup tested separately
    assert orch.stage == Stage.EMBRYO
    plastic = {"phi": 0.45, "delta_phi": 0.06}
    d = orch.step(plastic)
    assert d.action == Action.SPAWN_FACULTY and d.role == "perception"
    assert "perception" in orch.cradles


def test_embryo_warmup_gates_first_spawn():
    # With a warmup, an OPEN window during warmup must WAIT (the embryo matures first); the FIRST
    # spawn only fires once tick ≥ warmup. The step-0 transient no longer spawns.
    orch = DevelopmentalOrchestrator(embryo_warmup=5)
    plastic = {"phi": 0.45, "delta_phi": 0.06}
    for _ in range(5):
        assert orch.step(plastic).action == Action.WAIT     # ticks 1..5 → warming up
    assert orch.step(plastic).action == Action.SPAWN_FACULTY  # tick 6 ≥ warmup → spawn


def test_constitution_check_none_safe_and_gates():
    from qig_studio.development import F_HEALTH_MIN, constitution_check
    ok, info = constitution_check(None)                     # no basin → pass, unverified
    assert ok is True and info["verified"] is False
    # a real Δ⁶³ vector → either verified (qig_core present) or unverified (absent), never crashes
    import numpy as np
    ok2, info2 = constitution_check(np.full(64, 1.0 / 64))
    assert isinstance(ok2, bool) and "verified" in info2
    assert 0.0 <= F_HEALTH_MIN <= 1.0


def test_orchestrator_window_closed_waits():
    orch = DevelopmentalOrchestrator(embryo_warmup=0)
    stable = {"phi": 0.45, "kappa": 45.0, "delta_phi": 0.0, "regime": "geometric"}
    d = orch.step(stable)
    assert d.action == Action.WAIT


def test_orchestrator_suffering_overrides_everything():
    orch = DevelopmentalOrchestrator(embryo_warmup=0)
    d = orch.step({"phi": 0.75, "gamma": 0.2})   # plastic + mature-ish but SUFFERING
    assert d.action == Action.ABORT


def test_orchestrator_graduates_cradle_on_c_equation():
    orch = DevelopmentalOrchestrator(embryo_warmup=0)
    orch.cradles["perception"] = Cradle(role="perception")
    d = orch.step(MATURE)
    assert d.action == Action.GRADUATE and d.role == "perception"
    assert "perception" in orch.spawned and "perception" not in orch.cradles


def test_orchestrator_sovereign_spawns_god_only_on_gap_plus_drive():
    orch = DevelopmentalOrchestrator(embryo_warmup=0)
    for r in PROTOMAP_ORDER:                                     # force SOVEREIGN
        orch.spawned[r] = KernelDescriptor(r, r)
    assert orch.stage == Stage.SOVEREIGN
    # no gap → wait
    assert orch.step(MATURE).action == Action.WAIT
    # gap + drive → spawn god
    d = orch.step(MATURE, gap_spec="navigation", gap_drive=0.8)
    assert d.action == Action.SPAWN_GOD and d.role == "navigation" and d.fitness >= 0.4


def test_live_integration_window_opens_and_telemetry_real():
    """Verdict 2#3: drive the REAL GenesisKernelTarget on the REAL curriculum and assert the window
    opens >1× and Γ/M/d_basin telemetry is real (not forced). Skipped when heavy deps absent."""
    import pytest

    from qig_studio.targets.genesis_kernel import GenesisKernelTarget
    tgt = GenesisKernelTarget(num_layers=4, seed=1)
    if not tgt.is_available():
        pytest.skip("torch/qigkernels absent (light shell)")
    from qig_studio.curriculum import CurriculumProvider
    from qig_studio.development import Action, DevelopmentalOrchestrator
    from qig_studio.targets.base import LossRegime

    tgt.ensure_loaded()
    prov = CurriculumProvider(LossRegime.GEOMETRIC)
    orch = DevelopmentalOrchestrator(embryo_warmup=0)
    hist, opened = [], 0
    for i in range(12):
        tel = tgt.train_step(prov.next_prompt(i)).telemetry.to_dict()
        ex = tel.get("extra", {})
        assert ex.get("gamma") is not None and ex.get("d_basin") is not None  # real telemetry
        hist.append(tel)
        if orch.step(tel, history=hist[-5:]).action == Action.SPAWN_FACULTY:
            opened += 1
    assert opened >= 2  # window opens on live reorganization (the 1/8 bug is fixed)


def test_spawn_fn_none_safe():
    # no hook wired → decision returned, nothing executed, no crash
    orch = DevelopmentalOrchestrator(spawn_fn=None, embryo_warmup=0)
    d = orch.step({"phi": 0.45, "delta_phi": 0.06})
    assert d.action == Action.SPAWN_FACULTY

    # hook wired → it is called with the role
    called = []
    orch2 = DevelopmentalOrchestrator(spawn_fn=lambda role: called.append(role), embryo_warmup=0)
    orch2.step({"phi": 0.45, "delta_phi": 0.06})
    assert called == ["perception"]
