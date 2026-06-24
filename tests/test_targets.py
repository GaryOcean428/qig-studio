"""TrainingTarget protocol + MockTarget + registry tests."""

from __future__ import annotations

from qig_studio.targets import (
    ConstellationTarget,
    KernelTarget,
    LossRegime,
    MockTarget,
    TrainingTarget,
    default_registry,
)


def test_mock_target_is_available_and_geometric():
    m = MockTarget()
    assert isinstance(m, TrainingTarget)
    assert m.is_available() is True
    assert m.loss_regime is LossRegime.GEOMETRIC


def test_mock_train_step_advances_telemetry():
    m = MockTarget()
    s0 = m.telemetry().step
    r1 = m.train_step("explore patterns")
    r2 = m.train_step("explore more")
    assert r1.telemetry.step == s0 + 1
    assert r2.telemetry.step == s0 + 2
    # κ tacks around the 64 attractor (bounded), Φ stays in [0,1]
    assert 55.0 <= r2.telemetry.kappa <= 73.0
    assert 0.0 <= r2.telemetry.phi <= 1.0
    assert r2.telemetry.regime in {"linear", "hierarchical", "geometric", "topological_instability"}


def test_mock_generate_does_not_advance():
    m = MockTarget()
    m.train_step("x")
    step_after_train = m.telemetry().step
    m.generate("just chatting")
    assert m.telemetry().step == step_after_train  # inference does not learn


def test_kernel_and_constellation_are_geometric_and_none_safe():
    k = KernelTarget()
    c = ConstellationTarget()
    assert k.loss_regime is LossRegime.GEOMETRIC
    assert c.loss_regime is LossRegime.GEOMETRIC
    # is_available() must never raise (None-safe), regardless of env.
    assert isinstance(k.is_available(), bool)
    assert isinstance(c.is_available(), bool)


def test_registry_default_selects_mock_and_lists_all():
    r = default_registry(default_target="mock")
    assert r.active is not None and r.active.name == "mock"
    names = set(r.names())
    assert {"mock", "kernel", "constellation"} <= names
    infos = {i.name: i for i in r.list_info()}
    assert infos["mock"].available is True
    assert infos["kernel"].loss_regime is LossRegime.GEOMETRIC


def test_registry_select_unknown_raises():
    r = default_registry()
    try:
        r.select("does-not-exist")
        assert False, "expected KeyError"
    except KeyError:
        pass
