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


def test_registry_has_all_targets():
    r = default_registry()
    assert set(r.names()) == {"mock", "genesis", "mind", "kernel", "constellation", "geo-qwen", "qwen-local", "qwen-modal"}


def test_genesis_target_is_geometric_and_none_safe():
    from qig_studio.targets.base import LossRegime
    from qig_studio.targets.genesis_kernel import GenesisKernelTarget

    t = GenesisKernelTarget(num_layers=4)
    assert t.loss_regime is LossRegime.GEOMETRIC
    # None-safe: in the light app shell (no torch/qigkernels) it reports unavailable, never crashes.
    assert t.is_available() in (True, False)


def test_genesis_coords_path_wires_coordizer():
    """Coords path: a trained FisherCoordizer drives the kernel's enable_coords/CoordAdapter.
    Gated on the heavy stack (torch + qig_coordizer) — runs in the kernel venv, skipped in the
    light app shell."""
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("qig_coordizer")
    pytest.importorskip("qigkernels")
    import numpy as np
    from qig_coordizer import FisherCoordizer

    from qig_studio.targets.genesis_kernel import GenesisKernelTarget

    cz = FisherCoordizer(target_vocab_size=300)
    cz.train(b"the geometry is the truth; patterns flow through basins integrating space. " * 200,
             context_window=3, min_pair_count=2, verbose=False)

    t = GenesisKernelTarget(num_layers=2, hidden_dim=64, coordizer=cz)
    t.ensure_loaded()
    # coords path took: kernel built with enable_coords, vocab from the coordizer, coord_dim=64
    assert t._kernel.enable_coords is True
    assert t.vocab_size == len(cz.vocab)
    assert t.coord_dim == 64
    assert t.architecture()["input"] == "coords"

    # train_step runs with coords, CE finite (no arccos NaN through the CoordAdapter path)
    r = t.train_step("patterns flow through basins")
    assert r.telemetry.loss is not None and np.isfinite(r.telemetry.loss)
    assert np.isfinite(r.telemetry.phi)

    # speak: decode goes through the coordizer; self-observation + own-output telemetry present
    g = t.generate("hello", max_tokens=24)
    assert np.isfinite(g.telemetry.phi)
    for k in ("M_self_observation", "chose_to_stop", "generated_len", "mean_token_confidence"):
        assert k in g.telemetry.extra


def test_genesis_byte_path_unchanged_when_no_coordizer():
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("qigkernels")
    from qig_studio.targets.genesis_kernel import GenesisKernelTarget

    t = GenesisKernelTarget(num_layers=2, hidden_dim=64)
    t.ensure_loaded()
    assert t._kernel.enable_coords is False
    assert t.architecture()["input"] == "bytes"
    assert t.vocab_size == 256


def test_qwen_targets_are_language_and_none_safe():
    from qig_studio.targets import QwenLocalTarget, QwenModalTarget

    ql, qm = QwenLocalTarget(), QwenModalTarget()
    assert ql.loss_regime is LossRegime.LANGUAGE
    assert qm.loss_regime is LossRegime.LANGUAGE
    # None-safe: must return bool without raising even with no Ollama/Modal present.
    assert isinstance(ql.is_available(), bool)
    assert qm.is_available() is False  # no QIG_STUDIO_MODAL_URL configured


def test_language_curriculum_is_paired():
    from qig_studio.curriculum import CurriculumProvider

    c = CurriculumProvider(LossRegime.LANGUAGE)
    assert c.mode() == "paired"
    prompt, target = c.next_pair(1)
    assert isinstance(prompt, str) and isinstance(target, str) and target
