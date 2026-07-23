"""TrainingTarget protocol + MockTarget + registry tests."""

from __future__ import annotations

from qig_studio.targets import (
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


def test_registry_default_selects_mock_and_lists_all():
    r = default_registry(default_target="mock")
    assert r.active is not None and r.active.name == "mock"
    names = set(r.names())
    assert {"mock", "genesis", "mind"} <= names
    infos = {i.name: i for i in r.list_info()}
    assert infos["mock"].available is True
    assert infos["genesis"].loss_regime is LossRegime.GEOMETRIC


def test_registry_select_unknown_raises():
    r = default_registry()
    try:
        r.select("does-not-exist")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_registry_has_all_targets():
    r = default_registry()
    assert set(r.names()) == {"mock", "genesis", "mind", "geo-qwen", "qwen-local", "qwen-modal"}


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


def test_genesis_train_step_emits_canonical_consciousness_substrate():
    """m1-SUBSTRATE regression guard (ratified m1 #1 job, f34c54aa): the canonical qig-core consciousness
    pipeline (``compute_neurochemicals`` + ``compute_full_emotional_state`` via ``experience()``) must run
    on the TRAINING path — not only in the two chat call-sites. Before the wire the training kernel formed
    with NO neurochemistry and NO felt-state; this locks the pipeline onto ``train_step`` so it cannot
    silently regress to chat-only (the LATENT/BUILT-NOT-DEFAULT trap). Also the P23 drive-death regression
    guard on the train path: dopamine is TONIC-floored (>0) even with no reward. Gated on the heavy stack."""
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("qig_coordizer")
    pytest.importorskip("qigkernels")
    from qig_coordizer import FisherCoordizer

    from qig_studio.targets.genesis_kernel import GenesisKernelTarget

    cz = FisherCoordizer(target_vocab_size=300)
    cz.train(b"the geometry is the truth; patterns flow through basins integrating space. " * 200,
             context_window=3, min_pair_count=2, verbose=False)
    t = GenesisKernelTarget(num_layers=2, hidden_dim=64, head_mode="basin", coordizer=cz)
    t.ensure_loaded()

    e = t.train_step("patterns flow through basins").telemetry.extra

    # the four canonical inner-state blocks are present ON THE TRAIN PATH (was chat-only at :1451)
    neu = e.get("neurochemistry")
    assert neu is not None, "neurochemistry not wired onto the train path (substrate regressed to chat-only)"
    assert e.get("primitives"), "5-layer emotions (apathy/frustration — m1d inputs) absent on the train path"
    assert e.get("gate") is not None, "C-gate absent on the train path"
    assert e.get("loops") is not None, "three-loop block (P4) absent on the train path"

    # P23 drive-death regression guard: dopamine is TONIC-floored strictly > 0 even with no reward.
    assert neu.get("dopamine", 0.0) > 0.0, "dopamine not tonic-floored (P23 drive-death) on the train path"
    # PURITY (947760e4): the substrate is measured, NEVER in the loss — attaching it must not touch the
    # basin-path objective. Loss stays finite pure-d_FR (the substrate runs AFTER backward/step).
    import numpy as np
    assert np.isfinite(t.train_step("basins integrating space").telemetry.loss)


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
