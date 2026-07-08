"""Neocortex harness (Phase 3 ARM B + Phase 4 ARM A) — the arm-polymorphic cortex wrapper contract.

These tests pin the cheap, deterministic surface of :class:`~qig_studio.neocortex.Neocortex`: the name
mapping (which drives the checkpoint dir + the live-trace ``model`` chip), BOTH arms building their own
target (``qk`` → ``GenesisKernelTarget``; ``geo`` → ``GeoCortexTarget``), and that each wraps a target
rather than reimplementing the model. Heavy training (the 100k kernel/GeoModel, the loss, telemetry) is the
targets' own coverage — not re-exercised here.
"""
from __future__ import annotations

import pytest

from qig_studio.neocortex import Neocortex

# qigkernels.Kernel / geocoding.GeoModel are built lazily (ensure_loaded); Neocortex.__init__ constructs the
# target but imports nothing heavy until ensure_loaded(). EXCEPTION: arm="geo" runs the faithfulness +
# loss-value parity asserts at CONSTRUCTION when deps are present (geocoding+qigkernels), which is the whole
# point — they gate before bpb is trusted. Construction + name mapping are otherwise dependency-light. Skip
# the heavy-dep cases if torch+qigkernels are genuinely absent (the None-safe app-shell environment).
_HAVE_DEPS = Neocortex(arm="qk", num_layers=2).is_available()


def test_geo_arm_builds_geo_cortex_target() -> None:
    """Phase 4: ARM A (geocoding) now BUILDS a GeoCortexTarget (no longer raises). Construction also runs
    the coords-off faithfulness + loss-value parity gates when deps are present — building without raising
    IS the parity passing."""
    from qig_studio.targets.geo_cortex import GeoCortexTarget

    n = Neocortex(arm="geo", num_layers=2, lang_loss="fisher_rao")
    assert isinstance(n.target, GeoCortexTarget)
    assert n.target.lang_loss == "fisher_rao"
    assert n.target.num_layers == 2
    assert n.name == "neocortex-geo-2L"
    # the SAME pass-throughs the launcher + ARM B rely on are present for ARM A too
    for attr in ("train_step", "generate", "eval_text_bpb", "eval_text_fr", "save", "load",
                 "telemetry", "vocab_size", "num_layers", "ensure_loaded", "is_available", "architecture"):
        assert hasattr(n, attr), f"missing pass-through: {attr}"


@pytest.mark.skipif(not _HAVE_DEPS, reason="geocoding+qigkernels absent (None-safe app shell)")
def test_geo_arm_construction_runs_parity_gates() -> None:
    """When deps are present, building ARM A asserts BOTH coords-off parity gates (≤1e-5). Prove the target
    exposes them and they pass on the values used at construction."""
    n = Neocortex(arm="geo", num_layers=2, device="cpu")
    assert n.target.assert_faithful_to_qigkernels(atol=1e-5) <= 1e-5
    assert n.target.assert_loss_value_parity(atol=1e-5) <= 1e-5


def test_unknown_arm_raises() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        Neocortex(arm="nonsense")


def test_name_mapping_drives_the_model_chip() -> None:
    """name = neocortex-{arm}-{N}L, or neocortex-{arm}-1L-rec for the 1-block-recursive variant."""
    assert Neocortex(arm="qk", num_layers=8).name == "neocortex-qk-8L"
    assert Neocortex(arm="qk", num_layers=2).name == "neocortex-qk-2L"
    rec = Neocortex(arm="qk", num_layers=8, recursive=True)
    assert rec.name == "neocortex-qk-1L-rec"
    # recursive collapses to a SINGLE block (recursion = the kernel's internal min_recursion_depth).
    assert rec._num_layers == 1


def test_wraps_genesis_kernel_target_not_a_reimplementation() -> None:
    """ARM B is exactly ONE GenesisKernelTarget with role='neocortex' — the constellation's central kernel,
    standalone. We assert the wrapped type + role, proving reuse (no kernel/loss/telemetry rebuild here)."""
    from qig_studio.targets.genesis_kernel import GenesisKernelTarget

    n = Neocortex(arm="qk", num_layers=2, lang_loss="fisher_rao")
    assert isinstance(n.target, GenesisKernelTarget)
    assert n.target.role == "neocortex"
    assert n.target.lang_loss == "fisher_rao"
    assert n.target.num_layers == 2
    # pass-throughs exist and are wired to the target
    for attr in ("train_step", "eval_text_bpb", "eval_text_fr", "save", "load",
                 "telemetry", "vocab_size", "ensure_loaded", "generate"):
        assert hasattr(n, attr), f"missing pass-through: {attr}"


@pytest.mark.skipif(not _HAVE_DEPS, reason="torch+qigkernels absent (None-safe app shell)")
def test_byte_level_cortex_takes_a_real_train_step() -> None:
    """Smoke at the unit level: a tiny byte-level (no coordizer) 1-layer cortex builds and takes a real,
    finite train_step — Φ populated, loss finite. Keeps it cheap (byte vocab=256, 1 layer, cpu)."""
    import math

    n = Neocortex(arm="qk", num_layers=1, device="cpu")   # byte-level: no coordizer
    n.ensure_loaded()
    res = n.train_step("the kernel learns to integrate information")
    tel = res.telemetry
    assert tel.step >= 1
    assert tel.phi is not None and math.isfinite(float(tel.phi))
    assert tel.loss is not None and math.isfinite(float(tel.loss))
    assert n.name == "neocortex-qk-1L"
