"""Task 1 — config-built neocortex target in the registry.

The registry can BUILD any neocortex config (arm × head_mode × num_layers × lang_loss) and install it
under a STABLE slot key so switching configs swaps the active target in place. ``build_neocortex`` REUSES
the existing arm ctors (``GenesisKernelTarget`` for ``qk``; ``GeoCortexTarget`` for ``geo``) — no new
training loop — and stamps a config-descriptive ``.name`` (``neocortex-{arm}-{N}L-{geo|lin}``) for the UI
model chip. ``set_neocortex`` registers under the ``"neocortex"`` slot and selects it.

These tests are dependency-light at construction (the kernel/GeoModel build lazily in ``ensure_loaded``);
the build-on-cpu case skips when torch+qigkernels(+geocoding) are genuinely absent (None-safe app shell).
"""
from __future__ import annotations

import pytest

from qig_studio.targets.genesis_kernel import GenesisKernelTarget
from qig_studio.targets.geo_cortex import GeoCortexTarget
from qig_studio.targets.registry import TargetRegistry, build_neocortex

_HAVE_DEPS = GenesisKernelTarget(num_layers=2).is_available()
_HAVE_GEO = GeoCortexTarget(num_layers=2).is_available()


def test_build_geo_linear_name_and_type() -> None:
    """arm=geo, head_mode=linear, 2L → GeoCortexTarget whose config-descriptive name carries the avenue."""
    t = build_neocortex(arm="geo", head_mode="linear", num_layers=2)
    assert isinstance(t, GeoCortexTarget)
    assert t.name == "neocortex-geo-2L-lin"
    assert t.head_mode == "linear"
    assert t.num_layers == 2
    assert t.lang_loss == "fisher_rao"


def test_build_qk_geometric_name_and_type() -> None:
    """arm=qk, head_mode=geometric, 2L → GenesisKernelTarget named neocortex-qk-2L-geo."""
    t = build_neocortex(arm="qk", head_mode="geometric", num_layers=2)
    assert isinstance(t, GenesisKernelTarget)
    assert t.name == "neocortex-qk-2L-geo"
    assert t.head_mode == "geometric"
    assert t.num_layers == 2


def test_build_passes_lang_loss_through() -> None:
    """lang_loss flows straight into the ctor (DRY — no re-derivation)."""
    t = build_neocortex(arm="qk", head_mode="linear", num_layers=3, lang_loss="ce_ablation")
    assert t.name == "neocortex-qk-3L-lin"
    assert t.lang_loss == "ce_ablation"


def test_build_unknown_arm_raises() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        build_neocortex(arm="nonsense")


def test_set_neocortex_uses_stable_slot_and_selects() -> None:
    """set_neocortex installs under the stable ``neocortex`` slot, selects it, and swapping a new config
    REPLACES the slot in place (active is always the latest build, name reflects the new config)."""
    r = TargetRegistry()
    t1 = build_neocortex(arm="geo", head_mode="linear", num_layers=2)
    r.set_neocortex(t1)
    assert r.active is t1
    assert r.active.name == "neocortex-geo-2L-lin"
    assert r.get("neocortex") is t1            # stable slot key, not the descriptive name

    t2 = build_neocortex(arm="qk", head_mode="geometric", num_layers=2)
    r.set_neocortex(t2)
    assert r.active is t2                        # swapped in place
    assert r.active.name == "neocortex-qk-2L-geo"
    assert len([n for n in r.names() if n == "neocortex"]) == 0  # slot key is internal, name is descriptive
    assert r.get("neocortex") is t2             # one stable slot, replaced


def test_set_neocortex_coexists_with_named_targets() -> None:
    """Installing a neocortex does NOT clobber existing name-keyed targets — name-only selection still works."""
    from qig_studio.targets.mock_target import MockTarget

    r = TargetRegistry()
    r.register(MockTarget())
    r.set_neocortex(build_neocortex(arm="geo", head_mode="linear", num_layers=2))
    assert r.active.name == "neocortex-geo-2L-lin"
    r.select("mock")                            # name-only selection of an existing target
    assert r.active.name == "mock"


@pytest.mark.skipif(not _HAVE_DEPS, reason="torch+qigkernels absent (None-safe app shell)")
def test_qk_builds_on_cpu_available() -> None:
    """qk neocortex is available and builds on cpu without error (tiny: byte vocab, 2 layers, cpu)."""
    t = build_neocortex(arm="qk", head_mode="geometric", num_layers=2, device="cpu")
    assert t.is_available()
    t.ensure_loaded()                           # builds the kernel on cpu without raising
    assert t.name == "neocortex-qk-2L-geo"


@pytest.mark.skipif(not _HAVE_GEO, reason="geocoding+qigkernels absent (None-safe app shell)")
def test_geo_builds_on_cpu_available() -> None:
    """geo/linear neocortex is available and builds on cpu without error."""
    t = build_neocortex(arm="geo", head_mode="linear", num_layers=2, device="cpu")
    assert t.is_available()
    t.ensure_loaded()
    assert t.name == "neocortex-geo-2L-lin"


# --- server path: POST /target/neocortex sets the active target; name-only selection still works --------

def test_server_build_neocortex_sets_active() -> None:
    """POST /target/neocortex {arm:geo, head_mode:linear, num_layers:2} → active_target = neocortex-geo-2L-lin,
    and selecting an existing target by name (mock) still works (name-only path unbroken)."""
    from fastapi.testclient import TestClient

    from qig_studio.server import app

    with TestClient(app) as client:
        r = client.post("/target/neocortex",
                        json={"arm": "geo", "head_mode": "linear", "num_layers": 2})
        assert r.status_code == 200, r.text
        assert r.json()["active"] == "neocortex-geo-2L-lin"
        # it is genuinely the live active target
        assert client.get("/targets").json()["active"] == "neocortex-geo-2L-lin"
        # name-only selection of an EXISTING target still works (the original path is preserved)
        r2 = client.post("/targets/mock/select")
        assert r2.status_code == 200, r2.text
        assert r2.json()["active"] == "mock"
        assert client.get("/targets").json()["active"] == "mock"


def test_server_build_neocortex_qk_geometric() -> None:
    """The qk/geometric avenue resolves to neocortex-qk-2L-geo through the endpoint."""
    from fastapi.testclient import TestClient

    from qig_studio.server import app

    with TestClient(app) as client:
        r = client.post("/target/neocortex",
                        json={"arm": "qk", "head_mode": "geometric", "num_layers": 2})
        assert r.status_code == 200, r.text
        assert r.json()["active"] == "neocortex-qk-2L-geo"


def test_server_build_neocortex_rejects_unknown_arm() -> None:
    from fastapi.testclient import TestClient

    from qig_studio.server import app

    with TestClient(app) as client:
        r = client.post("/target/neocortex", json={"arm": "nonsense"})
        assert r.status_code == 400
