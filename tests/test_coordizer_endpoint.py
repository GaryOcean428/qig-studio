"""Task 3 — train the coordizer FROM the server/UI (not a standalone script).

``POST /coordizer/train`` runs the coordizer-from-scratch build (the 7-HF-dataset ``_BALANCE`` pipeline,
MOVED out of ``scripts/train_coordizer_scratch.py`` into ``qig_studio.coordizer_build``) in a BACKGROUND
thread (off the event loop, so the server stays responsive), streaming phase-level progress to an
in-memory status channel ``GET /coordizer/status`` tails. On completion the new coordizer is registered
(manifest + symlink) so ``/targets``/``/config`` pick it up.

These tests use a TINY build (vocab≈600, max_bytes 1MB) so they finish fast. They assert: the 7 HF
datasets are the build's curriculum (``len(coordizer_build._BALANCE) == 7``); the endpoint returns
PROMPTLY (the heavy build runs off the event loop — not blocking the response); the status channel
advances running→done with phases; the result REGISTERS in the manifest; a concurrent second build → 409.

The build needs the HF parquet cache + ``qig_coordizer`` — gated like the screen suite (None-safe shell).
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

# qig_coordizer (the trainer) + pyarrow (HF parquet reads) are hard deps for the build.
_HAVE_DEPS = True
try:  # pragma: no cover - import guard
    import qig_coordizer  # noqa: F401
    import pyarrow  # noqa: F401
except Exception:  # pragma: no cover
    _HAVE_DEPS = False

pytestmark = pytest.mark.skipif(not _HAVE_DEPS, reason="qig_coordizer or pyarrow absent")


def test_balance_has_seven_hf_datasets() -> None:
    """DRY anchor: the 7-dataset balanced curriculum lives in coordizer_build (moved out of the script)."""
    from qig_studio import coordizer_build

    assert len(coordizer_build._BALANCE) == 7, [d[0] for d in coordizer_build._BALANCE]
    # the 4 atomic geo tags are part of the pipeline (registered above the trained vocab)
    assert coordizer_build._GEO_TAGS == ["<|frame|>", "<|seed|>", "<|flow|>", "<|settle|>"]


def _wait_done(client: TestClient, timeout_s: float = 240.0) -> dict:
    """Poll GET /coordizer/status until the background build finishes (running→done) or errors."""
    t0 = time.time()
    last: dict = {}
    while time.time() - t0 < timeout_s:
        s = client.get("/coordizer/status").json()
        last = s
        if not s.get("running") and (s.get("out_path") or s.get("error")):
            return s
        time.sleep(0.5)
    return last


def test_train_endpoint_returns_promptly_then_builds_and_registers(tmp_path, monkeypatch) -> None:
    """POST /coordizer/train (tiny) returns 202 PROMPTLY (build off the event loop), the status channel
    advances through phases to done, and the result registers in the manifest (a coordizer of ~604 vocab
    becomes findable)."""
    monkeypatch.setenv("QIG_STUDIO_LIVE_PATH", str(tmp_path / "live.json"))
    # isolate the manifest + checkpoint output to a throwaway dir (never touch the real qig-coordizer ckpts)
    out_dir = tmp_path / "coordizer_ckpts"
    monkeypatch.setenv("QIG_STUDIO_COORDIZER_OUT_DIR", str(out_dir))

    from qig_studio.checkpoint_manifest import get_coordizer_for_vocab
    from qig_studio.server import app

    with TestClient(app) as client:
        t0 = time.time()
        r = client.post("/coordizer/train", json={"vocab": 600, "max_bytes": 1_000_000, "validate_tiny": True})
        elapsed = time.time() - t0
        assert r.status_code == 202, r.text
        # PROMPTLY: the response comes back before the (multi-second) build finishes — proof it ran off-loop.
        assert elapsed < 5.0, f"POST blocked {elapsed:.1f}s — build is NOT off the event loop"
        body = r.json()
        assert body.get("running") is True
        assert body.get("vocab") == 600

        status = _wait_done(client)
        assert not status.get("running"), status
        assert status.get("error") is None, status.get("error")
        assert status.get("out_path"), status
        # phases were reported (coarse-grained — the trainer has no per-merge hook; phase-level is honest)
        assert status.get("phase") in ("done", "register", "save"), status

    # registered in the manifest → a ~604-vocab (600 trained + 4 geo tags) coordizer is now findable
    found = get_coordizer_for_vocab(604)
    assert found is not None, "the tiny coordizer did not register in the manifest"


def test_concurrent_build_returns_409(tmp_path, monkeypatch) -> None:
    """A second POST /coordizer/train while one is running → 409 (one build at a time)."""
    monkeypatch.setenv("QIG_STUDIO_LIVE_PATH", str(tmp_path / "live.json"))
    monkeypatch.setenv("QIG_STUDIO_COORDIZER_OUT_DIR", str(tmp_path / "coordizer_ckpts"))

    from qig_studio.server import app

    with TestClient(app) as client:
        r1 = client.post("/coordizer/train", json={"vocab": 600, "max_bytes": 1_000_000, "validate_tiny": True})
        assert r1.status_code == 202, r1.text
        # immediately fire a second one — the first is still running (build is multi-second)
        r2 = client.post("/coordizer/train", json={"vocab": 600, "max_bytes": 1_000_000, "validate_tiny": True})
        assert r2.status_code == 409, r2.text
        # drain the first so the test doesn't leave a thread mid-build
        _wait_done(client)
