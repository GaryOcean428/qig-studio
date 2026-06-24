"""P0 server hardening: fail-closed auth (F2), LANGUAGE steps-cap (F3), qig-warp early-stop."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from qig_studio.config import Settings
from qig_studio.server import app
from qig_studio.targets.base import LossRegime
from qig_studio.targets.mock_target import MockTarget


def _events(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


class _LangMock(MockTarget):
    """Always-available LANGUAGE target for exercising the steps-cap path."""

    name = "langmock"
    loss_regime = LossRegime.LANGUAGE

    def is_available(self) -> bool:
        return True


# --- P0-F2: fail-closed auth ---------------------------------------------------

def test_auth_open_on_localhost_without_key():
    with TestClient(app) as c:
        assert c.post("/chat", json={"message": "hi"}).status_code == 200


def test_auth_enforced_when_key_set():
    with TestClient(app) as c:
        app.state.auth_key = "secret123"
        try:
            assert c.post("/chat", json={"message": "hi"}).status_code == 401
            ok = c.post("/chat", json={"message": "hi"}, headers={"X-Studio-Key": "secret123"})
            assert ok.status_code == 200
        finally:
            app.state.auth_key = None


def test_is_loopback_guard_logic():
    assert Settings(host="127.0.0.1").is_loopback
    assert Settings(host="localhost").is_loopback
    assert not Settings(host="0.0.0.0").is_loopback
    assert not Settings(host="10.0.0.5").is_loopback


# --- P0-F3: LANGUAGE steps-cap + confirm --------------------------------------

def test_language_target_steps_cap_and_confirm():
    with TestClient(app) as c:
        c.app.state.registry.register(_LangMock())
        c.app.state.registry.select("langmock")
        try:
            ev = _events(c.post("/train", json={"steps": 5}).text)
            assert any(e["type"] == "error" and "steps>1" in e["error"] for e in ev)

            ev2 = _events(c.post("/train", json={"steps": 1}).text)
            assert any(e["type"] == "error" and "confirm" in e["error"] for e in ev2)

            ev3 = _events(c.post("/train", json={"steps": 1, "confirm": True}).text)
            assert ev3[-1]["type"] == "done"
        finally:
            c.app.state.registry.select("mock")


# --- qig-warp convergence early-stop (package lever) --------------------------

def test_early_stop_flag_wired_and_terminates():
    with TestClient(app) as c:
        ev = _events(c.post("/train", json={"steps": 40, "early_stop": True}).text)
        assert ev[0]["type"] == "start"
        assert ev[0]["early_stop"] is True  # qig-warp present → lever active
        # converging mock telemetry → check_ci_stabilized should fire before 40 steps
        assert any(e["type"] == "early_stop" and e["lever"] == "qig_warp.check_ci_stabilized" for e in ev)


# --- UI/UX: directory nomination ----------------------------------------------

def test_config_get_and_set_dirs():
    with TestClient(app) as c:
        cfg = c.get("/config").json()
        assert {"output_dir", "curriculum_dir", "warp_early_stop", "auth_required"} <= set(cfg)
        r = c.post("/config", json={"output_dir": "/tmp/qs_out", "curriculum_dir": "/tmp/qs_curr"})
        assert r.status_code == 200 and r.json()["output_dir"] == "/tmp/qs_out"
        assert c.get("/config").json()["curriculum_dir"] == "/tmp/qs_curr"


def test_curriculum_loads_from_nominated_dir(tmp_path):
    from qig_studio.curriculum import CurriculumProvider

    (tmp_path / "phase.txt").write_text("custom prompt one\ncustom prompt two\n", encoding="utf-8")
    p = CurriculumProvider(LossRegime.GEOMETRIC, curriculum_dir=str(tmp_path))
    assert p.next_prompt(1) == "custom prompt one"
    assert p.next_prompt(2) == "custom prompt two"
    assert p.next_prompt(3) == "custom prompt one"  # wraps


def test_curriculum_malformed_jsonl_does_not_crash(tmp_path):
    """REL-1: non-dict JSON values (number/bool/bare-string) must not crash the loader."""
    from qig_studio.curriculum import CurriculumProvider

    (tmp_path / "bad.jsonl").write_text(
        '42\n"prompt and target"\ntrue\n{"prompt":"p","target":"t"}\nnull\n', encoding="utf-8"
    )
    p = CurriculumProvider(LossRegime.LANGUAGE, curriculum_dir=str(tmp_path))
    # the one valid dict record loads; malformed lines are skipped (no TypeError)
    assert p.next_pair(1) == ("p", "t")
