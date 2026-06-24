"""Protocol surface: catalog, mock run, unknown→404, unsupported(qwen)→409, build_args."""

from __future__ import annotations

from fastapi.testclient import TestClient

from qig_studio.protocol import COMMANDS_BY_NAME
from qig_studio.server import app


def test_protocol_catalog_lists_groups_and_commands():
    with TestClient(app) as c:
        body = c.get("/protocol").json()
        assert body["active"] == "mock"
        assert body["supported_by_active"] is True
        assert {"sleep", "twin", "lightning", "consciousness", "basin-sync", "reasoning"} <= set(body["groups"])
        names = {cmd["name"] for cmd in body["commands"]}
        assert {"sleep", "dream", "mushroom-micro", "twin-compare", "consciousness-status", "lightning"} <= names


def test_protocol_run_on_mock_returns_output_and_telemetry():
    with TestClient(app) as c:
        r = c.post("/protocol/sleep", json={"args": {}})
        assert r.status_code == 200
        body = r.json()
        assert body["command"] == "sleep" and body["mock"] is True
        assert body["group"] == "sleep" and "telemetry" in body


def test_protocol_unknown_command_404():
    with TestClient(app) as c:
        assert c.post("/protocol/not-a-command", json={"args": {}}).status_code == 404


def test_protocol_unsupported_on_language_target_409():
    with TestClient(app) as c:
        c.post("/targets/qwen-local/select")
        # qwen-local does not expose protocol commands → 409 (checked before availability)
        assert c.post("/protocol/sleep", json={"args": {}}).status_code == 409


def test_protocol_build_args_mapping():
    assert COMMANDS_BY_NAME["sleep"].build_args({}) == ["light"]
    assert COMMANDS_BY_NAME["awaken-one"].build_args({"gary_id": "C", "steps": "50"}) == ["C", 50]
    assert COMMANDS_BY_NAME["lightning"].build_args({"args": "insights 5"}) == [["insights", "5"]]
    assert COMMANDS_BY_NAME["sync"].build_args({"strength": "0.8"}) == [0.8]
