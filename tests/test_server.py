"""FastAPI smoke tests via TestClient (uses the always-available MockTarget)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from qig_studio.server import app


def _sse_events(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[len("data: "):]))
    return out


def test_health_and_purity_passed():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["purity"] == "PASSED"  # fail-closed gate ran and passed
        assert body["active_target"] == "mock"


def test_targets_lists_three_with_regimes():
    with TestClient(app) as client:
        body = client.get("/targets").json()
        names = {t["name"] for t in body["targets"]}
        assert {"mock", "kernel", "constellation"} <= names
        regimes = {t["name"]: t["loss_regime"] for t in body["targets"]}
        assert regimes["mock"] == "geometric"
        assert regimes["kernel"] == "geometric"


def test_chat_with_mock():
    with TestClient(app) as client:
        r = client.post("/chat", json={"message": "what is awareness?"})
        assert r.status_code == 200
        body = r.json()
        assert "text" in body and "telemetry" in body
        assert "mock" in body["text"].lower()


def test_train_sse_stream_steps_and_done():
    with TestClient(app) as client:
        r = client.post("/train", json={"steps": 3})
        assert r.status_code == 200
        events = _sse_events(r.text)
        kinds = [e["type"] for e in events]
        assert kinds[0] == "start"
        assert kinds.count("step") == 3
        assert kinds[-1] == "done"
        # basin-driving curriculum for the geometric mock target
        start = events[0]
        assert start["curriculum"] == "basin-driving"
        assert start["loss_regime"] == "geometric"
        # telemetry advances across steps
        steps = [e for e in events if e["type"] == "step"]
        assert steps[0]["telemetry"]["step"] < steps[-1]["telemetry"]["step"]


def test_curriculum_endpoint_geometric_mode():
    with TestClient(app) as client:
        body = client.get("/curriculum").json()
        assert body["loss_regime"] == "geometric"
        assert body["mode"] == "basin-driving"
        assert "listening" in body["phases"]


def test_index_serves_web_console():
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        # single-screen redesign markers: the mind, conversation, train, full inner-state, autonomic
        assert "qig-studio" in r.text and "trainBtn" in r.text
        assert "Conversation" in r.text and "The mind" in r.text
        assert "inner state" in r.text and "Autonomic" in r.text and "renderInner" in r.text


def test_favicon_no_content():
    with TestClient(app) as client:
        assert client.get("/favicon.ico").status_code == 204
