"""Task 2 — the A/B avenue screen as a SERVER capability through the wired training loop.

``POST /screen`` routes each of the 4 configs {qk,geo}×{geometric,linear} (fixed depth) through the SAME
wired training path ``POST /train`` uses (``server._train_core``), so the UI shows each config training
(``active_target`` flips to its ``neocortex-{arm}-{N}L-{geo|lin}`` name) and the telemetry is coherent.
After each config's short equal budget it evals held-out d_FR and ranks (or flags UNDER-POWERED).

These use the byte-mode targets (no coordizer needed — tiny), and assert the head_modes ACTUALLY differ
across configs (the env caveat: QIG_STUDIO_HEAD_MODE overrides the ctor, so the screen MUST set it per
config or every config pins to one head). DRY is asserted at the unit level in ``test_screen_helper`` /
the grep in the task verifier (screen.py is eval-only).
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from qig_studio.targets.genesis_kernel import GenesisKernelTarget

_HAVE_DEPS = GenesisKernelTarget(num_layers=2).is_available()
pytestmark = pytest.mark.skipif(not _HAVE_DEPS, reason="torch+qigkernels absent (None-safe app shell)")


def _events(resp) -> list[dict]:
    out = []
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[len("data: ") :]))
    return out


@pytest.mark.slow  # trains 4 tiny kernels on CPU — opt-in only; the REAL avenue result comes from a GPU run
def test_screen_streams_four_configs_with_flipping_active_target(tmp_path, monkeypatch) -> None:
    """POST /screen {layers:2, steps:4} streams a config event per avenue, each setting active_target to
    its neocortex name, and a final ranked (or under-powered) event."""
    # isolate the live channel + runs dir so the test never pollutes the production stream
    monkeypatch.setenv("QIG_STUDIO_LIVE_PATH", str(tmp_path / "live.json"))
    from qig_studio.server import app

    with TestClient(app) as client:
        r = client.post("/screen", json={"layers": 2, "steps": 4, "device": "cpu"})
        assert r.status_code == 200, r.text
        events = _events(r)

    kinds = [e.get("type") for e in events]
    assert "start" in kinds
    # one config-start event per avenue, each carrying the avenue name + active_target == that name
    config_starts = [e for e in events if e.get("type") == "config_start"]
    assert len(config_starts) == 4, kinds
    for e in config_starts:
        assert e["active_target"] == e["name"], e
        assert e["name"].startswith("neocortex-")

    # the 4 avenues are the {qk,geo}×{geometric,linear} matrix at depth 2
    names = {e["name"] for e in config_starts}
    assert names == {
        "neocortex-qk-2L-geo",
        "neocortex-qk-2L-lin",
        "neocortex-geo-2L-geo",
        "neocortex-geo-2L-lin",
    }, names

    # the head_modes ACTUALLY differ (the env caveat) — both geometric and linear appear, verified per build
    head_modes = {e["head_mode"] for e in config_starts}
    assert "geometric" in head_modes and "linear" in head_modes, head_modes

    # final event: a ranked d_FR list OR the under-power flag (both honest outcomes)
    done = [e for e in events if e.get("type") == "screen_done"]
    assert len(done) == 1, kinds
    payload = done[0]
    assert "ranking" in payload and "underpowered" in payload
    assert isinstance(payload["ranking"], list)
    assert "uniform_dFR_floor" in payload


@pytest.mark.slow  # trains kernels on CPU — opt-in only
def test_screen_writes_runs_json(tmp_path, monkeypatch) -> None:
    """The screen persists runs/screen_<date>.json with the configs + ranking."""
    monkeypatch.setenv("QIG_STUDIO_LIVE_PATH", str(tmp_path / "live.json"))
    monkeypatch.chdir(tmp_path)
    # the held-out set + the package must still resolve from the new cwd
    import shutil
    from pathlib import Path as _P

    src_eval = _P(__file__).resolve().parent.parent / "data" / "eval" / "heldout_bpb.json"
    (tmp_path / "data" / "eval").mkdir(parents=True, exist_ok=True)
    shutil.copy(src_eval, tmp_path / "data" / "eval" / "heldout_bpb.json")

    from qig_studio.server import app

    with TestClient(app) as client:
        r = client.post("/screen", json={"layers": 2, "steps": 2, "device": "cpu"})
        assert r.status_code == 200, r.text

    runs = list((tmp_path / "runs").glob("screen_*.json"))
    assert runs, "no runs/screen_*.json written"
    out = json.loads(runs[0].read_text())
    assert len(out["configs"]) == 4
    assert "ranking" in out and "underpowered" in out


@pytest.mark.slow  # trains kernels on CPU — opt-in only
def test_screen_live_channel_reflects_each_config(tmp_path, monkeypatch) -> None:
    """During /screen the shared live channel's current.source reflects the live config (coherent
    telemetry — the whole point). The LAST config to train owns the final current record."""
    live = tmp_path / "live.json"
    monkeypatch.setenv("QIG_STUDIO_LIVE_PATH", str(live))
    from qig_studio.server import app

    with TestClient(app) as client:
        r = client.post("/screen", json={"layers": 2, "steps": 2, "device": "cpu"})
        assert r.status_code == 200, r.text

    payload = json.loads(live.read_text())
    # every record written during the screen carries a neocortex avenue source (NOT an idle target)
    sources = {rec.get("source") for rec in payload.get("recent", [])}
    assert any(s and s.startswith("neocortex-") for s in sources), sources
    assert payload["current"]["source"].startswith("neocortex-")


# --- screen.py eval/ranking unit tests (no torch — DRY: eval-only helpers) ----------------------------


class _FakeTarget:
    """A tiny eval-only stand-in for a TrainingTarget arm (exercises eval_heldout_dFR without torch)."""

    def __init__(self, dfr_per_pos: float, bpb: float) -> None:
        self._dfr = dfr_per_pos
        self._bpb = bpb

    def eval_text_fr(self, text: str):
        n = max(1, len(text) - 1)
        return self._dfr * n, n

    def eval_text_bpb(self, text: str):
        nbytes = max(1, len(text.encode("utf-8")))
        return self._bpb * nbytes, nbytes


def test_eval_heldout_dFR_aggregates() -> None:
    from qig_studio.screen import eval_heldout_dFR

    ev = eval_heldout_dFR(_FakeTarget(0.5, 1.2), ["hello world", "another passage here"])
    assert ev["heldout_dFR"] == pytest.approx(0.5, abs=1e-6)
    assert ev["ce_bpb"] == pytest.approx(1.2, abs=1e-6)
    assert ev["n_pos"] > 0


def test_uniform_floor_near_pi_for_large_vocab() -> None:
    import math

    from qig_studio.screen import uniform_dFR_floor

    assert uniform_dFR_floor(32000) > 3.13  # ≈ π for a large vocab
    assert uniform_dFR_floor(2) == pytest.approx(2.0 * math.acos(math.sqrt(0.5)), abs=1e-5)


def test_rank_ranks_lower_dFR_first_and_detects_underpower() -> None:
    from qig_studio.screen import rank_configs, uniform_dFR_floor

    floor = uniform_dFR_floor(64)
    # one config clearly beats the floor, one is pinned
    good = {"name": "neocortex-qk-2L-geo", "heldout_dFR": floor - 0.5}
    pinned = {"name": "neocortex-geo-2L-lin", "heldout_dFR": floor - 0.001}
    res = rank_configs([pinned, good], floor)
    assert res["ranking"][0] == "neocortex-qk-2L-geo"  # lower d_FR ranks first
    assert res["winner"] == "neocortex-qk-2L-geo"
    assert res["underpowered"] is False

    # all pinned near the floor → under-powered, no winner
    res2 = rank_configs([pinned, {"name": "x", "heldout_dFR": floor - 0.005}], floor)
    assert res2["underpowered"] is True
    assert res2["winner"] is None
