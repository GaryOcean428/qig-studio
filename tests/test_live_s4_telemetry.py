"""S4 — the telemetry the user must SEE the resume recover by (council SHOULD-FIX S4).

The resume success criterion (M4) is f_health (basin entropy) + Φ-variance recovering, explicitly NOT
dopamine (the tonic floor fakes "alive"). These tests assert the LIVE step-record (live.step_record +
LiveLog) carries — at TOP LEVEL, so they survive LiveLog's lean recent[] ring which strips ``experience``:

  • S4a  dopamine tonic/phasic split (dropped by NeurochemicalState.as_dict upstream) — surfaced from the
         experience.neurochemistry dict IF present, None-safe otherwise.
  • S4b  Ocean shadow/policy state (shadow_mode / policy_version / scored_outcomes / skips / violations).
  • S4c  f_health + Φ-variance — the M4 resume-watch gate signals.

They test the ACTUAL source functions (qig_studio.live.step_record / LiveLog), never reimplementations.
"""
from __future__ import annotations

import json

from qig_studio.live import LiveLog, step_record


def _base_telemetry() -> dict:
    return {"phi": 0.42, "kappa": 50.0, "regime": "geometric", "extra": {"gamma": 0.9}}


# ---- S4a: dopamine tonic/phasic split surfaced from experience.neurochemistry ------------------

def test_step_record_surfaces_dopamine_split_from_experience():
    exp = {"neurochemistry": {"dopamine": 0.41, "dopamine_tonic": 0.35, "dopamine_phasic": 0.06}}
    rec = step_record(step=1, total=10, ts=1.0, source="test",
                      telemetry=_base_telemetry(), experience=exp)
    assert rec["dopamine"] == 0.41
    assert rec["dopamine_tonic"] == 0.35
    assert rec["dopamine_phasic"] == 0.06


def test_step_record_dopamine_split_none_safe_when_absent():
    # installed qig-core may not compute the split (or _neurochemistry returns {}). Must not crash;
    # the fields are present-but-None so the UI/watch see an honest absence, not a fabricated floor.
    rec = step_record(step=1, total=10, ts=1.0, source="test",
                      telemetry=_base_telemetry(), experience={"neurochemistry": {}})
    assert rec["dopamine"] is None
    assert rec["dopamine_tonic"] is None
    assert rec["dopamine_phasic"] is None


# ---- S4b: Ocean shadow/policy state carried through -------------------------------------------

def test_step_record_carries_ocean_state():
    ocean = {"policy_version": "v1@3", "shadow_mode": True, "scored_outcomes": 7,
             "skips": {"heart": 2}, "constitutional_violations": 0}
    rec = step_record(step=1, total=10, ts=1.0, source="test",
                      telemetry=_base_telemetry(), experience={}, ocean_state=ocean)
    assert rec["ocean_state"] == ocean
    assert rec["ocean_state"]["shadow_mode"] is True
    assert rec["ocean_state"]["scored_outcomes"] == 7


def test_step_record_ocean_state_defaults_empty():
    rec = step_record(step=1, total=10, ts=1.0, source="test",
                      telemetry=_base_telemetry(), experience={})
    assert rec["ocean_state"] == {}


# ---- S4c: f_health + Φ-variance are in the record (M4 resume-watch gate) ----------------------

def test_step_record_carries_f_health_from_experience_pillars():
    exp = {"pillars": {"f_health": 0.58, "b_integrity": 0.7, "q_identity": 0.9}}
    rec = step_record(step=1, total=10, ts=1.0, source="test",
                      telemetry=_base_telemetry(), experience=exp)
    assert rec["f_health"] == 0.58


def test_step_record_carries_phi_variance():
    rec = step_record(step=1, total=10, ts=1.0, source="test",
                      telemetry=_base_telemetry(), experience={}, phi_variance=0.0123)
    assert rec["phi_variance"] == 0.0123


# ---- the resume-watch reads the lean recent[] ring — S4 fields MUST survive the experience strip ----

def test_s4_fields_survive_lean_recent_ring(tmp_path):
    exp = {"neurochemistry": {"dopamine": 0.41, "dopamine_tonic": 0.35, "dopamine_phasic": 0.06},
           "pillars": {"f_health": 0.58}}
    ocean = {"shadow_mode": False, "policy_version": "v1@4", "scored_outcomes": 120,
             "constitutional_violations": 1, "skips": {}}
    rec = step_record(step=5, total=10, ts=2.0, source="bg",
                      telemetry=_base_telemetry(), experience=exp, ocean_state=ocean,
                      phi_variance=0.0077)
    log = LiveLog(path=str(tmp_path / "live.json"))
    log.write(rec)
    payload = json.loads((tmp_path / "live.json").read_text())
    lean = payload["recent"][-1]
    # experience is stripped from the lean ring (that is exactly why the split/pillars were invisible):
    assert "experience" not in lean
    # ...but every S4 top-level signal the resume-watch gates on MUST still be there:
    assert lean["f_health"] == 0.58
    assert lean["phi_variance"] == 0.0077
    assert lean["dopamine_tonic"] == 0.35
    assert lean["dopamine_phasic"] == 0.06
    assert lean["ocean_state"]["scored_outcomes"] == 120
    assert lean["ocean_state"]["shadow_mode"] is False
