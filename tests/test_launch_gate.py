"""Fail-closed launch-gate evaluator (Matrix D2) — a gate is ENFORCED, not remembered.

The evaluator is pure/torch-free: it validates a caller-measured checklist and raises LaunchGateFailure
naming every required item that is not satisfied. A measurement that could not run (None) FAILS CLOSED for a
run of record — never a silent pass. Version parity is EXACT (a dev build fails a release pin, Matrix E3).
"""
import pytest

from qig_studio.launch_gate import (INTEGRITY_ITEMS, LaunchGateFailure, evaluate_gate, sha_ok, version_ok)


def test_version_ok_exact_and_recordonly():
    assert version_ok("2.15.0", "2.15.0") is True
    assert version_ok("2.15.0", None) is True            # no pin → record-only, passes
    assert version_ok("2.13.5.dev5+ge6b1b0b", "2.15.0") is False   # E3: dev build fails a release pin
    assert version_ok("2.15.1", "2.15.0") is False
    assert version_ok(None, "2.15.0") is False


def test_sha_ok_exact_prefix_and_recordonly():
    full = "5977cf12ab34cd56ef"
    assert sha_ok(full, None) is True                    # no expected → passes
    assert sha_ok(full, full) is True
    assert sha_ok(full, "5977cf") is True                # prefix pin matches full digest
    assert sha_ok(full, "deadbeef") is False
    assert sha_ok(None, "5977cf") is False               # can't verify → fails


def _full_checklist(**overrides):
    base = {k: True for k in INTEGRITY_ITEMS}
    base.update({"coach_wired": True, "coach_live": True})
    base.update(overrides)
    return base


def test_all_required_true_passes():
    req = INTEGRITY_ITEMS + ("coach_wired", "coach_live")
    summary = evaluate_gate(_full_checklist(), required_items=req)
    assert summary["passed"] is True and summary["failures"] == []


def test_a_false_required_item_fails_closed():
    req = INTEGRITY_ITEMS + ("coach_wired",)
    with pytest.raises(LaunchGateFailure) as ei:
        evaluate_gate(_full_checklist(frame_consistent=False), required_items=req)
    assert "frame_consistent" in str(ei.value)


def test_none_measurement_fails_closed_not_silent_pass():
    req = INTEGRITY_ITEMS
    with pytest.raises(LaunchGateFailure) as ei:
        evaluate_gate(_full_checklist(anchor_honest=None), required_items=req)
    assert "anchor_honest" in str(ei.value)


def test_non_required_false_item_does_not_fail():
    # coach_live False but a smoke omits it from required_items → must NOT raise.
    req = INTEGRITY_ITEMS + ("coach_wired",)
    summary = evaluate_gate(_full_checklist(coach_live=False), required_items=req)
    assert summary["passed"] is True


def test_failure_message_lists_all_failed_items():
    req = INTEGRITY_ITEMS
    with pytest.raises(LaunchGateFailure) as ei:
        evaluate_gate(_full_checklist(anchor_honest=False, coordizer_sha_ok=False), required_items=req)
    msg = str(ei.value)
    assert "anchor_honest" in msg and "coordizer_sha_ok" in msg
