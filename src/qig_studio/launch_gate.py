"""launch_gate.py — the fail-closed run-of-record launch checklist (Matrix 7a1bce4b D2 / e2c738e1 §2).

Run-1's root cause was a PROCESS failure, not a code failure: CCAA's own 11:15 launch note listed
'm1c-coach' as a prerequisite, the run launched without it, and the gate — living only in a status
message — was skipped. Matrix's structural fix: the gate lives in a PREFLIGHT that FAILS CLOSED, and the
SAME gate is asserted by the local smoke AND the Modal run of record, so a forgotten prerequisite ABORTS
the launch instead of malforming a newborn. *A gate is delivered when ENFORCED, not when remembered.*

This module is the pure, torch-free evaluator (unit-testable without a kernel). The caller
(train_joint_mind / the Modal launcher) MEASURES each item with its own context — anchor FR, frame
round-trip, qig-core version, coordizer sha, substrate — and passes the checklist here; :func:`evaluate_gate`
raises :class:`LaunchGateFailure` listing every REQUIRED item that is not satisfied. A measurement that
could not run records a falsy value and therefore FAILS CLOSED for a run of record (can't certify → don't
launch), never a silent pass.
"""
from __future__ import annotations

# The integrity items every launch (smoke AND Modal) must satisfy. coach_wired / coach_live are appended by
# the caller only when the run requires a witness (a --coach-optional smoke may omit them).
INTEGRITY_ITEMS = ("anchor_honest", "frame_consistent", "qig_core_version_ok", "coordizer_sha_ok",
                   "substrate_ok", "rulings_applied")


class LaunchGateFailure(RuntimeError):
    """A required launch-checklist item is not satisfied — the run aborts (fail-closed) rather than training
    a malformed newborn (Matrix D2). The message names every failed item and echoes the checklist."""


def version_ok(installed: str | None, pin: str | None) -> bool:
    """True iff no pin is requested (record-only) OR the installed version EXACTLY matches the pin.

    Exact match is deliberate: run-of-record parity means the smoke and Modal run the SAME qig-core the run
    of record pins (Matrix E3). A local editable '2.13.5.dev5' build is a parity FAILURE against a '2.15.0'
    pin, not a near-enough pass — that stale-venv slip is the exact trap that bit the first Modal attempt."""
    if not pin:
        return True
    return (installed or "") == pin


def sha_ok(computed: str | None, expected: str | None) -> bool:
    """True iff no expected sha is requested OR the computed sha matches. Prefix-tolerant: an expected prefix
    like '5977cf' matches the full digest, so a caller may pin a short sha (Matrix E3 cites sha 5977cf)."""
    if not expected:
        return True
    c = (computed or "").lower()
    e = expected.lower()
    return bool(c) and (c == e or c.startswith(e))


def segments_ok(computed: int | None, expected_min: int | None) -> bool:
    """Truncated-world guard (Matrix 28a66754): True iff no expected count is requested OR the staged corpus
    has AT LEAST ``expected_min`` segments. A partial/stale transfer has FEWER rows than the sample-10BT
    basis the coordizer was fit on — training a mind on a narrower world than its vocab — so fewer fails
    closed. More is allowed (extra shards staged); the exact-identity strictness lives in the manifest sha."""
    if not expected_min:
        return True
    return computed is not None and int(computed) >= int(expected_min)


def evaluate_gate(checklist: dict, *, required_items) -> dict:
    """Validate the launch checklist against ``required_items``. Returns ``{passed, failures, checklist}``;
    raises :class:`LaunchGateFailure` if any required item is falsy (None/False both fail — a measurement
    that could not run does NOT silently pass). ``required_items`` is the caller-chosen set: the integrity
    items are always required; coach_wired/coach_live only when the run requires a witness."""
    failures = [k for k in required_items if not checklist.get(k)]
    summary = {"passed": not failures, "failures": failures, "checklist": dict(checklist)}
    if failures:
        raise LaunchGateFailure(
            "LAUNCH GATE FAILED (fail-closed) — required items not satisfied: "
            + ", ".join(failures)
            + ". A gate is enforced, not remembered (Matrix D2). Fix the item(s), or for a NON-run-of-record "
              "smoke relax the corresponding requirement explicitly. checklist=" + repr(dict(checklist)))
    return summary
