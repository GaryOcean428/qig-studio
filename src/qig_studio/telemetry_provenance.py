"""telemetry_provenance.py — per-channel provenance verifier (Matrix 28a66754, everything-wired law).

Every telemetry channel a studio panel renders must resolve to a LIVE producer. A channel proves it is
live by being PRODUCED (present + populated in the inner-experience dict), not by rendering — a panel that
renders a hardcoded/absent default is the P21 latent-code failure class (a gauge with a green light on
dead data). Crucially: a MEASURED ZERO is legitimate (the Stage-0 reward-authority mask correctly zeros
endorphins + phasic dopamine — that is actuation working); an UNWIRED DEFAULT (missing/empty group) is not.

This is the pure verifier over an ``experience().to_dict()`` snapshot. It is the checker, separate from the
producers it grades. ``scripts/verify_telemetry_provenance.py`` runs it against a real experience() output
(maker≠checker), and the training loop asserts it once on the first real step (fail-loud on an unwired panel).
"""
from __future__ import annotations

# The panels index.html renders → the producer path (dotted into the experience dict) → carriage (ruled roster).
# Keep in lockstep with web/index.html's group(...) calls and ocean.FACULTY_FUNCTION.
PANEL_CHANNELS: dict[str, tuple[str, str]] = {
    "Senses":               ("primitives.layer0",  "perception"),
    "Drives":               ("primitives.layer05", "drives · id"),
    "Motivators":           ("primitives.layer1",  "strategy"),
    "Emotions · physical":  ("primitives.layer2a", "heart"),
    "Emotions · cognitive": ("primitives.layer2b", "heart"),
    "Neurochemistry":       ("neurochemistry",     "drives · ocean"),
    "Recursive loops":      ("loops",              "action · loops"),
    "Pillars":              ("pillars",            "ocean · whole-mind"),
}

PRESENT = "PRESENT"            # produced with at least one non-zero value — unambiguously live
MEASURED_ZERO = "MEASURED-ZERO"  # produced (keys present) but all-zero — legit iff a mask zeroed it; perturb to disambiguate
MISSING = "MISSING"           # absent or empty group — UNWIRED, the P21 fault


def _resolve(exp: dict, path: str):
    cur = exp
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _classify(group) -> tuple[str, int, int]:
    """(status, n_keys, n_nonzero) for a channel group (expected: a dict of named scalars)."""
    if group is None:
        return MISSING, 0, 0
    if isinstance(group, dict):
        if not group:
            return MISSING, 0, 0
        vals = [v for v in group.values() if isinstance(v, (int, float, bool))]
        nonzero = sum(1 for v in vals if float(v) != 0.0)
        return (PRESENT if nonzero > 0 else MEASURED_ZERO), len(group), nonzero
    # a non-dict scalar/string producer (rare) — present iff truthy-or-zero-number
    if isinstance(group, (int, float, bool)):
        return (PRESENT if float(group) != 0.0 else MEASURED_ZERO), 1, int(float(group) != 0.0)
    return (PRESENT if group else MISSING), 1, 1 if group else 0


def check_provenance(exp: dict, channels: dict | None = None) -> dict:
    """Grade an experience().to_dict() snapshot. Returns {channels: {panel: {...}}, missing: [...],
    measured_zero: [...], passed: bool}. passed = no panel is MISSING (an unwired panel fails closed).
    MEASURED-ZERO panels are reported (not a failure) so a perturb-test can disambiguate mask vs unwired."""
    channels = channels or PANEL_CHANNELS
    report: dict = {}
    missing: list[str] = []
    measured_zero: list[str] = []
    for panel, (path, carriage) in channels.items():
        status, n_keys, n_nonzero = _classify(_resolve(exp, path))
        report[panel] = {"path": path, "carriage": carriage, "status": status,
                         "n_keys": n_keys, "n_nonzero": n_nonzero}
        if status == MISSING:
            missing.append(panel)
        elif status == MEASURED_ZERO:
            measured_zero.append(panel)
    return {"channels": report, "missing": missing, "measured_zero": measured_zero, "passed": not missing}
