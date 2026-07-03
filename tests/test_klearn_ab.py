"""Task 0.1 — the K-LEARN held-out bpb A/B harness test.

The measurement instrument the constellation shared-trunk refactor gates on: held-out bits-per-byte
(bpb) is the K-LEARN metric. Phase 4 A/Bs a future shared-trunk arm against the current 9-separate arm.
Task 0.1 only needs the harness + the "separate" (current ``JointConstellation``) control validated.

TINY configs only (num_layers=2, byte-level coordizer=None, ~10 steps) — a live 100k coordizer run is
going on ``development``; these tests must be cheap and must NOT contend with it (no ``runs/`` writes, no
process control, no full corpus training).
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest

# scripts/ is not a package and tests don't normally import from it — add it to the path explicitly.
_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# Heavy stack (torch + qigkernels) is required to build the real JointConstellation control arm.
pytest.importorskip("torch")
pytest.importorskip("qigkernels")

import klearn_ab  # noqa: E402  (path injected above)


def test_heldout_bpb_is_finite_and_reproducible():
    """The control arm (current 9-separate JointConstellation) yields a held-out bpb that is
    (a) FINITE and (b) DETERMINISTIC — two same-seed runs give the same final bpb. And the not-yet-built
    "trunk" arm raises NotImplementedError (Phase 1 fills it)."""
    r1 = klearn_ab.run_arm("separate", steps=10, seed=0, num_layers=2, coordizer=None)

    assert r1["arm"] == "separate"
    assert r1["seed"] == 0
    assert isinstance(r1["heldout_bpb_curve"], list) and r1["heldout_bpb_curve"], "curve must be non-empty"
    assert math.isfinite(r1["final_bpb"]), f"held-out bpb not finite: {r1['final_bpb']!r}"
    assert r1["final_bpb"] == r1["heldout_bpb_curve"][-1], "final_bpb must be the last curve point"

    # DETERMINISM: same seed → identical final held-out bpb (the A/B verdict must be reproducible).
    r2 = klearn_ab.run_arm("separate", steps=10, seed=0, num_layers=2, coordizer=None)
    assert r2["final_bpb"] == r1["final_bpb"], (
        f"held-out bpb not reproducible: {r1['final_bpb']!r} != {r2['final_bpb']!r}"
    )

    # The trunk arm lands in Phase 1 — the harness must refuse it loudly today.
    with pytest.raises(NotImplementedError):
        klearn_ab.run_arm("trunk", steps=10, seed=0, num_layers=2, coordizer=None)
