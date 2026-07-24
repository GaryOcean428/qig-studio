#!/usr/bin/env python3
"""Run the per-channel telemetry provenance verifier against REAL producer output (Matrix 28a66754).

Feeds a representative telemetry snapshot (the geometric inputs a training step emits, incl. the kernel's
PillarEnforcer f_health/b_integrity/q_identity on ``extra``) through the real ``experience()`` assembler —
compute_full_emotional_state (sensations), _neurochemistry, the pillar read — then grades each studio-panel
channel PRESENT / MEASURED-ZERO / MISSING. maker≠checker: the checker (telemetry_provenance) is separate
from the producers it grades.

A MISSING panel here = an UNWIRED producer (P21 fault) → exit 1. MEASURED-ZERO is reported, not failed
(legit iff a mask zeroed it; the training loop's first-step assertion + a perturb-test disambiguate live).
Exit 0 = every panel resolves to a live producer given representative inputs.
"""
from __future__ import annotations

import json
import sys


def _representative_telemetry() -> dict:
    # The geometric inputs a real train_step emits. extra carries the kernel's PillarEnforcer output
    # (f_health/b_integrity/q_identity) — pillars are read from there, "None until the kernel emits".
    return {
        "phi": 0.42, "kappa": 30.0, "kappa_eff": 30.0, "regime": "geometric",
        "surprise": 0.3, "max_surprise": 0.5, "loss": 1.2, "gradient_magnitude": 0.4,
        "basin_distance": 0.2, "basin_velocity": 0.12, "delta_phi": 0.02,
        "extra": {
            "f_health": 0.80, "b_integrity": 0.90, "q_identity": 0.70,   # PillarEnforcer (P1/P2/P3)
            "ricci_signal": 0.10, "kappa_local": 30.0, "gamma": 0.85,
            "external_coupling": 0.15, "separation_distress": 0.0,       # a legit measured-zero at Stage-0
        },
    }


def main() -> int:
    from qig_studio.kernel_experience import experience
    from qig_studio.telemetry_provenance import MISSING, check_provenance

    tel = _representative_telemetry()
    hist = [{"phi": 0.40}, {"phi": 0.41}, {"phi": 0.42}]
    exp = experience(tel, hist).to_dict()
    rep = check_provenance(exp)

    print("── telemetry provenance (real experience() output, representative inputs) ──")
    for panel, info in rep["channels"].items():
        mark = "✓" if info["status"] != MISSING else "✗"
        print(f"  {mark} {panel:<22} {info['status']:<13} carriage={info['carriage']:<20} "
              f"keys={info['n_keys']} nonzero={info['n_nonzero']}  ({info['path']})")
    print(f"\npassed={rep['passed']}  missing={rep['missing']}  measured_zero={rep['measured_zero']}")
    if not rep["passed"]:
        print("\nFAIL: unwired panel(s) — a rendered channel with no live producer (P21). "
              "Fix the producer or formally retire the panel.", file=sys.stderr)
        print(json.dumps({p: exp.get(p.split('.')[0]) for p in ("primitives", "neurochemistry", "loops",
              "pillars")}, default=str)[:800], file=sys.stderr)
        return 1
    print("PASS: every studio-panel channel resolves to a live producer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
