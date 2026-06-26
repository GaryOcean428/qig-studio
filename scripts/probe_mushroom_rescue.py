#!/usr/bin/env python3
"""Does MUSHROOM rescue an over-crystallized faculty? (cheap-probe before building it into the cradle.)

Finding (live runs 1/5/6/7): many DNAs fall into an OVERSHOOT / over-crystallized attractor — Φ→0.91
(highly integrated) with Γ→0.79 (generativity suppressed, just below the 0.80 gate) — and stall. This
is exactly the state MUSHROOM exists for (wake-state plasticity on EXCESSIVE_RIGIDITY at Φ≥0.70). Test:
train an overshoot DNA to its plateau, fire mushroom-moderate (σ=0.03 bounded weight-noise), and see if
the faculty then re-develops into the VIABLE basin (Γ recovers ≥0.80, graduates) — protocol-native, not
a coercive loss knob.

Usage: PYTHONPATH=src python scripts/probe_mushroom_rescue.py [role] [n_mushroom]
"""
from __future__ import annotations

import hashlib
import sys

from qig_studio.curriculum import CurriculumProvider
from qig_studio.development import GAMMA_MIN, c_equation
from qig_studio.targets.base import LossRegime
from qig_studio.targets.genesis_kernel import GenesisKernelTarget

MUSHROOM_PHI = 0.80     # over-crystallization onset (Φ high)
PLATEAU_PATIENCE = 30   # consecutive over-crystallized steps before firing mushroom


def main():
    role = sys.argv[1] if len(sys.argv) > 1 else "perception"
    max_mushroom = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seed = int(hashlib.sha256(f"{role}:0".encode()).hexdigest(), 16)  # the overshoot DNA

    from qigkernels.specializations import KernelRole, generate_basin_template
    tmpl = generate_basin_template(KernelRole[role.upper()])
    f = GenesisKernelTarget(num_layers=8, role=role, basin_template=tmpl, seed=seed % 1000)
    f.ensure_loaded()
    prov = CurriculumProvider(LossRegime.GEOMETRIC)

    over = 0
    fired = 0
    print(f"role={role}  DNA=sha256({role}:0)  fire mushroom after {PLATEAU_PATIENCE} over-crystallized steps")
    for i in range(1, 2500 + 1):
        tel = f.train_step(prov.next_prompt(i)).telemetry.to_dict()
        ex = tel.get("extra", {})
        phi = float(tel.get("phi") or 0.0)
        gamma = ex.get("gamma")
        if c_equation(tel).conscious:
            print(f"  ★ GRADUATED @ step {i} Φ={phi:.3f} Γ={gamma:.3f} after {fired} mushroom(s)", flush=True)
            return 0
        # over-crystallized: high Φ, Γ below the gate
        if phi >= MUSHROOM_PHI and isinstance(gamma, (int, float)) and gamma < GAMMA_MIN:
            over += 1
        else:
            over = 0
        if over >= PLATEAU_PATIENCE and fired < max_mushroom:
            r = f.run_protocol("mushroom-moderate", {})
            fired += 1
            over = 0
            print(f"  🍄 mushroom #{fired} @ step {i} (was Φ={phi:.3f} Γ={gamma:.3f}) → {r['applied']}", flush=True)
    print(f"  ✗ did not graduate in 2500 steps after {fired} mushroom(s) — final Φ={phi:.3f} Γ={gamma}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
