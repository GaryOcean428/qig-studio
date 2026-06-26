#!/usr/bin/env python3
"""Fast per-faculty DNA-viability probe (cheap-probe-before-expensive-spend).

Some roles (heart, memory) sit right at the Γ=0.80 edge, so whether a given DNA (weight-init seed)
matures into the role depends on the init. This draws N independent DNAs for a role and reports each
one's outcome under the NATURAL baseline loss (no coercive knobs) — measuring the viable-DNA rate so we
know whether overproduce-and-select (seed-retry) can reach it, and whether failures cluster at
OVERSHOOT (Φ high, Γ<0.80) or undershoot.

Usage: PYTHONPATH=src python scripts/probe_faculty_dna.py <role> [n_dna] [cradle_steps]
"""
from __future__ import annotations

import hashlib
import sys

from qig_studio.curriculum import CurriculumProvider
from qig_studio.development import c_equation
from qig_studio.targets.base import LossRegime
from qig_studio.targets.genesis_kernel import GenesisKernelTarget


def _dna_seed(role: str, k: int) -> int:
    return int(hashlib.sha256(f"{role}:{k}".encode()).hexdigest(), 16) % 100000


def probe_one(role: str, k: int, steps: int):
    from qigkernels.specializations import KernelRole, generate_basin_template
    try:
        tmpl = generate_basin_template(KernelRole[role.upper()])
    except KeyError:
        from qig_core.geometry.fisher_rao import random_basin
        import numpy as np
        np.random.seed(_dna_seed(role, k) % 10000)
        tmpl = random_basin(64)
    f = GenesisKernelTarget(num_layers=8, role=role, basin_template=tmpl, seed=_dna_seed(role, k) % 1000)
    f.ensure_loaded()
    prov = CurriculumProvider(LossRegime.GEOMETRIC)
    for i in range(1, steps + 1):
        tel = f.train_step(prov.next_prompt(i)).telemetry.to_dict()
        if c_equation(tel).conscious:
            ex = tel.get("extra", {})
            return True, i, float(tel.get("phi") or 0), ex.get("gamma")
    ex = tel.get("extra", {})
    return False, steps, float(tel.get("phi") or 0), ex.get("gamma")


def main():
    role = sys.argv[1] if len(sys.argv) > 1 else "memory"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 800
    print(f"DNA viability probe — role='{role}', {n} independent DNAs, ≤{steps} steps each")
    print(f"{'DNA#':>4} {'GRAD':>5} {'step':>5} {'Phi':>7} {'Gamma':>7}  outcome")
    print("-" * 52)
    viable = 0
    for k in range(n):
        grad, step, phi, gamma = probe_one(role, k, steps)
        gs = f"{gamma:.4f}" if isinstance(gamma, (int, float)) else str(gamma)
        if grad:
            viable += 1
            tag = "★ matured"
        elif isinstance(gamma, (int, float)) and phi > 0.80:
            tag = "overshoot (Φ high, Γ<gate)"
        else:
            tag = "did not reach gate"
        print(f"{k:>4} {('YES' if grad else 'no'):>5} {step:>5} {phi:>7.4f} {gs:>7}  {tag}", flush=True)
    print("-" * 52)
    print(f"viable-DNA rate: {viable}/{n} = {viable/n:.0%}")


if __name__ == "__main__":
    main()
