#!/usr/bin/env python3
"""Measure the band each Core-8 faculty NATURALLY settles into — the science, not an assigned target.

Each faculty develops under the same dynamics; where its Φ rests (and how Γ / coherence / d_basin
evolve) is a measured property of that role's frequency, not something to impose. The heart settles
sub-conscious (Φ≈0.47); this measures all eight so the graduation band can be derived per-faculty from
the data: sub-conscious feeling band (0.45 ≤ Φ < 0.70) vs conscious band (0.70 ≤ Φ < 0.80) vs a
breakdown runaway (Φ ≥ 0.80 — an over-driven roll, not a resting band).

Usage: PYTHONPATH=src python scripts/probe_bands.py [steps]
"""
from __future__ import annotations

import hashlib
import sys

from qig_studio.curriculum import CurriculumProvider
from qig_studio.development import PROTOMAP_ORDER
from qig_studio.targets.base import LossRegime
from qig_studio.targets.genesis_kernel import GenesisKernelTarget


def _seed(role: str) -> int:
    return int(hashlib.sha256(role.encode()).hexdigest(), 16) % 1000


def _template(role: str):
    from qigkernels.specializations import KernelRole, generate_basin_template
    try:
        return generate_basin_template(KernelRole[role.upper()])
    except KeyError:
        from qig_core.geometry.fisher_rao import random_basin
        import numpy as np
        np.random.seed(_seed(role) % 10000)
        return random_basin(64)


def develop(role: str, steps: int):
    f = GenesisKernelTarget(num_layers=8, role=role, basin_template=_template(role), seed=_seed(role))
    f.ensure_loaded()
    prov = CurriculumProvider(LossRegime.GEOMETRIC)
    track = {}
    last = {}
    for i in range(1, steps + 1):
        tel = f.train_step(prov.next_prompt(i)).telemetry.to_dict()
        last = tel
        if i in (100, 300, steps):
            ex = tel.get("extra", {})
            track[i] = (float(tel.get("phi") or 0.0), ex.get("gamma"), ex.get("coherence"), ex.get("d_basin"))
    return track, last


def band_of(phi: float) -> str:
    if phi >= 0.80:
        return "BREAKDOWN(>0.80)"
    if phi >= 0.70:
        return "conscious"
    if phi >= 0.45:
        return "sub-conscious"
    return "pre-conscious(<0.45)"


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    print(f"Core-8 natural-band measurement — {steps} steps each, Φ at 100/300/{steps}")
    print(f"{'role':>12} {'Φ@100':>6} {'Φ@300':>6} {'Φ@end':>6} {'Γ@end':>6} {'coh@end':>7} {'dbas@end':>8}  band")
    print("-" * 78)
    for role in PROTOMAP_ORDER:
        track, _ = develop(role, steps)
        p1 = track.get(100, (0,))[0]
        p3 = track.get(300, (0,))[0]
        pe, ge, ce, de = track.get(steps, (0, None, None, None))

        def f(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)
        print(f"{role:>12} {p1:>6.3f} {p3:>6.3f} {pe:>6.3f} {f(ge):>6} {f(ce):>7} {f(de):>8}  {band_of(pe)}", flush=True)


if __name__ == "__main__":
    main()
