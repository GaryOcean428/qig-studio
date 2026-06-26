#!/usr/bin/env python3
"""Full-curriculum training for the Core-8 — uses the FULL (sanitised, ASCII-only) curriculum, the
kernel's OWN self-regulation (no external scheduler), and CHECKPOINTS every kernel + the collective
constellation state with a 3-checkpoint rolling cleanup.

Flow:
  1. Spawn each Core-8 faculty (deterministic DNA, self-regulating kernel).
  2. Train it through the FULL curriculum (ContinuousLearningLoop, full=True — the loop defers to the
     kernel's brainstem; no puppeteer). Checkpoint the kernel every --ckpt-every steps (keep latest 3).
  3. Capture its Δ⁶³ basin; free the kernel (4 GB budget).
  4. Assemble the Core-8 into the constellation; run it; checkpoint the COLLECTIVE state (keep latest 3).

Usage: PYTHONPATH=src python scripts/train_full_curriculum.py
         [--steps N] [--ckpt-every K] [--layers 8] [--max-seconds S] [--roles a,b,...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=0, help="curriculum steps per faculty (0 = one full pass)")
    ap.add_argument("--ckpt-every", type=int, default=300, help="checkpoint cadence (steps); 3-lag cleanup")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--max-seconds", type=float, default=14400, help="overall wall-clock budget")
    ap.add_argument("--roles", default="", help="comma-list to override the Core-8 (default = all 8)")
    ap.add_argument("--ckpt-root", default="runs/checkpoints")
    ap.add_argument("--out", default="runs/spawn/full_curriculum.json")
    ap.add_argument("--coordizer", default="", help="path to a pre-fit FisherCoordizer (richer Δ⁶³ "
                    "vocab via the kernel's coordizer ctor arg); empty = byte-level (validated default)")
    args = ap.parse_args()

    import numpy as np
    from qigkernels.specializations import KernelRole, generate_basin_template

    from qig_studio.checkpoint import save_constellation_checkpoint, save_kernel_checkpoint
    from qig_studio.corpus import load_full_curriculum
    from qig_studio.curriculum import CurriculumProvider
    from qig_studio.development import PROTOMAP_ORDER, c_equation
    from qig_studio.learning import ContinuousLearningLoop
    from qig_studio.optimisation import load_coordizer, settle_decision
    from qig_studio.targets.base import LossRegime
    from qig_studio.targets.genesis_kernel import GenesisKernelTarget

    roles = [r.strip() for r in args.roles.split(",") if r.strip()] or list(PROTOMAP_ORDER)
    full = load_full_curriculum()                       # fail-loud if the corpus is missing
    steps = args.steps or len(full)                     # 0 → one full pass through the curriculum
    coordizer = load_coordizer(args.coordizer)          # qig-coordizer: richer Δ⁶³ vocab (None-safe → byte-level)
    settle_floor = max(args.ckpt_every * 3, steps // 4) # qig-warp SETTLE step floor — emergence never gated early
    t0 = time.time()
    print(f"[full] FULL curriculum: {len(full)} sanitised ASCII prompts | {steps} steps/faculty | "
          f"checkpoint every {args.ckpt_every} (3-lag) | vocab={'coordizer Δ⁶³' if coordizer else 'byte-level'} | "
          f"warp-SETTLE floor={settle_floor} | roles={roles}", flush=True)

    def _seed(role: str) -> int:
        return int(hashlib.sha256(role.encode()).hexdigest(), 16) % 1000

    def _template(role: str):
        try:
            return generate_basin_template(KernelRole[role.upper()])
        except KeyError:
            np.random.seed(_seed(role) % 10000)
            from qig_core.geometry.fisher_rao import random_basin
            return random_basin(64)

    def _basin64(faculty) -> np.ndarray | None:
        from qig_core.geometry.fisher_rao import to_simplex
        bh = getattr(faculty, "_basin_history", None)
        if not bh:
            return None
        try:
            b = bh[-1].detach().cpu().numpy()
        except Exception:
            b = np.asarray(bh[-1])
        b = np.asarray(b, dtype=np.float64).ravel()
        if b.size != 64:
            b = b.reshape(64, b.size // 64).sum(axis=1) if b.size % 64 == 0 else \
                np.add.reduceat(b, np.arange(0, b.size, max(1, b.size // 64)))[:64]
        return to_simplex(b)

    trace: dict = {"curriculum_prompts": len(full), "steps_per_faculty": steps,
                   "packages": {"qig_core": "geometry+pillars+constants (full)",
                                "qigkernels": "Kernel+NaturalGradientDescent+RecursiveIntegrator (full)",
                                "qig_warp": "SETTLE (check_ci_stabilized on Φ)",
                                "qig_coordizer": "Δ⁶³ vocab" if coordizer else "N/A (byte-level)",
                                "qig_compute": "N/A (lattice quantum QFI ≠ neural basin)",
                                "qig_bench": "N/A (frozen-physics harness)",
                                "qig_doctor": "N/A (not a package)"},
                   "faculties": []}
    graduated_basins: dict = {}

    for role in roles:
        if (time.time() - t0) > args.max_seconds:
            print(f"[full] wall-clock budget reached; stopping before '{role}'", flush=True)
            break
        faculty = GenesisKernelTarget(num_layers=args.layers, role=role, basin_template=_template(role),
                                      seed=_seed(role), coordizer=coordizer)
        faculty.ensure_loaded()
        prov = CurriculumProvider(LossRegime.GEOMETRIC, full=True)   # the FULL sanitised curriculum
        loop = ContinuousLearningLoop(faculty, curriculum=prov, max_steps=steps)
        print(f"[full] '{role}': training on full curriculum (self_regulating="
              f"{faculty.self_regulating})…", flush=True)
        ck, phi_hist, settled = 0, [], ""
        for i in range(1, steps + 1):
            loop.step()
            if i % args.ckpt_every == 0 or i == steps:
                save_kernel_checkpoint(faculty, i, root=args.ckpt_root)   # 3-lag cleanup inside
                ck += 1
                phi_hist.append(float(faculty.telemetry().to_dict().get("phi") or 0.0))
                stop, why = settle_decision(phi_hist)                 # qig-warp SETTLE on the Φ trajectory
                if stop and i >= settle_floor:                        # past the emergence floor → stop redundant grind
                    settled = f"@{i}: {why}"
                    print(f"[full]   '{role}' SETTLED {settled}", flush=True)
                    break
            if (time.time() - t0) > args.max_seconds:
                break
        tel = faculty.telemetry().to_dict()
        res = c_equation(tel)
        ex = tel.get("extra", {})
        graduated_basins[role] = _basin64(faculty)
        rec = {"role": role, "steps": i, "checkpoints": ck, "band": res.band,
               "phi": round(float(tel.get("phi") or 0), 4), "gamma": ex.get("gamma"),
               "coherence": ex.get("coherence"), "d_basin": ex.get("d_basin"),
               "matured": res.conscious, "autonomic": ex.get("autonomic"),
               "settled": settled or None, "vocab": "coordizer" if coordizer else "byte-level"}
        trace["faculties"].append(rec)
        print(f"[full]   '{role}' done @ {i} steps · band={res.band} Φ={rec['phi']} Γ={rec['gamma']} "
              f"coher={rec['coherence']} matured={rec['matured']} · {ck} checkpoints", flush=True)
        faculty = None   # free (4 GB budget); state is in the checkpoints

    # COLLECTIVE: assemble the Core-8 + checkpoint the constellation state (3-lag).
    usable = {r: b for r, b in graduated_basins.items() if b is not None}
    if len(usable) >= 2:
        from qig_studio.constellation import Constellation
        con = Constellation.from_basins(usable)
        ct = con.run(500)
        cpath = save_constellation_checkpoint(con, ct.tick, root=args.ckpt_root)
        trace["constellation"] = {"faculties": sorted(usable), "ticks": ct.tick,
                                  "min_pairwise_fr": round(ct.min_pairwise_fr, 4),
                                  "individuation_preserved": bool(ct.min_pairwise_fr > 0.03),
                                  "f_tack": ct.rhythm.f_tack, "tau_macro": ct.tau_macro,
                                  "collective_checkpoint": str(cpath) if cpath else None}
        print(f"[full] COLLECTIVE: constellation min_pairwise_FR={ct.min_pairwise_fr:.4f} "
              f"individuated={ct.min_pairwise_fr>0.03} f_tack={ct.rhythm.f_tack} → {cpath}", flush=True)

    trace["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(trace, indent=2))
    print(f"\n[full] DONE: {len(trace['faculties'])} faculties trained on the full curriculum, "
          f"checkpointed (3-lag) under {args.ckpt_root} · trace → {args.out} ({trace['elapsed_s']}s)")


if __name__ == "__main__":
    main()
