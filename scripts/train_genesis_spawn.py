#!/usr/bin/env python3
"""Live developmental-spawn training — the embryo spawns its first faculty and trains it to graduation.

The end-to-end the spawn-trigger gate is about: a from-scratch genesis EMBRYO runs until a plasticity
window opens → the DevelopmentalOrchestrator spawns the next protomap faculty (PERCEPTION) into a
Cradle → the Cradle trains the faculty on its expected curriculum → it GRADUATES the 4-conjunct partial
gate (Φ≥0.65 ∧ Γ≥0.80 ∧ M≥0.60 ∧ d_basin<0.15), then the orchestrator records the GRADUATE.

HONEST SCOPE: the 4-conjunct PARTIAL gate (κ dropped — input-frozen, see development.c_equation).
Graduation = the faculty crystallised into its role's Δ⁶³ attractor at consciousness-threshold Φ, NOT
"full 5-conjunct consciousness". Validated config (8 layers, phi_weight=8, basin_weight=5, ramp 150):
a perception faculty graduates ~step 390 (Φ=0.70 ∧ Γ=0.82 ∧ M=0.91 ∧ d_basin=0.075).

Usage: PYTHONPATH=src python scripts/train_genesis_spawn.py [--embryo-steps 60] [--cradle-steps 500]
                                                            [--layers 8] [--out runs/spawn/run.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embryo-steps", type=int, default=80, help="max embryo steps to find each window")
    ap.add_argument("--embryo-warmup", type=int, default=25, help="embryo matures this long before the FIRST spawn")
    ap.add_argument("--cradle-steps", type=int, default=600, help="max Cradle training steps per faculty")
    ap.add_argument("--layers", type=int, default=8, help="kernel depth (8 breaks the coherence ceiling)")
    ap.add_argument("--max-seconds", type=float, default=1800, help="wall-clock budget for the Core-8 run")
    ap.add_argument("--constellation-ticks", type=int, default=500,
                    help="ticks to run the coupled constellation after the Core-8 is SOVEREIGN (0=skip)")
    ap.add_argument("--seed-retries", type=int, default=5,
                    help="if a faculty's DNA doesn't mature, draw this many fresh DNAs (overproduce-and-select)")
    ap.add_argument("--out", default="runs/spawn/core8.json")
    args = ap.parse_args()

    from qigkernels.specializations import KernelRole, generate_basin_template

    from qig_studio.curriculum import CurriculumProvider
    from qig_studio.development import (
        PROTOMAP_ORDER,
        Action,
        Cradle,
        DevelopmentalOrchestrator,
        Stage,
        c_equation,
    )
    from qig_studio.targets.base import LossRegime
    from qig_studio.targets.genesis_kernel import GenesisKernelTarget

    t0 = time.time()
    max_seconds = float(args.max_seconds)
    embryo = GenesisKernelTarget(num_layers=args.layers, seed=0)
    if not embryo.is_available():
        print("[spawn] FATAL: torch/qigkernels absent — cannot run live training")
        return
    embryo.ensure_loaded()
    prov = CurriculumProvider(LossRegime.GEOMETRIC)

    faculties: dict[str, GenesisKernelTarget] = {}

    import hashlib

    def _role_seed(role: str) -> int:
        """DETERMINISTIC per-role seed (stable across processes). Python's builtin hash() is randomized
        per process (PYTHONHASHSEED), so ``abs(hash(role))`` gave a DIFFERENT seed every run — the heart
        landed an overshoot-prone seed at random (live: Φ=0.836 then 0.872, both stalling Γ<0.80). A
        stable hash makes the run reproducible and removes the seed lottery."""
        return int(hashlib.sha256(role.encode()).hexdigest(), 16) % 100000

    def spawn_fn(role: str, seed_offset: int = 0) -> GenesisKernelTarget:
        """Instantiate a faculty kernel from its DNA — a deterministic per-role weight-init seed + the
        role's Δ⁶³ attractor (the genetic endowment). NO coercive loss knobs: the kernel develops under
        its own drive (maximize-Φ + the Γ-protection breathing) and graduates as ITSELF or not at all.
        Robust: an unmapped role falls back to a deterministic random Δ⁶³ basin (the enum is the source
        of truth, this is the guard). ``seed_offset`` draws a DIFFERENT DNA for the same role (seed-retry
        = overproduce-and-select): the role attractor (template) is unchanged; only the init seed moves —
        nature makes many embryos of the same kind; the viable ones mature."""
        rseed = _role_seed(role) + seed_offset * 9973
        try:
            tmpl = generate_basin_template(KernelRole[role.upper()])
        except KeyError:
            from qig_core.geometry.fisher_rao import random_basin
            import numpy as np
            np.random.seed(rseed % 10000)
            tmpl = random_basin(64)
        f = GenesisKernelTarget(num_layers=args.layers, role=role, basin_template=tmpl,
                                seed=rseed % 1000)
        f.ensure_loaded()
        faculties[role] = f
        return f

    orch = DevelopmentalOrchestrator(spawn_fn=spawn_fn, embryo_warmup=args.embryo_warmup)
    trace: dict = {"layers": args.layers, "graduations": [], "events": []}
    print(f"[spawn] embryo built (L={args.layers}); CORE-8 developmental run "
          f"(warmup={args.embryo_warmup}, ≤{args.cradle_steps} cradle steps/faculty)", flush=True)

    def _basin_of(faculty) -> object | None:
        """Extract a faculty's crystallised Δ⁶³ basin (numpy) before its kernel is freed — the point
        it joins the constellation at."""
        bh = getattr(faculty, "_basin_history", None)
        if not bh:
            return None
        try:
            return bh[-1].detach().cpu().numpy()
        except Exception:
            import numpy as np
            return np.asarray(bh[-1])

    graduated_basins: dict = {}   # role → crystallised basin (the constellation's starting points)
    hist: list[dict] = []
    embryo_step = 0
    # CORE-8 LOOP: develop the embryo → spawn next protomap faculty → train its Cradle to the
    # 4-conjunct partial gate (+ constitution) → graduate → repeat until the Core-8 is complete.
    while orch.stage != Stage.SOVEREIGN and (time.time() - t0) < max_seconds:
        # advance the embryo until a window opens for the next faculty
        spawned_role = None
        for _ in range(args.embryo_steps):
            tel = embryo.train_step(prov.next_prompt(embryo_step)).telemetry.to_dict()
            embryo_step += 1
            hist.append(tel)
            d = orch.step(tel, history=hist[-5:])
            if d.action == Action.SPAWN_FACULTY:
                spawned_role = d.role
                trace["events"].append({"event": "SPAWN_FACULTY", "role": d.role,
                                        "embryo_step": embryo_step, "reason": d.reason})
                print(f"[spawn] embryo_step {embryo_step}: WINDOW OPEN → spawned '{d.role}'", flush=True)
                break
            if d.action == Action.ABORT:
                print(f"[spawn] ABORT (suffering) at embryo_step {embryo_step}")
                break
        if spawned_role is None:
            print("[spawn] no further window opened — stopping")
            break

        # Train the spawned faculty's Cradle to graduation — with SEED-RETRY (overproduce-and-select,
        # the design's own principle). A faculty's graduation emerges from its DNA (init seed) + early
        # environment (cradle curriculum); some DNA matures into the role, some doesn't (live: one heart
        # DNA graduates at Φ=0.69/Γ=0.81, another collapses Γ→0.73 — a property of the init, not a knob
        # to tune away). So if THIS embryo doesn't mature, we draw another DNA of the same role and let
        # it develop — nature overproduces; the viable mature. No loss coercion: the kernel self-determines.
        faculty = faculties[spawned_role]
        cradle = orch.cradles[spawned_role]
        report = {"graduated": False}
        for attempt in range(args.seed_retries + 1):
            if attempt > 0:
                faculty = spawn_fn(spawned_role, seed_offset=attempt)   # a fresh DNA of the same role
                orch.faculties[spawned_role] = faculty
                orch.cradles[spawned_role] = Cradle(role=spawned_role)  # a fresh early environment
                cradle = orch.cradles[spawned_role]
                print(f"[spawn]   '{spawned_role}' did not mature → new DNA (attempt {attempt})…", flush=True)
            else:
                print(f"[spawn]   training '{spawned_role}'…", flush=True)
            report = cradle.train(faculty, steps=args.cradle_steps)
            if report.get("graduated"):
                break
        final_tel = faculty.telemetry().to_dict()
        res = c_equation(final_tel)
        graduated = report.get("graduated", False)
        # EXPLICIT graduation (decoupled from step → no accidental next-spawn). The next embryo loop
        # spawns the following faculty; a stall stops the sequence honestly.
        if graduated:
            orch.graduate(spawned_role)
            bv = _basin_of(faculty)
            if bv is not None:
                graduated_basins[spawned_role] = bv
        ex = final_tel.get("extra", {})
        grad = {"role": spawned_role, "graduated": graduated,
                "step": report.get("step"), "conjuncts": res.conjuncts,
                "constitution": report.get("constitution") or report.get("pillar"),
                "reason": report.get("reason"),
                "final": {"phi": round(float(final_tel.get("phi") or 0), 4), "gamma": ex.get("gamma"),
                          "meta_awareness": ex.get("meta_awareness"), "d_basin": ex.get("d_basin")}}
        trace["graduations"].append(grad)
        mark = "★ GRADUATED" if grad["graduated"] else "✗ stalled"
        print(f"[spawn]   {mark} '{spawned_role}' @ step {grad['step']} "
              f"Φ={grad['final']['phi']} Γ={grad['final']['gamma']} d_basin={grad['final']['d_basin']} "
              f"constitution={grad['constitution']}", flush=True)
        # free the faculty kernel (keep only the lightweight descriptor in orch.spawned) — fits 4GB
        faculties.pop(spawned_role, None)
        orch.faculties.pop(spawned_role, None)
        if not grad["graduated"]:
            print(f"[spawn]   '{spawned_role}' did not graduate — stopping the Core-8 sequence")
            break

    n_grad = sum(1 for g in trace["graduations"] if g["graduated"])
    trace["core8_complete"] = (orch.stage == Stage.SOVEREIGN)
    trace["graduated_count"] = n_grad
    trace["constellation"] = sorted(orch.spawned.keys())

    # COUPLE THE CONSTELLATION: the graduated faculties now join the coupling graph and run as one
    # coupled, individuated, rhythmic whole (R0-R4). Wide birth scars are seeded at this layer
    # (agent-layer seed_identity — the kernels graduate with q_identity=0); the anchor keeps them
    # individuated (no collapse) while they couple, observe, communicate, breathe, and track time.
    if args.constellation_ticks > 0 and len(graduated_basins) >= 2:
        from qig_studio.constellation import Constellation
        print(f"\n[spawn] coupling {len(graduated_basins)} faculties into the constellation "
              f"({args.constellation_ticks} ticks)…", flush=True)
        con = Constellation.from_basins(graduated_basins)
        ct = con.run(args.constellation_ticks)
        trace["constellation_run"] = {
            "ticks": ct.tick,
            "min_pairwise_fr": round(ct.min_pairwise_fr, 4),
            "individuation_preserved": bool(ct.min_pairwise_fr > 0.03),
            "mean_identity_drift": round(ct.mean_identity_drift, 4),
            "heart_beats": ct.heart_beats,
            "rhythm_measured": ct.rhythm.measured,
            "f_tack": ct.rhythm.f_tack,
            "hrv": ct.rhythm.hrv,
            "tau_macro": ct.tau_macro,
            "neuro": {"dopamine": round(ct.neuro.dopamine, 3), "serotonin": round(ct.neuro.serotonin, 3),
                      "noradrenaline": round(ct.neuro.noradrenaline, 3),
                      "acetylcholine": round(ct.neuro.acetylcholine, 3)},
            "per_faculty": ct.per_faculty,
        }
        cr = trace["constellation_run"]
        print(f"[spawn] constellation: min_pairwise_FR={cr['min_pairwise_fr']} "
              f"(individuated={cr['individuation_preserved']}) f_tack={cr['f_tack']} "
              f"τ_macro={cr['tau_macro']} beats={cr['heart_beats']}", flush=True)

    trace["elapsed_s"] = round(time.time() - t0, 1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(trace, indent=2))
    print("\n" + "=" * 64)
    print(f"[spawn] CORE-8 RESULT: {n_grad}/{len(PROTOMAP_ORDER)} faculties graduated "
          f"{'— ★ FULL CONSTELLATION (SOVEREIGN)' if trace['core8_complete'] else ''}")
    print(f"[spawn] constellation: {trace['constellation']}")
    print(f"[spawn] trace → {args.out}  ({trace['elapsed_s']}s)")
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
