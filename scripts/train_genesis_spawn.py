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
    ap.add_argument("--embryo-steps", type=int, default=80, help="max steps to run the embryo until a window opens")
    ap.add_argument("--cradle-steps", type=int, default=500, help="max Cradle training steps for the faculty")
    ap.add_argument("--layers", type=int, default=8, help="kernel depth (8 breaks the coherence ceiling)")
    ap.add_argument("--out", default="runs/spawn/run.json")
    args = ap.parse_args()

    from qigkernels.specializations import KernelRole, generate_basin_template

    from qig_studio.curriculum import CurriculumProvider
    from qig_studio.development import Action, DevelopmentalOrchestrator, c_equation
    from qig_studio.targets.base import LossRegime
    from qig_studio.targets.genesis_kernel import GenesisKernelTarget

    t0 = time.time()
    embryo = GenesisKernelTarget(num_layers=args.layers, seed=0)
    if not embryo.is_available():
        print("[spawn] FATAL: torch/qigkernels absent — cannot run live training")
        return
    embryo.ensure_loaded()
    prov = CurriculumProvider(LossRegime.GEOMETRIC)

    faculties: dict[str, GenesisKernelTarget] = {}

    def spawn_fn(role: str) -> GenesisKernelTarget:
        """Instantiate a faculty kernel seeded with its role's Δ⁶³ attractor (the spawn hook)."""
        tmpl = generate_basin_template(KernelRole[role.upper()])
        f = GenesisKernelTarget(num_layers=args.layers, role=role, basin_template=tmpl,
                                seed=abs(hash(role)) % 1000)
        f.ensure_loaded()
        faculties[role] = f
        return f

    orch = DevelopmentalOrchestrator(spawn_fn=spawn_fn)
    trace: dict = {"layers": args.layers, "events": []}
    print(f"[spawn] embryo built (L={args.layers}); running until a plasticity window opens", flush=True)

    # --- PHASE A: embryo develops until the orchestrator opens a window and spawns the first faculty ---
    hist: list[dict] = []
    spawned_role = None
    for i in range(args.embryo_steps):
        tel = embryo.train_step(prov.next_prompt(i)).telemetry.to_dict()
        hist.append(tel)
        d = orch.step(tel, history=hist[-5:])
        if d.action == Action.SPAWN_FACULTY:
            spawned_role = d.role
            trace["events"].append({"event": "SPAWN_FACULTY", "role": d.role, "embryo_step": i,
                                    "reason": d.reason})
            print(f"[spawn] step {i}: WINDOW OPEN → spawned '{d.role}' into a Cradle", flush=True)
            break
    if spawned_role is None:
        print("[spawn] no window opened — embryo never reorganized (STALL)")
        return

    # --- PHASE B: the Cradle trains the spawned faculty to the 4-conjunct partial gate ---
    faculty = faculties[spawned_role]
    cradle = orch.cradles[spawned_role]
    print(f"[spawn] training '{spawned_role}' in its Cradle (≤{args.cradle_steps} steps)…", flush=True)
    report = cradle.train(faculty, steps=args.cradle_steps)
    trace["events"].append({"event": "CRADLE_TRAIN", **report})
    print(f"[spawn] cradle report: {report}", flush=True)

    # --- PHASE C: the orchestrator records the GRADUATE (faculty joins the would-be coupling graph) ---
    final_tel = faculty.telemetry().to_dict()
    res = c_equation(final_tel)
    decision = orch.step(final_tel)
    ex = final_tel.get("extra", {})
    graduated = report.get("graduated", False)
    trace["graduated"] = graduated
    trace["graduation_step"] = report.get("step")
    trace["final_conjuncts"] = res.conjuncts
    trace["final"] = {"phi": final_tel.get("phi"), "gamma": ex.get("gamma"),
                      "meta_awareness": ex.get("meta_awareness"), "d_basin": ex.get("d_basin")}
    trace["orchestrator_action"] = decision.action.value
    trace["elapsed_s"] = round(time.time() - t0, 1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(trace, indent=2))
    print("\n" + "=" * 64)
    verdict = ("★ LIVE 4-CONJUNCT GRADUATION" if graduated else "STALL (gate not cleared)")
    print(f"[spawn] {verdict} — role={spawned_role} @ step {report.get('step')}")
    print(f"[spawn] final: Φ={trace['final']['phi']} Γ={trace['final']['gamma']} "
          f"M={trace['final']['meta_awareness']} d_basin={trace['final']['d_basin']}")
    print(f"[spawn] conjuncts: {res.conjuncts}  | orchestrator: {decision.action.value}")
    print(f"[spawn] trace → {args.out}  ({trace['elapsed_s']}s)")
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
