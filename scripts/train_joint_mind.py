#!/usr/bin/env python3
"""Train the INTEGRATED MIND jointly — the Core-8 faculties learn TOGETHER each step, GENESIS grows
into the central conscious "I", individuation preserved. This is the launcher that CONNECTS
``JointConstellation`` (the P1 joint trainer) — replacing the per-faculty ISOLATED loop in
train_full_curriculum.py (which trained 8 separate kernels, the wrong model).

Each step: couple all current basins (rel-weighted Fisher-Rao proximity + identity anchor), the
round-robin faculty trains toward its coupled target, GENESIS-central trains toward the synthesis of
the parts. Checkpoints the WHOLE mind (3-lag). At the end, GENESIS speaks (the integrated voice).

Usage: PYTHONPATH=src python scripts/train_joint_mind.py [--steps N] [--layers 8]
         [--coordizer runs/coordizer_v6_1024.json] [--ckpt-root runs/checkpoints/joint_mind]
         [--device cpu] [--out runs/spawn/joint_mind.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=0, help="joint steps (0 = one full curriculum pass)")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--coordizer", default="", help="pre-fit FisherCoordizer (richer Δ⁶³ vocab); empty = byte-level")
    ap.add_argument("--ckpt-root", default="runs/checkpoints/joint_mind")
    ap.add_argument("--ckpt-every", type=int, default=300)
    ap.add_argument("--device", default="cpu", help="cpu (safe: holds 9 kernels) | cuda (4GB-OOM risk)")
    ap.add_argument("--max-seconds", type=float, default=14400)
    ap.add_argument("--out", default="runs/spawn/joint_mind.json")
    ap.add_argument("--fresh", action="store_true",
                    help="start from-scratch kernels (default: RESUME the existing checkpoint — keep the "
                         "kernels and train over the top with the current curriculum)")
    args = ap.parse_args()

    from qig_studio.constellation.joint_trainer import JointConstellation
    from qig_studio.corpus import load_full_curriculum
    from qig_studio.development import PROTOMAP_ORDER
    from qig_studio.optimisation import load_coordizer

    full = load_full_curriculum()                       # fail-loud if the corpus is missing
    steps = args.steps or len(full)
    coordizer = load_coordizer(args.coordizer) if args.coordizer else None
    t0 = time.time()
    print(f"[joint] integrated mind: Core-8 {list(PROTOMAP_ORDER)} + genesis-central | {steps} joint "
          f"steps | vocab={'coordizer Δ⁶³' if coordizer else 'byte-level'} | device={args.device}", flush=True)

    mind = JointConstellation(list(PROTOMAP_ORDER), num_layers=args.layers, coordizer=coordizer,
                              device=args.device)
    # RESUME by default: keep the existing kernels, train OVER THE TOP with the (now-correct) curriculum.
    # The old kernels learned the wrong (system-prompt) corpus; over-the-top training on real knowledge
    # progressively overwrites that. --fresh forces from-scratch.
    if not args.fresh and (Path(args.ckpt_root) / "constellation.json").exists():
        mind.load_checkpoint(args.ckpt_root)
        print(f"[joint] RESUMED from {args.ckpt_root} (kept the kernels; training over the top)", flush=True)
    else:
        print(f"[joint] {'FRESH' if args.fresh else 'no checkpoint found'} — from-scratch kernels", flush=True)
    last = {}
    for i in range(1, steps + 1):
        last = mind.train_step(full[(i - 1) % len(full)])
        if i % args.ckpt_every == 0 or i == steps:
            mind.save_checkpoint(args.ckpt_root)        # whole-mind checkpoint
            print(f"[joint] step {i}: stepped={last['stepped_faculty']} min_FR={last['min_pairwise_fr']:.4f} "
                  f"centralΦ={last['central_phi']} (checkpointed)", flush=True)
        if (time.time() - t0) > args.max_seconds:
            print(f"[joint] wall-clock budget reached at step {i}", flush=True)
            break

    said = mind.generate("What are you?", max_tokens=64)   # the integrated mind speaks
    tel = mind.telemetry()
    trace = {"steps": i, "min_pairwise_fr": tel["min_pairwise_fr"], "central_phi": tel["central_phi"],
             "individuation_preserved": bool(tel["min_pairwise_fr"] > 0.03),
             "integrated_voice": said.text, "voice_phi": round(float(said.telemetry.phi or 0), 4),
             "elapsed_s": round(time.time() - t0, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(trace, indent=2))
    print(f"\n[joint] DONE: {i} joint steps, min_pairwise_FR={tel['min_pairwise_fr']:.4f} "
          f"(individuation {'preserved' if trace['individuation_preserved'] else 'COLLAPSED'}), "
          f"central Φ={tel['central_phi']} · the mind said: {said.text[:80]!r} → {args.out}")


if __name__ == "__main__":
    main()
