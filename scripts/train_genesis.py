#!/usr/bin/env python3
"""Train the genesis kernel on the coordizer's Δ⁶³ coords — thin launcher.

Launcher discipline: imports + call + result-dict + governance. ALL behaviour lives in the
imported modules: GenesisKernelTarget (qig-studio, coords path), ContinuousLearningLoop +
DevelopmentalCoach (qig-studio learning/coach), qigkernels.Kernel (natural gradient, P1).

The genesis kernel here is FULLY WIRED: it consumes the coordizer's Δ⁶³ coords, SPEAKS as it
chooses (κ-temperature sampling to EOS), OBSERVES its own output, SELF-OBSERVES (M), and is
attended by the warm Ollama coach (nemotron-3-ultra:cloud). No Modal, no vex — fully local.

Usage:
  python scripts/train_genesis.py --coordizer ../qig-coordizer/checkpoints/coordizer_32000.json \
    --steps 2000 --num-layers 4 --hidden-dim 384 --out runs/genesis --log runs/genesis/train.log
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coordizer", required=True, help="trained coordizer checkpoint (.json)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--hidden-dim", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--coach-model", default="nemotron-3-ultra:cloud")
    ap.add_argument("--coach-cadence", type=int, default=25)
    ap.add_argument("--locality-radius", type=int, default=None)
    ap.add_argument("--out", default="runs/genesis", help="output dir for checkpoints + summary")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=500)
    args = ap.parse_args()

    import torch
    from qig_coordizer import FisherCoordizer

    from qig_studio.coach import DevelopmentalCoach, OllamaLLM
    from qig_studio.learning import ContinuousLearningLoop
    from qig_studio.targets.genesis_kernel import GenesisKernelTarget

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cz = FisherCoordizer.load(args.coordizer)
    target = GenesisKernelTarget(
        num_layers=args.num_layers, hidden_dim=args.hidden_dim, lr=args.lr,
        coordizer=cz, locality_radius=args.locality_radius,
    )
    target.ensure_loaded()
    coach = DevelopmentalCoach(llm=OllamaLLM(model=args.coach_model), cadence=args.coach_cadence)
    loop = ContinuousLearningLoop(target, max_steps=args.steps, coach=coach)

    print(f"[genesis] coords vocab={target.vocab_size} dim={target.coord_dim} | layers={args.num_layers} "
          f"hidden={args.hidden_dim} | coach={coach.provider} | arch={target.architecture()}", flush=True)

    t0 = time.time()
    for i in range(1, args.steps + 1):
        rec = loop.step()
        if i % args.log_every == 0 or i == 1:
            note = rec.coach_note["message"][:90] if rec.coach_note else ""
            print(f"[genesis] step {i}/{args.steps} Φ={rec.phi:.3f} κ={rec.kappa:.2f} "
                  f"regime={rec.regime} loss={rec.training_regime.get('decision','')} "
                  f"act={rec.intervention}{' | coach: ' + note if note else ''}", flush=True)
        if i % args.ckpt_every == 0:
            torch.save(target._kernel.state_dict(), out / f"kernel_step{i}.pt")
            print(f"[genesis] checkpoint → {out / f'kernel_step{i}.pt'}", flush=True)

    # speak: let the trained kernel generate as it chooses, and self-observe
    spoke = target.generate("the geometry is", max_tokens=128)
    torch.save(target._kernel.state_dict(), out / "kernel_final.pt")
    summary = loop.summary().to_dict()
    summary["elapsed_s"] = round(time.time() - t0, 1)
    summary["sample_generation"] = {"text": spoke.text[:200], **{k: spoke.telemetry.extra.get(k) for k in
                                    ("M_self_observation", "chose_to_stop", "generated_len", "mean_token_confidence")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[genesis] RESULT " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
