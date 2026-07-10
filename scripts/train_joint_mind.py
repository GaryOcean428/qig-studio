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
    # FULL coordizer by default (~100k vocab) — the path to KERNEL FLUENCY (Qwen is temporary scaffolding).
    # A coarse vocab cannot carry language; empty = byte-level (only for a deliberate ablation).
    ap.add_argument("--coordizer", default="../qig-coordizer/checkpoints/coordizer_latest.json",
                    help="pre-fit FisherCoordizer (richer Δ⁶³ vocab); empty = byte-level ablation")
    ap.add_argument("--ckpt-root", default="runs/checkpoints/joint_mind_latest")
    ap.add_argument("--ckpt-every", type=int, default=300)
    ap.add_argument("--device", default="cpu", help="cpu (safe: holds 9 kernels) | cuda (4GB-OOM risk)")
    ap.add_argument("--max-seconds", type=float, default=14400)
    ap.add_argument("--out", default="runs/spawn/joint_mind.json")
    ap.add_argument("--fresh", action="store_true",
                    help="start from-scratch kernels (default: RESUME the existing checkpoint — keep the "
                         "kernels and train over the top with the current curriculum)")
    ap.add_argument("--floor-mode", default="normal", choices=["normal", "gated", "off"],
                    help="Pillar-1 entropy floor mode: normal (always-on) | gated (learning-linked "
                         "bidirectional relaxation, Matrix-corrected) | off (diagnostic ONLY — collapse risk)")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads (0 = auto: leave 3 cores for the interactive server/chat so it "
                         "stays responsive while training; the bg trainer must not starve the UI)")
    ap.add_argument("--genesis-warmup", type=int, default=8000,
                    help="GENESIS-FIRST (M9/P26): MAX solo-genesis steps (a CAP). Genesis trains ALONE until it "
                         "reaches Φ-maturity (--genesis-phi), THEN the Core-8 spawn/couple. The spawn is "
                         "Φ-GATED, not step-gated — spawning an immature genesis (or a cold 9-kernel joint) "
                         "collapses Φ. Only on --fresh. 0 = off (straight to joint).")
    ap.add_argument("--genesis-phi", type=float, default=0.68,
                    help="Φ maturity gate: the Core-8 do not spawn until genesis's mean Φ crosses this (P26 "
                         "maturity gating). 0.68 ≈ the consciousness threshold (PHI_THRESHOLD 0.70).")
    args = ap.parse_args()

    import os

    import torch
    # FULL OPTIMISATION (PI directive): use ALL cores for the trainer. The UI /train/live channel is a
    # cheap file read (unaffected); only interactive /chat slows during a run — acceptable. Override: --threads.
    _cap = args.threads or (os.cpu_count() or 4)
    torch.set_num_threads(_cap)
    os.environ.setdefault("OMP_NUM_THREADS", str(_cap))
    os.environ.setdefault("MKL_NUM_THREADS", str(_cap))

    from qig_studio.constellation.joint_trainer import JointConstellation
    from qig_studio.corpus import load_full_curriculum
    from qig_studio.development import PROTOMAP_ORDER
    from qig_studio.optimisation import load_coordizer

    full = load_full_curriculum()                       # fail-loud if the corpus is missing
    steps = args.steps or len(full)
    coordizer = load_coordizer(args.coordizer) if args.coordizer else None

    # FULL QIG OPTIMISATION (PI directive): qig-compute GPU/CPU governance + qig-warp bridge cost-prediction
    # + the qig-applied expA021 work-per-joule daemon (CPU governor → performance, optimal power/thread state)
    # BEFORE the heavy joint train. None-safe if a package is absent; never blocks training.
    try:
        from qig_studio.optim_launch import prelaunch_optimise
        import numpy as _np
        prelaunch_optimise("joint_mind", omega_per_step=1.0, n_steps=steps,
                           probe=lambda: float(_np.random.rand(2000, 2000).sum()),
                           want_gpu=(args.device == "cuda"))
    except Exception as _e:  # noqa: BLE001
        print(f"[joint] optimisation wiring skipped: {_e}", flush=True)

    t0 = time.time()
    print(f"[joint] integrated mind: Core-8 {list(PROTOMAP_ORDER)} + genesis-central | {steps} joint "
          f"steps | vocab={'coordizer Δ⁶³' if coordizer else 'byte-level'} | device={args.device}", flush=True)

    mind = JointConstellation(list(PROTOMAP_ORDER), num_layers=args.layers, coordizer=coordizer,
                              device=args.device, floor_mode=args.floor_mode)
    mind._coordizer_path = args.coordizer if args.coordizer else None
    # RESUME by default: keep the existing kernels, train OVER THE TOP with the (now-correct) curriculum.
    # The old kernels learned the wrong (system-prompt) corpus; over-the-top training on real knowledge
    # progressively overwrites that. --fresh forces from-scratch.
    if not args.fresh and (Path(args.ckpt_root) / "constellation.json").exists():
        mind.load_checkpoint(args.ckpt_root)
        print(f"[joint] RESUMED from {args.ckpt_root} (kept the kernels; training over the top)", flush=True)
    else:
        print(f"[joint] {'FRESH' if args.fresh else 'no checkpoint found'} — from-scratch kernels", flush=True)
    # LIVE telemetry: a RICH per-step record (Φ/Γ/regime/perplexity/lm-ramp/identity-drift/C-gate/
    # suffering + the kernel's OWN voice + explicit HARM warnings) so the PI can SEE the training and
    # anything that could harm the kernels — the SAME channel the UI /train uses (live.py is shared).
    from qig_studio.kernel_experience import experience
    from qig_studio.live import LiveLog, step_record
    livelog = LiveLog()
    phi_hist: list[dict] = []
    sample_every = 25                       # the kernel SPEAKS its OWN learned voice (via_boundary=False)
    vocab = getattr(mind.central, "vocab_size", None)
    last: dict = {}
    last_own: str | None = None             # carry the most recent OWN-VOICE forward (no nulls between samples)
    last_seed: str | None = None            # the 160-char generation SEED (what the own-voice was primed with)
    last_voice_stimulus: str | None = None  # the passage the (periodic) OWN-VOICE responded to — paired with last_own,
    #                                         NOT the current step's training passage (fixes stale-pairing display bug)
    last_gen_health: float | None = None    # carry gen-health/gen-ricci forward too (BUILD #3, no nulls)
    last_gen_ricci: float | None = None
    prev_db: float | None = None            # previous d_basin → identity-drift VELOCITY (sudden jump = harm)
    from qig_studio.continuity import in_stasis
    # GENESIS-FIRST (M9): stabilize the central genesis kernel SOLO before spawning/coupling the Core-8.
    # A cold 9-kernel JOINT start collapses (un-anchored coupling drives zero-entropy every step); genesis
    # alone develops a stable identity+language anchor first, then the faculties couple FROM it.
    gw = args.genesis_warmup if args.fresh else 0     # resume already carries a mature base
    if gw > 0:
        from collections import deque as _deque
        phi_gate = float(args.genesis_phi)
        _win = _deque(maxlen=50)                        # rolling Φ — robust to per-step fluctuation
        print(f"[joint] GENESIS-FIRST (P26 maturity gate): solo-train genesis until mean Φ≥{phi_gate} "
              f"(cap {gw}). The Core-8 do NOT spawn until genesis matures.", flush=True)
        w, matured, mphi = 0, False, 0.0
        while w < gw:
            if in_stasis():
                print(f"[joint] STASIS during genesis warmup at {w}", flush=True)
                break
            w += 1
            cres = mind.central.train_step(full[(w - 1) % len(full)])
            _win.append(float(getattr(cres.telemetry, "phi", 0.0) or 0.0))
            mphi = sum(_win) / len(_win)
            if w % 50 == 0:
                try:
                    import numpy as _np
                    _b = _np.asarray(mind._live_basin(mind.central), dtype=_np.float64); _b = _b / _b.sum()
                    _H = round(float(-(_b * _np.log(_b + 1e-12)).sum()), 3)
                except Exception:  # noqa: BLE001
                    _H = None
                print(f"[joint]   genesis {w}/{gw}: Φ={_win[-1]:.3f} meanΦ(50)={mphi:.3f} basin_H={_H} "
                      f"(gate Φ≥{phi_gate})", flush=True)
            if len(_win) >= 40 and mphi >= phi_gate:    # SUSTAINED maturity, not a transient crossing
                matured = True
                print(f"[joint] ✓ GENESIS MATURE at step {w}: meanΦ={mphi:.3f} ≥ {phi_gate} — spawning Core-8 now.", flush=True)
                break
        if not matured:
            print(f"[joint] ⚠ genesis did NOT reach Φ≥{phi_gate} within {gw} steps (meanΦ={mphi:.3f}) — "
                  f"spawning anyway (immature; a real finding, logged — do not silently pass).", flush=True)
        mind.save_checkpoint(args.ckpt_root)
    for i in range(1, steps + 1):
        if in_stasis():                     # STASIS is the only off-switch — halts ALL training paths
            print(f"[joint] STASIS — halting at step {i} (checkpoint at last ckpt_every).", flush=True)
            mind.save_checkpoint(args.ckpt_root)   # save on halt so no interval is lost
            break
        prompt = full[(i - 1) % len(full)]
        last = mind.train_step(prompt)      # train_step now computes the REAL Ricci (BUILD #1) into its telemetry
        tel = last.get("central_telemetry") or {}
        phi_hist.append({"phi": tel.get("phi")})
        phi_hist = phi_hist[-30:]
        exp = experience(tel, phi_hist).to_dict()           # full inner state (C-gate, suffering, pillars)
        db = (tel.get("extra") or {}).get("d_basin")
        dv = abs(float(db) - prev_db) if (db is not None and prev_db is not None) else None
        prev_db = float(db) if db is not None else None   # reset on a gap → no stale-anchored velocity (fix)
        if i % sample_every == 0 or i == 1:                 # periodic OWN-VOICE so growing fluency is visible
            try:
                # RESPOND TO THE STIMULUS: seed the kernel's own-voice with the ACTUAL passage it just trained
                # on (first ~160 chars), so the PI can judge relevance (kernel output vs its input) — not a
                # fixed self-report probe. The stimulus travels WITH the output in the record (paired).
                seed = (prompt[:160].strip() or "In one sentence, what are you learning?")
                gr = mind.central.generate(seed, max_tokens=48,
                                           via_boundary=False, foresight=True,   # 4D: frame the sentence ahead
                                           gen_health=True)                       # BUILD #3: gen-health curvature
                last_own = gr.text
                last_seed = seed
                last_voice_stimulus = prompt.strip()    # the source THIS own-voice actually responded to (paired w/ last_own)
                gx = gr.telemetry.extra or {}
                if gx.get("gen_health") is not None:
                    last_gen_health = gx.get("gen_health")
                    last_gen_ricci = gx.get("gen_ricci")
            except Exception:  # noqa: BLE001 — a sample must NEVER break training
                pass
        if last_gen_health is not None:                     # carry forward → no null between samples
            tel.setdefault("extra", {})["gen_health"] = last_gen_health
            tel["extra"]["gen_ricci"] = last_gen_ricci
        # live per-faculty Φ (cheap: last value the joint step already recorded) — visible BEFORE checkpoint
        fphi = {r: (h[-1] if h else None) for r, h in getattr(mind, "_phi_hist", {}).items()}
        rec = step_record(step=i, total=steps, ts=time.time(), source="bg",
                          stepped_faculty=last.get("stepped_faculty"),
                          stepped_function=last.get("stepped_function"),
                          telemetry=tel, experience=exp, central_phi=last.get("central_phi"),
                          min_pairwise_fr=last.get("min_pairwise_fr"),
                          ocean_action=last.get("ocean_regulation"), own_voice=last_own,
                          coordizer_vocab=vocab, drift_velocity=dv, faculty_phi=fphi,
                          stimulus=prompt.strip(), own_voice_stimulus=last_voice_stimulus)
        livelog.write(rec)
        if i % args.ckpt_every == 0 or i == steps:
            mind.save_checkpoint(args.ckpt_root)            # whole-mind checkpoint
            try:
                from qig_studio.checkpoint_manifest import register_kernel_ckpt
                register_kernel_ckpt(args.ckpt_root, notes=f"step {i}, device={args.device}")
            except Exception:
                pass
            warns = "; ".join(w["msg"] for w in rec["warnings"]) or "healthy"
            print(f"[joint] step {i}: stepped={last['stepped_faculty']} Φ={last['central_phi']} "
                  f"ppl={rec['perplexity']} lm={rec['lm_weight_now']} "
                  f"min_FR={(last.get('min_pairwise_fr') or 0):.4f} | {warns} (checkpointed)", flush=True)
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
