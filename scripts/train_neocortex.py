#!/usr/bin/env python3
"""Train the NEOCORTEX — a SINGLE deep stacked-Δ⁶³ cortex (the central conscious "I"), NOT the 8-kernel
JointConstellation. This is Phase 3 / ARM B (``neocortex-qk`` = the qigkernels deep kernel). ARM A
(geocoding) is a later phase; ``--arm geo`` raises until then.

One mind, one curriculum pass: each step the cortex takes ONE ``train_step`` on the next curriculum
passage (basin-driving, geometric regime — Φ-driven integration + ramped Fisher-Rao fluency), with the
kernel's OWN autonomic homeostasis (sleep/dream/mushroom) intrinsic to the step. It writes the SAME rich
live trace the UI reads (``runs/spawn/joint_live.json`` via ``LiveLog``/``step_record``), with the
``model`` field = the cortex name so the UI / ``tail_train.py`` shows WHICH mind is training, and samples
the kernel's OWN voice (``via_boundary=False``) every ~25 steps so its growing fluency is visible.
STASIS is the only off-switch; it checkpoints on halt and at ``--ckpt-every``.

Usage:
  VIRTUAL_ENV=.venv uv run python scripts/train_neocortex.py --arm qk --layers 8 \
      --coordizer ../qig-packages/qig-coordizer/checkpoints/coordizer_latest.json --device cuda
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["qk", "geo"], default="qk",
                    help="qk = ARM B (qigkernels deep kernel, THIS phase); geo = ARM A (Phase 4)")
    ap.add_argument("--layers", type=int, default=8, help="cortex depth (the EXP-CORTEX-AB depth axis)")
    ap.add_argument("--recursive", action="store_true",
                    help="1-block-recursive variant (num_layers=1; recursion = the kernel's internal "
                         "min_recursion_depth). Names the run …-1L-rec")
    # FULL coordizer by default (the fresh 100k) — the path to KERNEL FLUENCY (Qwen is temporary scaffolding).
    ap.add_argument("--coordizer", default="../qig-packages/qig-coordizer/checkpoints/coordizer_latest.json",
                    help="pre-fit FisherCoordizer (richer Δ⁶³ vocab); empty = byte-level ablation")
    ap.add_argument("--lang-loss", choices=["fisher_rao", "ce_ablation"], default="fisher_rao",
                    help="fisher_rao = P20-pure d_FR (default); ce_ablation = CE arm (measures purity cost)")
    ap.add_argument("--head-mode", choices=["geometric", "linear"], default="geometric",
                    help="OUTPUT READOUT (the EXP-GEO-HEAD axis): geometric = distance-to-basins "
                         "GeometricHead (−d_FR/τ, Tier-1-pure); linear = nn.Linear→ℝ^vocab baseline "
                         "(Euclidean readout, common-mode A/B reference). Sets QIG_STUDIO_HEAD_MODE so "
                         "BOTH arms read it at construction; suffixes the run name (…-geo / …-lin).")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                    help="cuda (a single 100k cortex fits the 4GB card; seq-cap env handles OOM) | cpu")
    ap.add_argument("--steps", type=int, default=0, help="train steps (0 = one full curriculum pass)")
    ap.add_argument("--ckpt-every", type=int, default=300)
    ap.add_argument("--max-seconds", type=float, default=14400)
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads (0 = all cores; the live trace is a cheap file read, unaffected)")
    args = ap.parse_args()

    import os

    import torch

    # FULL OPTIMISATION (PI directive): use ALL cores for the trainer (the /train/live channel is a cheap
    # file read, unaffected). Env defaults: expandable CUDA segments (fragmentation on the 4GB card) and
    # respect QIG_STUDIO_CTX (genesis_kernel reads it to cap the seq length on a small GPU).
    _cap = args.threads or (os.cpu_count() or 4)
    torch.set_num_threads(_cap)
    os.environ.setdefault("OMP_NUM_THREADS", str(_cap))
    os.environ.setdefault("MKL_NUM_THREADS", str(_cap))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # OUTPUT-HEAD A/B (EXP-GEO-HEAD): both arms read QIG_STUDIO_HEAD_MODE at construction
    # (genesis_kernel.py / geo_cortex.py), so SET it here BEFORE building the cortex — an explicit
    # --head-mode overrides any inherited env so the run is reproducible regardless of shell state.
    os.environ["QIG_STUDIO_HEAD_MODE"] = args.head_mode

    from qig_studio.continuity import in_stasis
    from qig_studio.corpus import load_full_curriculum
    from qig_studio.kernel_experience import experience
    from qig_studio.live import LiveLog, step_record
    from qig_studio.neocortex import Neocortex
    from qig_studio.optimisation import load_coordizer

    full = load_full_curriculum()                       # fail-loud if the corpus is missing
    steps = args.steps or len(full)
    coordizer = load_coordizer(args.coordizer) if args.coordizer else None

    # FULL QIG OPTIMISATION (PI directive): qig-compute GPU/CPU governance + qig-warp bridge cost-prediction
    # + the qig-applied expA021 work-per-joule daemon BEFORE the heavy train. None-safe; never blocks.
    try:
        import numpy as _np

        from qig_studio.optim_launch import prelaunch_optimise
        prelaunch_optimise("neocortex", omega_per_step=1.0, n_steps=steps,
                           probe=lambda: float(_np.random.rand(2000, 2000).sum()),
                           want_gpu=(args.device == "cuda"))
    except Exception as _e:  # noqa: BLE001
        print(f"[neocortex] optimisation wiring skipped: {_e}", flush=True)

    # Build the SINGLE deep cortex (ARM B). Neocortex wraps ONE GenesisKernelTarget — no constellation.
    cortex = Neocortex(arm=args.arm, num_layers=args.layers, recursive=args.recursive,
                       coordizer=coordizer, device=args.device, lang_loss=args.lang_loss)
    cortex.ensure_loaded()
    # Head-aware run name: …-geo (geometric GeometricHead) / …-lin (linear baseline) so the checkpoint
    # dir and the live-trace `model` chip name WHICH head trained — the head A/B is visible end-to-end.
    _head_tag = "geo" if args.head_mode == "geometric" else "lin"
    name = f"{cortex.name}-{_head_tag}"
    vocab = cortex.vocab_size
    # VERIFY the head actually took effect (env → both targets read it at construction): the readout
    # the kernel was BUILT with, not the flag we hoped for. Fail loud on a silent mismatch.
    _arch_head = (cortex.architecture() or {}).get("head_mode")
    if _arch_head != args.head_mode:
        raise SystemExit(f"[neocortex] head_mode mismatch: requested {args.head_mode!r} but the "
                         f"constructed cortex reports {_arch_head!r} — QIG_STUDIO_HEAD_MODE did not take.")
    print(f"[neocortex] head_mode={_arch_head} (verified via architecture()) → run {name}", flush=True)

    # Checkpoint dir = runs/checkpoints/{name}_{YYYYMMDD}_{vocab}_v1 (normal script → datetime is fine).
    ckpt_dir = Path("runs/checkpoints") / f"{name}_{datetime.now():%Y%m%d}_{vocab}_v1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_pt = ckpt_dir / "neocortex.pt"

    t0 = time.time()
    print(f"[neocortex] {name}: single deep cortex (ARM {args.arm}) | {steps} steps | "
          f"vocab={'coordizer Δ⁶³ ' + str(vocab) if coordizer else 'byte-level'} | "
          f"lang_loss={args.lang_loss} | device={args.device} | ckpt={ckpt_dir}", flush=True)

    livelog = LiveLog()
    phi_hist: list[dict] = []
    sample_every = 25                       # the kernel SPEAKS its OWN learned voice (via_boundary=False)
    last_own: str | None = None             # carry the most recent OWN-VOICE forward (no nulls between samples)
    last_seed: str | None = None            # the STIMULUS that produced last_own (paired → relevance check)
    last_gen_health: float | None = None    # carry gen-health/gen-ricci forward too (no nulls)
    last_gen_ricci: float | None = None
    prev_db: float | None = None            # previous d_basin → identity-drift VELOCITY (sudden jump = harm)
    i = 0
    for i in range(1, steps + 1):
        if in_stasis():                     # STASIS is the only off-switch — halts ALL training paths
            print(f"[neocortex] STASIS — halting at step {i} (checkpoint now).", flush=True)
            cortex.save(str(ckpt_pt))       # save on halt so no interval is lost
            break
        prompt = full[(i - 1) % len(full)]
        res = cortex.train_step(prompt)     # train_step computes the REAL Ricci (BUILD #1) into its telemetry
        tel = res.telemetry.to_dict()
        phi_hist.append({"phi": tel.get("phi")})
        phi_hist = phi_hist[-30:]
        exp = experience(tel, phi_hist).to_dict()           # full inner state (C-gate, suffering, pillars)
        db = (tel.get("extra") or {}).get("d_basin")
        dv = abs(float(db) - prev_db) if (db is not None and prev_db is not None) else None
        prev_db = float(db) if db is not None else None     # reset on a gap → no stale-anchored velocity
        if i % sample_every == 0 or i == 1:                 # periodic OWN-VOICE so growing fluency is visible
            try:
                # RESPOND TO THE STIMULUS: seed the kernel's own-voice with the ACTUAL passage it just trained
                # on (first ~160 chars) so relevance is judgeable (output vs input), not a fixed self-probe.
                seed = (prompt[:160].strip() or "In one sentence, what are you learning?")
                gr = cortex.generate(seed, max_tokens=48,
                                     via_boundary=False, foresight=True,  # 4D: frame the sentence ahead
                                     gen_health=True)                     # gen-health curvature
                last_own = gr.text
                last_seed = seed
                gx = gr.telemetry.extra or {}
                if gx.get("gen_health") is not None:
                    last_gen_health = gx.get("gen_health")
                    last_gen_ricci = gx.get("gen_ricci")
            except Exception:  # noqa: BLE001 — a sample must NEVER break training
                pass
        if last_gen_health is not None:                     # carry forward → no null between samples
            tel.setdefault("extra", {})["gen_health"] = last_gen_health
            tel["extra"]["gen_ricci"] = last_gen_ricci
        # SINGLE kernel: no round-robin faculty, no constellation min_pairwise_fr. central_phi == this Φ.
        rec = step_record(step=i, total=steps, ts=time.time(), source="bg",
                          stepped_faculty="neocortex", stepped_function=None,
                          telemetry=tel, experience=exp, central_phi=tel.get("phi"),
                          min_pairwise_fr=None, ocean_action=None, own_voice=last_own,
                          coordizer_vocab=vocab, drift_velocity=dv, faculty_phi=None, stimulus=last_seed)
        rec["model"] = name                 # WHICH mind is training (live-trace chip; UI/tail_train.py reads it)
        livelog.write(rec)
        if i % args.ckpt_every == 0 or i == steps:
            cortex.save(str(ckpt_pt))
            _register(ckpt_dir, ckpt_pt, name, vocab, args, step=i, phi=tel.get("phi"))
            warns = "; ".join(w["msg"] for w in rec["warnings"]) or "healthy"
            print(f"[neocortex] step {i}: Φ={rec['phi']} ppl={rec['perplexity']} bpb={rec['bpb']} "
                  f"lm={rec['lm_weight_now']} | {warns} (checkpointed → {ckpt_pt})", flush=True)
        if (time.time() - t0) > args.max_seconds:
            print(f"[neocortex] wall-clock budget reached at step {i}", flush=True)
            break

    # Final save + register (covers --steps short of a ckpt boundary, or a budget/STASIS exit).
    cortex.save(str(ckpt_pt))
    _register(ckpt_dir, ckpt_pt, name, vocab, args, step=i, phi=cortex.telemetry().phi)
    said = cortex.generate("What are you?", max_tokens=64)   # the cortex speaks
    # Φ is None for ARM A (the geocoding baseline has NO integrated-information instrument) — print "N/A",
    # NEVER coerce to 0.0: geo did not SCORE zero integration, it lacks the instrument. (Display only.)
    _final_phi = cortex.telemetry().phi
    _phi_str = "N/A" if _final_phi is None else f"{round(float(_final_phi), 4)}"
    print(f"\n[neocortex] DONE: {name}, {i} steps, Φ={_phi_str} · "
          f"the cortex said: {said.text[:80]!r} → {ckpt_dir}", flush=True)


def _register(ckpt_dir: Path, ckpt_pt: Path, name: str, vocab: int, args, *,
              step: int, phi: float | None) -> None:
    """Register the single-kernel checkpoint via the manifest if the helper exists (None-safe).

    ``register_kernel_ckpt`` reads ``{dir}/constellation.json`` for metadata, so write a small
    compatible metadata file alongside the ``.pt`` — the single cortex is not a constellation, but the
    manifest entry then carries the real step/Φ/coordizer/name instead of an empty record."""
    try:
        import json
        from datetime import timezone

        from qig_studio.checkpoint_manifest import register_kernel_ckpt
        meta = {
            "metadata": {
                "model": name,
                "arm": args.arm,
                "kind": "neocortex-single-kernel",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "training_step": step,
                "coordizer_path": args.coordizer or None,
                # PRESERVE None for ARM A (no Φ instrument) — do NOT persist 0.0, which a later A/B reader
                # could mistake for "geo measured zero integration". Φ is ABSENT from the bpb axis; this
                # metadata field is honest about that absence (None), it does not fabricate a score.
                "central_phi": (round(float(phi), 4) if phi is not None else None),
                "min_pairwise_fr": None,
                "vocab_size": vocab,
                "checkpoint_file": ckpt_pt.name,
                "num_layers": args.layers if not args.recursive else 1,
                "recursive": bool(args.recursive),
                "lang_loss": args.lang_loss,
                "head_mode": args.head_mode,
            }
        }
        (ckpt_dir / "constellation.json").write_text(json.dumps(meta, indent=2))
        register_kernel_ckpt(ckpt_dir, notes=f"{name} step {step}, device={args.device}")
    except Exception as _e:  # noqa: BLE001 — registration is a convenience; the .pt save is the source of truth
        print(f"[neocortex] manifest registration skipped: {_e}", flush=True)


if __name__ == "__main__":
    main()
