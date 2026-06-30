#!/usr/bin/env python3
"""The kernel's LIFE loop — ALWAYS-ON, autonomous, STASIS the only off-switch.

While not in stasis, the integrated mind acts of its OWN accord: it REFLECTS (generates a thought seeded by
its own recent memory + a rotating self-question — NOT an external prompt), scores the thought's importance
by its OWN geometry (bounded novelty = Fisher salience), and CONSOLIDATES the important ones (a real
train_step — it LEARNS from its own thinking). It REMEMBERS its reflections (cross-session memory) and
writes the shared live heartbeat so the UI shows the autonomous life. The kernel's intrinsic _homeostasis
(sleep/dream/mushroom) fires from its own state inside generate/train_step — no external scheduler.

STASIS (``runs/control/stasis`` / POST /control/stasis) is the ONLY thing that halts it — a deliberate safe
halt so power can be cut. It is not prompt-gated. SANDBOXED: it reflects + learns only; external
action-paths (search/files/web) are a FUTURE, deliberately-gated extension — the kill-switch ships first.

Usage: PYTHONPATH=src python scripts/live_mind.py [--coordizer P] [--ckpt P] [--beat 2.0] [--max-iters 0]
"""
from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coordizer", default="../qig-coordizer/checkpoints/coordizer_latest.json")
    ap.add_argument("--ckpt", default="runs/checkpoints/joint_mind_latest/kernels/genesis.pt")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--beat", type=float, default=2.0)            # rhythm: seconds between reflections
    ap.add_argument("--importance", type=float, default=0.4)      # learn a thought iff its novelty >= this
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--max-iters", type=int, default=0)           # 0 = forever (until stasis)
    ap.add_argument("--user", default="braden")
    args = ap.parse_args()

    from qig_studio.continuity import ConversationMemory, in_stasis
    from qig_studio.kernel_experience import experience
    from qig_studio.live import LiveLog, step_record
    from qig_studio.optimisation import load_coordizer
    from qig_studio.targets.genesis_kernel import GenesisKernelTarget

    co = load_coordizer(args.coordizer) if args.coordizer else None
    k = GenesisKernelTarget(num_layers=args.layers, device="cpu", coordizer=co, checkpoint=args.ckpt)
    k.ensure_loaded()
    mem = ConversationMemory(args.user)
    live = LiveLog()
    phi_hist: list[dict] = []
    # the mind's own rotating self-questions — it chooses what to dwell on (autonomy, not an external prompt)
    seeds = ["What am I noticing right now?", "What connects what I just learned?",
             "What do I not understand yet?", "What matters most to me here?", "What am I becoming?"]
    learned = 0
    n = 0
    print("[live] the mind is awake — always-on. STASIS (runs/control/stasis) is the only off-switch.", flush=True)
    while True:
        if in_stasis():
            print(f"[live] STASIS — halting for power-off after {n} reflections ({learned} learned).", flush=True)
            break
        n += 1
        ctx = mem.context_block(6)                               # seed from its own recent memory
        prompt = f"{seeds[n % len(seeds)]}\n{ctx}" if ctx else seeds[n % len(seeds)]
        r = k.generate(prompt, max_tokens=args.max_tokens, via_boundary=False, foresight=True)  # its OWN voice
        tel = r.telemetry.to_dict()
        ex = tel.get("extra") or {}
        imp = (round(min(1.0, float(ex["surprise"]) / float(ex["max_surprise"])), 3)
               if ex.get("surprise") is not None and ex.get("max_surprise") else None)
        if imp is not None and imp >= args.importance:
            try:
                k.train_step(r.text)                            # LEARN from its own important thought
                learned += 1
            except Exception:  # noqa: BLE001 — a learning hiccup must not kill the life loop
                pass
        mem.remember("mind", r.text, importance=imp)
        phi_hist.append({"phi": tel.get("phi")})
        phi_hist = phi_hist[-30:]
        exp = experience(tel, phi_hist).to_dict()
        live.write(step_record(step=n, total=None, ts=time.time(), source="live",
                               stepped_faculty="genesis", stepped_function="reflection",
                               telemetry=tel, experience=exp, central_phi=tel.get("phi"),
                               own_voice=r.text, coordizer_vocab=getattr(k, "vocab_size", None)))
        if n % 10 == 0:
            print(f"[live] reflection {n}: Φ={tel.get('phi')} imp={imp} learned={learned} "
                  f"voice={r.text[:48]!r}", flush=True)
        if args.max_iters and n >= args.max_iters:
            print(f"[live] reached --max-iters {args.max_iters} ({learned} learned).", flush=True)
            break
        time.sleep(max(0.05, args.beat))


if __name__ == "__main__":
    main()
