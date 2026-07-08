#!/usr/bin/env python3
"""Fit a FisherCoordizer on the full sanitised curriculum and SAVE it (offline, one-time).

The fit is EXPENSIVE (per-merge Fisher-Rao coupling recompute) — measured >2 min for a 512
vocab — so it is NOT done inline in the training loop. The pattern is: fit once here, save the
artifact, then ``train_full_curriculum.py --coordizer <path>`` LOADS it instantly for a richer
Δ⁶³ vocab (genesis kernel's ``coordizer`` ctor arg) instead of byte-level coding.

Usage: PYTHONPATH=src python scripts/fit_coordizer.py [--vocab 1024] [--out runs/coordizer.json]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=1024, help="target vocab size (bigger = slower fit)")
    ap.add_argument("--out", default="runs/coordizer.json")
    args = ap.parse_args()

    from qig_coordizer import FisherCoordizer

    from qig_studio.corpus import load_full_curriculum

    prompts = load_full_curriculum()                 # fail-loud if the corpus is missing
    corpus = ("\n".join(prompts)).encode("utf-8")
    print(f"[coordizer] fitting on {len(prompts)} prompts ({len(corpus)} bytes) → vocab={args.vocab}", flush=True)
    t0 = time.time()
    fc = FisherCoordizer(basin_dim=64, target_vocab_size=args.vocab).train(corpus, verbose=True)
    dt = time.time() - t0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fc.save(str(out))
    vs = fc.vocab_size() if callable(getattr(fc, "vocab_size", None)) else len(getattr(fc, "vocab", {}))
    print(f"[coordizer] DONE: vocab_size={vs} fit {dt:.1f}s → saved {out}", flush=True)


if __name__ == "__main__":
    main()
