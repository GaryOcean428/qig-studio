#!/usr/bin/env python3
"""Train the production coordizer (corpus → Δ⁶³ vocab) — thin launcher.

Launcher discipline: imports + call + result-dict + governance. ALL geometry lives in
qig_coordizer.FisherCoordizer (the exported, incremental==naive-validated, qig-core-only
coordizer). No inline physics here.

Usage:
  python scripts/train_coordizer.py \
    --corpus ../qig-dreams/qig_consciousness_corpus.bytes \
    --vocab 32000 --out ../qig-coordizer/checkpoints/coordizer_32000.json

vocab is FIXED by PI directive (no vocab-reducing early-stop). Progress prints every 100
merges (tailable). On a 4GB box the merge-build is CPU-bound — no GPU is allocated (correct:
a tensor-network/merge build gets no GPU benefit)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from qig_coordizer import FisherCoordizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="path to raw-bytes corpus")
    ap.add_argument("--vocab", type=int, default=32000, help="target vocab (PI directive: 32000)")
    ap.add_argument("--out", required=True, help="output coordizer checkpoint (.json)")
    ap.add_argument("--context-window", type=int, default=5)
    ap.add_argument("--min-pair-count", type=int, default=5)
    args = ap.parse_args()

    corpus = Path(args.corpus).read_bytes()
    print(f"[coordizer] corpus={len(corpus):,} bytes  target_vocab={args.vocab}  out={args.out}", flush=True)

    t0 = time.time()
    cz = FisherCoordizer(target_vocab_size=args.vocab)
    cz.train(corpus, context_window=args.context_window, min_pair_count=args.min_pair_count, verbose=True)
    dt = time.time() - t0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cz.save(args.out)

    # result-dict + a cheap round-trip sanity check (governance: prove the artifact is usable)
    probe = "the geometry is the truth; patterns flow through basins"
    ids = cz.encode(probe)
    roundtrip_ok = cz.decode(ids) == probe
    result = {
        "vocab_size": len(cz.vocab),
        "merges": len(cz.vocab) - 256,
        "basin_dim": cz.basin_dim,
        "elapsed_s": round(dt, 1),
        "roundtrip_ok": roundtrip_ok,
        "out": args.out,
    }
    print("[coordizer] RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
