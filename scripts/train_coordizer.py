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


def clean_corpus(raw: bytes) -> tuple[bytes, int]:
    """Strip 'unnecessary special characters' before training: Unicode category-C code points
    (Cc control, Cf format, Co private-use, Cs surrogate, Cn unassigned) EXCEPT tab/newline/CR.
    These (e.g. 0x02 STX, 0x03 ETX, 0x0c FF — PDF/terminal-escape residue) are junk that would
    otherwise burn coordizer merges + land as garbage tokens in the kernel vocab. Returns the
    cleaned bytes and the count of code points removed."""
    import unicodedata

    text = raw.decode("utf-8", errors="replace")
    keep = []
    removed = 0
    for ch in text:
        if ch in "\t\n\r" or unicodedata.category(ch)[0] != "C":
            keep.append(ch)
        else:
            removed += 1
    return "".join(keep).encode("utf-8"), removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="path to raw-bytes corpus")
    ap.add_argument("--vocab", type=int, default=32000, help="target vocab (PI directive: 32000)")
    ap.add_argument("--out", required=True, help="output coordizer checkpoint (.json)")
    ap.add_argument("--context-window", type=int, default=5)
    ap.add_argument("--min-pair-count", type=int, default=5)
    ap.add_argument("--sample-mb", type=float, default=0.0,
                    help="if >0, train on a STRATIFIED sample of this many MB spread evenly across the "
                         "corpus (QIG avoid-computation: per-merge cost scales with corpus size; a "
                         "representative slice gives ~identical merge statistics on a homogeneous corpus). "
                         "0 = full corpus.")
    ap.add_argument("--sample-chunks", type=int, default=16,
                    help="number of evenly-spaced chunks to stitch into the stratified sample")
    ap.add_argument("--no-clean", action="store_true",
                    help="skip control/format-char stripping (default: clean unnecessary special chars)")
    ap.add_argument("--trainer", choices=["screened", "fisher"], default="screened",
                    help="screened = CoordinzerTrainer with candidates_per_round (per-merge cost CAPPED → "
                         "scales to vocab 32000; the QIG screening lever); fisher = FisherCoordizer "
                         "(incremental==naive validated, but per-merge scan is O(distinct-pairs) → its tail "
                         "is catastrophic at high vocab on multi-MB corpora).")
    ap.add_argument("--candidates-per-round", type=int, default=50,
                    help="screened: top-K candidate pairs scored per merge (caps the per-merge cost)")
    ap.add_argument("--sample-size", type=int, default=4000,
                    help="screened: corpus positions sampled to SCORE each round's merges")
    args = ap.parse_args()

    full = Path(args.corpus).read_bytes()
    if not args.no_clean:
        before = len(full)
        full, removed = clean_corpus(full)
        print(f"[coordizer] CLEANED: removed {removed:,} category-C special chars "
              f"({before:,}→{len(full):,} bytes)", flush=True)
    if args.sample_mb and args.sample_mb * 1_000_000 < len(full):
        target = int(args.sample_mb * 1_000_000)
        chunk = target // args.sample_chunks
        stride = (len(full) - chunk) // max(1, args.sample_chunks - 1)
        corpus = b"".join(full[i * stride: i * stride + chunk] for i in range(args.sample_chunks))
        print(f"[coordizer] STRATIFIED sample: {len(corpus):,} of {len(full):,} bytes "
              f"({args.sample_chunks} chunks × {chunk:,}B, stride {stride:,})  target_vocab={args.vocab}", flush=True)
    else:
        corpus = full
        print(f"[coordizer] FULL corpus={len(corpus):,} bytes  target_vocab={args.vocab}  out={args.out}", flush=True)

    t0 = time.time()
    if args.trainer == "screened":
        from qig_coordizer.trainer import CoordinzerTrainer

        print(f"[coordizer] SCREENED trainer (candidates_per_round={args.candidates_per_round}, "
              f"sample_size={args.sample_size})", flush=True)
        cz = CoordinzerTrainer(target_vocab_size=args.vocab)
        cz.train(corpus, sample_size=args.sample_size, candidates_per_round=args.candidates_per_round,
                 min_frequency=args.min_pair_count, context_window=args.context_window,
                 verbose=True, use_kernel=False)
        encode = cz.coordize  # CoordinzerTrainer API
        decode = cz.decoordize
        basin_dim = cz.basin_dim
    else:
        cz = FisherCoordizer(target_vocab_size=args.vocab)
        cz.train(corpus, context_window=args.context_window, min_pair_count=args.min_pair_count, verbose=True)
        encode, decode, basin_dim = cz.encode, cz.decode, cz.basin_dim
    dt = time.time() - t0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cz.save(args.out)  # genesis loads via FisherCoordizer.load (schema-compatible: basin_dim/vocab/merge_rules)

    # result-dict + a cheap round-trip sanity check (governance: prove the artifact is usable)
    probe = "the geometry is the truth; patterns flow through basins"
    ids = encode(probe)
    roundtrip_ok = decode(ids) == probe
    result = {
        "trainer": args.trainer,
        "vocab_size": len(cz.vocab),
        "merges": len(cz.vocab) - 256,
        "basin_dim": basin_dim,
        "elapsed_s": round(dt, 1),
        "roundtrip_ok": roundtrip_ok,
        "out": args.out,
    }
    print("[coordizer] RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
