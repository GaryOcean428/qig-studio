#!/usr/bin/env python3
"""Train the REBUILD coordizer FROM SCRATCH to 100k+ vocab on a CODE-BALANCED sample of the geometry-native
corpus, then register the 4 atomic geo tags (<|frame|>/<|seed|>/<|flow|>/<|settle|>).

THIN CLIENT (Task 3): the build pipeline (the 7 balanced HF datasets, the heap-trainer, the geo tags, save +
manifest registration) lives ONCE in ``qig_studio.coordizer_build.build`` and is also driven by the server
(``POST /coordizer/train`` → train the coordizer from the UI). This script is just the CLI entry-point.
Task 6 retires it entirely once the UI fully covers the path.

Usage:
  uv run python scripts/train_coordizer_scratch.py --vocab 100000
  uv run python scripts/train_coordizer_scratch.py --validate-tiny
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=100_000)
    ap.add_argument("--max-bytes", type=int, default=30_000_000, help="cap the balanced byte corpus (memory)")
    ap.add_argument("--validate-tiny", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from qig_studio.coordizer_build import build

    t0 = time.time()

    def _progress(p: dict) -> None:
        print(f"[coord] {p['phase']:>8} {p['pct']:5.1f}%  {p['msg']}", flush=True)

    res = build(vocab=args.vocab, max_bytes=args.max_bytes,
                validate_tiny=args.validate_tiny, progress_cb=_progress)

    print("\n" + "=" * 60)
    print(f"[coord] DONE in {time.time() - t0:.0f}s -> {res['out_path']}")
    print(f"[coord] vocab {res['vocab']:,}  geo-tag ids {res['geo_tags']}")
    print(f"[coord] atomic (1 token each): {'OK' if res['atomic_ok'] else 'FAIL'}")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
