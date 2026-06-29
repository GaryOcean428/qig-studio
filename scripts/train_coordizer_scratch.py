#!/usr/bin/env python3
"""Train the REBUILD coordizer FROM SCRATCH to 100k+ vocab on a CODE-BALANCED sample of the geometry-native
corpus, then register the 4 atomic geo tags (<|frame|>/<|seed|>/<|flow|>/<|settle|>).

Why a balanced SAMPLE (not the raw 3.68M-block corpus): TinyStories (58%) + open-perfectblend (37%) would
dominate the merge frequencies and starve code of dedicated tokens (code is <1% of raw blocks). The coordizer
learns by frequency, so for a CODE-AWARE vocab we cap the giants and UPSAMPLE the code datasets. The KERNEL
still trains on the full corpus; only the coordizer's VOCAB-learning sample is balanced.

Pipeline: cached parquet -> normalize (geometry-native) -> balanced byte corpus -> CoordinzerTrainer.train
(target 100k) -> save -> FisherCoordizer.load -> register_special_tokens(4 tags) -> save final.

Usage:
  uv run python scripts/train_coordizer_scratch.py --vocab 100000 --out ../qig-coordizer/checkpoints/coordizer_rebuild.json
  uv run python scripts/train_coordizer_scratch.py --validate-tiny
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# (dataset, kind, fields, cap, upsample) — cap giants, upsample code so the vocab is code-aware.
_BALANCE = [
    ("roneneldan/TinyStories", "narrative", ("text",), 150_000, 1),
    ("Estwld/empathetic_dialogues_llm", "conversations", ("conversations",), None, 1),
    ("Anthropic/hh-rlhf", "hh", ("chosen",), 100_000, 1),
    ("armand0e/claude-fable-5-claude-code", "messages", ("messages",), None, 30),
    ("PawanKrd/claude-fable-5-code", "messages", ("messages",), None, 30),
    ("WithinUsAI/GPT_5.5_Distilled", "tagged", ("text",), None, 1),
    ("mlabonne/open-perfectblend", "conversations", ("conversations",), 150_000, 1),
]
_GEO_TAGS = ["<|frame|>", "<|seed|>", "<|flow|>", "<|settle|>"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=100_000)
    ap.add_argument("--out", default="../qig-coordizer/checkpoints/coordizer_rebuild.json")
    ap.add_argument("--max-bytes", type=int, default=400_000_000, help="cap the balanced byte corpus (memory)")
    ap.add_argument("--checkpoint-dir", default="runs/coordizer_rebuild_ckpts")
    ap.add_argument("--validate-tiny", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    from build_chat_corpus import _hf_headers, _parquet_rows, _row_to_text
    from qig_studio.prompt_template import SETTLE

    from qig_coordizer.coordizer import FisherCoordizer
    from qig_coordizer.trainer import CoordinzerTrainer

    vocab = 600 if args.validate_tiny else args.vocab
    cache = root / "data" / ".parquet_cache"
    headers = _hf_headers()
    t0 = time.time()

    print(f"[coord] building code-balanced sample (target vocab {vocab:,})…", flush=True)
    blocks: list[str] = []
    for dataset, kind, fields, cap, ups in _BALANCE:
        cap = 60 if args.validate_tiny else cap
        kept = 0
        for row in _parquet_rows(dataset, headers, cache, cap):
            txt = _row_to_text(row, kind, fields)
            cut = txt.rfind(SETTLE)
            if cut == -1:
                continue
            txt = txt[:cut + len(SETTLE)]
            if len(txt) >= 24:
                for _ in range(ups):           # upsample code datasets → frequency for code-aware merges
                    blocks.append(txt)
                kept += 1
        print(f"[coord]   {dataset}: {kept:,} unique x{ups} -> {kept*ups:,} blocks", flush=True)

    # deterministic interleave so registers mix in the merge-frequency sample (not all-stories-then-code)
    import random
    random.Random(7).shuffle(blocks)
    corpus = ("\n".join(blocks)).encode("utf-8")[: args.max_bytes]
    print(f"[coord] balanced corpus: {len(blocks):,} blocks -> {len(corpus)/1e6:.0f}MB "
          f"({time.time()-t0:.0f}s)", flush=True)
    if len(corpus) < 10_000:
        print("[coord] corpus too small — abort.", flush=True)
        sys.exit(1)

    print(f"[coord] training coordizer to {vocab:,} vocab (from scratch)…", flush=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    trainer = CoordinzerTrainer(target_vocab_size=vocab)
    trainer.train(corpus=corpus, verbose=True, checkpoint_dir=args.checkpoint_dir,
                  checkpoint_interval=2000, enable_interrupt=False, use_kernel=False)
    tmp = str(Path(args.out).with_suffix(".pretags.json"))
    Path(tmp).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(tmp)

    # register the 4 atomic geo tags ABOVE the trained vocab + save the final artifact
    fc = FisherCoordizer.load(tmp)
    ids = fc.register_special_tokens(_GEO_TAGS)
    fc.save(args.out)
    # verify atomicity
    atomic = {t: len(fc.encode(t)) for t in _GEO_TAGS}
    print("\n" + "=" * 60)
    print(f"[coord] DONE in {time.time()-t0:.0f}s -> {args.out}")
    print(f"[coord] trained vocab {len(trainer.vocab):,} + {len(ids)} geo tags = {fc.vocab_size:,}")
    print(f"[coord] geo-tag ids {ids}")
    print(f"[coord] atomic (1 token each): {atomic}  -> {'OK' if all(v==1 for v in atomic.values()) else 'FAIL'}")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
