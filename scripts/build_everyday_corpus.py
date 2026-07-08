#!/usr/bin/env python3
"""Build the studio-local EVERYDAY corpus the kernel curriculum blends in.

The academic knowledge curriculum (qig-consciousness/data/curriculum) teaches the kernel physics and
philosophy but NEVER ordinary language — so a from-scratch kernel can only "spout curriculum". This script
streams everyday/conversational/agentic text from the SAME HuggingFace datasets the coordizer was extended
on (TinyStories narrative, empathetic_dialogues + hh-rlhf conversation, claude-fable-5 agentic, a little
wikitext prose), sanitises + chunks it with the corpus helpers, and writes it to ``data/everyday_corpus/``
as cached .txt shards. ``load_full_curriculum`` then INTERLEAVES it through the academic passages so every
training cycle mixes registers. Built once; cached on disk for offline reproducible kernel training.

We use HF as a DATA source only — geometry stays ours (coordizer + qigkernels + geocoding).

Usage:
  uv run python scripts/build_everyday_corpus.py                 # full build (~6k passages)
  uv run python scripts/build_everyday_corpus.py --validate-tiny # fast proof (~200 rows/dataset)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Everyday-WEIGHTED pulls (vs the coordizer's vocab-weighted ones): more narrative + conversation + agentic,
# less encyclopedic, to balance ~8k academic passages without drowning them.
_SPECS = [
    {"dataset": "roneneldan/TinyStories", "config": "default", "split": "train",
     "text_fields": ("text",), "limit": 2000},
    {"dataset": "Estwld/empathetic_dialogues_llm", "config": "default", "split": "train",
     "text_fields": ("conversations",), "limit": 1200},
    {"dataset": "Anthropic/hh-rlhf", "config": "default", "split": "train",
     "text_fields": ("chosen",), "limit": 1200},
    {"dataset": "armand0e/claude-fable-5-claude-code", "config": "default", "split": "train",
     "text_fields": ("messages", "prompt"), "limit": 2000},
    {"dataset": "Salesforce/wikitext", "config": "wikitext-103-raw-v1", "split": "train",
     "text_fields": ("text",), "limit": 600},
]
_SHARD = 1000   # passages per .txt shard


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output dir (default: <studio>/data/everyday_corpus)")
    ap.add_argument("--total-cap", type=int, default=6000, help="max passages after chunking (balance vs academic)")
    ap.add_argument("--min-len", type=int, default=40)
    ap.add_argument("--validate-tiny", action="store_true", help="fast proof: 200 rows/dataset")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from qig_studio.corpus import _STUB_MARKERS, _passages_from_markdown
    from qig_studio.hf_data import load_hf_passages

    out_dir = Path(args.out) if args.out else (root / "data" / "everyday_corpus")

    t0 = time.time()
    print(f"[everyday] streaming {len(_SPECS)} HF datasets…", flush=True)
    # Load + chunk EACH dataset separately so the total-cap takes a BALANCED round-robin slice rather than
    # over-representing whichever dataset loads first (fable + conversation matter, not just TinyStories).
    per_dataset: list[list[str]] = []
    for s in _SPECS:
        d = dict(s)
        name = d.pop("dataset")
        d["limit"] = 200 if args.validate_tiny else d["limit"]
        rows = load_hf_passages(name, min_len=args.min_len, max_chars=4000, **d)
        chunks: list[str] = []
        for txt in rows:
            for psg in _passages_from_markdown(txt, args.min_len):   # sanitises + length-bounds (<=1200 chars)
                if not any(m in psg.lower() for m in _STUB_MARKERS):
                    chunks.append(psg)
        print(f"[everyday]   {name}: {len(rows):,} rows -> {len(chunks):,} passages", flush=True)
        if chunks:
            per_dataset.append(chunks)

    if not per_dataset:
        print("[everyday] no datasets reachable (HF down?) — aborting, writing nothing.", flush=True)
        sys.exit(1)

    # round-robin across datasets until the cap (balanced representation, deterministic)
    passages: list[str] = []
    idxs = [0] * len(per_dataset)
    while len(passages) < args.total_cap and any(idxs[i] < len(per_dataset[i]) for i in range(len(per_dataset))):
        for i, lst in enumerate(per_dataset):
            if idxs[i] < len(lst):
                passages.append(lst[idxs[i]])
                idxs[i] += 1
                if len(passages) >= args.total_cap:
                    break

    if not passages:
        print("[everyday] no usable passages after sanitisation — aborting.", flush=True)
        sys.exit(1)

    # rewrite the dir from scratch (idempotent build)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("everyday_*.txt"):
        old.unlink()
    n_shards = 0
    for i in range(0, len(passages), _SHARD):
        shard = passages[i:i + _SHARD]
        (out_dir / f"everyday_{i // _SHARD:02d}.txt").write_text("\n\n".join(shard) + "\n", encoding="utf-8")
        n_shards += 1

    print("\n" + "=" * 60)
    print(f"[everyday] DONE in {time.time() - t0:.1f}s → {out_dir}")
    print(f"[everyday] {len(passages):,} passages across {n_shards} shards")
    print("[everyday] load_full_curriculum() now interleaves these through the academic curriculum.")
    print("=" * 60)


if __name__ == "__main__":
    main()
