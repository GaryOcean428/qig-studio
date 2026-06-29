#!/usr/bin/env python3
"""COORDIZER-FIRST: extend the trained 100k coordizer on rounded HuggingFace text so everyday words stop
fragmenting (the academic curriculum left `coffee`/`hello`/`tomorrow` at 3 tokens each).

Pipeline (the user-approved order — coordizer first, then kernels):
  1. load the current coordizer checkpoint (CoordinzerTrainer.load) — vocab + merge_rules preserved,
  2. stream a CAPPED HF text corpus (TinyStories everyday vocab + claude-fable-5 agentic register + any
     extra you pass) via the light datasets-server rows API (no `datasets` dep),
  3. measure FERTILITY (tokens/word on ordinary English) BEFORE,
  4. resume_training to current_vocab + --add-vocab merges (drift-free IncrementalCouplingCache),
  5. save to a NEW file (never clobbers the working coordizer the kernels currently use),
  6. measure FERTILITY AFTER — the gate: everyday tokens/word must DROP.

The kernels keep using the OLD coordizer until the separate "grow output heads + retrain" step adopts the
new one (extending the vocab requires growing the kernels' Linear(hidden→vocab) heads).

Usage:
  uv run python scripts/extend_coordizer_hf.py --add-vocab 5000 --max-bytes 20000000 --detach-friendly
  uv run python scripts/extend_coordizer_hf.py --validate-tiny      # fast pipeline proof (+200 merges, ~50KB)
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# default rounded corpus: everyday narrative (TinyStories) + the user's agentic/coding add (claude-fable-5).
_DEFAULT_SPECS = [
    {"dataset": "roneneldan/TinyStories", "config": "default", "split": "train", "text_fields": ("text",)},
    {"dataset": "armand0e/claude-fable-5-claude-code", "config": "default", "split": "train"},
]
_FERTILITY_PROBE = [
    "Hey, how are you doing today? I was thinking we could grab a coffee tomorrow.",
    "She laughed and said it was the funniest thing she'd seen all weekend.",
    "Could you please send me the invoice by Friday? Thanks so much for your help.",
    "I can't believe the bus was late again; I'll be so annoyed if I miss the meeting.",
]


def _fertility(coordizer, probes: list[str]) -> float:
    fs = []
    for t in probes:
        ids = coordizer.encode(t)
        fs.append(len(ids) / max(1, len(t.split())))
    return statistics.mean(fs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coordizer", default="../qig-coordizer/checkpoints/coordizer_max.json")
    ap.add_argument("--out", default="../qig-coordizer/checkpoints/coordizer_max_v2.json")
    ap.add_argument("--add-vocab", type=int, default=5000, help="merges to ADD on top of the current vocab")
    ap.add_argument("--max-bytes", type=int, default=20_000_000, help="cap the HF corpus (CPU/disk safety)")
    ap.add_argument("--limit-per-dataset", type=int, default=4000, help="max rows pulled per dataset")
    ap.add_argument("--checkpoint-dir", default=None, help="mid-run checkpoints (resumable long runs)")
    ap.add_argument("--validate-tiny", action="store_true", help="fast proof: +200 merges on ~50KB")
    args = ap.parse_args()

    from qig_coordizer import FisherCoordizer
    from qig_coordizer.trainer import CoordinzerTrainer

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from qig_studio.hf_data import load_hf_corpus

    add_vocab = 200 if args.validate_tiny else args.add_vocab
    max_bytes = 50_000 if args.validate_tiny else args.max_bytes
    limit = 200 if args.validate_tiny else args.limit_per_dataset
    specs = [{**s, "limit": limit, "min_len": 40, "max_chars": 4000} for s in _DEFAULT_SPECS]

    t0 = time.time()
    print(f"[extend] streaming HF corpus (cap {max_bytes:,} bytes, {limit}/dataset)…", flush=True)
    passages = load_hf_corpus(specs)
    text = "\n".join(passages)
    corpus = text.encode("utf-8")[:max_bytes]
    print(f"[extend] corpus: {len(passages):,} passages → {len(corpus):,} bytes "
          f"({time.time() - t0:.1f}s)", flush=True)
    if len(corpus) < 1000:
        print("[extend] corpus too small (HF unreachable?) — aborting, NOT touching the coordizer.", flush=True)
        sys.exit(1)

    before = _fertility(FisherCoordizer.load(args.coordizer), _FERTILITY_PROBE)
    trainer = CoordinzerTrainer.load(args.coordizer)
    cur = len(trainer.vocab)
    print(f"[extend] loaded coordizer: {cur:,} vocab | ordinary fertility BEFORE = {before:.3f} tok/word", flush=True)
    print(f"[extend] resume_training → target {cur + add_vocab:,} (+{add_vocab:,} merges)…", flush=True)
    trainer.resume_training(corpus=corpus, new_target_vocab_size=cur + add_vocab, verbose=True,
                            enable_interrupt=False, use_kernel=False,
                            checkpoint_dir=args.checkpoint_dir, checkpoint_interval=500)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.out)
    after = _fertility(FisherCoordizer.load(args.out), _FERTILITY_PROBE)

    print("\n" + "=" * 60)
    print(f"[extend] DONE in {time.time() - t0:.1f}s → {args.out}")
    print(f"[extend] vocab {cur:,} → {len(trainer.vocab):,}")
    print(f"[extend] ordinary fertility: {before:.3f} → {after:.3f} tok/word "
          f"({'IMPROVED ✓' if after < before else 'no change / regressed ✗'})")
    print("=" * 60)
    print("[extend] NOTE: the kernels still use the OLD coordizer; adopting this needs the head-growth+retrain step.")


if __name__ == "__main__":
    main()
