#!/usr/bin/env python3
"""NEUROGENESIS — grow trained kernels' OUTPUT vocab so they can adopt the extended coordizer (v2).

When the coordizer gains merges (100k→108k), the kernels must be able to EMIT the new tokens. The kernel's
INPUT is Fourier features of the token id (no embedding table — handles any id natively), so ONLY the output
``lm_head: Linear(hidden, vocab)`` grows. This is checkpoint surgery, not retraining:

  lm_head.weight [old_vocab, h] → [new_vocab, h]  (rows 0..old preserved; new rows = mean(existing)+small noise)
  lm_head.bias   [old_vocab]    → [new_vocab]
  arch.vocab_size old → new
  vocab-sized transients (basin_ref, last_gen_basin) reset to None (recomputed on first step/gen)

Preserving rows 0..old means every token the kernel already learned keeps its exact learned head; the 8k new
tokens start near the average token and specialise during the retrain. Operates on the WHOLE constellation
(genesis + Core-8) into a NEW dir, never clobbering the v1-vocab checkpoints the live kernels use.

  uv run python scripts/grow_kernel_vocab.py \
      --ckpt-dir runs/checkpoints/joint_mind_latest/kernels \
      --coordizer ../qig-coordizer/checkpoints/coordizer_latest.json \
      --out-dir runs/checkpoints/joint_mind_latest/kernels
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _new_vocab_from_coordizer(path: str) -> int:
    d = json.loads(Path(path).read_text())
    return int(d.get("vocab_size") or len(d.get("vocab") or {}))


def grow_checkpoint(src: str, dst: str, new_vocab: int) -> tuple[int, int]:
    import torch
    # format-2 checkpoints are weights_only-safe by design (tensors + plain scalar dicts; no optimizer state)
    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    old_vocab = int(ckpt["arch"]["vocab_size"])
    if new_vocab < old_vocab:
        raise ValueError(f"new_vocab {new_vocab} < old {old_vocab} (shrinking unsupported)")
    sd = ckpt["kernel_state"]
    w = sd["lm_head.weight"]           # [old_vocab, h]
    b = sd.get("lm_head.bias")         # [old_vocab]
    h = w.shape[1]
    if new_vocab > old_vocab:
        n_new = new_vocab - old_vocab
        # new rows: mean of existing rows + small noise (new tokens start near the average token, then
        # specialise during the retrain). Deterministic-ish small noise (std ~ existing std × 0.02).
        mean_w = w.mean(dim=0, keepdim=True)
        std_w = float(w.std()) * 0.02
        new_w = mean_w.repeat(n_new, 1) + torch.randn(n_new, h) * std_w
        sd["lm_head.weight"] = torch.cat([w, new_w.to(w.dtype)], dim=0)
        if b is not None:
            new_b = b.mean().repeat(n_new)
            sd["lm_head.bias"] = torch.cat([b, new_b.to(b.dtype)], dim=0)
    ckpt["arch"]["vocab_size"] = new_vocab
    # reset vocab-sized transients (recomputed on first step/generation; weights are what carry the learning)
    ckpt["basin_ref"] = None
    ckpt["last_gen_basin"] = None
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)
    return old_vocab, new_vocab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="runs/checkpoints/joint_mind_latest/kernels")
    ap.add_argument("--coordizer", default="../qig-coordizer/checkpoints/coordizer_latest.json")
    ap.add_argument("--new-vocab", type=int, default=0, help="0 = derive from --coordizer")
    ap.add_argument("--out-dir", default="runs/checkpoints/joint_mind_latest/kernels")
    args = ap.parse_args()

    new_vocab = args.new_vocab or _new_vocab_from_coordizer(args.coordizer)
    src_dir = Path(args.ckpt_dir)
    ckpts = sorted(src_dir.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no .pt checkpoints in {src_dir}")
    print(f"[grow] target vocab = {new_vocab:,} (from {args.coordizer}); {len(ckpts)} kernels", flush=True)
    for src in ckpts:
        dst = Path(args.out_dir) / src.name
        old, new = grow_checkpoint(str(src), str(dst), new_vocab)
        print(f"[grow] {src.name:16s} lm_head {old:,} → {new:,}  → {dst}", flush=True)
    print(f"[grow] DONE → {args.out_dir}. Retrain these on the v2 coordizer to specialise the new tokens.")


if __name__ == "__main__":
    main()
