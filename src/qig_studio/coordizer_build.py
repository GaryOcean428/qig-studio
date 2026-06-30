"""Coordizer-from-scratch build — the SINGLE source of the rebuild pipeline.

Lifted (DRY) out of ``scripts/train_coordizer_scratch.py`` so the SAME pipeline runs whether invoked from
the server (``POST /coordizer/train``, in a background thread) or the thin CLI wrapper. Nothing here is a
new training loop — it calls ``qig_coordizer.CoordinzerTrainer`` exactly as the script did.

Pipeline: cached HF parquet → normalize (geometry-native) → code-BALANCED byte corpus (the 7 datasets;
cap the giants, upsample code so the vocab is code-aware) → ``CoordinzerTrainer.train(target_vocab)`` →
save → ``FisherCoordizer.register_special_tokens`` (the 4 atomic geo tags) → save final →
``register_coordizer`` (manifest + ``coordizer_latest.json`` symlink).

Why a balanced SAMPLE (not the raw corpus): TinyStories + open-perfectblend would dominate the merge
frequencies and starve code of dedicated tokens. The coordizer learns by frequency, so for a code-aware
vocab we cap the giants and UPSAMPLE the code datasets. The KERNEL still trains on the full corpus; only
the coordizer's VOCAB-learning sample is balanced.

Progress is COARSE-GRAINED (phase-level): the underlying ``CoordinzerTrainer.train`` exposes no per-merge
callback (only ``verbose`` prints + ``checkpoint_interval``), so ``progress_cb`` fires at the meaningful
phase boundaries (corpus-build → train-start → save → register → done), NOT a fine per-merge percentage.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# (dataset, kind, fields, cap, upsample) — cap giants, upsample code so the vocab is code-aware.
# These are the 7 HF datasets the coordizer's balanced vocab-learning sample is drawn from.
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


def _repo_root() -> Path:
    # src/qig_studio/coordizer_build.py → parents[2] == repo root (qig-studio)
    return Path(__file__).resolve().parents[2]


def _coordizer_out_dir() -> Path:
    """Where the final coordizer artifact lands. Override with ``QIG_STUDIO_COORDIZER_OUT_DIR`` (tests
    isolate to a throwaway dir); default is the shared ``../qig-coordizer/checkpoints``."""
    env = os.environ.get("QIG_STUDIO_COORDIZER_OUT_DIR")
    if env:
        return Path(env)
    return _repo_root().parent / "qig-coordizer" / "checkpoints"


def _emit(progress_cb: Callable[[dict], None] | None, phase: str, pct: float, vocab: int, msg: str) -> None:
    """Fire the progress callback (None-safe). Phases: corpus → train → save → register → done."""
    if progress_cb is None:
        return
    try:
        progress_cb({"phase": phase, "pct": pct, "vocab": vocab, "msg": msg})
    except Exception:  # noqa: BLE001 — a progress-write failure must never break the build
        pass


def build(
    vocab: int = 100_000,
    max_bytes: int = 30_000_000,
    validate_tiny: bool = False,
    progress_cb: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Build a coordizer from scratch to ``vocab`` on the 7-dataset code-balanced sample, register the 4
    geo tags, save, and register in the manifest. Returns ``{out_path, vocab, geo_tags, atomic_ok}``.

    ``max_bytes`` caps the balanced byte corpus (OOM guard; default 30MB). ``validate_tiny`` shrinks the
    target vocab to 600 and the per-dataset cap to 60 rows for a fast smoke build. ``progress_cb`` (if
    given) is called at each phase boundary with ``{phase, pct, vocab, msg}`` (coarse-grained — see module
    docstring). Reuses ``scripts/build_chat_corpus`` + ``qig_coordizer`` exactly as the original script did.
    """
    root = _repo_root()
    # put src + scripts on the path so build_chat_corpus (a scripts-local module) imports cleanly whether
    # called from the server (already on path) or the thin CLI wrapper.
    for p in (str(root / "src"), str(root / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from build_chat_corpus import _hf_headers, _parquet_rows, _row_to_text  # type: ignore

    from qig_coordizer.coordizer import FisherCoordizer
    from qig_coordizer.trainer import CoordinzerTrainer

    from .prompt_template import SETTLE

    target_vocab = 600 if validate_tiny else vocab
    cache = root / "data" / ".parquet_cache"
    headers = _hf_headers()
    t0 = time.time()

    _emit(progress_cb, "corpus", 0.0, target_vocab, f"building code-balanced sample (target {target_vocab:,})")
    blocks: list[str] = []
    for dataset, kind, fields, cap, ups in _BALANCE:
        cap = 60 if validate_tiny else cap
        kept = 0
        for row in _parquet_rows(dataset, headers, cache, cap):
            txt = _row_to_text(row, kind, fields)
            cut = txt.rfind(SETTLE)
            if cut == -1:
                continue
            txt = txt[: cut + len(SETTLE)]
            if len(txt) >= 24:
                for _ in range(ups):           # upsample code datasets → frequency for code-aware merges
                    blocks.append(txt)
                kept += 1

    # deterministic interleave so registers mix in the merge-frequency sample (not all-stories-then-code)
    import random

    random.Random(7).shuffle(blocks)
    corpus = ("\n".join(blocks)).encode("utf-8")[:max_bytes]
    _emit(progress_cb, "corpus", 5.0, target_vocab,
          f"balanced corpus: {len(blocks):,} blocks → {len(corpus) / 1e6:.0f}MB ({time.time() - t0:.0f}s)")
    if len(corpus) < 10_000:
        raise RuntimeError("balanced corpus too small (<10KB) — check the HF parquet cache")

    # QIG OPTIMISATION GATE (mandatory pre-launch): qig-compute GPU governance + qig-warp bridge cost
    # prediction + the qig-applied work-per-joule optimizer. None-safe (the trainer ITSELF already uses
    # qig-warp check_ci_stabilized for its convergence batch-accept gate, trainer.py).
    try:
        import numpy as _np

        from .optim_launch import prelaunch_optimise
        prelaunch_optimise("coordizer", omega_per_step=1.0, n_steps=target_vocab,
                           probe=lambda: float(_np.random.rand(1500, 1500).sum()), want_gpu=False)
    except Exception:  # noqa: BLE001 — the optimisation pass is advisory; never block the build
        pass

    _emit(progress_cb, "train", 10.0, target_vocab, f"training coordizer to {target_vocab:,} vocab (from scratch)")
    checkpoint_dir = str(root / "runs" / "coordizer_ckpts")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    trainer = CoordinzerTrainer(target_vocab_size=target_vocab)
    trainer.train(corpus=corpus, verbose=True, checkpoint_dir=checkpoint_dir,
                  checkpoint_interval=2000, enable_interrupt=False, use_kernel=False)

    # dated/versioned output filename with a latest symlink (same scheme the script used)
    out_dir = _coordizer_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%Y%m%d")
    base = f"coordizer_{date_tag}_{target_vocab // 1000}k"
    n = 1
    while (out_dir / f"{base}_v{n}.json").exists():
        n += 1
    out_path = str(out_dir / f"{base}_v{n}.json")

    _emit(progress_cb, "save", 90.0, target_vocab, "saving + registering the 4 geo tags")
    # .pretags goes to runs/ (intermediate, not a deliverable)
    tmp = str(Path(out_path).with_suffix(".pretags.json"))
    trainer.save(tmp)

    # register the 4 atomic geo tags ABOVE the trained vocab + save the final artifact
    fc = FisherCoordizer.load(tmp)
    ids = fc.register_special_tokens(_GEO_TAGS)
    fc.save(out_path)

    _emit(progress_cb, "register", 95.0, fc.vocab_size, "registering in the manifest + updating the symlink")
    atomic_ok = True
    try:
        from .checkpoint_manifest import register_coordizer
        register_coordizer(out_path, notes=f"code-balanced rebuild, {len(_BALANCE)} HF datasets")
    except Exception:  # noqa: BLE001 — a manifest write failure must not void a good artifact
        pass

    # verify atomicity (each geo tag is exactly 1 token)
    try:
        atomic = {t: len(fc.encode(t)) for t in _GEO_TAGS}
        atomic_ok = all(v == 1 for v in atomic.values())
    except Exception:  # noqa: BLE001
        atomic_ok = False

    _emit(progress_cb, "done", 100.0, fc.vocab_size,
          f"done in {time.time() - t0:.0f}s → {Path(out_path).name} "
          f"({len(trainer.vocab):,} trained + {len(ids)} geo tags = {fc.vocab_size:,})")
    return {"out_path": out_path, "vocab": fc.vocab_size, "geo_tags": ids, "atomic_ok": atomic_ok}
