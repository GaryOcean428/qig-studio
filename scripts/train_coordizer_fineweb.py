#!/usr/bin/env python3
"""DoD-1b — train a TEXT-HEAVY coordizer from scratch on HuggingFaceFW/fineweb (STREAMED), under the
COORDIZER ARTIFACT STANDARD (matrix 6e2a5ff6 / directive 69aa0f59).

WHY THIS EXISTS (the named-reversal test): the DoD-1 arms verdict ran on ``coordizer_20260705_64k_v1``, a
CODE-BALANCED 64k vocab. The registered caveat was that a code-balanced vocab on a text-only eval may have
disadvantaged the ``gk`` arm — a pre-committed "named reversal scenario". This artifact IS that test's
instrument: a fineweb-trained TEXT-HEAVY 64k coordizer. Matched to 64k vocab ON PURPOSE so the ONLY changed
variable vs 64k_v1 is the corpus (code-balanced -> web-text). Re-running the bake-off on this coordizer tells
us whether the geo>gk gap moves under a text-appropriate vocab.

WHAT IT SHARES WITH THE CANONICAL BUILD: identical BPE primitives — ``Normalizer(pretokenize=True,
max_segment_bytes=128)`` segment-frequency substrate + ``CoordinzerTrainer`` + the 4 atomic geo tags. The
ONLY differences are the corpus (streamed fineweb, not the 7-dataset code-balanced parquet) and that this
writes the full ARTIFACT STANDARD (provenance filename + sidecar .manifest.json + embedded header +
sha256). The canonical server-driven ``coordizer_build.build`` is deliberately left untouched (it is the
64k_v1 reproduction path).

Usage (from qig-studio/, with the repo .venv that has httpx + qig_coordizer):
  ../.venv/bin/python scripts/train_coordizer_fineweb.py --vocab 64000 --passages 300000
  ../.venv/bin/python scripts/train_coordizer_fineweb.py --validate-tiny      # fast smoke (600 vocab)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROWS_API = "https://datasets-server.huggingface.co/rows"
DATASET = "HuggingFaceFW/fineweb"
CONFIG = "sample-10BT"          # curated 10B-token sample; 'default' 501s on the rows API (too large)
SPLIT = "train"
PAGE = 100                      # rows API hard cap per call
GEO_TAGS = ["<|frame|>", "<|seed|>", "<|flow|>", "<|settle|>"]
MAX_SEG_BYTES = 128             # matches coordizer_build._MAX_SEG_BYTES (identical char-safe run-on cap)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fetch_page(httpx, offset: int, n: int, *, page_delay: float) -> list:
    """One rows-API page with bounded exponential backoff + jitter on 429/5xx (qig-agent-comms recovery).
    Returns the rows list; [] signals honest end-of-stream (empty page). Raises on exhausted retries so the
    caller can decide (never silently truncates a partial corpus into a 'done')."""
    import random
    delay = 2.0
    for attempt in range(8):
        try:
            r = httpx.get(ROWS_API, params={"dataset": DATASET, "config": CONFIG, "split": SPLIT,
                                            "offset": offset, "length": n}, timeout=60.0)
            if r.status_code == 200:
                if page_delay:
                    time.sleep(page_delay)          # polite base pacing between successful pages
                return (r.json() or {}).get("rows") or []
            if r.status_code in (429, 500, 502, 503, 504):
                sleep_s = min(delay, 60.0) + random.uniform(0, 1.5)
                print(f"[fineweb] {r.status_code} at offset {offset} (attempt {attempt + 1}/8) — "
                      f"backoff {sleep_s:.1f}s", flush=True)
                time.sleep(sleep_s)
                delay *= 2
                continue
            raise RuntimeError(f"rows API status {r.status_code} at offset {offset}")
        except (RuntimeError,):
            raise
        except Exception as e:  # noqa: BLE001 — transient network; back off and retry
            sleep_s = min(delay, 60.0) + random.uniform(0, 1.5)
            print(f"[fineweb] net error at offset {offset} (attempt {attempt + 1}/8): {e!r} — "
                  f"backoff {sleep_s:.1f}s", flush=True)
            time.sleep(sleep_s)
            delay *= 2
    raise RuntimeError(f"rows API exhausted 8 retries at offset {offset} — aborting (partial corpus not accepted)")


def _stream_fineweb(seg_norm, seg_freq: Counter, *, passages: int, min_len: int, max_chars: int,
                    trainer_id: str, page_delay: float, min_viable: int) -> dict:
    """Page fineweb via the datasets-server rows API and accumulate a UNIQUE-SEGMENT -> FREQUENCY table
    (flat RAM, never materialises the corpus). Returns provenance {rows_seen, passages_kept, dumps, revision,
    truncated}. Fail-loud below ``min_viable`` passages; ABOVE it, a rate-limit / stream-end is accepted as
    a viable-but-truncated corpus with the shortfall LOGGED (no silent cap — the manifest records it)."""
    import httpx

    kept = 0
    seen = 0
    offset = 0
    dumps: Counter = Counter()
    truncated = None
    t0 = time.time()
    while kept < passages:
        n = PAGE
        try:
            rows = _fetch_page(httpx, offset, n, page_delay=page_delay)
        except RuntimeError as e:
            # rate-limit / API exhaustion: accept a viable-but-truncated corpus rather than throw away
            # ~27k already-streamed passages; below the viability floor it is still a hard failure.
            if kept >= min_viable:
                truncated = f"stream stopped early at offset {offset} ({kept:,} passages): {e}"
                print(f"[fineweb] {truncated} — ACCEPTING viable partial corpus (>= {min_viable:,})", flush=True)
                break
            raise
        if not rows:
            print(f"[fineweb] no rows at offset {offset} — end of stream", flush=True)
            break
        for item in rows:
            row = item.get("row") or {}
            txt = str(row.get("text") or "")
            seen += 1
            if len(txt) < min_len:
                continue
            txt = txt[:max_chars]
            d = row.get("dump")
            if d:
                dumps[str(d)] += 1
            for seg in seg_norm.to_byte_segments(txt):
                seg_freq[bytes(seg)] += 1
            kept += 1
        offset += len(rows)
        if offset % 5000 == 0 or kept >= passages:
            print(f"[fineweb] offset {offset:,} · kept {kept:,}/{passages:,} · "
                  f"unique-segs {len(seg_freq):,} · {time.time() - t0:.0f}s", flush=True)
        if len(rows) < n:
            break
    if kept == 0:
        raise RuntimeError("fineweb stream yielded 0 usable passages — refusing to build on an empty corpus")
    return {"rows_seen": seen, "passages_kept": kept, "dumps": dict(dumps.most_common(12)),
            "config": CONFIG, "split": SPLIT, "truncated": truncated}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=64_000,
                    help="target vocab; default 64000 == matched to 64k_v1 (isolate the corpus variable)")
    ap.add_argument("--passages", type=int, default=300_000, help="fineweb passages to stream into the vocab")
    ap.add_argument("--min-len", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--page-delay", type=float, default=1.0,
                    help="polite base delay (s) between successful rows-API pages (rate-limit courtesy)")
    ap.add_argument("--min-viable", type=int, default=20_000,
                    help="accept a rate-limited partial corpus above this many passages (logged as truncated)")
    ap.add_argument("--validate-tiny", action="store_true", help="smoke: 600 vocab, 2000 passages")
    args = ap.parse_args()

    root = _repo_root()
    sys.path.insert(0, str(root / "src"))
    from qig_coordizer.coordizer import FisherCoordizer
    from qig_coordizer.normalizer import Normalizer
    from qig_coordizer.trainer import CoordinzerTrainer

    target_vocab = 600 if args.validate_tiny else args.vocab
    passages = 2_000 if args.validate_tiny else args.passages
    trainer_id = "CCAa"
    t0 = time.time()

    print(f"[fineweb] DoD-1b coordizer build · vocab={target_vocab:,} · passages={passages:,} · "
          f"corpus={DATASET}[{CONFIG}] STREAMED", flush=True)

    # 1) stream fineweb -> segment-frequency substrate (identical primitive to coordizer_build.build)
    seg_norm = Normalizer(pretokenize=True, max_segment_bytes=MAX_SEG_BYTES)
    seg_freq: Counter = Counter()
    min_viable = 200 if args.validate_tiny else args.min_viable
    stream_prov = _stream_fineweb(seg_norm, seg_freq, passages=passages, min_len=args.min_len,
                                  max_chars=args.max_chars, trainer_id=trainer_id,
                                  page_delay=args.page_delay, min_viable=min_viable)

    # 2) compact (flat_tokens, seg_bounds, weights) — deterministic sorted order (reproducible artifact)
    flat_tokens: list[int] = []
    seg_bounds: list[int] = []
    weights: list[int] = []
    for seg_bytes in sorted(seg_freq):
        seg_bounds.append(len(flat_tokens))
        flat_tokens.extend(seg_bytes)
        weights.append(seg_freq[seg_bytes])
    n_unique = len(seg_bounds)
    compact = len(flat_tokens)
    physical = sum(w * ((seg_bounds[i + 1] if i + 1 < n_unique else compact) - seg_bounds[i])
                   for i, w in enumerate(weights))
    print(f"[fineweb] substrate: {n_unique:,} unique-segs · compact {compact:,} tok · "
          f"physical {physical / 1e6:.0f}M tok · collapse {physical / max(1, compact):.0f}x", flush=True)
    if compact < 5_000:
        raise RuntimeError("segment substrate too small (<5K tokens) — check the fineweb stream")
    del seg_freq

    # 3) train BPE to target vocab (same trainer + params as the canonical build)
    print(f"[fineweb] training coordizer to {target_vocab:,} vocab from scratch…", flush=True)
    ckpt_dir = str(root / "runs" / "coordizer_ckpts")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    trainer = CoordinzerTrainer(target_vocab_size=target_vocab, pretokenize=True,
                                max_segment_bytes=MAX_SEG_BYTES)
    trainer.train(corpus=b"", corpus_segments=(flat_tokens, seg_bounds, weights),
                  verbose=True, checkpoint_dir=ckpt_dir, checkpoint_interval=2000,
                  enable_interrupt=False, use_kernel=False)

    # 4) provenance filename carries the ACTUAL trained vocab, NOT the target — a rate-limited corpus can
    # yield fewer merges than requested (e.g. target 64k -> 32k when the stream is throttled), and the
    # filename MUST NOT lie about vocab (artifact-standard truthfulness; the '64k' filename on a 32k artifact
    # was exactly this bug). If actual << target, that is logged as a known-bias too.
    actual_vocab = len(trainer.vocab)  # trained merges + base bytes (before the +4 geo tags)
    vocab_reached_target = actual_vocab >= int(0.97 * target_vocab)
    out_dir = _repo_root().parent / "qig-packages" / "qig-coordizer" / "checkpoints"
    if not out_dir.exists():  # layout-independent fallback (flat vs qig-packages/ grouping)
        alt = _repo_root().parent / "qig-coordizer" / "checkpoints"
        out_dir = alt if alt.exists() else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    mix_slug = "fineweb-sample10bt" + ("-tiny" if args.validate_tiny else "")
    base = f"coordizer_{date_tag}_{round(actual_vocab / 1000)}k_{mix_slug}"
    nver = 1
    while (out_dir / f"{base}_v{nver}.json").exists():
        nver += 1
    out_path = out_dir / f"{base}_v{nver}.json"

    # 5) register the 4 atomic geo tags above the trained vocab, save final
    tmp = str(out_path.with_suffix(".pretags.json"))
    trainer.save(tmp)
    fc = FisherCoordizer.load(tmp)
    ids = fc.register_special_tokens(GEO_TAGS)
    fc.save(str(out_path))

    # 6) ARTIFACT STANDARD — sidecar manifest + embedded header + sha256 (6e2a5ff6 A/B/C)
    built_iso = datetime.now(timezone.utc).isoformat()
    manifest = {
        "artifact": out_path.name,
        "built_utc": built_iso,
        "trainer": trainer_id,
        "vocab_size": fc.vocab_size,
        "basin_dim": 64,
        "geo_tags": {t: i for t, i in zip(GEO_TAGS, ids)},
        "corpus": {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "streamed": True,
            "encode_once": True,
            "passages_kept": stream_prov["passages_kept"],
            "rows_seen": stream_prov["rows_seen"],
            "top_dumps": stream_prov["dumps"],
            "compact_tokens": compact,
            "physical_tokens_est": physical,
            "min_len_chars": args.min_len,
            "max_chars": args.max_chars,
        },
        "training_config": {
            "target_vocab": target_vocab,
            "pretokenize": True,
            "max_segment_bytes": MAX_SEG_BYTES,
            "checkpoint_interval": 2000,
            "deterministic_segment_order": "sorted (reproducible; merge order-independent)",
        },
        "intended_use": (
            "DoD-1b named-reversal control: a TEXT-HEAVY (web text) coordizer matched to 64k vocab so the "
            "arms bake-off (gk/geo/hybrid) can be re-run with the corpus as the ONLY changed variable vs the "
            "code-balanced coordizer_20260705_64k_v1. Fit for text-domain held-out d_FR + ce_bpb evaluation."
        ),
        "known_biases": [
            "text-heavy: web prose only, NO code — may favor text-native arms and DISADVANTAGE code-sensitive "
            "arms relative to the code-balanced 64k_v1 (this is the effect under test, not a defect).",
            "fineweb sample-10BT is English-dominant, CommonCrawl-derived; carries web-text register + "
            "boilerplate that fineweb's own filtering reduces but does not eliminate.",
        ] + ([f"TRUNCATED: {stream_prov['truncated']} — corpus is a viable partial (>= min_viable), NOT the "
              "full requested passage count; vocab coverage of rarer web-text may be thinner than a full run."]
             if stream_prov.get("truncated") else []),
        "standard": "coordizer-artifact-standard 6e2a5ff6 (filename+sidecar+embedded+run-manifest-field)",
    }
    # (C) embed the same header inside the artifact (load() ignores unknown top-level keys — verified).
    # The embedded header intentionally OMITS sha256 (a file cannot honestly contain its own final digest);
    # the sidecar's sha256 covers the actual on-disk artifact and is the verification anchor.
    with open(out_path) as f:
        art = json.load(f)
    art["_manifest"] = manifest
    with open(out_path, "w") as f:
        json.dump(art, f, indent=2)

    # (B) sidecar manifest with sha256 over the FINAL artifact file (with embedded header)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    sidecar_manifest = dict(manifest)
    sidecar_manifest["sha256"] = sha
    sidecar_manifest["sha256_covers"] = "the final artifact file including its embedded _manifest header"
    sidecar = out_path.with_suffix(".manifest.json")
    with open(sidecar, "w") as f:
        json.dump(sidecar_manifest, f, indent=2)

    # 7) verify the embedded header did not break the loader + geo-tag atomicity (mirrors build())
    fc2 = FisherCoordizer.load(str(out_path))
    atomic = {t: len(fc2.encode(t)) for t in GEO_TAGS}
    atomic_ok = all(v == 1 for v in atomic.values())
    roundtrip_ok = len(fc2.encode("The Fisher-Rao geometry of web text.")) > 0

    # 8) DELIBERATELY NOT registered as coordizer_latest. register_coordizer() unconditionally repoints the
    # 'latest' symlink + manifest, but this is an EXPERIMENT control artifact — making it 'latest' would
    # mis-wire the whole studio (every existing checkpoint is trained on coordizer_20260705_64k_v1; a
    # different-vocab 'latest' breaks their load, observed 2026-07-22). The artifact is fully discoverable
    # by its explicit provenance filename + sidecar .manifest.json; experiment scripts reference it by name.
    print(f"[fineweb] NOT registered as 'latest' (experiment control artifact; canonical latest untouched)", flush=True)

    print("\n" + "=" * 72, flush=True)
    print(f"[fineweb] DONE in {time.time() - t0:.0f}s", flush=True)
    print(f"[fineweb] artifact : {out_path}", flush=True)
    print(f"[fineweb] manifest : {sidecar}", flush=True)
    print(f"[fineweb] vocab {fc.vocab_size:,} (target was {target_vocab:,}; "
          f"{'REACHED' if vocab_reached_target else 'UNDERSHOT — rate-limited corpus, NOT a matched-64k control'}) · "
          f"geo-tags {ids} · atomic {'OK' if atomic_ok else 'FAIL'} · roundtrip {'OK' if roundtrip_ok else 'FAIL'}", flush=True)
    print(f"[fineweb] sha256(file) {hashlib.sha256(out_path.read_bytes()).hexdigest()}", flush=True)
    print(f"[fineweb] intended_use: {manifest['intended_use'][:90]}…", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
