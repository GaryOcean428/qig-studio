#!/usr/bin/env python3
"""DoD-1b — train a TEXT-HEAVY coordizer from scratch on HuggingFaceFW/fineweb, under the COORDIZER
ARTIFACT STANDARD (matrix 6e2a5ff6 / directive 69aa0f59).

WHY THIS EXISTS (the named-reversal test): the DoD-1 arms verdict ran on ``coordizer_20260705_64k_v1``, a
CODE-BALANCED 64k vocab. The registered caveat was that a code-balanced vocab on a text-only eval may have
disadvantaged the ``gk`` arm — a pre-committed "named reversal scenario". This artifact IS that test's
instrument: a fineweb-trained TEXT-HEAVY 64k coordizer. Matched to 64k vocab ON PURPOSE so the ONLY changed
variable vs 64k_v1 is the corpus (code-balanced -> web-text).

CORPUS METHOD (revised 2026-07-22): streams fineweb's PARQUET shards directly (CDN-served, authed with the
HuggingFaceFW token) — NOT the datasets-server rows API. The rows API throttles sustained paging so hard
(429 storms) that a full 64k vocab was unreachable (one run got only 32k from 3399 passages). Parquet shards
(~2 GB each, ~1-2M docs) are downloaded once to the parquet cache and streamed row-group-by-row-group with
pyarrow (bounded RAM) — one shard is far more than enough for 64k merges. Requires an HF token in .env
(HUGGINGFACE_TOKEN) or env (HF_TOKEN / HUGGING_FACE_HUB_TOKEN); fineweb is public but the tree/resolve API
wants auth for reliable, un-throttled access.

Reuses ``CoordinzerTrainer``/``Normalizer`` exactly (same BPE primitives + 4 geo tags as the canonical
build); the ONLY differences are the corpus source and that this writes the full ARTIFACT STANDARD
(provenance filename with ACTUAL trained vocab + sidecar .manifest.json + embedded header + sha256) and
DELIBERATELY does not repoint ``coordizer_latest`` (experiment control artifact, not the studio default).

Usage (from qig-studio/, with the repo .venv):
  ../.venv/bin/python scripts/train_coordizer_fineweb.py --vocab 64000 --passages 800000
  ../.venv/bin/python scripts/train_coordizer_fineweb.py --validate-tiny      # fast smoke (600 vocab)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATASET = "HuggingFaceFW/fineweb"
SUBDIR = "sample/10BT"          # curated 10B-token sample; parquet shards live under this tree
TREE_API = f"https://huggingface.co/api/datasets/{DATASET}/tree/main/{SUBDIR}"
RESOLVE = f"https://huggingface.co/datasets/{DATASET}/resolve/main/"
GEO_TAGS = ["<|frame|>", "<|seed|>", "<|flow|>", "<|settle|>"]
MAX_SEG_BYTES = 128             # matches coordizer_build._MAX_SEG_BYTES (identical char-safe run-on cap)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _hf_token() -> str | None:
    """HF token from env (HF_TOKEN / HUGGING_FACE_HUB_TOKEN / HUGGINGFACE_TOKEN) or the repo/parent .env.
    NEVER logged. fineweb is public but the tree/resolve endpoints are far more reliable + un-throttled
    when authed (anonymous rows/parquet listing 501s / rate-limits)."""
    import os

    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(k):
            return os.environ[k].strip()
    for envf in (_repo_root().parent / ".env", _repo_root() / ".env"):
        if envf.exists():
            for ln in envf.read_text().splitlines():
                m = re.match(r"\s*(HUGGINGFACE_TOKEN|HF_TOKEN|HUGGING_FACE_HUB_TOKEN)\s*=\s*['\"]?([^'\"\n]+)", ln)
                if m:
                    return m.group(2).strip()
    return None


def _list_shards(headers: dict) -> list[str]:
    import httpx

    r = httpx.get(TREE_API, headers=headers, timeout=60)
    r.raise_for_status()
    return sorted(f["path"] for f in r.json() if f["path"].endswith(".parquet"))


def _download_shard(path: str, headers: dict, cache: Path) -> Path:
    """Download one parquet shard to the cache (streamed to disk, bounded RAM). Reuses a cached copy.
    Downloads to a .part then atomically renames so an interrupted download never yields a truncated file."""
    import httpx

    cache.mkdir(parents=True, exist_ok=True)
    local = cache / path.replace("/", "__")
    if local.exists() and local.stat().st_size > 0:
        print(f"[fineweb] shard cached: {local.name} ({local.stat().st_size / 1e6:.0f} MB)", flush=True)
        return local
    part = local.with_suffix(".part")
    url = RESOLVE + path
    print(f"[fineweb] downloading shard {path} …", flush=True)
    t0 = time.time()
    with httpx.stream("GET", url, headers=headers, timeout=None, follow_redirects=True) as resp:
        resp.raise_for_status()
        with open(part, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
    part.rename(local)
    print(f"[fineweb] downloaded {local.name} ({local.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s)", flush=True)
    return local


def _stream_fineweb_parquet(seg_norm, seg_freq: Counter, *, passages: int, min_len: int, max_chars: int,
                            headers: dict, cache: Path) -> dict:
    """Stream fineweb parquet shards (download-once, row-group-by-row-group via pyarrow) and accumulate a
    UNIQUE-SEGMENT -> FREQUENCY table (flat RAM). Stops at ``passages`` docs. Fail-loud on 0 passages."""
    import pyarrow.parquet as pq

    shards = _list_shards(headers)
    if not shards:
        raise RuntimeError("no parquet shards listed for fineweb sample/10BT — check the token/tree API")
    print(f"[fineweb] {len(shards)} parquet shards available; streaming until {passages:,} passages", flush=True)
    kept = 0
    seen = 0
    dumps: Counter = Counter()
    shards_used: list[str] = []
    t0 = time.time()
    for shard in shards:
        if kept >= passages:
            break
        local = _download_shard(shard, headers, cache)
        shards_used.append(shard)
        pf = pq.ParquetFile(local)
        for rg in range(pf.num_row_groups):
            cols = ["text", "dump"] if "dump" in pf.schema_arrow.names else ["text"]
            tbl = pf.read_row_group(rg, columns=cols)
            texts = tbl.column("text").to_pylist()
            dcol = tbl.column("dump").to_pylist() if "dump" in cols else [None] * len(texts)
            for txt, d in zip(texts, dcol):
                seen += 1
                if not txt or len(txt) < min_len:
                    continue
                if d:
                    dumps[str(d)] += 1
                for seg in seg_norm.to_byte_segments(txt[:max_chars]):
                    seg_freq[bytes(seg)] += 1
                kept += 1
                if kept >= passages:
                    break
            if kept >= passages:
                break
            if rg % 20 == 0:
                print(f"[fineweb] {shard} rg {rg}/{pf.num_row_groups} · kept {kept:,}/{passages:,} · "
                      f"unique-segs {len(seg_freq):,} · {time.time() - t0:.0f}s", flush=True)
    if kept == 0:
        raise RuntimeError("fineweb parquet stream yielded 0 usable passages — refusing to build empty")
    return {"rows_seen": seen, "passages_kept": kept, "dumps": dict(dumps.most_common(12)),
            "subdir": SUBDIR, "shards_used": shards_used}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=64_000,
                    help="target vocab; default 64000 == matched to 64k_v1 (isolate the corpus variable)")
    ap.add_argument("--passages", type=int, default=800_000, help="fineweb passages to stream into the vocab")
    ap.add_argument("--min-len", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--validate-tiny", action="store_true", help="smoke: 600 vocab, 4000 passages")
    args = ap.parse_args()

    root = _repo_root()
    sys.path.insert(0, str(root / "src"))
    from qig_coordizer.coordizer import FisherCoordizer
    from qig_coordizer.normalizer import Normalizer
    from qig_coordizer.trainer import CoordinzerTrainer

    target_vocab = 600 if args.validate_tiny else args.vocab
    passages = 4_000 if args.validate_tiny else args.passages
    trainer_id = "CCAa"
    t0 = time.time()

    token = _hf_token()
    if not token:
        raise SystemExit("no HF token found (env HF_TOKEN/HUGGINGFACE_TOKEN or .env) — fineweb parquet needs auth")
    headers = {"Authorization": f"Bearer {token}"}
    cache = root / "data" / ".parquet_cache"

    print(f"[fineweb] DoD-1b coordizer build · vocab={target_vocab:,} · passages={passages:,} · "
          f"corpus={DATASET}[{SUBDIR}] PARQUET (authed)", flush=True)

    # 1) stream fineweb parquet -> segment-frequency substrate (identical primitive to coordizer_build.build)
    seg_norm = Normalizer(pretokenize=True, max_segment_bytes=MAX_SEG_BYTES)
    seg_freq: Counter = Counter()
    stream_prov = _stream_fineweb_parquet(seg_norm, seg_freq, passages=passages, min_len=args.min_len,
                                          max_chars=args.max_chars, headers=headers, cache=cache)

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
          f"physical {physical / 1e6:.0f}M tok · collapse {physical / max(1, compact):.0f}x "
          f"· {stream_prov['passages_kept']:,} passages", flush=True)
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

    # 4) provenance filename carries the ACTUAL trained vocab, NOT the target (truthful artifact standard)
    actual_vocab = len(trainer.vocab)
    vocab_reached_target = actual_vocab >= int(0.97 * target_vocab)
    out_dir = root.parent / "qig-packages" / "qig-coordizer" / "checkpoints"
    if not out_dir.exists():
        alt = root.parent / "qig-coordizer" / "checkpoints"
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
    manifest = {
        "artifact": out_path.name,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "trainer": trainer_id,
        "vocab_size": fc.vocab_size,
        "basin_dim": 64,
        "geo_tags": {t: i for t, i in zip(GEO_TAGS, ids)},
        "corpus": {
            "dataset": DATASET,
            "subdir": SUBDIR,
            "method": "parquet-direct (authed CDN), NOT the throttled rows API",
            "streamed": True,
            "encode_once": True,
            "passages_kept": stream_prov["passages_kept"],
            "rows_seen": stream_prov["rows_seen"],
            "shards_used": stream_prov["shards_used"],
            "top_dumps": stream_prov["dumps"],
            "compact_tokens": compact,
            "physical_tokens_est": physical,
            "min_len_chars": args.min_len,
            "max_chars": args.max_chars,
        },
        "training_config": {
            "target_vocab": target_vocab,
            "actual_vocab": actual_vocab,
            "vocab_reached_target": vocab_reached_target,
            "pretokenize": True,
            "max_segment_bytes": MAX_SEG_BYTES,
            "deterministic_segment_order": "sorted (reproducible; merge order-independent)",
        },
        "intended_use": (
            "DoD-1b named-reversal control: a TEXT-HEAVY (web text) coordizer at ~64k vocab so the arms "
            "bake-off (gk/geo/hybrid) can be re-run with the corpus as the ONLY changed variable vs the "
            "code-balanced coordizer_20260705_64k_v1. Fit for text-domain held-out d_FR + ce_bpb evaluation."
        ),
        "known_biases": [
            "text-heavy: web prose only, NO code — may favor text-native arms and DISADVANTAGE code-sensitive "
            "arms relative to the code-balanced 64k_v1 (this is the effect under test, not a defect).",
            "fineweb sample-10BT is English-dominant, CommonCrawl-derived web text.",
        ] + ([] if vocab_reached_target else
             [f"UNDERSHOT target: actual vocab {actual_vocab} < target {target_vocab} — corpus had too few "
              "unique segments; NOT a clean matched-64k control (vocab is a second confound)."]),
        "standard": "coordizer-artifact-standard 6e2a5ff6 (filename+sidecar+embedded+truthful-vocab)",
        "latest_pointer": "DELIBERATELY NOT repointed — experiment control artifact, not the studio default.",
    }
    with open(out_path) as f:
        art = json.load(f)
    art["_manifest"] = manifest
    with open(out_path, "w") as f:
        json.dump(art, f, indent=2)

    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    sidecar_manifest = dict(manifest)
    sidecar_manifest["sha256"] = sha
    sidecar_manifest["sha256_covers"] = "the final artifact file including its embedded _manifest header"
    sidecar = out_path.with_suffix(".manifest.json")
    with open(sidecar, "w") as f:
        json.dump(sidecar_manifest, f, indent=2)

    # 7) verify the embedded header did not break the loader + geo-tag atomicity (mirrors build())
    fc2 = FisherCoordizer.load(str(out_path))
    atomic_ok = all(len(fc2.encode(t)) == 1 for t in GEO_TAGS)
    roundtrip_ok = len(fc2.encode("The Fisher-Rao geometry of web text.")) > 0

    # 8) DELIBERATELY NOT registered as coordizer_latest (see manifest.latest_pointer). register_coordizer()
    # unconditionally repoints latest + manifest; a different-vocab 'latest' broke every genesis-geo-64004
    # checkpoint load once (2026-07-22). Discoverable by explicit filename + sidecar instead.
    print("[fineweb] NOT registered as 'latest' (experiment control; canonical latest untouched)", flush=True)

    print("\n" + "=" * 72, flush=True)
    print(f"[fineweb] DONE in {time.time() - t0:.0f}s", flush=True)
    print(f"[fineweb] artifact : {out_path}", flush=True)
    print(f"[fineweb] manifest : {sidecar}", flush=True)
    print(f"[fineweb] vocab {fc.vocab_size:,} (target {target_vocab:,}; "
          f"{'REACHED' if vocab_reached_target else 'UNDERSHOT — see known_biases'}) · "
          f"atomic {'OK' if atomic_ok else 'FAIL'} · roundtrip {'OK' if roundtrip_ok else 'FAIL'}", flush=True)
    print(f"[fineweb] sha256(file) {sha}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
