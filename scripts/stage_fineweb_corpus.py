#!/usr/bin/env python3
"""Stage the FULL FineWeb sample/10BT corpus under a CLEAR naming convention + a manifest.

The recurring pain: the flat ``data/.parquet_cache/sample__10BT__000_00000.parquet`` mixed the FineWeb
basis in with the 7-repo blend shards under opaque HF-path names, and only 1 of 15 shards was staged — so
"is this the full corpus?" and "docs or passages?" were unanswerable. This establishes ONE convention.

Layout (clear, self-describing):
    data/corpora/fineweb-sample10bt/
        fineweb-sample10bt.000-of-015.parquet     # shard index / total is IN the name
        ...
        fineweb-sample10bt.014-of-015.parquet
        MANIFEST.json                             # dataset, slice, per-shard sha+rows, totals, complete flag

MANIFEST.json is the single source of truth for corpus identity (the truncated-world guard reads it):
completeness is EXPLICIT (``complete`` = all 15 shards staged), the count is DOCUMENTS (parquet rows) with
a passages estimate alongside so the metric is never ambiguous again.

Idempotent + resumable: skips shards already staged (by size), reuses an existing legacy shard by renaming
it in, downloads the rest (batched), and rewrites the manifest each run. Auth: HF token from env/.env
(never logged). Usage: python scripts/stage_fineweb_corpus.py [--shards N] [--workers K]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
from pathlib import Path

CORPUS = "fineweb-sample10bt"
N_TOTAL = 15                    # full sample/10BT shard count (HF tree, 2026-07-24)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _corpus_dir() -> Path:
    d = _repo_root() / "data" / "corpora" / CORPUS
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clear_name(idx: int) -> str:
    return f"{CORPUS}.{idx:03d}-of-{N_TOTAL:03d}.parquet"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(dstdir: Path) -> dict:
    import pyarrow.parquet as pq

    shards = sorted(dstdir.glob(f"{CORPUS}.*-of-*.parquet"))
    entries, lines, total_rows = [], [], 0
    for s in shards:
        rows = int(pq.ParquetFile(s).metadata.num_rows)
        sha = _sha256(s)
        total_rows += rows
        entries.append({"name": s.name, "sha256": sha, "num_rows": rows})
        lines.append(f"{s.name}:{sha}:{rows}")
    manifest_sha = hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest() if lines else None
    man = {
        "corpus": CORPUS, "dataset": "HuggingFaceFW/fineweb", "slice": "sample/10BT",
        "n_shards_total": N_TOTAL, "n_shards_staged": len(shards),
        "complete": len(shards) == N_TOTAL,
        "total_documents": total_rows,                      # parquet rows = DOCUMENTS (the guard's count)
        "passages_estimate_note": "the studio chunks each document into ~1.5-1.7 passages (min_len 200, "
                                  "max_chars 4000); 'segment/passage' counts elsewhere are POST-CHUNK and "
                                  "are NOT this document count — do not conflate.",
        "manifest_sha256": manifest_sha, "shards": entries,
        "naming_convention": f"{CORPUS}.<idx:03>-of-<total:03>.parquet",
    }
    (dstdir / "MANIFEST.json").write_text(json.dumps(man, indent=2))
    return man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=N_TOTAL, help="stage the first N shards (default all 15)")
    ap.add_argument("--workers", type=int, default=3, help="concurrent shard downloads (batched)")
    ap.add_argument("--manifest-only", action="store_true", help="just (re)write the manifest from staged shards")
    args = ap.parse_args()

    sys.path.insert(0, str(_repo_root() / "src"))
    from qig_studio.fineweb_source import RESOLVE, hf_token, list_shards

    dst = _corpus_dir()
    if args.manifest_only:
        man = _write_manifest(dst)
        print(json.dumps({k: man[k] for k in ("n_shards_staged", "complete", "total_documents",
                                              "manifest_sha256")}, indent=2))
        return 0

    tok = hf_token()
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    hf_shards = sorted(list_shards(headers))[: args.shards]

    # reuse a legacy shard already on disk (the old .parquet_cache/sample__10BT__000_00000.parquet)
    legacy = _repo_root() / "data" / ".parquet_cache" / "sample__10BT__000_00000.parquet"
    tgt0 = dst / _clear_name(0)
    if legacy.exists() and not tgt0.exists():
        print(f"[stage] reusing legacy shard 000 → {tgt0.name} (no re-download)", flush=True)
        legacy.rename(tgt0)

    def _fetch(i_path):
        import httpx
        i, hf_path = i_path
        tgt = dst / _clear_name(i)
        if tgt.exists() and tgt.stat().st_size > 0:
            return f"skip {tgt.name} (present)"
        part = tgt.with_suffix(".part")
        with httpx.stream("GET", RESOLVE + hf_path, headers=headers, timeout=None,
                          follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(part, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    f.write(chunk)
        part.rename(tgt)
        return f"fetched {tgt.name} ({tgt.stat().st_size/1e9:.2f} GB)"

    todo = [(i, p) for i, p in enumerate(hf_shards)]
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(_fetch, todo):
            print(f"[stage] {msg}", flush=True)

    man = _write_manifest(dst)
    print(f"[stage] DONE — {man['n_shards_staged']}/{N_TOTAL} shards, complete={man['complete']}, "
          f"total_documents={man['total_documents']:,}, manifest_sha={man['manifest_sha256'][:12]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
