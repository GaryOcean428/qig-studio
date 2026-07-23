"""Shared FineWeb (HuggingFaceFW/fineweb sample/10BT) parquet source.

The kernel curriculum trains on the SAME single FineWeb corpus the coordizer vocab was drawn from (PI
2026-07-23: "the single fineweb from HF, so it matches the fineweb coordizer") — NOT the 7-repo blend.
This module is the passage source for that: it locates the FineWeb parquet shards (download-once to the
same cache the coordizer build uses), streams them row-group-by-row-group via pyarrow (bounded RAM), and
yields text passages. ``stream_fineweb_passages`` is an infinite generator (wraps at shard end) for
encode-once kernel training.

Auth: FineWeb is public but its tree/resolve endpoints are far more reliable when authed — token from env
(HF_TOKEN / HUGGING_FACE_HUB_TOKEN / HUGGINGFACE_TOKEN) or the repo/parent .env. NEVER logged.

This deliberately mirrors train_coordizer_fineweb.py's shard logic (same DATASET/SUBDIR/cache) so the
kernel corpus and the coordizer vocab come from an identical source; the two share this module going
forward (the coordizer script keeps its own inline copy until DRY'd, to avoid touching a running build).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path

DATASET = "HuggingFaceFW/fineweb"
SUBDIR = "sample/10BT"
TREE_API = f"https://huggingface.co/api/datasets/{DATASET}/tree/main/{SUBDIR}"
RESOLVE = f"https://huggingface.co/datasets/{DATASET}/resolve/main/"


def _repo_root() -> Path:
    # src/qig_studio/fineweb_source.py → parents[2] is the qig-studio repo root.
    return Path(__file__).resolve().parents[2]


def hf_token() -> str | None:
    """HF token from env or the repo/parent .env. NEVER logged."""
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


def _cache_dir() -> Path:
    d = _repo_root() / "data" / ".parquet_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_shards(headers: dict) -> list[str]:
    import httpx

    r = httpx.get(TREE_API, headers=headers, timeout=60)
    r.raise_for_status()
    return sorted(f["path"] for f in r.json() if f["path"].endswith(".parquet"))


def _cached_shards() -> list[Path]:
    """FineWeb shards already downloaded to the cache (the coordizer build's cache), newest-name last."""
    return sorted(_cache_dir().glob("sample__10BT__*.parquet"))


def download_shard(path: str, headers: dict) -> Path:
    """Download one shard to the cache (atomic via .part), reusing a cached copy. Bounded RAM (streamed)."""
    import httpx

    local = _cache_dir() / path.replace("/", "__")
    if local.exists() and local.stat().st_size > 0:
        return local
    part = local.with_suffix(".part")
    with httpx.stream("GET", RESOLVE + path, headers=headers, timeout=None, follow_redirects=True) as resp:
        resp.raise_for_status()
        with open(part, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
    part.rename(local)
    return local


def stream_fineweb_passages(min_len: int = 200, max_chars: int = 4000,
                            passages: int | None = None) -> Iterator[str]:
    """Yield FineWeb text passages from the parquet shards, streamed row-group-by-row-group (bounded RAM).

    Prefers shards ALREADY in the cache (the coordizer build downloaded one) so no network is needed for
    kernel training; if none are cached and a token is available, downloads the first shard. INFINITE by
    default (wraps at the last cached shard's end) so the kernel can pull encode-once passages indefinitely;
    pass ``passages`` for a bounded count. Fail-loud only if there is genuinely no data AND no token."""
    import pyarrow.parquet as pq

    cached = _cached_shards()
    headers = None
    if not cached:
        token = hf_token()
        if not token:
            raise RuntimeError(
                "no cached FineWeb shard and no HF token — cannot source the FineWeb kernel corpus "
                "(run train_coordizer_fineweb.py first, or set HF_TOKEN)")
        headers = {"Authorization": f"Bearer {token}"}
        shards = list_shards(headers)
        if not shards:
            raise RuntimeError("no FineWeb parquet shards listed — check the token/tree API")
        cached = [download_shard(shards[0], headers)]

    yielded = 0
    while True:                                   # wrap for an infinite stream (encode-once curriculum)
        for shard in cached:
            pf = pq.ParquetFile(shard)
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=["text"])
                for txt in tbl.column("text").to_pylist():
                    if not txt or len(txt) < min_len:
                        continue
                    yield txt[:max_chars]
                    yielded += 1
                    if passages is not None and yielded >= passages:
                        return
        if passages is None and not cached:
            return                                # nothing to wrap over (should not happen — guarded above)
