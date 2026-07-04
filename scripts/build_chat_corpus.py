#!/usr/bin/env python3
"""Build the REBUILD corpus from the FULL HuggingFace datasets (PI: "full datasets, all 7"), normalised to
the studio's GEOMETRY-NATIVE template (prompt_template.py: frame/seed/flow/settle), lightly cleaned (code +
unicode PRESERVED), sharded to data/chat_corpus/. Full datasets = ~3.95M rows / ~2.76GB parquet, so we
stream the parquet files directly (the paginated rows-API can't do millions) and write shards incrementally
(bounded memory). Parquet is downloaded once to data/.parquet_cache/ (gitignored) and reused.

Datasets (kind drives normalisation → role/content turns or raw text):
  roneneldan/TinyStories                 narrative      -> seedless flow      (2.14M)
  Estwld/empathetic_dialogues_llm        conversations  -> chat               (25k)
  Anthropic/hh-rlhf (chosen)              hh text        -> chat               (169k)
  armand0e/claude-fable-5-claude-code    messages       -> chat (agentic)     (63)
  PawanKrd/claude-fable-5-code            messages       -> chat (CODE)        (603)
  WithinUsAI/GPT_5.5_Distilled            <|user|> text  -> retag chat         (18k)
  mlabonne/open-perfectblend              conversations  -> chat (incl. CODE)  (1.42M)

Usage:
  uv run python scripts/build_chat_corpus.py                 # full build (long; downloads ~2.76GB once)
  uv run python scripts/build_chat_corpus.py --max-per 2000  # cap rows/dataset (fast dev build)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from pathlib import Path

_SPECS = [
    ("roneneldan/TinyStories", "narrative", ("text",)),
    ("Estwld/empathetic_dialogues_llm", "conversations", ("conversations",)),
    ("Anthropic/hh-rlhf", "hh", ("chosen",)),
    ("armand0e/claude-fable-5-claude-code", "messages", ("messages",)),
    ("PawanKrd/claude-fable-5-code", "messages", ("messages",)),
    ("WithinUsAI/GPT_5.5_Distilled", "tagged", ("text",)),
    ("mlabonne/open-perfectblend", "conversations", ("conversations",)),
]
_SHARD = 2000
_ZERO_WIDTH = "".join(["​", "‌", "‍", "﻿", "⁠", "­"])
_HH_RE = re.compile(r"\n\n(Human|Assistant):\s*", re.S)


def _hf_headers() -> dict:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        tp = Path.home() / ".cache" / "huggingface" / "token"
        tok = tp.read_text().strip() if tp.exists() else None
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _clean(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    out = [ch for ch in text if ch not in _ZERO_WIDTH and (ord(ch) >= 32 or ch in "\n\t")]
    s = "".join(out)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _hh_to_messages(text: str) -> list[dict]:
    parts = _HH_RE.split(text)
    msgs: list[dict] = []
    for i in range(1, len(parts) - 1, 2):
        role = "user" if parts[i] == "Human" else "assistant"
        body = (parts[i + 1] or "").strip()
        if body:
            msgs.append({"role": role, "content": body})
    return msgs


def _as_dicts(seq) -> list[dict]:
    """Normalise a turn-list whose elements may be dicts OR JSON-strings (armand0e/fable-claude-code stores
    `messages` as a list of JSON strings in parquet) into a list of dicts."""
    import json
    out: list[dict] = []
    for m in (seq if isinstance(seq, (list, tuple)) else []):
        if isinstance(m, dict):
            out.append(m)
        elif isinstance(m, str):
            try:
                d = json.loads(m)
                if isinstance(d, dict):
                    out.append(d)
            except Exception:  # noqa: BLE001 — not JSON; skip this element
                pass
    return out


def _row_to_text(row: dict, kind: str, fields: tuple[str, ...]) -> str:
    from qig_studio.prompt_template import as_completion, format_chat, retag
    val = next((row[f] for f in fields if f in row and row[f] is not None), None)
    if val is None:
        return ""
    if kind == "narrative":
        return as_completion(_clean(val if isinstance(val, str) else str(val)))
    if kind == "messages":
        return format_chat([{"role": m.get("role") or m.get("from"),
                             "content": _clean(m.get("content") or m.get("value") or "")}
                            for m in _as_dicts(val)])
    if kind == "conversations":
        return format_chat([{"role": m.get("from") or m.get("role"),
                             "content": _clean(m.get("value") or m.get("content") or "")}
                            for m in _as_dicts(val)])
    if kind == "hh":
        return format_chat([{**m, "content": _clean(m["content"])} for m in _hh_to_messages(str(val))])
    if kind == "tagged":
        return retag(_clean(str(val)))
    return ""


def _parquet_urls(dataset: str, headers: dict) -> list[str]:
    import httpx
    r = httpx.get("https://datasets-server.huggingface.co/parquet",
                  params={"dataset": dataset}, headers=headers, timeout=60.0)
    r.raise_for_status()
    return [f["url"] for f in (r.json().get("parquet_files") or []) if f.get("split") == "train"]


def _parquet_rows(dataset: str, headers: dict, cache: Path, max_per: int | None):
    """Yield row dicts from the dataset's FULL train parquet (download-once to cache, stream in batches).

    Robust to a failing/empty network listing: if ``_parquet_urls`` returns nothing (the HF
    datasets-server intermittently returns an empty parquet list — e.g. open-perfectblend), FALL BACK
    to the already-cached ``<dataset>__*.parquet`` files. Without this a dataset with ~1.5GB of cached
    shards is silently dropped from the corpus (the perfectblend 0-rows bug, 2026-07-04)."""
    import httpx
    import pyarrow.parquet as pq
    prefix = dataset.replace("/", "__") + "__"
    try:
        urls = _parquet_urls(dataset, headers)
    except Exception:  # noqa: BLE001 — network listing failure must not drop a cached dataset
        urls = []
    locals_to_read: list[Path] = []
    for url in urls:
        local = cache / (prefix + url.rstrip("/").split("/")[-1])
        if not local.exists() or local.stat().st_size == 0:
            tmp = local.with_suffix(local.suffix + ".part")
            with httpx.stream("GET", url, headers=headers, timeout=600.0, follow_redirects=True) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_bytes(1 << 20):
                        fh.write(chunk)
            tmp.rename(local)
        locals_to_read.append(local)
    if not locals_to_read:  # network listing empty/failed → use whatever shards are already cached
        locals_to_read = sorted(p for p in cache.glob(prefix + "*.parquet") if p.stat().st_size > 0)
    n = 0
    for local in locals_to_read:
        for batch in pq.ParquetFile(local).iter_batches(batch_size=2000):
            for row in batch.to_pylist():
                yield row
                n += 1
                if max_per and n >= max_per:
                    return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-len", type=int, default=24)
    ap.add_argument("--max-per", type=int, default=0, help="cap rows/dataset (0 = FULL dataset)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from qig_studio.prompt_template import SETTLE

    out_dir = Path(args.out) if args.out else (root / "data" / "chat_corpus")
    cache = root / "data" / ".parquet_cache"
    cache.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("chat_*.txt"):
        old.unlink()
    headers = _hf_headers()
    max_per = args.max_per or None
    SEP = "\n\f\n"

    buf: list[str] = []
    shard_i = 0
    totals: dict[str, int] = {}

    def flush() -> None:
        nonlocal shard_i
        if not buf:
            return
        (out_dir / f"chat_{shard_i:04d}.txt").write_text(SEP.join(buf) + "\n", encoding="utf-8")
        shard_i += 1
        buf.clear()

    for dataset, kind, fields in _SPECS:
        kept = 0
        try:
            for row in _parquet_rows(dataset, headers, cache, max_per):
                txt = _row_to_text(row, kind, fields)
                cut = txt.rfind(SETTLE)                       # trim to last completed seed->flow->settle
                if cut == -1:
                    continue
                txt = txt[:cut + len(SETTLE)]
                if len(txt) >= args.min_len:
                    buf.append(txt)
                    kept += 1
                    if len(buf) >= _SHARD:
                        flush()
        except Exception as e:  # noqa: BLE001 — one dataset failing must not lose the rest
            print(f"[chat]   {dataset}: ERROR after {kept} blocks: {str(e)[:80]}", flush=True)
        totals[dataset] = kept
        print(f"[chat]   {dataset} ({kind}): {kept:,} blocks", flush=True)
    flush()

    total = sum(totals.values())
    print(f"\n[chat] DONE -> {out_dir} | {total:,} blocks across {shard_i} shards", flush=True)
    if not total:
        sys.exit(1)


if __name__ == "__main__":
    main()
