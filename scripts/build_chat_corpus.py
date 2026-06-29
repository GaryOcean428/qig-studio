#!/usr/bin/env python3
"""Build the REBUILD corpus: all HF datasets we use (NO academic markdown), normalised to the studio's
STANDARD PROMPT TEMPLATE (prompt_template.py: <|system|>/<|user|>/<|assistant|>/<|end|>), lightly cleaned
(code + unicode PRESERVED — not the ASCII-only academic sanitiser), sharded to data/chat_corpus/.

This is the foundation for the from-scratch coordizer + kernel rebuild (PI 2026-06-29): everyday + agentic
+ CODE first, academic re-curated separately later. Datasets (rows-API, light httpx — no `datasets` dep):
  - roneneldan/TinyStories                         narrative      -> assistant completion
  - Estwld/empathetic_dialogues_llm                conversations  -> chat
  - Anthropic/hh-rlhf (chosen)                      Human/Assistant text -> chat
  - armand0e/claude-fable-5-claude-code            messages       -> chat   (agentic)
  - PawanKrd/claude-fable-5-code                    messages       -> chat   (CODE agentic)
  - WithinUsAI/GPT_5.5_Distilled                    <|user|>/<|assistant|> text -> re-templated chat
  - mlabonne/open-perfectblend                      conversations (ShareGPT) -> chat (incl. evol-codealpaca CODE)

Usage:
  uv run python scripts/build_chat_corpus.py                 # full build
  uv run python scripts/build_chat_corpus.py --validate-tiny # fast proof (120 rows/dataset)
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

# (dataset, kind, fields, limit). kind drives normalisation → a list of {role,content} turns (or raw text).
_SPECS = [
    ("roneneldan/TinyStories", "narrative", ("text",), 2500),
    ("Estwld/empathetic_dialogues_llm", "conversations", ("conversations",), 1500),
    ("Anthropic/hh-rlhf", "hh", ("chosen",), 1500),
    ("armand0e/claude-fable-5-claude-code", "messages", ("messages",), 2500),
    ("PawanKrd/claude-fable-5-code", "messages", ("messages",), 4000),
    ("WithinUsAI/GPT_5.5_Distilled", "tagged", ("text",), 4000),
    ("mlabonne/open-perfectblend", "conversations", ("conversations",), 5000),
]
_SHARD = 1000

_ZERO_WIDTH = "".join(["​", "‌", "‍", "﻿", "⁠", "­"])
_HH_RE = re.compile(r"\n\n(Human|Assistant):\s*", re.S)


def _clean(text: str) -> str:
    """Light clean — strip zero-width/control (keep \n \t), collapse blank-line runs. PRESERVES code,
    symbols and unicode (the from-scratch coordizer tokenises them; no ASCII transliteration)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    out = [ch for ch in text if ch not in _ZERO_WIDTH and (ord(ch) >= 32 or ch in "\n\t")]
    s = "".join(out)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _hh_to_messages(text: str) -> list[dict]:
    """Anthropic hh-rlhf 'chosen' is '\\n\\nHuman: …\\n\\nAssistant: …' — split into role turns."""
    parts = _HH_RE.split(text)
    msgs: list[dict] = []
    # parts = [pre, role, body, role, body, …]; ignore pre
    for i in range(1, len(parts) - 1, 2):
        role = "user" if parts[i] == "Human" else "assistant"
        body = (parts[i + 1] or "").strip()
        if body:
            msgs.append({"role": role, "content": body})
    return msgs


def _row_to_text(row: dict, kind: str, fields: tuple[str, ...]) -> str:
    from qig_studio.prompt_template import as_completion, format_chat, retag
    val = next((row[f] for f in fields if f in row and row[f] is not None), None)
    if val is None:
        return ""
    if kind == "narrative":
        return as_completion(_clean(val if isinstance(val, str) else str(val)))
    if kind == "messages":
        msgs = val if isinstance(val, list) else []
        return format_chat([{"role": m.get("role") or m.get("from"),
                             "content": _clean(m.get("content") or m.get("value") or "")} for m in msgs
                            if isinstance(m, dict)])
    if kind == "conversations":
        msgs = val if isinstance(val, list) else []
        return format_chat([{"role": m.get("from") or m.get("role"),
                             "content": _clean(m.get("value") or m.get("content") or "")} for m in msgs
                            if isinstance(m, dict)])
    if kind == "hh":
        return format_chat([{**m, "content": _clean(m["content"])} for m in _hh_to_messages(str(val))])
    if kind == "tagged":                       # already <|user|>/<|assistant|> — retag ON THE FLY to ours
        return retag(_clean(str(val)))
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-len", type=int, default=24)
    ap.add_argument("--validate-tiny", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    out_dir = Path(args.out) if args.out else (root / "data" / "chat_corpus")
    blocks: list[str] = []
    for dataset, kind, fields, limit in _SPECS:
        lim = 120 if args.validate_tiny else limit
        # pull RAW rows (no field flattening) so kind-specific normalisation sees the real structure
        raw = _raw_rows(dataset, fields, lim)
        from qig_studio.prompt_template import SETTLE
        kept = 0
        for row in raw:
            txt = _row_to_text(row, kind, fields)
            cut = txt.rfind(SETTLE)                       # trim trailing UNANSWERED turns: a training block must
            if cut == -1:                                # end at a completed seed->flow->settle navigation
                continue                                 # (no completed exchange at all → skip)
            txt = txt[:cut + len(SETTLE)]
            if len(txt) >= args.min_len:
                blocks.append(txt)
                kept += 1
        print(f"[chat]   {dataset} ({kind}): {len(raw)} rows -> {kept} blocks", flush=True)

    if not blocks:
        print("[chat] no blocks (HF unreachable?) — aborting.", flush=True)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("chat_*.txt"):
        old.unlink()
    # one block per record; shards separate blocks by a blank line + a form-feed sentinel so the loader can
    # split on the sentinel (template text itself contains blank lines).
    SEP = "\n\f\n"
    for i in range(0, len(blocks), _SHARD):
        (out_dir / f"chat_{i // _SHARD:03d}.txt").write_text(SEP.join(blocks[i:i + _SHARD]) + "\n", encoding="utf-8")
    print(f"\n[chat] DONE -> {out_dir} | {len(blocks):,} templated blocks, {((len(blocks)-1)//_SHARD)+1} shards", flush=True)


def _raw_rows(dataset: str, fields: tuple[str, ...], limit: int) -> list[dict]:
    """Fetch RAW row dicts (not flattened) via the datasets-server rows API, paginated at 100."""
    import time as _t

    import httpx
    out: list[dict] = []
    got = 0
    while got < limit:
        n = min(100, limit - got)
        rows = None
        for attempt in range(4):                 # RETRY transient hiccups (rate-limit/5xx) so a flaky page
            try:                                  # never silently drops a whole dataset
                r = httpx.get("https://datasets-server.huggingface.co/rows",
                              params={"dataset": dataset, "config": "default", "split": "train",
                                      "offset": got, "length": n}, timeout=45.0)
                if r.status_code == 200:
                    rows = (r.json() or {}).get("rows") or []
                    break
                if r.status_code in (429, 500, 502, 503):
                    _t.sleep(2.0 * (attempt + 1))
                    continue
                break                             # hard error (404 etc.) — stop
            except Exception:  # noqa: BLE001
                _t.sleep(2.0 * (attempt + 1))
        if not rows:
            break
        out.extend(item.get("row") or {} for item in rows)
        got += len(rows)
        if len(rows) < n:
            break
    return out


if __name__ == "__main__":
    main()
