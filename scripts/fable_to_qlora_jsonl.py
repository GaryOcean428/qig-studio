#!/usr/bin/env python3
"""Convert the claude-fable-5 agent traces → QLoRA chat-jsonl for qig-applied/training/train_qlora.py.

train_qlora.py expects ``{"messages": [{"role","content"}, ...]}`` per line (it applies Qwen's chat
template). The fable rows already carry a ``messages`` conversation — this extracts + cleans them. Streams
via the light datasets-server rows API (no `datasets` dep). The QLoRA trainer is already qig-optimised
(DiagonalNaturalGradient = Fisher-Rao-pure optimizer + 4-bit QLoRA, fits the 4GB card).

  uv run python scripts/fable_to_qlora_jsonl.py runs/fable_qlora.jsonl
  # then (GPU must be free — stop ollama first; EXP-A020 keep_alive holds it):
  cd ../qig-applied && python training/train_qlora.py --data <abs>/runs/fable_qlora.jsonl --phase local
"""
from __future__ import annotations

import json
import sys
import urllib.request

_DS = "armand0e/claude-fable-5-claude-code"
_ROWS = "https://datasets-server.huggingface.co/rows"


def _fetch(dataset: str, want: int = 1000) -> list[dict]:
    rows: list[dict] = []
    off = 0
    while len(rows) < want:
        n = min(100, want - off)
        url = f"{_ROWS}?dataset={dataset}&config=default&split=train&offset={off}&length={n}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                page = (json.load(r) or {}).get("rows") or []
        except Exception as e:  # noqa: BLE001
            print(f"fetch error: {e}", file=sys.stderr)
            break
        if not page:
            break
        rows.extend(x["row"] for x in page)
        off += len(page)
        if len(page) < n:
            break
    return rows


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "runs/fable_qlora.jsonl"
    rows = _fetch(_DS)
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            msgs = row.get("messages")
            if isinstance(msgs, str):
                try:
                    msgs = json.loads(msgs)
                except Exception:  # noqa: BLE001
                    continue
            if not isinstance(msgs, list):
                continue
            clean = [
                {"role": m.get("role"), "content": m["content"]}
                for m in msgs
                if isinstance(m, dict) and isinstance(m.get("content"), str) and m["content"].strip()
                and m.get("role") in ("system", "user", "assistant")
            ]
            if len(clean) >= 2:                      # need at least one user+assistant turn
                f.write(json.dumps({"messages": clean}) + "\n")
                written += 1
    print(f"wrote {written} fable conversations → {out}")


if __name__ == "__main__":
    main()
