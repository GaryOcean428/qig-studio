"""HuggingFace dataset ingestion — TEXT only, as a curriculum/corpus source.

We use HF as a DATA source (not its tokenizers or models — the geometry stays ours: qig-coordizer +
qigkernels + geocoding). Rather than the heavy ``datasets`` library, this streams rows from the HF
**datasets-server rows API** (paginated JSON over https) — light (httpx only), capped, and never downloads
more than asked. Returns a list of text passages usable by BOTH the coordizer resume-training (vocab) and
the kernel curriculum (training).

None-safe: any network/parse failure returns what it has so far (or []), never raises into a training loop.
"""
from __future__ import annotations

ROWS_API = "https://datasets-server.huggingface.co/rows"
_PAGE = 100   # the rows API caps length per call at 100


def _flatten(v: object) -> str:
    """Text out of any row value: str (parse if it's a JSON list/dict), message-dict {role,content}→content,
    list→joined, dict→joined values. Handles chat/agent-trace rows (fable `messages`, empathetic
    `conversations`) that arrive as nested structures or JSON strings."""
    import json
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in "[{":                       # a JSON-encoded list/dict (e.g. fable `messages`)
            try:
                return _flatten(json.loads(s))
            except Exception:  # noqa: BLE001 — not JSON; use the raw string
                return v
        return v
    if isinstance(v, dict):
        c = v.get("content")
        if isinstance(c, str) and c.strip():    # a chat message {role, content}
            return c
        return "\n".join(_flatten(x) for x in v.values())
    if isinstance(v, list):
        return "\n".join(_flatten(x) for x in v)
    return ""


def _row_text(row: dict, fields: tuple[str, ...]) -> str:
    """Text from a row: first matching field (structure-aware), else flatten all values."""
    for f in fields:
        if f in row:
            t = _flatten(row[f]).strip()
            if t:
                return t
    return "\n".join(_flatten(v) for v in row.values()).strip()


def load_hf_passages(
    dataset: str,
    *,
    config: str = "default",
    split: str = "train",
    text_fields: tuple[str, ...] = ("text", "content", "story", "dialog", "conversation", "messages"),
    limit: int = 1000,
    min_len: int = 40,
    max_chars: int = 4000,
    offset: int = 0,
) -> list[str]:
    """Stream up to ``limit`` text passages from a HF dataset via the datasets-server rows API.

    ``max_chars`` clips each passage (the unbounded-context kernel can take long ones, but clip for sanity);
    ``min_len`` drops trivially short rows. Paginated at 100/call. Light + capped — no ``datasets`` dep."""
    import httpx
    out: list[str] = []
    got = 0
    while got < limit:
        n = min(_PAGE, limit - got)
        params = {"dataset": dataset, "config": config, "split": split, "offset": offset + got, "length": n}
        try:
            r = httpx.get(ROWS_API, params=params, timeout=30.0)
            if r.status_code != 200:
                break
            rows = (r.json() or {}).get("rows") or []
        except Exception:  # noqa: BLE001 — return what we have; never break a training pipeline
            break
        if not rows:
            break
        for item in rows:
            txt = _row_text(item.get("row") or {}, text_fields).strip()
            if len(txt) >= min_len:
                out.append(txt[:max_chars])
        got += len(rows)
        if len(rows) < n:    # last page
            break
    return out


def load_hf_corpus(specs: list[dict], *, total_cap: int | None = None) -> list[str]:
    """Load + concatenate passages from several datasets. Each spec: {dataset, config?, split?, limit?,
    text_fields?}. ``total_cap`` bounds the combined passage count (CPU/disk safety)."""
    passages: list[str] = []
    for s in specs:
        d = dict(s)
        name = d.pop("dataset")
        passages.extend(load_hf_passages(name, **d))
        if total_cap and len(passages) >= total_cap:
            return passages[:total_cap]
    return passages
