"""Honest measurement plumbing — council ruling 2026-07-15 (P0, gates everything).

Fixes the v2.1 measurement failure (passage-level hash split -> same-document leakage inflated
the mid-panel 0.081 vs distant 0.003) with the council's corrected design:

  1. DOCUMENT-level split — hash the SOURCE DOC id, never the passage.
  2. NEAR-DUPLICATE bucketing — n-gram Jaccard > threshold forces docs into the same bucket
     (plain doc-id hashing misses boilerplate/mirrors/quoted text).
  3. TWO-TIER held set — DEV/STOP panel (watched freely for early-stop + gap guard) +
     LOCKED SCIENCE panel (evaluated ONCE at stop; adaptive selection on it voids the number).
  4. RULERS from TRAIN DOCS ONLY — uniform chance, unigram, bigram; verdict bands
     below-unigram / unigram..bigram / significantly-above-bigram.
  5. NEGATIVE CONTROL — Markov-scrambled same-doc token panel; if a model scores ~= real held
     on scrambled text, the metric is broken (validity gate).
  6. FROZEN MANIFEST — sha256 of every panel + the split seed, written BEFORE any training run;
     runs must assert the manifest hash they trained against.

Maker != checker at the document level, by construction, with an audit trail.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "load_curriculum_documents",
    "build_split",
    "compute_rulers",
    "markov_scrambled_panel",
    "SplitManifest",
]

_DEF_JACCARD = 0.8
_NGRAM = 8  # character n-grams for near-dupe detection (robust to token choice)


# ── document-aware corpus loading ─────────────────────────────────────────────
def load_curriculum_documents(corpus_dir: str | Path, *, min_len: int = 40) -> list[dict]:
    """Load the curriculum as DOCUMENTS: [{doc_id, path, passages:[str]}].

    Reuses qig_studio.corpus passage extraction + sanitisation per file, but PRESERVES the
    document identity the flat loader throws away — the identity the split must hash on.
    Skips the same superseded/damaged/stub docs the flat loader skips."""
    from .corpus import _DAMAGED_DOCS, _STUB_DOCS, _STUB_MARKERS, _SUPERSEDED_QIG_DOCS, _passages_from_markdown

    p = Path(corpus_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"curriculum dir not found: {p}")
    docs: list[dict] = []
    for f in sorted(p.glob("*.md")) + sorted(p.glob("*.txt")):
        if f.name in _SUPERSEDED_QIG_DOCS or f.name in _DAMAGED_DOCS or f.name in _STUB_DOCS:
            continue
        passages = []
        for psg in _passages_from_markdown(f.read_text(encoding="utf-8", errors="replace"), min_len):
            if any(m in psg.lower() for m in _STUB_MARKERS):
                continue
            passages.append(psg)
        if passages:
            docs.append({"doc_id": f.name, "path": str(f), "passages": passages})
    if not docs:
        raise ValueError(f"no usable documents under {p}")
    return docs


# ── near-duplicate bucketing (doc-family) ─────────────────────────────────────
def _char_ngrams(text: str, n: int = _NGRAM, cap: int = 20000) -> set[str]:
    t = " ".join(text.split())[:cap].lower()
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _bucket_families(docs: list[dict], threshold: float = _DEF_JACCARD) -> dict[str, int]:
    """Union-find doc-families: any doc pair with Jaccard > threshold shares a bucket, so
    boilerplate/mirrors cannot straddle the split. O(D^2) on n-gram sets — fine at D~200."""
    grams = [_char_ngrams(" ".join(d["passages"][:20])) for d in docs]
    parent = list(range(len(docs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            if _jaccard(grams[i], grams[j]) > threshold:
                parent[find(i)] = find(j)
    return {d["doc_id"]: find(i) for i, d in enumerate(docs)}


# ── the split ─────────────────────────────────────────────────────────────────
@dataclass
class SplitManifest:
    seed: int
    jaccard_threshold: float
    train_docs: list[str]
    dev_docs: list[str]
    science_docs: list[str]
    train_passages: int
    dev_panel: list[str] = field(repr=False)
    science_panel: list[str] = field(repr=False)
    rulers: dict[str, float] = field(default_factory=dict)
    sha256: str = ""
    built: str = ""

    def to_json(self) -> dict:
        d = {k: getattr(self, k) for k in (
            "seed", "jaccard_threshold", "train_docs", "dev_docs", "science_docs",
            "train_passages", "dev_panel", "science_panel", "rulers", "built")}
        d["sha256"] = hashlib.sha256(
            json.dumps({k: d[k] for k in sorted(d) if k != "sha256"},
                       sort_keys=True).encode()).hexdigest()
        return d


def build_split(docs: list[dict], *, seed: int = 0, dev_frac: float = 0.10,
                science_frac: float = 0.10, dev_panel_n: int = 60, science_panel_n: int = 60,
                train_passage_cap: int | None = None,
                jaccard_threshold: float = _DEF_JACCARD) -> tuple[list[str], SplitManifest]:
    """DOCUMENT-family split -> (train_passages, manifest with dev + LOCKED science panels).

    Families (near-dupe buckets) are shuffled with `seed` and dealt to science/dev/train so
    no family straddles the split. Panels are randomized SAMPLES across their tier's docs
    (never curriculum head order). Assertion: zero doc overlap across tiers."""
    fam = _bucket_families(docs, jaccard_threshold)
    families: dict[int, list[dict]] = {}
    for d in docs:
        families.setdefault(fam[d["doc_id"]], []).append(d)
    fam_ids = sorted(families)
    rng = random.Random(seed)
    rng.shuffle(fam_ids)

    n_sci = max(1, int(len(fam_ids) * science_frac))
    n_dev = max(1, int(len(fam_ids) * dev_frac))
    sci_f, dev_f, train_f = fam_ids[:n_sci], fam_ids[n_sci:n_sci + n_dev], fam_ids[n_sci + n_dev:]

    def _docs(fids: list[int]) -> list[dict]:
        return [d for f in fids for d in families[f]]

    sci_docs, dev_docs, train_docs = _docs(sci_f), _docs(dev_f), _docs(train_f)
    assert not ({d["doc_id"] for d in sci_docs} & {d["doc_id"] for d in dev_docs}
                | {d["doc_id"] for d in sci_docs} & {d["doc_id"] for d in train_docs}
                | {d["doc_id"] for d in dev_docs} & {d["doc_id"] for d in train_docs}), \
        "SPLIT VIOLATION: document appears in more than one tier"

    def _panel(tier_docs: list[dict], n: int, salt: str) -> list[str]:
        pool = [p for d in tier_docs for p in d["passages"]]
        r = random.Random(f"{seed}:{salt}")
        r.shuffle(pool)
        return pool[:n]

    train_passages = [p for d in train_docs for p in d["passages"]]
    random.Random(f"{seed}:train").shuffle(train_passages)
    if train_passage_cap:
        train_passages = train_passages[:train_passage_cap]

    man = SplitManifest(
        seed=seed, jaccard_threshold=jaccard_threshold,
        train_docs=[d["doc_id"] for d in train_docs],
        dev_docs=[d["doc_id"] for d in dev_docs],
        science_docs=[d["doc_id"] for d in sci_docs],
        train_passages=len(train_passages),
        dev_panel=_panel(dev_docs, dev_panel_n, "dev"),
        science_panel=_panel(sci_docs, science_panel_n, "science"),
        built=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    return train_passages, man


# ── rulers (train docs ONLY) ──────────────────────────────────────────────────
def compute_rulers(encode, train_passages: list[str], panel: list[str],
                   *, vocab_size: int, ctx: int = 256) -> dict[str, float]:
    """Uniform / unigram / bigram held-out top-1 rulers, statistics from TRAIN ONLY."""
    from collections import Counter, defaultdict
    uni: Counter = Counter()
    big: dict[int, Counter] = defaultdict(Counter)
    for p in train_passages:
        ids = encode(p)[:ctx]
        uni.update(ids)
        for a, b in zip(ids, ids[1:]):
            big[a][b] += 1
    modal = uni.most_common(1)[0][0] if uni else 0
    u_c = b_c = pos = 0
    for p in panel:
        ids = encode(p)[:ctx]
        for a, b in zip(ids, ids[1:]):
            pos += 1
            if b == modal:
                u_c += 1
            pred = big[a].most_common(1)[0][0] if big.get(a) else modal
            if pred == b:
                b_c += 1
    return {"uniform": 1.0 / max(1, vocab_size),
            "unigram": u_c / max(1, pos),
            "bigram": b_c / max(1, pos),
            "n_positions": pos}


# ── negative control (metric-validity gate) ───────────────────────────────────
def markov_scrambled_panel(encode, decode, panel: list[str], *, seed: int = 0,
                           ctx: int = 256) -> list[str]:
    """Markov(1)-scrambled twins of the panel: resample each passage's token sequence from its
    OWN bigram chain — preserves local token statistics, destroys long-range/document meaning.
    A model scoring ~= real-panel on these is reading statistics, not meaning (metric broken)."""
    from collections import defaultdict
    rng = random.Random(seed)
    out: list[str] = []
    for p in panel:
        ids = encode(p)[:ctx]
        if len(ids) < 4:
            out.append(p)
            continue
        chain: dict[int, list[int]] = defaultdict(list)
        for a, b in zip(ids, ids[1:]):
            chain[a].append(b)
        cur = ids[0]
        scr = [cur]
        for _ in range(len(ids) - 1):
            nxt = rng.choice(chain[cur]) if chain.get(cur) else rng.choice(ids)
            scr.append(nxt)
            cur = nxt
        try:
            out.append(decode(scr))
        except Exception:  # decode not available -> keep id-level twin via sentinel join
            out.append(" ".join(map(str, scr)))
    return out
