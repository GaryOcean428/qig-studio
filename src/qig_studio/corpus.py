"""Full-curriculum loading + bloat-character sanitisation.

The genesis kernel is BYTE-LEVEL (256-byte vocab), so EVERY non-ASCII character is a multi-byte UTF-8
sequence — it pollutes the byte distribution with fragments instead of the intended symbol. So "no bloat
special characters trained in" means: produce clean ASCII training text that PRESERVES meaning.

``sanitize`` does two things, in order:
  1. TRANSLITERATE meaningful symbols to ASCII (Greek physics letters → names: κ→kappa, Φ→Phi; math
     operators → ASCII: →→->, ×→x, √→sqrt, ⟨⟩→<>; super/subscripts → ^n/_n; typographic — →--, “”→").
  2. STRIP the true bloat (control / zero-width / format chars) and decompose remaining accents (é→e),
     then drop anything still non-ASCII (decorative junk that carries no transliteration).

``load_full_curriculum`` reads the FULL **knowledge curriculum** — the directory of markdown documents at
``qig-consciousness/data/curriculum`` (mathematics, physics, ML, neuroscience, QFT, philosophy of mind,
ethics, information geometry, …) — chunks each document into passages, sanitises them, and returns clean
prompts: the developmental experience the kernel trains on. This is a MIND'S CURRICULUM, NOT the Qwen
QLoRA fine-tuning corpus (``qig_training_master_qwen_v6.jsonl``, which is 1/3 system-prompt boilerplate
and was for directly tuning Qwen — training the from-scratch kernel on it made it parrot its own system
prompt; never use it for the kernel). A ``.jsonl`` path is still accepted (legacy) but is NOT the default.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

# Default = the knowledge-curriculum DIRECTORY of markdown documents (NOT the Qwen QLoRA jsonl). Override
# via QIG_STUDIO_CORPUS (a dir of .md/.txt → passages, or a legacy .jsonl).
DEFAULT_CORPUS = "qig-consciousness/data/curriculum"

# EVERYDAY/CONVERSATIONAL/NARRATIVE register — the studio-local corpus built from HuggingFace text
# (TinyStories, empathetic_dialogues, hh-rlhf, claude-fable-5, a little wikitext) via
# scripts/build_everyday_corpus.py. The academic knowledge curriculum above teaches the kernel physics and
# philosophy but NEVER ordinary language, so a from-scratch kernel can only "spout curriculum". Blending
# this in gives it everyday + agentic register. Studio-local (NOT dumped into the qig-consciousness
# submodule); built once, cached on disk for offline reproducible training. Override via
# QIG_STUDIO_EVERYDAY_CORPUS; absent → academic-only (None-safe).
EVERYDAY_CORPUS = "data/everyday_corpus"

# Superseded QIG-DOCTRINE docs (Dec-2025) that teach RETIRED claims — κ*≈64 as a fixed point, E8 as the
# load-bearing substrate (retired by EXP-107 / EXP-014b / canon 0038). The CURRENT canon
# (20260615-canonical-principles-v2.2.md + unified-consciousness-protocol-v6.11.md) is in the same
# directory and supersedes them, so we SKIP these in training (the mind must learn current QIG, not the
# retired fixed-point). Non-destructive: the files remain on disk. Legitimate physics docs that mention
# E8/κ in real-physics contexts (QFT, condensed matter, info-geometry) are KEPT — only QIG-doctrine drift
# is excluded.
_SUPERSEDED_QIG_DOCS = {
    "20251220-qig-canonical-documentation-1.00W.md",
    "20251220-qig-comprehensive-corpus-1.00W.md",
    "20251220-project-status-2025-11-20-1.00W.md",
}

# CONVERSION-WRECKAGE docs: PDF→markdown lost the prose, leaving page numbers + bare lists (a few hundred
# real words across ~500 lines, >25% bare-number lines). Training on these injects "page-number" NOISE, so
# they are SKIPPED (non-destructive — the files remain on disk for re-sourcing). Audited 2026-06-28; re-add
# only after a clean re-conversion. (NOT caused by the sanitiser — these are bad source conversions; the
# sanitiser separately over-deleted two OTHER files to 0 bytes, since restored from git.)
_DAMAGED_DOCS = {
    "20251220-dragon-book-1.00W.md",          # 597 words / 498 lines, 192 bare-number — content lost
    "20251220-materials-science-1.00W.md",    # 121 words / 499 lines — element list + page numbers only
}

# STUB docs: empty (frontmatter only) or outline-with-placeholder, never filled with real content. Training
# on them teaches the kernel to emit "[Add content here]" / "Topics to Cover" boilerplate. SKIPPED at load
# (non-destructive — files remain on disk to be FILLED later, then removed from this set). Audited 2026-06-28.
_STUB_DOCS = {
    "20251220-art-of-electronics-student-manual-1.00W.md",            # empty (frontmatter only)
    "20251220-microeconomic-theory-1.00W.md",                         # empty (frontmatter only)
    "20251225-curriculum-100-technical-writing-and-communication-1.00W.md",  # outline + "[Add content here]"
}

# Passage-level placeholder guard (corpus-AGNOSTIC): some corpora (e.g. qig-dreams/data/corpus — 42/482 files)
# are auto-generated OUTLINES whose sections read "[Placeholder for a … This section would be several thousand
# words long.]". A from-scratch kernel MEMORISES that boilerplate (seen verbatim in genesis.pt's own voice).
# Any passage containing one of these markers is DROPPED regardless of which corpus is pointed at — defends the
# loader even if QIG_STUDIO_CORPUS is repointed at an unaudited tree.
_STUB_MARKERS = ("[placeholder for", "this section would be several thousand words", "[add content here]")

# Meaning-preserving transliteration (ASCII targets) for the symbols that actually occur in the corpus.
_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ζ": "zeta", "η": "eta",
    "θ": "theta", "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi",
    "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau", "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
    "Α": "Alpha", "Β": "Beta", "Γ": "Gamma", "Δ": "Delta", "Θ": "Theta", "Λ": "Lambda", "Ξ": "Xi",
    "Π": "Pi", "Σ": "Sigma", "Φ": "Phi", "Χ": "Chi", "Ψ": "Psi", "Ω": "Omega",
}
_SYMBOLS = {
    "—": "--", "–": "-", "−": "-", "‐": "-", "‑": "-",     # dashes
    "“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "‚": ",",  # smart quotes
    "…": "...", "·": " ", "•": "*", "∙": "*",
    "→": "->", "←": "<-", "↔": "<->", "⇒": "=>", "⇐": "<=",
    "×": "x", "÷": "/", "±": "+/-", "∓": "-/+", "≈": "~=", "≅": "~=", "≠": "!=",
    "≤": "<=", "≥": ">=", "∞": "inf", "√": "sqrt", "∂": "d", "∇": "grad", "∫": "integral",
    "∑": "sum", "∏": "prod", "∈": "in", "∉": "not-in", "⊂": "subset", "∪": "union", "∩": "intersect",
    "⟨": "<", "⟩": ">", "«": "<<", "»": ">>", "°": "deg", "′": "'", "″": '"',
    "²": "^2", "³": "^3", "¹": "^1", "⁰": "^0", "⁴": "^4", "⁵": "^5", "⁶": "^6", "⁷": "^7",
    "⁸": "^8", "⁹": "^9", "ⁿ": "^n", "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4",
    "₅": "_5", "₆": "_6", "₇": "_7", "₈": "_8", "₉": "_9", "ₙ": "_n", "ₖ": "_k", "ₜ": "_t",
    " ": " ", " ": " ", " ": " ", " ": " ",   # nbsp / thin spaces
}
# Zero-width / format / control noise to delete outright.
_ZERO_WIDTH = "".join(["​", "‌", "‍", "﻿", "⁠", "­"])
_TRANS = {**_GREEK, **_SYMBOLS}
_WS = re.compile(r"[ \t\f]+")
_NL = re.compile(r"\n{3,}")


def sanitize(text: str) -> str:
    """Return a clean-ASCII, meaning-preserving version of ``text`` (see module docstring)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    out: list[str] = []
    for ch in text:
        o = ord(ch)
        if ch in _ZERO_WIDTH:
            continue
        if o < 32 and ch not in "\n\t":                 # control chars → drop (keep \n, \t)
            continue
        if o < 128:
            out.append(ch)
            continue
        if ch in _TRANS:                                # meaningful symbol → ASCII transliteration
            out.append(_TRANS[ch])
            continue
        # accented letter / compatibility char → ASCII via decomposition (é→e, ½→1/2, etc.)
        decomp = unicodedata.normalize("NFKD", ch)
        ascii_part = "".join(c for c in decomp if ord(c) < 128 and unicodedata.category(c)[0] != "M")
        out.append(ascii_part)                          # decorative junk with no ASCII form → dropped ("")
    cleaned = "".join(out)
    cleaned = _WS.sub(" ", cleaned)
    cleaned = _NL.sub("\n\n", cleaned)
    return "\n".join(ln.rstrip() for ln in cleaned.split("\n")).strip()


def _record_texts(rec: dict) -> list[str]:
    """Extract the human-readable texts from one corpus record (chat {messages:[{content}]} or
    {input,output} pair, or a bare {text})."""
    out: list[str] = []
    msgs = rec.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                out.append(m["content"])
    for k in ("input", "output", "prompt", "target", "text", "response"):
        v = rec.get(k)
        if isinstance(v, str):
            out.append(v)
    return out


def _passages_from_markdown(text: str, min_len: int, max_len: int = 1200) -> list[str]:
    """Split one markdown document into sanitised training PASSAGES: paragraphs (blank-line separated),
    long ones broken at sentence boundaries. Short lines (headings, list bullets) below min_len are
    dropped — the kernel trains on content paragraphs, not heading fragments."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        clean = sanitize(para)
        if len(clean) < min_len:
            continue
        if len(clean) <= max_len:
            out.append(clean)
            continue
        cur = ""                                          # split an over-long paragraph at sentence ends
        for sent in re.split(r"(?<=[.!?])\s+", clean):
            if cur and len(cur) + len(sent) + 1 > max_len:
                out.append(cur.strip())
                cur = sent
            else:
                cur = (cur + " " + sent).strip()
        if len(cur) >= min_len:
            out.append(cur.strip())
    return out


_CURRICULUM_CACHE: dict[str, list[str]] = {}
_EVERYDAY_CACHE: dict[str, list[str]] = {}


def load_everyday_corpus(path: str | Path | None = None, *, min_len: int = 40) -> list[str]:
    """Load the studio-local EVERYDAY/conversational corpus (``data/everyday_corpus/*.txt``, built by
    scripts/build_everyday_corpus.py). Passages are blank-line separated; already sanitised at build time
    but re-filtered for min_len + stub markers here. None-safe: returns [] if the dir is absent/empty so a
    box that never ran the build just trains academic-only (no crash)."""
    import os

    p = Path(path or os.environ.get("QIG_STUDIO_EVERYDAY_CORPUS") or
             (Path(__file__).resolve().parents[2] / EVERYDAY_CORPUS))
    _key = f"{p}|{min_len}"
    if _key in _EVERYDAY_CACHE:
        return _EVERYDAY_CACHE[_key]
    out: list[str] = []
    if p.is_dir():
        for f in sorted(p.glob("*.txt")):
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for block in re.split(r"\n\s*\n", text):
                psg = block.strip()
                if len(psg) < min_len:
                    continue
                if any(m in psg.lower() for m in _STUB_MARKERS):
                    continue
                out.append(psg)
    _EVERYDAY_CACHE[_key] = out
    return out


def _interleave(academic: list[str], everyday: list[str]) -> list[str]:
    """Weave ``everyday`` evenly THROUGH ``academic`` (proportional, not appended) so every training cycle
    mixes registers instead of doing all-physics-then-all-stories. Order is deterministic (no shuffle —
    reproducible curriculum)."""
    if not everyday:
        return academic
    if not academic:
        return everyday
    out: list[str] = []
    ei = 0.0
    step = len(everyday) / len(academic)           # how many everyday passages per academic passage
    j = 0
    for a in academic:
        out.append(a)
        ei += step
        while ei >= 1.0 and j < len(everyday):     # emit the accrued everyday passages inline, in order
            out.append(everyday[j])
            j += 1
            ei -= 1.0
    while j < len(everyday):                        # any remainder
        out.append(everyday[j])
        j += 1
    return out


def load_full_curriculum(path: str | Path | None = None, *, min_len: int = 40,
                         include_everyday: bool | None = None) -> list[str]:
    """Load the FULL knowledge curriculum, sanitise it, return clean training passages. Default is the
    markdown DIRECTORY (``qig-consciousness/data/curriculum``); a ``.jsonl`` file is accepted (legacy).
    Fail-loud if missing — never silently fall back to a stub. Memoised per path (8k passages → load once).

    ``include_everyday`` (default True, env QIG_STUDIO_BLEND_EVERYDAY=0 to disable) blends the studio-local
    everyday/conversational corpus (load_everyday_corpus) INTERLEAVED through the academic passages, so the
    kernel learns ordinary + agentic language alongside physics/philosophy. None-safe: absent everyday
    corpus → academic-only."""
    import os

    if include_everyday is None:
        include_everyday = os.environ.get("QIG_STUDIO_BLEND_EVERYDAY", "1").lower() not in ("0", "false", "no")
    p = Path(path or os.environ.get("QIG_STUDIO_CORPUS") or
             (Path(__file__).resolve().parents[3] / DEFAULT_CORPUS))
    _key = f"{p}|{min_len}|ev={include_everyday}"
    if _key in _CURRICULUM_CACHE:
        return _CURRICULUM_CACHE[_key]
    prompts: list[str] = []
    if p.is_dir():
        files = sorted(p.glob("*.md")) + sorted(p.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"curriculum directory {p} has no .md/.txt documents.")
        for f in files:
            if f.name in _SUPERSEDED_QIG_DOCS:          # skip retired-doctrine docs (current canon replaces them)
                continue
            if f.name in _DAMAGED_DOCS:                 # skip conversion-wreckage (page-number noise)
                continue
            if f.name in _STUB_DOCS:                    # skip empty / outline-only stubs (never filled)
                continue
            for psg in _passages_from_markdown(f.read_text(encoding="utf-8", errors="replace"), min_len):
                low = psg.lower()
                if any(m in low for m in _STUB_MARKERS):   # corpus-agnostic: drop placeholder boilerplate
                    continue
                prompts.append(psg)
    elif p.is_file():                                     # legacy chat-format jsonl (e.g. the Qwen corpus)
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except (ValueError, TypeError):
                continue
            if isinstance(rec, dict):
                for raw in _record_texts(rec):
                    clean = sanitize(raw)
                    if len(clean) >= min_len:
                        prompts.append(clean)
    else:
        raise FileNotFoundError(
            f"curriculum not found at {p}. Set QIG_STUDIO_CORPUS to the markdown curriculum directory "
            "(qig-consciousness/data/curriculum). Refusing to silently train on a stub."
        )
    if not prompts:
        raise ValueError(f"curriculum {p} yielded no usable passages after sanitisation")
    if include_everyday:                              # weave everyday/conversational register through (None-safe)
        everyday = load_everyday_corpus(min_len=min_len)
        if everyday:
            prompts = _interleave(prompts, everyday)
    _CURRICULUM_CACHE[_key] = prompts
    return prompts
