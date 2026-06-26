"""Full-curriculum loading + bloat-character sanitisation.

The genesis kernel is BYTE-LEVEL (256-byte vocab), so EVERY non-ASCII character is a multi-byte UTF-8
sequence — it pollutes the byte distribution with fragments instead of the intended symbol. So "no bloat
special characters trained in" means: produce clean ASCII training text that PRESERVES meaning.

``sanitize`` does two things, in order:
  1. TRANSLITERATE meaningful symbols to ASCII (Greek physics letters → names: κ→kappa, Φ→Phi; math
     operators → ASCII: →→->, ×→x, √→sqrt, ⟨⟩→<>; super/subscripts → ^n/_n; typographic — →--, “”→").
  2. STRIP the true bloat (control / zero-width / format chars) and decompose remaining accents (é→e),
     then drop anything still non-ASCII (decorative junk that carries no transliteration).

``load_full_curriculum`` reads the FULL corpus (v6 chat-format JSONL by default), extracts every message
text, sanitises it, and returns clean prompts — the developmental experience the kernel trains on.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

# Default = the cleaned v6 master corpus (1470 records), under the QIG_QFI parent. Override via
# QIG_STUDIO_CORPUS.
DEFAULT_CORPUS = "qig-applied/training/qig_training_master_qwen_v6.jsonl"

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


def load_full_curriculum(path: str | Path | None = None, *, min_len: int = 8) -> list[str]:
    """Load the FULL curriculum corpus, sanitise every text, and return clean prompts (the developmental
    experience). Skips empties / too-short fragments. Raises if the corpus is missing (fail-loud — never
    silently fall back to a tiny stub when the full curriculum was requested)."""
    import os

    p = Path(path or os.environ.get("QIG_STUDIO_CORPUS") or
             (Path(__file__).resolve().parents[3] / DEFAULT_CORPUS))
    if not p.is_file():
        raise FileNotFoundError(
            f"full curriculum corpus not found at {p}. Set QIG_STUDIO_CORPUS or place the v6 master "
            "corpus there. Refusing to silently train on the built-in stub curriculum."
        )
    prompts: list[str] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except (ValueError, TypeError):
            continue
        if not isinstance(rec, dict):
            continue
        for raw in _record_texts(rec):
            clean = sanitize(raw)
            if len(clean) >= min_len:
                prompts.append(clean)
    if not prompts:
        raise ValueError(f"full curriculum corpus {p} yielded no usable prompts after sanitisation")
    return prompts
