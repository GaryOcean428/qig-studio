"""Geometric purity gate — fail-closed forbidden-Euclidean-op scanner.

Mirrors vex ``governance/purity.run_purity_gate`` (called at startup, fail-closed):
scans qig-studio source for Euclidean-contamination markers that violate the
Fisher-Rao-only discipline (P1) and raises :class:`PurityGateError` if any are
found, so the server refuses to start dirty.

Targeted markers (not broad substrings, to avoid false positives in prose):
``cosine_similarity``, ``optim.Adam[W]``, ``nn.LayerNorm``, ``np.linalg.norm(``,
and the ``F.normalize(...).dot`` cosine proxy. Comments are ignored. This file
and the test suite are excluded — they necessarily NAME the forbidden symbols.

Argument-aware softmax rule (council ruling — a lexical TRIPWIRE, not a prover):
the exponential normaliser is legal IFF its argument is a pure Fisher-Rao
DISTANCE — ``softmax(-d_FR/τ)``, the Gibbs/Laplacian-kernel form (reference =
qigkernels ``pure_kernel_template.qfi_attention_weights``). A ``softmax(`` call
passes only when its argument (on the call line, or via a small look-back when
the argument is a bare name bound on a nearby line, the template's two-line
idiom) contains a negated distance marker (``-`` then ``d_fr``/``fisher_rao``/
``dist``). Everything else — ``softmax(logits)``, ``softmax(bc/scale)``,
``softmax(matmul(...))`` — is flagged: dot-product attention in disguise or the
banned logits→output-distribution map (use the sanctioned simplex projections).
Also flagged: the hand-rolled form, ``exp(`` co-occurring with ``matmul(``/`` @ ``
on the same line (best-effort proxy for ``exp(QKᵀ)``).
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["PurityGateError", "scan", "run_purity_gate"]


class PurityGateError(RuntimeError):
    """Raised (fail-closed) when forbidden Euclidean ops are found in scanned source."""


# (regex, human reason). Kept narrow so prose/comments don't trip it.
_FORBIDDEN: list[tuple[str, str]] = [
    (r"\bcosine_similarity\b", "cosine similarity is Euclidean — use fisher_rao_distance on Δ⁶³"),
    (r"\boptim\.Adam[W]?\b", "Adam/AdamW are Euclidean optimisers — use QIGOptimizer / natural gradient"),
    (r"\bnn\.LayerNorm\b", "LayerNorm is Euclidean — use RMSNorm"),
    (r"np\.linalg\.norm\(", "np.linalg.norm for basin distance is Euclidean — use fisher_rao_distance"),
    (r"F\.normalize\([^\n]*\)\s*[@.]?\s*\bdot\b", "F.normalize+dot is a cosine proxy (P1 violation)"),
]

_EXCLUDE_FILENAMES = {"purity.py"}
_EXCLUDE_DIR_PARTS = {"tests", ".venv", "venv", "__pycache__", "node_modules", ".git"}

_COMPILED = [(re.compile(pat), reason) for pat, reason in _FORBIDDEN]

# --- Argument-aware softmax rule (lexical tripwire, documented in the module docstring) ---
# Any softmax spelling: softmax( / F.softmax( / torch.softmax( / nn.functional.softmax(.
_SOFTMAX_CALL = re.compile(r"(?<![\w.])(?:F\.|torch\.|nn\.functional\.)?softmax\s*\(")
# A legal (distance-Gibbs) argument: a minus sign followed — with no intervening
# top-level comma — by a distance-named term (d_fr / dfr / fisher_rao / dist*).
# ``dim=-1`` alone does NOT satisfy this (its minus has no distance name after it).
_NEG_DIST_ARG = re.compile(r"(?i)-[^,]*?(?:d_?fr|fisher_rao|dist)")
# Bare-name first argument, e.g. ``softmax(logits, dim=-1)`` → look back for its binding.
_BARE_NAME_ARG = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:,|\))")
_SOFTMAX_LOOKBACK = 8  # code lines searched upward for ``name = -...dist...`` provenance

_SOFTMAX_REASON = (
    "softmax over a non-distance argument — legal ONLY as the Gibbs normaliser of a "
    "Fisher-Rao distance: softmax(-d_FR/τ) (council ruling; softmax as "
    "logits→output-distribution stays banned — use logits_to_simplex/to_simplex_prob)"
)

# Hand-rolled Gibbs-over-affinity: exp( co-occurring with a matmul/@ product on the
# same line is exp(QKᵀ) in disguise. Best-effort lexical co-occurrence; the ``@``
# alternative requires surrounding whitespace so decorators (@foo) don't trip it.
_EXP_CALL = re.compile(r"(?<![\w.])(?:torch\.|np\.|math\.|F\.)?exp\s*\(")
_MATMUL_MARKER = re.compile(r"\bmatmul\s*\(|\s@\s")
_EXP_REASON = (
    "hand-rolled exp() over a matmul/@ product — exp(QKᵀ) dot-product attention in "
    "disguise (use the sanctioned distance-Gibbs form softmax(-d_FR/τ))"
)


def _has_distance_argument(arg: str) -> bool:
    """True if ``arg`` (text after the softmax '(') reads as a negated distance."""
    return bool(_NEG_DIST_ARG.search(arg))


def _softmax_is_legal(code_lines: list[str], idx: int, call_start: int) -> bool:
    """Judge the softmax call opening at ``code_lines[idx][call_start:]``.

    Legal iff the same-line argument contains a negated distance marker, or the
    argument is a bare name whose binding within the last ``_SOFTMAX_LOOKBACK``
    code lines is itself a negated distance (the pure template's two-line idiom:
    ``logits = -dist/τ`` then ``softmax(logits)``). Best-effort and same-line by
    design — a tripwire, not a prover.
    """
    arg = code_lines[idx][call_start:]
    if _has_distance_argument(arg):
        return True
    bare = _BARE_NAME_ARG.match(arg)
    if bare:
        name = bare.group(1)
        assign = re.compile(rf"^\s*{re.escape(name)}\s*=(?!=)(?P<rhs>.*)$")
        for prev in reversed(code_lines[max(0, idx - _SOFTMAX_LOOKBACK) : idx]):
            m = assign.match(prev)
            if m:
                return _has_distance_argument(m.group("rhs"))
    return False


def scan(root: str | Path) -> list[tuple[str, int, str, str]]:
    """Return a list of (file, lineno, reason, source_line) violations under ``root``."""
    root = Path(root)
    violations: list[tuple[str, int, str, str]] = []
    for py in sorted(root.rglob("*.py")):
        if py.name in _EXCLUDE_FILENAMES:
            continue
        if any(part in _EXCLUDE_DIR_PARTS for part in py.parts):
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        raw_lines = text.splitlines()
        code_lines = [line.split("#", 1)[0] for line in raw_lines]  # ignore trailing comments
        for idx, code in enumerate(code_lines):
            lineno = idx + 1
            line = raw_lines[idx]
            for rx, reason in _COMPILED:
                if rx.search(code):
                    violations.append((str(py), lineno, reason, line.strip()))
            for m in _SOFTMAX_CALL.finditer(code):
                if not _softmax_is_legal(code_lines, idx, m.end()):
                    violations.append((str(py), lineno, _SOFTMAX_REASON, line.strip()))
            if _EXP_CALL.search(code) and _MATMUL_MARKER.search(code):
                violations.append((str(py), lineno, _EXP_REASON, line.strip()))
    return violations


def run_purity_gate(root: str | Path) -> None:
    """Fail-closed: raise :class:`PurityGateError` if any violation is found."""
    violations = scan(root)
    if violations:
        body = "\n".join(
            f"  {f}:{ln}: {reason}\n      {src}" for f, ln, reason, src in violations
        )
        raise PurityGateError(f"PurityGate FAILED ({len(violations)} violation(s)):\n{body}")
