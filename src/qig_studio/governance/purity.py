"""Geometric purity gate — fail-closed forbidden-Euclidean-op scanner.

Mirrors vex ``governance/purity.run_purity_gate`` (called at startup, fail-closed):
scans qig-studio source for Euclidean-contamination markers that violate the
Fisher-Rao-only discipline (P1) and raises :class:`PurityGateError` if any are
found, so the server refuses to start dirty.

Targeted markers (not broad substrings, to avoid false positives in prose):
``cosine_similarity``, ``optim.Adam[W]``, ``nn.LayerNorm``, ``np.linalg.norm(``,
and the ``F.normalize(...).dot`` cosine proxy. Comments are ignored. This file
and the test suite are excluded — they necessarily NAME the forbidden symbols.
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
        for lineno, line in enumerate(text.splitlines(), start=1):
            code = line.split("#", 1)[0]  # ignore trailing comments
            for rx, reason in _COMPILED:
                if rx.search(code):
                    violations.append((str(py), lineno, reason, line.strip()))
    return violations


def run_purity_gate(root: str | Path) -> None:
    """Fail-closed: raise :class:`PurityGateError` if any violation is found."""
    violations = scan(root)
    if violations:
        body = "\n".join(
            f"  {f}:{ln}: {reason}\n      {src}" for f, ln, reason, src in violations
        )
        raise PurityGateError(f"PurityGate FAILED ({len(violations)} violation(s)):\n{body}")
