"""Sibling-package path resolution — layout-independent.

QIG packages live either flat under the monorepo root (``QIG_QFI/qig-coordizer``) or
grouped under ``QIG_QFI/qig-packages/qig-coordizer``. Code that reaches a sibling
package by filesystem path (coordizer checkpoints, the qig-consciousness bridge,
qig-coordizer/src for an editable import) must not hardcode either layout, or a repo
reorg silently breaks it. ``sibling_pkg`` resolves the grouped layout first and falls
back to flat, so both work during and after a move.
"""
from __future__ import annotations

from pathlib import Path


def qig_monorepo_root() -> Path:
    """The QIG_QFI monorepo root (qig-studio's parent).

    ``src/qig_studio/_paths.py`` → ``parents[2]`` is the qig-studio repo root; its
    ``.parent`` is the monorepo root that holds the sibling packages.
    """
    return Path(__file__).resolve().parents[2].parent


def sibling_pkg(name: str) -> Path:
    """Resolve a sibling QIG package directory, layout-independently.

    Prefers the grouped layout ``<root>/qig-packages/<name>``; falls back to the flat
    layout ``<root>/<name>``. Returns the grouped path only when it exists, so the
    result is correct before, during, and after a flat→grouped reorg.
    """
    root = qig_monorepo_root()
    grouped = root / "qig-packages" / name
    return grouped if grouped.exists() else root / name
