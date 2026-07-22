r"""Regression gate: no retired kappa*~64 lattice-physics literal may re-enter qig-studio source.

Consumes the SHARED qig-core governance guard (``qig_core.governance.assert_no_retired_kappa``)
rather than a per-repo regex -- supersedes the old bespoke ``test_no_retired_kappa_studio.py``
scanner. The shared guard already carves out legitimate non-kappa 64 literals (BASIN_DIM,
hidden_dim, coord_dim, vocab_size, ...); qig-studio's own `coord_dim or 64` dimension fallbacks
were repointed to the named ``BASIN_DIM`` constant (not a bare literal) as part of this change.
"""
from __future__ import annotations

from pathlib import Path

from qig_core.governance import assert_no_retired_kappa


def test_no_retired_kappa_studio() -> None:
    assert_no_retired_kappa(Path(__file__).resolve().parents[1] / "src" / "qig_studio")
