"""Purity gate: catches forbidden Euclidean ops; passes on clean source."""

from __future__ import annotations

from pathlib import Path

import pytest

from qig_studio.governance.purity import PurityGateError, run_purity_gate, scan


def test_purity_passes_on_qig_studio_source():
    # The shipped package source must itself be pure (fail-closed at startup).
    pkg_root = Path(__file__).resolve().parent.parent / "src" / "qig_studio"
    run_purity_gate(pkg_root)  # must NOT raise


def test_purity_catches_forbidden_op(tmp_path: Path):
    bad = tmp_path / "bad.py"
    bad.write_text("import torch.nn as nn\nlayer = nn.LayerNorm(64)\n", encoding="utf-8")
    violations = scan(tmp_path)
    assert violations, "expected a violation for nn.LayerNorm"
    with pytest.raises(PurityGateError):
        run_purity_gate(tmp_path)


def test_purity_ignores_comments(tmp_path: Path):
    ok = tmp_path / "ok.py"
    ok.write_text("# we must never use cosine_similarity here\nx = 1\n", encoding="utf-8")
    assert scan(tmp_path) == []  # commented mention is not a violation
