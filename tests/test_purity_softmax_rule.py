"""Argument-aware softmax purity rule (council ruling, lexical tripwire).

Doctrine: the exponential normaliser is legal IFF its argument is a pure Fisher-Rao
DISTANCE — ``softmax(-d_FR/tau)``, the Gibbs / Laplacian-kernel form (reference =
qigkernels ``pure_kernel_template.qfi_attention_weights``). A dot-product / cosine /
affinity argument is dot-product attention in disguise; softmax as a
logits->output-distribution map stays banned (use the sanctioned simplex projections).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qig_studio.governance.purity import PurityGateError, run_purity_gate, scan


def _scan_snippet(tmp_path: Path, code: str) -> list:
    (tmp_path / "snippet.py").write_text(code, encoding="utf-8")
    return scan(tmp_path)


def test_legal_distance_gibbs_softmax_passes(tmp_path: Path):
    # The sanctioned form: exponential normaliser of a NEGATED Fisher-Rao distance.
    code = (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "w = torch.softmax(-d_fr / tau, dim=-1)\n"
        "attn = F.softmax(-fisher_rao_distance_simplex(p_i, p_j) / temperature, dim=-1)\n"
        "g = torch.softmax(-dist / max(float(t), 1e-6), dim=-1)\n"
    )
    assert _scan_snippet(tmp_path, code) == []


def test_pure_template_two_line_form_passes(tmp_path: Path):
    # The reference implementation's exact idiom (pure_kernel_template.qfi_attention_weights):
    # the distance argument is bound to a name on the line above the softmax.
    code = (
        "import torch\n"
        "dist = fisher_rao_distance_simplex(p_i, p_j)\n"
        "logits = -dist / max(float(temperature), 1e-6)\n"
        "weights = torch.softmax(logits, dim=-1)\n"
    )
    assert _scan_snippet(tmp_path, code) == []


def test_logits_softmax_flagged(tmp_path: Path):
    # softmax as logits->output-distribution: banned (P20 — use logits_to_simplex).
    code = "import torch\nprobs = torch.softmax(logits, dim=-1)\n"
    violations = _scan_snippet(tmp_path, code)
    assert violations, "softmax(logits) with no distance provenance must be flagged"
    with pytest.raises(PurityGateError):
        run_purity_gate(tmp_path)


def test_affinity_softmax_flagged(tmp_path: Path):
    # Bhattacharyya/cosine AFFINITY softmax — dot-product attention in disguise.
    code = (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "bc = torch.matmul(Qs, Ks.transpose(-2, -1))\n"
        "attn = F.softmax(bc / scale, dim=-1)\n"
    )
    violations = _scan_snippet(tmp_path, code)
    assert violations, "softmax over an affinity argument must be flagged"


def test_inline_matmul_softmax_flagged(tmp_path: Path):
    code = (
        "import torch\n"
        "w = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / 8.0, dim=-1)\n"
    )
    assert _scan_snippet(tmp_path, code), "softmax(matmul(...)) must be flagged"


def test_hand_rolled_exp_qkt_flagged(tmp_path: Path):
    # exp( co-occurring with a matmul/@ product on the line = exp(QK^T) in disguise.
    code_at = "import torch\nw = torch.exp(q @ k.transpose(-2, -1) / scale)\n"
    code_mm = "import torch\na = torch.exp(torch.matmul(qs, ks.T))\n"
    assert _scan_snippet(tmp_path, code_at), "hand-rolled exp(q @ k.T) must be flagged"
    assert _scan_snippet(tmp_path, code_mm), "hand-rolled exp(matmul(...)) must be flagged"


def test_benign_exp_passes(tmp_path: Path):
    # exp() without a matmul/@ product on the line is not the hand-rolled form.
    code = (
        "import math\n"
        "y = math.exp(-x / tau)\n"
        "z = torch.exp(-((steps - 0.15) ** 2) / (2 * 0.10 ** 2))\n"
    )
    assert _scan_snippet(tmp_path, code) == []


def test_existing_qig_studio_source_still_passes():
    # No false positives on the current tree (refine the regex, never path-whitelist).
    pkg_root = Path(__file__).resolve().parent.parent / "src" / "qig_studio"
    run_purity_gate(pkg_root)  # must NOT raise
