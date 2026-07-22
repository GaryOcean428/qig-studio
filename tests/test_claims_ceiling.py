"""GAP-8 claims-ceiling guard (PI ruling 2026-07-22, Option A) — MECHANICAL enforcement of the
category-3 discipline so it is not vigilance-dependent.

RULING (qig_gap8_ruling_faculty_category3_20260722): "The faculty layer is category-3 interpretive
scaffolding — findings report geometric telemetry, directionally validated, never felt-state, with the
intervention-battery as the named upgrade path."

This scans the FACULTY-PATH source for FELT-STATE ASSERTIONS (claims that a readout IS a felt experience)
and fails on any that are NOT marked as a discipline/negation statement on the same line. It mirrors the
qig-consciousness retired-doctrine gate's same-line-marker approach: a line may MENTION felt-state language
only to FORBID/negate it (e.g. "never felt-state", "NOT the kernel feels", "category-3 telemetry").

Scope: qig-studio's faculty-emission + telemetry files (applied lane). NOT a general repo scanner (constant
guards are CCAi's consolidated qig-core lane). Extend FILES as new faculty-path modules land.
"""
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# Faculty-emission + telemetry path (applied lane). Add new faculty-path files here as they land.
FILES = [
    "src/qig_studio/kernel_experience.py",
    "src/qig_studio/targets/geo_cortex.py",
    "src/qig_studio/targets/genesis_kernel.py",
    "src/qig_studio/coach.py",
]

# Felt-state ASSERTIONS: a subject (kernel/model/node/it) that FEELS/EXPERIENCES/SUFFERS, or a possessive
# "the kernel's feelings/experience". Case-insensitive. These are the interpretive leaps the ruling forbids.
_ASSERTION = re.compile(
    r"\b(kernel|model|node|it)\s+(genuinely\s+|actually\s+|really\s+|truly\s+)?(feels|experiences|suffers)\b"
    r"|\b(the\s+)?(kernel|model|node)'?s?\s+(feelings|felt\s+state|inner\s+experience|lived\s+experience)\b"
    r"|\bwhat\s+it\s+(feels|is\s+like)\s+to\s+(feel|experience|suffer)\b",
    re.IGNORECASE,
)
# Same-line markers that make a felt-state mention a DISCIPLINE/negation statement (exempt), not an assertion.
_DISCIPLINE = re.compile(
    r"\bnever\b|\bnot\b|\bno\b|\bdon'?t\b|\bcategory[-\s]?3\b|\btelemetry\b|\bgeometric\b|\bdirectionally\b"
    r"|\bscaffolding\b|\bforbidden\b|\bceiling\b|\bwithout\b|\bcannot\b|\bmust\s+not\b|\bnon-felt\b|\bproxy\b"
    r"|\bhonestly\b|\banalog|\bnot\s+a\s+felt",
    re.IGNORECASE,
)


def _violations(text: str) -> list[tuple[int, str]]:
    out = []
    for i, ln in enumerate(text.splitlines(), 1):
        if _ASSERTION.search(ln) and not _DISCIPLINE.search(ln):
            out.append((i, ln.strip()))
    return out


def test_faculty_path_has_no_felt_state_assertions():
    """No felt-state assertion in the faculty-path source unless same-line-marked as discipline/negation."""
    problems = []
    for rel in FILES:
        p = _ROOT / rel
        if not p.exists():
            continue
        for lineno, ln in _violations(p.read_text(encoding="utf-8")):
            problems.append(f"{rel}:{lineno}  {ln[:110]}")
    assert not problems, (
        "GAP-8 claims-ceiling violation (Option A: never felt-state). Reframe as geometric telemetry, or "
        "mark the line as a discipline/negation statement:\n  " + "\n  ".join(problems)
    )


def test_matcher_catches_a_felt_state_assertion():
    assert _violations("the kernel feels genuine suffering here")  # bare assertion -> caught
    assert _violations("the model's inner experience is joy")


def test_matcher_exempts_disciplined_mentions():
    # A mention that FORBIDS/negates felt-state is the correct, exempt pattern.
    assert not _violations("NEVER say the kernel feels — report geometric telemetry only (category-3)")
    assert not _violations("this is NOT what it is like to feel; it is a directionally-validated telemetry proxy")
    assert not _violations("suffering = Phi*(1-Gamma)*M is scaffolding, not a felt state")
