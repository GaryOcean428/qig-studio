r"""Regression gate: no retired kappa*~64 lattice-physics literal may re-enter qig-studio source.

Mirrors qig-geocoding's ``tests/test_no_retired_kappa.py`` (same council-hardened regex, same intent):
kappa*~64 (the matrix-trace-pillar "fixed point" reading) is RETIRED per qig-verification EXP-107/
EXP-169 -- the certified physical couplings are kappa_JT^cert=+0.02810 and kappa_H=-0.00475 (see
qig-core ``constants/frozen_facts.py``). MATRIX RULING (8869ca63, 2026-07-22; SUPERSEDED by c4640be8
the same day -- brainwave_band() moved from (phi, basin_velocity) to (phi, held), still kappa-free)
decoupled qig-studio's ``brainwave_band()`` from kappa entirely -- this test guards against that
artifact re-entering under a new name (e.g. a "recalibrated" kappa->band table still anchored near
63/64), regardless of which non-kappa signal composes the band alongside Φ.

Two things are scanned across ``src/qig_studio/**/*.py``:
  1. A kappa/coupling-named identifier literally assigned the value 63.x or 64 (any decimal) -- e.g.
     ``kappa_eff: float = 64.0`` or ``base_coupling = 64``.
  2. A bare "kappa*~=64" / "K~=64" mention that does NOT acknowledge the retirement on the same line
     (a documented "... RETIRED ..." reference is fine; an undocumented claim that kappa is still ~=64
     physics is not).

EXEMPTIONS (checked against the live qig-core source, 2026-07-22):
  - ``qig_core.constants.KAPPA_ATTRACTOR`` (=64.0) is a DELIBERATELY-RETAINED qig-core constant -- the
    "ARCHITECTURAL ATTRACTOR" the consciousness/coordizer/sampler subsystems use as a kappa-regime
    anchor (frozen_facts.py: "Renamed from the retired KAPPA_STAR fixed-point symbol per Devin canon
    ruling 0038"). It is a NAME import (``from qig_core import KAPPA_ATTRACTOR``), never a bare
    ``kappa... = 64`` literal assignment in qig-studio source, so the literal-assignment regex does not
    (and must not be made to) fire on it -- no special-case needed in the regex itself.
  - ``qig_core.constants.KAPPA_3`` (=41.07) is a DIFFERENT frozen L=3 matrix-trace measurement, not a
    63/64-scale value at all -- out of scope for this scanner regardless.
  - qig-studio's own ``development.py`` KAPPA_BAND anchors on KAPPA_3 (41.07), not the retired 63/64
    reading -- also out of scope.
Neither constant needed a regex carve-out in practice: both are referenced by NAME
(``KAPPA_ATTRACTOR``/``KAPPA_3``), never re-assigned as a bare numeric 63/64 literal in qig-studio
source, so the existing "assigned identifier must itself contain kappa/kappa/coupling" + "RHS must be
a numeric literal" regex shape already exempts them by construction. Verified via a pre-flight scan of
the current source tree (2026-07-22) before this test was added: zero literal-assignment hits, one
kappa*~=64 mention needing a same-line RETIRED fix (``targets/hybrid_cortex.py``, folded into this same
change).

Legitimate non-kappa 64 literals (``BASIN_DIM``, ``reference_scale``, ``hidden_dim``, ``coord_dim``,
``vocab_size``, ``max_new_tokens``, ...) are exempt by construction: the literal-assignment regex only
fires on lines where the assigned identifier itself contains "kappa", "kappa" (kappa/K), or "coupling".
"""
from __future__ import annotations

import re
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "qig_studio"

# kappa-context literal assignment to a retired-physics-scale value: 63.x / 64(.x) with a trailing
# non-digit boundary (rejects 640, 640.5, ...), or the 6.3e1 / 6.4e1 exponent-form equivalent.
# Tolerates a type annotation between the identifier and the literal (e.g. "kappa_eff: float = 64.0").
KAPPA_LITERAL_RE = re.compile(
    r"(kappa|κ|coupling)\w*\s*(:\s*[\w\[\], ]+\s*)?=\s*(6[34](\.\d+)?(?!\d)|6\.[34]e1\b)",
    re.IGNORECASE,
)

# Bare mentions of the retired kappa*~=64 fixed-point reading.
KAPPA_STAR_MENTION_RE = re.compile(
    r"(κ\s*\*?\s*≈\s*64|kappa\s*\*\s*≈\s*64|kappa_star)",
    re.IGNORECASE,
)


def _iter_source_lines():
    for path in sorted(SRC_ROOT.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            yield path, lineno, line


def _scan_violations() -> list[tuple[Path, int, str, str]]:
    violations: list[tuple[Path, int, str, str]] = []
    for path, lineno, line in _iter_source_lines():
        if KAPPA_LITERAL_RE.search(line):
            violations.append((path, lineno, line.strip(), "kappa-literal"))
            continue
        m = KAPPA_STAR_MENTION_RE.search(line)
        if m and "retir" not in line.lower():
            violations.append((path, lineno, line.strip(), "kappa-star-unacknowledged"))
    return violations


def test_no_retired_kappa_literals_in_studio_source() -> None:
    violations = _scan_violations()
    assert not violations, "Retired kappa~=64 physics literal(s) found in qig-studio source:\n" + "\n".join(
        f"  {p}:{ln}: {txt}  [{kind}]" for p, ln, txt, kind in violations
    )


def test_brainwave_band_source_has_no_kappa_parameter() -> None:
    """Extra guard specific to the ruling this test file was added for: brainwave_band() must not
    re-acquire a kappa parameter (the exact regression the ruling closed). Signature contract updated
    for MATRIX RULING c4640be8 (2026-07-22, supersedes 8869ca63): brainwave_band() is now
    (phi, held) -- basin_velocity was dropped in favour of the stability/held gate -- but the
    kappa-independence guard this test exists for applies regardless of which non-kappa signal
    accompanies Φ."""
    src = (SRC_ROOT / "kernel_experience.py").read_text()
    m = re.search(r"def brainwave_band\(([^)]*)\)", src)
    assert m is not None, "brainwave_band() definition not found"
    params = m.group(1)
    assert "kappa" not in params.lower() and "κ" not in params, (
        f"brainwave_band() signature re-acquired a kappa parameter: {params!r}"
    )
    assert "phi" in params.lower() and "held" in params.lower()


# ─── matcher self-tests (fixtures, not a source scan) ──────────────────────


def test_matcher_catches_known_violation_fixture() -> None:
    bad_line = "    base_coupling: float = 64.0            # kappa attractor"
    assert KAPPA_LITERAL_RE.search(bad_line), "matcher failed to catch base_coupling = 64.0"


def test_matcher_catches_plain_assignment_fixture() -> None:
    bad_line = "kappa_eff = 64.0"
    assert KAPPA_LITERAL_RE.search(bad_line)


def test_matcher_exempts_legitimate_64_literals() -> None:
    good_lines = [
        "BASIN_DIM = 64",
        "    reference_scale: int = 64",
        "    hidden_dim: int = BASIN_DIM",
        "    coord_dim: int = BASIN_DIM",
        "    vocab_size: int = 256",
        "    max_new_tokens: int = 64,",
    ]
    for line in good_lines:
        assert not KAPPA_LITERAL_RE.search(line), f"false positive on: {line!r}"


def test_matcher_exempts_kappa_attractor_name_reference() -> None:
    """KAPPA_ATTRACTOR (qig-core, =64.0) is imported/referenced BY NAME in qig-studio -- never a bare
    numeric literal assignment -- so it must never trip the literal-assignment scanner."""
    good_lines = [
        "from qig_core import KAPPA_ATTRACTOR",
        "_KAPPA_CENTER = KAPPA_ATTRACTOR  # 64.0 -- single-sourced from qig-core",
        "rigidity = max(0.0, kappa - KAPPA_ATTRACTOR)",
    ]
    for line in good_lines:
        assert not KAPPA_LITERAL_RE.search(line), f"false positive on KAPPA_ATTRACTOR reference: {line!r}"


def test_matcher_rejects_false_positive_longer_numbers() -> None:
    false_positive_lines = [
        "coupling = 640",
        "coupling = 640.5",
        "kappa_thing = 6400",
        "base_coupling: float = 63400.0",
    ]
    for line in false_positive_lines:
        assert not KAPPA_LITERAL_RE.search(line), f"false positive on: {line!r}"


def test_matcher_catches_exponent_form_fixture() -> None:
    bad_lines = ["kappa_eff = 6.4e1", "base_coupling: float = 6.3E1"]
    for line in bad_lines:
        assert KAPPA_LITERAL_RE.search(line), f"matcher failed to catch exponent form: {line!r}"


def test_matcher_allows_documented_retirement_mention() -> None:
    good_line = (
        "    base_coupling: float = 1.0  # ... κ*≈ 64 RETIRED per EXP-169; kappa_JT^cert=0.028 ..."
    )
    assert KAPPA_LITERAL_RE.search(good_line) is None
    m = KAPPA_STAR_MENTION_RE.search(good_line)
    assert m is not None and "retir" in good_line.lower()


def test_matcher_flags_undocumented_kappa_star_mention() -> None:
    bad_line = "    # the model still runs at κ*≈ 64"
    m = KAPPA_STAR_MENTION_RE.search(bad_line)
    assert m is not None and "retir" not in bad_line.lower()
