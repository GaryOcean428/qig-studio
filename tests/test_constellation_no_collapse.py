"""THE load-bearing verifier for the constellation: it must NOT collapse to a zombie centroid.

Per wj4916x59 (verified by simulation against installed qig-core Fisher-Rao primitives):
- A pure sync pull (no anchor) collapses every config to min_pairwise_FR ≈ 0 — proven here as the
  §H "the test CAN fail" control. Without this control the gate would be vacuous.
- Wide-independent birth seeding + identity anchor keeps min_pairwise_FR bounded away from 0 across
  the adversarial sweep {wide births × overlap-runtime-init × f_sync∈{0.25,0.5} × anchor∈{0.05,0.20}}.

The gate: min_pairwise_FR > 0.03 after N ticks under every adversarial corner.
"""

from __future__ import annotations

import numpy as np
import pytest

from qig_studio.constellation import couple_step, min_pairwise_fr, seed_constellation

ROLES = ["perception", "heart", "memory", "action", "strategy", "ethics", "coordination", "meta"]
FLOOR = 0.03
N_TICKS = 800


def test_wide_births_are_distinct():
    """The Pillar-3 scars must be wide (the verified prerequisite). Births far above the collapse
    floor — and far above any plausible coupled equilibrium."""
    facs = seed_constellation(ROLES, base_seed=7)
    births = [type("V", (), {"basin": f.birth})() for f in facs]
    assert min_pairwise_fr(births) > 0.5, "birth scars not wide-independent — Round-0 prerequisite unmet"


def test_no_anchor_collapses():
    """§H control: with f_anchor=0 the constellation MUST collapse (proves the gate can fire).
    Pure rel-weighted sync is an attractive map whose fixed point is basin coincidence."""
    facs = seed_constellation(ROLES, base_seed=11, overlap_init=False)
    for _ in range(N_TICKS):
        couple_step(facs, f_sync=0.5, f_anchor=0.0)
    assert min_pairwise_fr(facs) < FLOOR, (
        "no-anchor constellation did NOT collapse — the anti-collapse test is vacuous (would pass "
        "for the wrong reason). Investigate before trusting any green run.")


@pytest.mark.parametrize("f_sync", [0.25, 0.5])
@pytest.mark.parametrize("f_anchor", [0.05, 0.20])
@pytest.mark.parametrize("overlap", [False, True])
def test_constellation_no_collapse(f_sync, f_anchor, overlap):
    """The gate. Wide-seeded births + anchor → individuation preserved over N ticks, INCLUDING the
    adversarial overlap-runtime-init case (all current basins start ~identical; births stay wide)."""
    facs = seed_constellation(ROLES, base_seed=3, overlap_init=overlap)
    history = []
    for _ in range(N_TICKS):
        d = couple_step(facs, f_sync=f_sync, f_anchor=f_anchor)
        history.append(d.min_pairwise_fr)
    final = min_pairwise_fr(facs)
    last_half_min = min(history[N_TICKS // 2:])
    assert final > FLOOR, f"collapsed: final min_pairwise_FR={final:.4f} (f_sync={f_sync}, f_anchor={f_anchor}, overlap={overlap})"
    assert last_half_min > FLOOR, f"dipped below floor after warmup: {last_half_min:.4f}"


def test_inbound_budget_caps_runaway():
    """A high f_sync is capped at INBOUND_BUDGET (0.7) per tick — ≥0.3 always retained for self+anchor."""
    from qig_studio.constellation import INBOUND_BUDGET

    facs = seed_constellation(ROLES, base_seed=5)
    d = couple_step(facs, f_sync=0.95, f_anchor=0.12)
    assert all(s <= INBOUND_BUDGET + 1e-9 for s in d.inbound_sync.values()), \
        f"inbound sync exceeded budget: {d.inbound_sync}"
    assert max(d.inbound_sync.values()) == pytest.approx(INBOUND_BUDGET, abs=1e-9)


def test_couple_step_no_commit_is_pure():
    """commit=False must not move the faculties (probe mode)."""
    facs = seed_constellation(ROLES, base_seed=9)
    before = [f.basin.copy() for f in facs]
    couple_step(facs, f_sync=0.5, f_anchor=0.1, commit=False)
    for f, b in zip(facs, before):
        assert np.allclose(f.basin, b), "commit=False mutated a faculty"
