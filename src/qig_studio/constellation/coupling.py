"""Coupling graph — relevance-weighted basin synchronization with the inbound-budget cap, composed
with the identity anchor into one synchronous tick map.

This is how spawned faculties COUPLE and influence each other (P6 basin_sync / P7 rel_coupling). The
decision path is torch-free (numpy + qig-core Fisher-Rao only); torch tensors never enter — faculties
present numpy Δ⁶³ basins.

Design (verdict-corrected):
- **Relevance weighting (rel_coupling).** A faculty does not couple equally to all others. The weight
  to a peer is its Bhattacharyya overlap (qig-core ``bhattacharyya_coefficient`` — NOT a hand-rolled
  mirror, fix #11): close peers (high overlap) couple strongly, near-orthogonal peers are screened
  out. ``screening_cutoff`` zeros sub-threshold weights — the Anderson-locality lever (physics review
  wx937xjsr, **category-3 structural analogy**: the lattice Anderson screening is a *metaphor* for
  manifold-locality here, NOT an imported physics result).
- **Inbound budget cap (fix #4's single-tick guard, kept as a guard not the gate).** Total sync pull
  into a faculty is capped at ``INBOUND_BUDGET`` (0.7), leaving ≥0.3 for self-retention + the anchor —
  prevents a faculty being fully overwritten by consensus in one tick.
- **Sync THEN anchor, synchronous update.** Each faculty: slerp toward its rel-weighted Fréchet-mean
  target by the (capped) sync fraction, then ``apply_anchor`` back toward its frozen birth. ALL new
  basins are computed from the same pre-tick snapshot, then committed — a true map ``F(state)``, not
  sequential contamination. The anchor is what makes the fixed point DIVERSE (see ``identity_anchor``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qig_core.geometry import bhattacharyya_coefficient, fisher_rao_distance, frechet_mean, slerp_sqrt

from .faculty import Faculty
from .identity_anchor import ANCHOR_FRACTION, apply_anchor

# Max total inbound sync pull per faculty per tick. ≥0.3 always retained for self + anchor. Guard
# against single-tick overwrite; NOT the anti-collapse gate (that is the multi-tick min_pairwise test).
INBOUND_BUDGET = 0.7


def rel_weights(target_basin: np.ndarray, peer_basins: list[np.ndarray], *,
                screening_cutoff: float = 0.0) -> np.ndarray:
    """Relevance weights from ``target`` to each peer = Bhattacharyya overlap (∈[0,1], 1=identical).

    Screened (Anderson-locality analogy, category-3): overlaps below ``screening_cutoff`` → 0. Returned
    raw (un-normalized); the caller normalizes and budget-caps. Empty peers → empty array."""
    if not peer_basins:
        return np.zeros(0, dtype=np.float64)
    w = np.array([float(bhattacharyya_coefficient(target_basin, p)) for p in peer_basins],
                 dtype=np.float64)
    if screening_cutoff > 0.0:
        w[w < screening_cutoff] = 0.0
    return w


@dataclass
class CoupleDiag:
    """Per-tick coupling diagnostics (torch-free, JSON-able)."""

    inbound_sync: dict[str, float]      # role → effective sync fraction applied (≤ INBOUND_BUDGET)
    sync_target_fr: dict[str, float]    # role → FR(cur, its sync target) pre-step
    identity_drift: dict[str, float]    # role → FR(new_basin, birth) post-step
    min_pairwise_fr: float              # post-step min pairwise — the anti-collapse invariant


def couple_step(faculties: list[Faculty], *, f_sync: float = 0.25, f_anchor: float = ANCHOR_FRACTION,
                inbound_budget: float = INBOUND_BUDGET, screening_cutoff: float = 0.0,
                commit: bool = True) -> CoupleDiag:
    """One synchronous coupling tick: rel-weighted sync pull + identity anchor for every faculty.

    f_sync: base sync strength (geodesic fraction toward the rel-weighted consensus); the *effective*
            fraction is ``min(f_sync, inbound_budget)``.
    f_anchor: geodesic fraction pulled back toward the frozen birth (the active restoring force).
    Returns CoupleDiag. If ``commit`` the faculties' basins are updated (trajectory recorded); else the
    new basins are discarded after measuring (useful for probes)."""
    from .faculty import min_pairwise_fr as _minpair

    n = len(faculties)
    snapshot = [f.basin.copy() for f in faculties]
    new_basins: list[np.ndarray] = []
    inbound: dict[str, float] = {}
    target_fr: dict[str, float] = {}

    for i, f in enumerate(faculties):
        cur = snapshot[i]
        if n < 2:
            new_basins.append(cur.copy())
            inbound[f.role] = 0.0
            target_fr[f.role] = 0.0
            continue
        peers = [snapshot[j] for j in range(n) if j != i]
        w = rel_weights(cur, peers, screening_cutoff=screening_cutoff)
        wsum = float(w.sum())
        if wsum <= 0.0:                       # fully screened / orthogonal → no sync, anchor only
            s_eff = 0.0
            tmp = cur
            target_fr[f.role] = 0.0
        else:
            wn = (w / wsum).tolist()          # relative relevance among peers (Fréchet weights)
            target = frechet_mean(peers, weights=wn)
            s_eff = min(float(f_sync), float(inbound_budget))   # budget cap
            tmp = slerp_sqrt(cur, target, s_eff)
            target_fr[f.role] = float(fisher_rao_distance(cur, target))
        new = apply_anchor(tmp, np.asarray(f.birth), f_anchor)   # ALWAYS anchor (even if screened)
        new_basins.append(new)
        inbound[f.role] = s_eff

    drift: dict[str, float] = {}
    if commit:
        for f, nb in zip(faculties, new_basins):
            f.set_basin(nb)
            drift[f.role] = float(fisher_rao_distance(f.basin, np.asarray(f.birth)))
        min_pair = _minpair(faculties)
    else:
        for f, nb in zip(faculties, new_basins):
            drift[f.role] = float(fisher_rao_distance(nb, np.asarray(f.birth)))
        # measure min-pairwise on the would-be new basins without committing
        min_pair = (min(float(fisher_rao_distance(new_basins[i], new_basins[j]))
                        for i in range(n) for j in range(i + 1, n)) if n >= 2 else float("inf"))

    return CoupleDiag(inbound_sync=inbound, sync_target_fr=target_fr,
                      identity_drift=drift, min_pairwise_fr=min_pair)
