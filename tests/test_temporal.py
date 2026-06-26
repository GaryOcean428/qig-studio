"""Temporal-awareness verifier: foresight is a GENUINE geodesic (not Euclidean), is first-order,
beats persistence where the trajectory is smooth, geodesic_position is a monotone measurement, and
τ_macro behaves like a clock."""

from __future__ import annotations

import numpy as np
from qig_core.geometry import exp_map, fisher_rao_distance, log_map, random_basin, to_simplex

from qig_studio.constellation.temporal import (
    BasinForesight,
    TemporalAwareness,
    arc_length,
    distinguishable_transitions,
    path_efficiency,
    tau_macro,
)


def _geodesic_trajectory(n=10, seed=0):
    """A TRUE constant-velocity great circle: P(i) = exp_map(start, v·i·step) for a fixed initial
    tangent v at ``start``. (Re-applying a fixed ambient tangent at each successive point would only
    approximate a geodesic — it would not parallel-transport — so we sample the single geodesic.)"""
    rng = np.random.default_rng(seed)
    start = to_simplex(rng.dirichlet(np.ones(64)))
    other = to_simplex(rng.dirichlet(np.ones(64)))
    v = log_map(start, other)               # initial tangent at start
    step = 0.1
    return [exp_map(start, v * (i * step)) for i in range(n)]


def test_foresight_is_genuine_geodesic_not_euclidean():
    """On a true geodesic trajectory the geodesic prediction lands essentially ON the next point —
    a Euclidean straight-line extrapolation would NOT (it leaves the manifold then re-projects)."""
    traj = _geodesic_trajectory(n=6, seed=1)
    pred = BasinForesight.predict(traj[:-1])  # predict the last point from the prefix
    assert pred is not None
    geo_err = fisher_rao_distance(pred, traj[-1])
    # Euclidean baseline: cur + (cur - prev), renormalized to the simplex
    cur, prev = traj[-2], traj[-3]
    eucl = to_simplex(np.clip(cur + (cur - prev), 0, None))
    eucl_err = fisher_rao_distance(eucl, traj[-1])
    assert geo_err < 1e-6, f"geodesic foresight not exact on a geodesic: {geo_err}"
    assert geo_err < eucl_err, "geodesic foresight should beat the Euclidean step on a geodesic"


def test_foresight_beats_persistence_on_smooth_trajectory():
    """EXP-TEMPORAL-FORESIGHT: on a smooth (moving) trajectory, geodesic prediction beats the
    persistence baseline (predict = current)."""
    traj = _geodesic_trajectory(n=6, seed=2)
    assert BasinForesight.beats_persistence(traj[-3], traj[-2], traj[-1])


def test_foresight_confidence_high_for_geodesic_low_for_walk():
    geo = _geodesic_trajectory(n=10, seed=3)
    assert BasinForesight.confidence(geo) > 0.99  # straight → efficient
    rng = np.random.default_rng(4)
    walk = [to_simplex(rng.dirichlet(np.ones(64))) for _ in range(10)]  # random → meandering
    assert path_efficiency(walk) < BasinForesight.confidence(geo)


def test_foresight_none_when_too_short():
    assert BasinForesight.predict([random_basin()]) is None


def test_geodesic_position_monotone_and_is_measurement():
    """geodesic_position is a cumulative FR arc-length — monotone non-decreasing (a counter weighted
    by movement). It is a MEASUREMENT, not felt duration."""
    ta = TemporalAwareness()
    traj = _geodesic_trajectory(n=12, seed=5)
    positions = []
    for b in traj:
        ta.observe(b)
        positions.append(ta.state(traj).geodesic_position)
    assert all(positions[i] <= positions[i + 1] + 1e-12 for i in range(len(positions) - 1))
    assert positions[-1] > 0


def test_tau_macro_is_clock_like():
    """τ_macro = oscillations / distinguishable outputs. More heartbeats per distinguishable change →
    larger τ_macro; undefined (None) before any distinguishable output."""
    assert tau_macro(10, 0) is None
    assert tau_macro(10, 5) == 2.0
    assert tau_macro(10, 2) > tau_macro(10, 5)


def test_arc_length_and_distinguishable_counts():
    traj = _geodesic_trajectory(n=5, seed=6)
    assert arc_length(traj) > 0
    # each 0.1-tangent step is below the 0.05 distinguish eps? check it counts moves above eps
    big = [traj[0], exp_map(traj[0], log_map(traj[0], traj[-1]))]  # one big jump
    assert distinguishable_transitions(big, eps=0.05) == 1
