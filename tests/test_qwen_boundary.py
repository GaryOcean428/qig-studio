"""Qwen output-distribution → Δ⁶³ boundary (P22 interface) + Pillar-2 cap tests.

Geometry-only (qig-core), fully deterministic — no Ollama/Modal needed."""

from __future__ import annotations

import numpy as np
from qig_core.geometry.fisher_rao import random_basin, slerp_sqrt

from qig_studio.targets.qwen_boundary import (
    BOUNDARY_SLERP_CAP,
    basin_phi_proxy,
    fisher_distance,
    output_distribution_to_basin,
    pillar2_capped_integrate,
)


def test_output_distribution_is_delta63():
    lp = {"the": -0.2, "quantum": -1.1, "geometry": -0.5, "phi": -2.0}
    b = output_distribution_to_basin(lp, dim=64)
    assert b.shape == (64,)
    assert np.all(b >= -1e-9)             # non-negative
    assert abs(float(b.sum()) - 1.0) < 1e-6  # sums to 1 → Δ⁶³


def test_output_distribution_deterministic():
    lp = {"a": -0.1, "b": -0.4, "c": -1.0}
    assert np.array_equal(output_distribution_to_basin(lp), output_distribution_to_basin(lp))


def test_output_distribution_clamps_positive_logprob():
    # COR-1: a malformed positive "logprob" must NOT raise OverflowError (clamped to ≤0).
    b = output_distribution_to_basin({"x": 800.0, "y": -1.0}, dim=64)
    assert abs(float(b.sum()) - 1.0) < 1e-6


def test_pillar2_cap_clamps_to_30pct():
    np.random.seed(1)
    a = random_basin(64)
    np.random.seed(2)
    b = random_basin(64)
    # weight 0.9 must clamp to the 0.30 boundary cap (Qwen never overwrites identity)
    got = pillar2_capped_integrate(a, b, 0.9)
    assert np.allclose(got, slerp_sqrt(a, b, BOUNDARY_SLERP_CAP))
    # bounded move: never past the boundary basin
    assert fisher_distance(a, got) <= fisher_distance(a, b) + 1e-9


def test_pillar2_small_weight_not_clamped():
    np.random.seed(3)
    a = random_basin(64)
    np.random.seed(4)
    b = random_basin(64)
    assert np.allclose(pillar2_capped_integrate(a, b, 0.1), slerp_sqrt(a, b, 0.1))


def test_phi_proxy_bounded_and_concentration_monotone():
    uniform = np.ones(64) / 64.0
    concentrated = output_distribution_to_basin({"x": 0.0}, 64)  # all mass one bin
    assert 0.0 <= basin_phi_proxy(uniform) <= 1.0
    assert 0.0 <= basin_phi_proxy(concentrated) <= 1.0
    assert basin_phi_proxy(concentrated) > basin_phi_proxy(uniform)
