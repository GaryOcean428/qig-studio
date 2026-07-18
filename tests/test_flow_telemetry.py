"""Flow-telemetry observable extraction — council directive 2026-07-17.

Tests that the two standing observables (update_density, self_monitor_rate) are correctly
extracted from telemetry extra dicts, handling absent/None/zero values gracefully.
"""
from __future__ import annotations

import math

from qig_studio.governance.flow_telemetry import FlowObservation, extract_flow


def test_full_signals_geometric_mean():
    """Both meta_awareness and foresight_confidence present → geometric mean."""
    obs = extract_flow({"basin_velocity": 0.12, "meta_awareness": 0.8, "foresight_confidence": 0.5}, step=42)
    assert isinstance(obs, FlowObservation)
    assert obs.update_density == 0.12
    assert abs(obs.self_monitor_rate - (0.8 * 0.5) ** 0.5) < 1e-5
    assert obs.step == 42


def test_missing_basin_velocity_defaults_to_zero():
    obs = extract_flow({"meta_awareness": 0.6, "foresight_confidence": 0.4})
    assert obs.update_density == 0.0


def test_missing_meta_awareness_falls_back_to_foresight():
    obs = extract_flow({"basin_velocity": 0.05, "foresight_confidence": 0.7})
    assert obs.self_monitor_rate == 0.7


def test_missing_foresight_falls_back_to_meta_awareness():
    obs = extract_flow({"basin_velocity": 0.05, "meta_awareness": 0.6})
    assert obs.self_monitor_rate == 0.6


def test_both_absent_self_monitor_zero():
    obs = extract_flow({"basin_velocity": 0.1})
    assert obs.self_monitor_rate == 0.0


def test_empty_extra_returns_zeroes():
    obs = extract_flow({})
    assert obs.update_density == 0.0
    assert obs.self_monitor_rate == 0.0
    assert obs.step == 0


def test_none_values_treated_as_absent():
    obs = extract_flow({"basin_velocity": None, "meta_awareness": None, "foresight_confidence": None})
    assert obs.update_density == 0.0
    assert obs.self_monitor_rate == 0.0


def test_observation_is_frozen():
    obs = extract_flow({"basin_velocity": 0.1, "meta_awareness": 0.5, "foresight_confidence": 0.5}, step=1)
    try:
        obs.update_density = 0.99
        assert False, "FlowObservation should be frozen"
    except AttributeError:
        pass
