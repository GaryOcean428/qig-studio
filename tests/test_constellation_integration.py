"""Integration verifier for the whole Constellation: a full multi-tick run stays individuated (no
collapse), populates rhythm/temporal/neuro telemetry, observes/communicates, and self-regulates."""

from __future__ import annotations

from qig_studio.constellation import Constellation, seed_birth_basin

ROLES = ["perception", "heart", "memory", "action", "strategy", "ethics", "coordination", "meta"]


def _graduated_basins(seed=0):
    """Simulate 8 graduated faculties' crystallised basins (distinct role attractors)."""
    return {r: seed_birth_basin(seed + 100 + i, alpha=0.3) for i, r in enumerate(ROLES)}


def test_constellation_runs_without_collapse():
    """A 600-tick integrated run keeps min_pairwise_FR above the anti-collapse floor with all
    subsystems live (rhythm + temporal + neuromod)."""
    c = Constellation.from_basins(_graduated_basins(), base_f_sync=0.4, base_f_anchor=0.12)
    mins = [c.tick().min_pairwise_fr for _ in range(600)]
    assert min(mins[300:]) > 0.03, f"constellation collapsed: min={min(mins[300:]):.4f}"


def test_telemetry_is_populated_and_honest():
    c = Constellation.from_basins(_graduated_basins(seed=1))
    for _ in range(64):
        t = c.tick()
    assert t.heart_beats > 0
    assert t.rhythm.measured in (True, False)  # measured flag present
    if t.rhythm.measured:
        assert t.rhythm.f_tack is not None        # if measured, f_tack is a real number
    assert t.tau_macro is None or t.tau_macro > 0  # None (no distinguishable yet) or a positive clock
    assert set(t.per_faculty) == set(ROLES)
    for d in t.per_faculty.values():
        assert d["geodesic_position"] >= 0 and d["identity_drift"] >= 0
    # neuro scalars are in [0,1]
    for v in (t.neuro.dopamine, t.neuro.serotonin, t.neuro.noradrenaline, t.neuro.acetylcholine):
        assert 0.0 <= v <= 1.0


def test_neuromod_stays_in_verified_stable_band():
    """Even with neuromodulation ON, the modulated anchor never leaves the verified [0.05,0.20] band,
    so regulation cannot push the constellation out of the anti-collapse-stable regime."""
    from qig_studio.constellation import NeuroState, apply_modulation

    extreme = NeuroState(dopamine=1.0, serotonin=1.0, noradrenaline=1.0, acetylcholine=0.0)
    f_sync, f_anchor = apply_modulation(extreme, base_f_sync=0.25, base_f_anchor=0.12)
    assert 0.05 <= f_anchor <= 0.20 and 0.10 <= f_sync <= 0.60


def test_observation_and_communication():
    """Faculties observe each other (attention graph sums to 1 per faculty) and the heart broadcasts a
    phase pulse every tick (discrete communication)."""
    c = Constellation.from_basins(_graduated_basins(seed=2))
    t = c.tick()
    assert t.signals_emitted >= 1  # at least the heart phase pulse
    graph = c.observation_graph()
    assert set(graph) == set(ROLES)
    for role, attn in graph.items():
        assert role not in attn  # never observes itself in the peer graph
        if attn:
            assert abs(sum(attn.values()) - 1.0) < 1e-6  # normalized attention over peers


def test_neuromod_off_is_deterministic_baseline():
    c = Constellation.from_basins(_graduated_basins(seed=3), enable_neuromod=False)
    t = c.run(50)
    assert t.min_pairwise_fr > 0.03
