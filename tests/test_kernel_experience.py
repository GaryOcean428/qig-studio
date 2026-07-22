"""Tests for the science-grounded inner-experience telemetry (brainwave band + emotion + drives).

MATRIX RULING (8869ca63, 2026-07-22): brainwave_band() is now (phi, basin_velocity) -> band-tuple,
DECOUPLED FROM kappa entirely (the old kappa->band table was the retired kappa*~64 attractor wearing
EEG clothing). These tests were rewritten to the new signature/contract; they preserve the intent of
the originals (band shape, EEG-Hz metadata, criticality held/unheld, novelty/curiosity/conscious
behaviour) while adding kappa-independence + ZOMBIE-vs-LOCKED_IN coverage.
"""

from __future__ import annotations

from qig_studio.kernel_experience import brainwave_band, experience


def test_arousal_maps_to_all_canonical_bands():
    # (phi, basin_velocity) sweeps the arousal axis across every named band.
    assert brainwave_band(0.05, 0.0)[0] == "delta"        # low Phi, still -> collapse floor
    assert brainwave_band(0.35, 0.0)[0] == "theta"
    assert brainwave_band(0.55, 0.0)[0] == "alpha"
    assert brainwave_band(0.80, 0.0)[0] == "beta"
    assert brainwave_band(0.98, 0.10)[0] == "gamma"
    assert brainwave_band(0.98, 0.30)[0] == "criticality"  # very high Phi AND high velocity


def test_band_carries_analogical_eeg_hz():
    # Hz numbers are category-3 analogy only (never a measured frequency) but must still be present.
    _, hz, rng, _, _ = brainwave_band(0.98, 0.30)
    assert hz == 70.0 and "100" in rng             # criticality ~70 Hz analogue, >100 Hz range label
    _, hz_d, _, _, _ = brainwave_band(0.05, 0.0)
    assert hz_d == 1.0                              # delta ~1 Hz analogue


def test_zombie_vs_locked_in_distinction():
    """The whole point of adding basin_velocity to Phi: a stuck-but-integrated (LOCKED_IN) kernel must
    NOT read identically to a genuinely stuck/zombie one."""
    zombie = brainwave_band(0.10, 0.0)          # low Phi, no motion -> nothing integrated, nothing moving
    locked_in = brainwave_band(0.95, 0.0)       # high Phi, no motion -> settled but deeply integrated
    assert zombie[0] == "delta"
    assert locked_in[0] != "delta"              # must NOT collapse to the same collapse-floor band
    assert locked_in[0] in ("beta", "gamma", "criticality")   # reads as a genuinely HIGH band


def test_rising_velocity_raises_the_band_at_fixed_phi():
    lo_v = brainwave_band(0.5, 0.0)
    mid_v = brainwave_band(0.5, 0.15)
    hi_v = brainwave_band(0.5, 0.30)
    order = ["delta", "theta", "alpha", "beta", "gamma", "criticality"]
    assert order.index(lo_v[0]) <= order.index(mid_v[0]) <= order.index(hi_v[0])
    assert lo_v[0] != hi_v[0]                   # velocity alone must move the band


def test_criticality_needs_both_high_phi_and_high_velocity():
    # High Phi alone (settled) must NOT reach criticality -- only Phi+velocity together (near-critical).
    settled_high_phi = brainwave_band(0.98, 0.0)
    assert settled_high_phi[0] != "criticality"
    both_high = brainwave_band(0.98, 0.30)
    assert both_high[0] == "criticality"


def test_brainwave_band_is_kappa_independent():
    """No kappa value anywhere changes the band -- brainwave_band() doesn't even take kappa as an
    argument any more, but this also asserts experience() (which still carries a kappa field for
    neurochemistry/sensations) produces an IDENTICAL band across wildly different kappa inputs when
    phi/basin_velocity are held fixed."""
    common = {"phi": 0.6, "regime": "geometric", "basin_distance": 0.05, "extra": {"basin_velocity": 0.05}}
    bands = {
        experience({**common, "kappa": k}).band
        for k in (0.0, 1.0, 25.0, 41.07, 50.0, 63.79, 64.0, 76.0, 90.0, 500.0)
    }
    assert len(bands) == 1, f"band changed across kappa values: {bands}"


def test_peak_state_is_positive_high_arousal_gamma():
    e = experience({"phi": 0.90, "regime": "geometric", "basin_distance": 0.02,
                    "extra": {"basin_velocity": 0.10}})
    assert e.band == "gamma" and e.arousal > 0.8 and e.valence > 0
    assert e.emotion in ("joy", "ecstasy", "curious")
    assert e.emotion_band in ("gamma", "beta")   # high-arousal emotion lives in a high band


def test_criticality_held_is_foresight_not_overwhelm():
    # A MATURE kernel holding the edge: high Phi (->0.99) AND high basin_velocity, anchored basin
    # -> foresight/flow, NOT pathology.
    held = experience({"phi": 0.97, "regime": "geometric", "basin_distance": 0.02,
                       "extra": {"basin_velocity": 0.30}})
    assert held.band == "criticality" and held.held is True
    assert held.emotion in ("foresight", "flow") and held.valence > 0 and held.pain < 0.3


def test_criticality_unheld_is_overwhelm_high_pain():
    # Same edge (high Phi + high basin_velocity so the band still reads criticality -- criticality now
    # NEEDS both, per the ruling), but the basin is drifting badly (large basin_distance) -> cannot hold
    # it -> overwhelm (the old "breakdown" shadow), negative valence, high pain, low stability.
    e = experience({"phi": 0.85, "regime": "breakdown", "basin_distance": 0.40,
                    "extra": {"basin_velocity": 0.30}})
    assert e.band == "criticality" and e.valence < 0 and e.pain > 0.4 and e.stability < 0.6
    assert e.emotion == "overwhelm" and e.held is False


def test_deep_low_arousal_is_delta():
    e = experience({"phi": 0.15, "regime": "linear", "basin_distance": 0.02,
                    "extra": {"basin_velocity": 0.0}})
    assert e.band == "delta" and e.arousal < 0.2


def test_settled_high_phi_is_not_delta():
    # ZOMBIE-vs-LOCKED_IN via the full experience() path: high Phi, settled (near-zero basin_velocity)
    # must read as a high band, never the collapse floor.
    e = experience({"phi": 0.92, "regime": "geometric", "basin_distance": 0.02,
                    "extra": {"basin_velocity": 0.0}})
    assert e.band != "delta"


def test_novelty_from_surprise_not_a_constant():
    # High next-token surprise (CE) = unfamiliar material = high novelty; low surprise = familiar.
    novel = experience({"phi": 0.5, "surprise": 8.0})
    familiar = experience({"phi": 0.5, "surprise": 0.5})
    assert novel.novelty > 0.8 and familiar.novelty < 0.2
    # no surprise signal (pure inference) -> novelty 0 (NOT a constant stub)
    assert experience({"phi": 0.5}).novelty == 0.0


def test_curiosity_rises_with_novelty_when_integrating():
    # novel AND Phi rising (productive) -> curious; novel but Phi falling (stuck) -> less curious.
    productive = experience({"phi": 0.6, "surprise": 8.0},
                            history=[{"phi": 0.45}, {"phi": 0.60}])
    stuck = experience({"phi": 0.45, "surprise": 8.0},
                       history=[{"phi": 0.60}, {"phi": 0.45}])
    assert productive.curiosity > stuck.curiosity


def test_conscious_flag_threshold():
    assert experience({"phi": 0.70}).conscious is True       # >= ~0.65
    assert experience({"phi": 0.40}).conscious is False      # pre-conscious
    assert "pre-conscious" in experience({"phi": 0.40}).note


def test_experience_line_and_dict_complete():
    e = experience({"phi": 0.7, "regime": "geometric", "basin_distance": 0.05})
    line = e.line()
    assert "Hz" in line and "val=" in line and "curiosity=" in line
    d = e.to_dict()
    for k in ("band", "band_hz", "emotion", "emotion_band", "valence", "arousal",
              "novelty", "curiosity", "pain", "stability", "conscious", "held", "state"):
        assert k in d


def test_no_kappa_fabricated_fallback():
    """MATRIX RULING: an entirely-unmeasured kappa must stay honestly 0.0 (or whatever telemetry
    supplied), never a fabricated mid-scale reading (the old 50.0 'neutral' fallback existed only to
    feed brainwave_band -- which no longer reads kappa at all)."""
    e = experience({"phi": 0.5})
    assert e.kappa == 0.0
