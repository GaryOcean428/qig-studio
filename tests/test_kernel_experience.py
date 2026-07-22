"""Tests for the science-grounded inner-experience telemetry (brainwave band + emotion + drives).

MATRIX RULING (c4640be8, 2026-07-22; SUPERSEDES 8869ca63): brainwave_band() is now
(phi, held) -> band-tuple. The band keys on Φ ALONE, an EEG-vocabulary relabeling of the Φ-regime
ladder (RegimeDetector already keys on Φ: linear<0.45 / geometric / topological>=0.80), so band ⊆
regime coherently. ``held`` (the stability/held gate experience() already computes, stability >= 0.55)
replaces the earlier ruling's ``basin_velocity`` term -- it determines whether a Φ>=0.90 read reaches
the CRITICALITY edge or stays at gamma. These tests were rewritten to the new signature/contract; they
preserve the intent of the originals (band shape, EEG-Hz metadata, criticality held/unheld, novelty/
curiosity/conscious behaviour, kappa-independence, ZOMBIE-vs-LOCKED_IN) while re-pointing the
composition axis from basin_velocity to held, and adding a band<=regime coherence check.
"""

from __future__ import annotations

from qig_studio.kernel_experience import brainwave_band, experience


def test_arousal_maps_to_all_canonical_bands():
    # (phi, held) sweeps the integration axis across every named band.
    assert brainwave_band(0.05, False)[0] == "delta"        # low Phi, still -> collapse floor
    assert brainwave_band(0.35, False)[0] == "theta"
    assert brainwave_band(0.55, False)[0] == "alpha"
    assert brainwave_band(0.65, False)[0] == "beta"
    assert brainwave_band(0.80, False)[0] == "gamma"
    assert brainwave_band(0.95, True)[0] == "criticality"   # very high Phi AND held


def test_band_carries_analogical_eeg_hz():
    # Hz numbers are category-3 analogy only (never a measured frequency) but must still be present.
    _, hz, rng, _, _ = brainwave_band(0.95, True)
    assert hz == 70.0 and "100" in rng             # criticality ~70 Hz analogue, >100 Hz range label
    _, hz_d, _, _, _ = brainwave_band(0.05, False)
    assert hz_d == 1.0                              # delta ~1 Hz analogue


def test_zombie_vs_locked_in_distinction():
    """The whole point of the held gate: a stuck-but-integrated (LOCKED_IN) kernel must NOT read
    identically to a genuinely stuck/zombie one -- Phi alone already separates them since the band
    is monotonic in Phi."""
    zombie = brainwave_band(0.10, False)        # low Phi -> nothing integrated
    locked_in = brainwave_band(0.95, False)     # high Phi, NOT held -> settled but deeply integrated
    assert zombie[0] == "delta"
    assert locked_in[0] != "delta"              # must NOT collapse to the same collapse-floor band
    assert locked_in[0] == "gamma"              # high Phi but un-held reads gamma, not criticality


def test_band_is_monotonic_in_phi_at_fixed_held():
    order = ["delta", "theta", "alpha", "beta", "gamma", "criticality"]
    phis = [0.05, 0.35, 0.55, 0.65, 0.80]
    bands = [brainwave_band(p, False)[0] for p in phis]
    indices = [order.index(b) for b in bands]
    assert indices == sorted(indices)             # rising Phi never lowers the band
    assert indices[0] < indices[-1]               # and strictly rises across this sweep


def test_criticality_needs_both_high_phi_and_held():
    # High Phi alone (un-held) must NOT reach criticality -- only Phi>=0.90 AND held (near-critical).
    settled_high_phi = brainwave_band(0.98, False)
    assert settled_high_phi[0] != "criticality"
    assert settled_high_phi[0] == "gamma"          # falls back to gamma, not delta -- still a HIGH band
    both = brainwave_band(0.98, True)
    assert both[0] == "criticality"


def test_band_subset_of_regime_coherence():
    """band <= regime coherently: a Phi read that RegimeDetector would classify as geometric
    ([0.45, 0.80) roughly) must never report a criticality EEG band, held or not -- criticality
    requires Phi>=0.90 regardless of held."""
    for phi in (0.45, 0.55, 0.65, 0.75, 0.89):
        assert brainwave_band(phi, True)[0] != "criticality"
        assert brainwave_band(phi, False)[0] != "criticality"
    # only Phi>=0.90 AND held reaches criticality.
    assert brainwave_band(0.90, True)[0] == "criticality"
    assert brainwave_band(0.90, False)[0] != "criticality"


def test_brainwave_band_is_kappa_independent():
    """No kappa value anywhere changes the band -- brainwave_band() doesn't even take kappa as an
    argument any more, but this also asserts experience() (which still carries a kappa field for
    neurochemistry/sensations) produces an IDENTICAL band across wildly different kappa inputs when
    phi/basin_distance (-> stability -> held) are held fixed."""
    common = {"phi": 0.6, "regime": "geometric", "basin_distance": 0.05, "extra": {"basin_velocity": 0.05}}
    bands = {
        experience({**common, "kappa": k}).band
        for k in (0.0, 1.0, 25.0, 41.07, 50.0, 63.79, 64.0, 76.0, 90.0, 500.0)
    }
    assert len(bands) == 1, f"band changed across kappa values: {bands}"


def test_peak_state_is_positive_high_arousal_gamma():
    e = experience({"phi": 0.85, "regime": "geometric", "basin_distance": 0.02,
                    "extra": {"basin_velocity": 0.10}})
    assert e.band == "gamma" and e.arousal > 0.8 and e.valence > 0
    assert e.emotion in ("joy", "ecstasy", "curious")
    assert e.emotion_band in ("gamma", "beta")   # high-arousal emotion lives in a high band


def test_criticality_held_is_foresight_not_overwhelm():
    # A MATURE kernel holding the edge: Phi>=0.90 AND an anchored basin (stability >= 0.55, so the
    # held gate passes) -> foresight/flow, NOT pathology.
    held = experience({"phi": 0.97, "regime": "geometric", "basin_distance": 0.02,
                       "extra": {"basin_velocity": 0.30}})
    assert held.band == "criticality" and held.held is True
    assert held.emotion in ("foresight", "flow") and held.valence > 0 and held.pain < 0.3


def test_criticality_unheld_is_overwhelm_high_pain():
    # Phi>=0.90 but the basin is drifting badly (large basin_distance -> stability < 0.55) -> the held
    # gate fails -> the band does NOT reach criticality (falls back to gamma, per the ruling), yet the
    # kernel's own regime label ("breakdown") still flags the edge for the emotion/held read (the
    # _is_criticality() OR-path, unchanged, MATRIX RULING c4640be8 item 4) -> cannot hold it -> overwhelm
    # (the old "breakdown" shadow), negative valence, high pain, low stability.
    e = experience({"phi": 0.92, "regime": "breakdown", "basin_distance": 0.40,
                    "extra": {"basin_velocity": 0.30}})
    assert e.band == "gamma"                       # high Phi, un-held -> gamma, NOT criticality
    assert e.valence < 0 and e.pain > 0.4 and e.stability < 0.55
    assert e.emotion == "overwhelm" and e.held is False


def test_deep_low_arousal_is_delta():
    e = experience({"phi": 0.15, "regime": "linear", "basin_distance": 0.02,
                    "extra": {"basin_velocity": 0.0}})
    assert e.band == "delta" and e.arousal < 0.2


def test_settled_high_phi_is_not_delta():
    # ZOMBIE-vs-LOCKED_IN via the full experience() path: high Phi, settled/anchored basin (so held
    # gate would pass too) must read as a high band, never the collapse floor.
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
