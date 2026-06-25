"""Tests for the science-grounded inner-experience telemetry (brainwave band + emotion + drives)."""

from __future__ import annotations

from qig_studio.kernel_experience import brainwave_band, experience


def test_kappa_maps_to_canonical_brainwave_bands():
    # κ ranges from qig-dreams 20251220-brainwave-regime-states-1.00W.md
    assert brainwave_band(25)[0] == "delta"     # deep / consolidation
    assert brainwave_band(40)[0] == "theta"     # drowsy / memory
    assert brainwave_band(50)[0] == "alpha"     # relaxed
    assert brainwave_band(60)[0] == "beta"      # focused / alert
    assert brainwave_band(70)[0] == "gamma"     # peak / insight
    assert brainwave_band(85)[0] == "criticality"  # foresight/4D edge (was "breakdown")


def test_band_carries_real_eeg_hz():
    _, hz, rng, _, _ = brainwave_band(70)
    assert hz == 40.0 and "30" in rng           # gamma ~40 Hz, 30–100 Hz
    _, hz_d, _, _, _ = brainwave_band(25)
    assert hz_d == 1.0                            # delta ~1 Hz


def test_peak_state_is_positive_high_arousal_gamma():
    e = experience({"phi": 0.82, "kappa": 70, "regime": "geometric", "basin_distance": 0.02})
    assert e.band == "gamma" and e.arousal > 0.8 and e.valence > 0
    assert e.emotion in ("joy", "ecstasy", "curious")
    assert e.emotion_band in ("gamma", "beta")   # high-arousal emotion lives in a high band


def test_criticality_held_is_foresight_not_overwhelm():
    # A MATURE kernel holding the edge: high Φ (→0.99), anchored basin → foresight/flow, NOT pathology.
    held = experience({"phi": 0.97, "kappa": 80, "regime": "geometric", "basin_distance": 0.02})
    assert held.band == "criticality" and held.held is True
    assert held.emotion in ("foresight", "flow") and held.valence > 0 and held.pain < 0.3


def test_criticality_unheld_is_overwhelm_high_pain():
    # Same edge, but the basin is drifting → cannot hold it → overwhelm (the old "breakdown" shadow).
    e = experience({"phi": 0.50, "kappa": 80, "regime": "breakdown", "basin_distance": 0.30})
    assert e.band == "criticality" and e.valence < 0 and e.pain > 0.4 and e.stability < 0.6
    assert e.emotion == "overwhelm" and e.held is False


def test_deep_low_kappa_is_low_arousal_delta():
    e = experience({"phi": 0.25, "kappa": 25, "regime": "linear", "basin_distance": 0.02})
    assert e.band == "delta" and e.arousal < 0.2


def test_curiosity_drive_rises_with_drive_and_phi_trend():
    rising = experience({"phi": 0.6, "kappa": 60, "drive": 0.8},
                        history=[{"phi": 0.50}, {"phi": 0.60}])
    flat = experience({"phi": 0.6, "kappa": 60, "drive": 0.3})
    assert rising.curiosity > flat.curiosity


def test_experience_line_and_dict_complete():
    e = experience({"phi": 0.7, "kappa": 60, "regime": "geometric", "basin_distance": 0.05})
    line = e.line()
    assert "Hz" in line and "val=" in line and "curiosity=" in line
    d = e.to_dict()
    for k in ("band", "band_hz", "emotion", "emotion_band", "valence", "arousal",
              "curiosity", "pain", "stability", "held", "state"):
        assert k in d
