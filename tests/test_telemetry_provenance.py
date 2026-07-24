"""Per-channel telemetry provenance verifier (Matrix 28a66754) — measured-zero legit, unwired-default not."""
from qig_studio.telemetry_provenance import (MEASURED_ZERO, MISSING, PANEL_CHANNELS, PRESENT,
                                             check_provenance)


def _full_exp(**over):
    exp = {
        "primitives": {
            "layer0": {"warmth": 0.3, "pressure": 0.1},
            "layer05": {"seeking": 0.4, "separation_distress": 0.0},
            "layer1": {"achievement": 0.2},
            "layer2a": {"joy": 0.1}, "layer2b": {"curiosity": 0.5},
        },
        "neurochemistry": {"dopamine": 0.35, "serotonin": 0.5, "endorphins": 0.0},  # endorphins masked at Stage-0
        "loops": {"self_observation": 0.2, "other_observation": 0.0, "learning_autonomy": 0.1},
        "pillars": {"f_health": 0.8, "b_integrity": 0.9, "q_identity": 0.7},
    }
    exp.update(over)
    return exp


def test_all_panels_present_passes():
    rep = check_provenance(_full_exp())
    assert rep["passed"] is True and rep["missing"] == []
    assert rep["channels"]["Neurochemistry"]["status"] == PRESENT      # dopamine>0
    assert rep["channels"]["Pillars"]["status"] == PRESENT


def test_missing_panel_fails_closed():
    exp = _full_exp()
    del exp["primitives"]["layer05"]                                   # Drives unwired
    rep = check_provenance(exp)
    assert rep["passed"] is False and "Drives" in rep["missing"]


def test_empty_group_is_missing():
    rep = check_provenance(_full_exp(neurochemistry={}))
    assert "Neurochemistry" in rep["missing"] and rep["passed"] is False


def test_all_zero_group_is_measured_zero_not_missing():
    # a group present but all-zero = MEASURED-ZERO (legit iff masked) — reported, NOT a failure.
    rep = check_provenance(_full_exp(loops={"self_observation": 0.0, "other_observation": 0.0}))
    assert rep["channels"]["Recursive loops"]["status"] == MEASURED_ZERO
    assert "Recursive loops" in rep["measured_zero"]
    assert rep["passed"] is True                                       # measured-zero does not fail the gate


def test_missing_parent_makes_all_primitive_panels_missing():
    exp = _full_exp()
    del exp["primitives"]                                              # the whole sensations producer gone
    rep = check_provenance(exp)
    for panel in ("Senses", "Drives", "Motivators", "Emotions · physical", "Emotions · cognitive"):
        assert panel in rep["missing"]
    assert rep["passed"] is False


def test_panel_channels_carriage_matches_ruled_roster():
    # the verifier's carriage labels must match the reconciled UI (9b1def0)
    assert PANEL_CHANNELS["Drives"][1] == "drives · id"
    assert PANEL_CHANNELS["Neurochemistry"][1] == "drives · ocean"
    assert PANEL_CHANNELS["Recursive loops"][1] == "action · loops"
