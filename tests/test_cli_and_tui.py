"""CLI dispatch + TUI module import (without the optional 'textual' extra)."""

from __future__ import annotations

from qig_studio import tui
from qig_studio.__main__ import main


def test_tui_module_imports_without_textual():
    # Importing the module must NOT require textual (lazy in run_tui).
    assert hasattr(tui, "run_tui")


def test_fmt_telemetry_renders():
    s = tui._fmt_telemetry(
        {
            "phi": 0.70,
            "kappa": 64.0,
            "regime": "geometric",
            "basin_distance": 0.1,
            "loss": 1.2,
            "step": 5,
            "delta_phi": 0.01,
        }
    )
    assert "Φ" in s and "geometric" in s and "step 5" in s


def test_cli_unknown_command_returns_usage_code():
    assert main(["definitely-not-a-command"]) == 2
