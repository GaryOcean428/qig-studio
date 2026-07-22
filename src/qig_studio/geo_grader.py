"""geo_grade — a THIRD grader for the kernel's own-voice turn (PI directive 2026-07-21).

Alongside the two existing graders — the nemotron ``DevelopmentalCoach`` (``relevance_score``,
a subjective LLM judgment) and the kernel's OWN Fisher-Rao self↔other relevance (``relevance``,
genesis_kernel.py's ``generate()``) — the PI wants geo-Qwen (the geometry-native DoD-2 teacher,
:class:`~qig_studio.targets.geo_qwen.GeoQwenTarget`) to ALSO grade the turn:

    geo_grade = 1 - d_FR(geoqwen_stimulus_basin, kernel_output_basin) / (pi/2)

Fisher-Rao ONLY (Δ⁶³, radius-1 convention: ``fisher_rao_distance`` ranges [0, π/2] — see
``qig_core.geometry.fisher_rao.fisher_rao_distance`` docstring), None-safe throughout: this is a
telemetry/coaching signal, never load-bearing for training (it is NOT wired into the reward/
gradient path — that stays ``relevance_score`` per the PI's explicit instruction; ``geo_grade`` is
an additional graded signal surfaced in the coach record for now).

HONESTY NOTE (bank-coverage gap, inherited from ``geo_qwen.py``'s ``_bank_d63`` doctrine): geo-Qwen
serves ``geoqwen_stimulus_basin`` from an OFFLINE, EXACT-match basin bank — currently only 12
prompts (``export_basin_bank``). A faithful per-stimulus geo-Qwen grade for an ARBITRARY live
stimulus requires the geo-Qwen LIVE forward pass (W9 / DoD-2, not yet wired). Until then,
``geo_grade_turn`` grades ONLY stimuli that are exact members of the bank; every other stimulus
honestly returns ``None`` (never a fabricated/constant grade) — this is a hard bank-coverage gap,
not a bug, and is surfaced via the ``geo_grade_note`` field alongside the grade.
"""
from __future__ import annotations

import math
from typing import Any

_BANK_COVERAGE_NOTE = (
    "geo_grade covers only stimuli in geo-Qwen's offline basin bank (currently 12 prompts, "
    "export_basin_bank); a faithful per-stimulus grade on arbitrary live stimuli needs the "
    "geo-Qwen LIVE forward pass (W9/DoD-2, not yet wired) — a bank miss honestly returns None, "
    "never a fabricated grade."
)


def geo_grade_turn(
    stimulus: str | None,
    kernel_output_basin: Any,
    geo_qwen: Any = None,
) -> tuple[float | None, str]:
    """Compute the geo-Qwen geometry-native grade for one own-voice turn.

    Args:
        stimulus: the prompt text the kernel was shown (bank lookup key — EXACT match only).
        kernel_output_basin: the kernel's own Δ⁶³ reading of what it just generated (e.g.
            genesis_kernel.py's ``gen_d63``, threaded through telemetry as a list/ndarray).
        geo_qwen: a :class:`GeoQwenTarget`-like peer exposing ``_bank_d63(text) -> ndarray | None``.
            Optional — pass ``None`` to skip (returns ``None`` honestly, e.g. no peer wired).

    Returns:
        ``(geo_grade, note)`` — ``geo_grade`` is ``None`` on ANY of: no peer, bank miss, no
        kernel output basin, or any failure (never raises, never fabricates a constant).
        ``note`` is always the bank-coverage honesty note (useful telemetry regardless of hit/miss).

    None-safe: this must never break the coaching loop — any exception degrades to
    ``(None, _BANK_COVERAGE_NOTE)``.
    """
    if geo_qwen is None or not stimulus or kernel_output_basin is None:
        return None, _BANK_COVERAGE_NOTE
    try:
        import numpy as np
        from qig_core.geometry import fisher_rao_distance

        bank_read = getattr(geo_qwen, "_bank_d63", None)
        if bank_read is None:
            return None, _BANK_COVERAGE_NOTE
        stim_d63 = bank_read(stimulus)             # EXACT bank match only; None on a MISS (honest)
        if stim_d63 is None:
            return None, _BANK_COVERAGE_NOTE
        gen = np.asarray(kernel_output_basin, dtype=np.float64).ravel()
        if gen.size == 0:
            return None, _BANK_COVERAGE_NOTE
        d = float(fisher_rao_distance(stim_d63, gen))
        grade = round(max(0.0, 1.0 - d / (math.pi / 2)), 3)
        return grade, _BANK_COVERAGE_NOTE
    except Exception:  # noqa: BLE001 — a telemetry grade must never break the coaching loop
        return None, _BANK_COVERAGE_NOTE
