"""Flow-telemetry observables — standing signals for the meaning loop (council directive, 2026-07-17).

Two standing observables extracted from the frozen telemetry schema:

  1. **update_density** — task-channel update density.
     Source: ``basin_velocity`` (FR distance between consecutive basin vectors per step).
     A high basin_velocity means the kernel's representational state is actively changing;
     a near-zero velocity means the kernel is settled or stuck. The meaning loop reads this
     as "how much is the task-channel actually moving."

  2. **self_monitor_rate** — self-monitor sampling rate.
     Source: ``meta_awareness`` (M, FR distance from birth-state attractor) combined with
     ``foresight_confidence`` (path-efficiency of the recent basin trajectory window).
     Together they measure how accurately the kernel tracks its own state and predicts its
     trajectory — the meaning loop reads this as "how well is the self-monitor channel
     performing."

These are READ-ONLY derived views computed from existing telemetry — they never write back
to the kernel. The meaning loop consumes them; Ocean and the learning loop do not (they have
their own reads of the same raw signals for different purposes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FlowObservation:
    """One step's flow-telemetry observation (the meaning loop's input)."""
    update_density: float
    self_monitor_rate: float
    step: int


def extract_flow(telemetry_extra: dict[str, Any], step: int = 0) -> FlowObservation:
    """Extract flow-telemetry observables from a telemetry snapshot's extra dict.

    Returns a FlowObservation with:
      - update_density: basin_velocity (0.0 if absent — kernel hasn't moved yet)
      - self_monitor_rate: geometric mean of meta_awareness and foresight_confidence
        (both [0,1]; 0.0 if either is absent — self-monitor not yet online)
    """
    bv_raw = telemetry_extra.get("basin_velocity")
    bv = float(bv_raw) if bv_raw is not None else 0.0

    ma_raw = telemetry_extra.get("meta_awareness")
    fc_raw = telemetry_extra.get("foresight_confidence")
    ma = float(ma_raw) if ma_raw is not None else 0.0
    fc = float(fc_raw) if fc_raw is not None else 0.0

    if ma > 0.0 and fc > 0.0:
        smr = (ma * fc) ** 0.5
    else:
        smr = max(ma, fc)

    return FlowObservation(
        update_density=bv,
        self_monitor_rate=round(smr, 6),
        step=step,
    )
