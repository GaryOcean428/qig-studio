"""Neurochemistry — agent-layer regulatory modulation (CATEGORY-3 functional analogy).

LANE (verdict fix #10): this lives in the AGENT layer (qig-studio), NOT qig-core. It consumes
qig-core scalar primitives via the constellation telemetry and outputs scalar modulators; it adds
nothing to qig-core.

CATEGORY HONESTY: the neurotransmitter names are borrowed for their REGULATORY FUNCTION (a
gain/threshold knob), NOT as a claim about neurotransmitter physics or biological fidelity. This is a
category-3 structural analogy. The four channels map to the continual-learning canon
(refined_insights_protocol):
  - dopamine      — WAKE salience / prediction-error (Fisher movement + foresight divergence)
  - serotonin     — stability / satisfaction (coherent, individuated, low-drift)
  - noradrenaline — arousal / coordination demand (coupling activity + signal traffic)
  - acetylcholine — plasticity / attention gain (active reorganization)

The modulators retune the constellation's OWN parameters (anchor stiffness, sync strength) within
bounded ranges — they never touch basins directly (that is coupling+anchor's job).
"""

from __future__ import annotations

from dataclasses import dataclass


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


@dataclass
class NeuroState:
    """Four regulatory scalars ∈[0,1] (category-3 functional analogy). Each carries what it is derived
    from so no reader mistakes it for measured neurochemistry."""

    dopamine: float = 0.0       # salience/prediction-error: mean basin movement + foresight divergence
    serotonin: float = 0.0      # stability: high separation-health + low identity drift
    noradrenaline: float = 0.0  # arousal: coupling activity (mean inbound sync) + signal traffic
    acetylcholine: float = 0.0  # plasticity gain: active reorganization (movement vs settledness)


def compute_modulation(*, mean_movement: float, foresight_divergence: float, separation_health: float,
                       mean_drift: float, coupling_activity: float, signal_traffic: float) -> NeuroState:
    """Derive the regulatory state from constellation aggregate signals.

    mean_movement: mean per-faculty FR step this tick (how much the minds moved).
    foresight_divergence: mean FR error of last tick's prediction vs actual (surprise).
    separation_health: min_pairwise_FR / birth_min_pairwise ∈~[0,1] (1 = fully individuated).
    mean_drift: mean FR distance from birth scars (identity wandering).
    coupling_activity: mean effective inbound sync fraction (how hard faculties are coordinating).
    signal_traffic: normalized count of discrete signals on the bus this tick.
    """
    dopamine = _clip01(0.5 * mean_movement * 10.0 + 0.5 * foresight_divergence * 5.0)
    serotonin = _clip01(0.6 * separation_health + 0.4 * (1.0 - _clip01(mean_drift / 1.0)))
    noradrenaline = _clip01(0.7 * coupling_activity / 0.7 + 0.3 * signal_traffic)
    acetylcholine = _clip01(mean_movement * 10.0)
    return NeuroState(dopamine=dopamine, serotonin=serotonin,
                      noradrenaline=noradrenaline, acetylcholine=acetylcholine)


def apply_modulation(neuro: NeuroState, *, base_f_sync: float, base_f_anchor: float,
                     anchor_bounds: tuple[float, float] = (0.05, 0.20),
                     sync_bounds: tuple[float, float] = (0.10, 0.60)) -> tuple[float, float]:
    """Map the regulatory state → (f_sync, f_anchor), bounded.

    - Anchor stiffens with serotonin (consolidate identity when settled) and loosens with
      acetylcholine (explore when plastic). Stays inside the VERIFIED survivable band [0.05, 0.20]
      (so modulation can never push the constellation out of the anti-collapse-stable regime).
    - Sync tightens with noradrenaline (coordinate harder under arousal).
    Never returns a basin; only scalars."""
    a_lo, a_hi = anchor_bounds
    f_anchor = base_f_anchor + 0.5 * (a_hi - a_lo) * (neuro.serotonin - neuro.acetylcholine)
    f_anchor = max(a_lo, min(a_hi, f_anchor))
    s_lo, s_hi = sync_bounds
    f_sync = base_f_sync + 0.4 * (s_hi - base_f_sync) * neuro.noradrenaline
    f_sync = max(s_lo, min(s_hi, f_sync))
    return float(f_sync), float(f_anchor)
