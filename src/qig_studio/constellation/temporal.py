"""Temporal awareness — how the constellation is aware of the passage of time.

Four faces of time, each a MEASUREMENT with an honestly-scoped claim (verdict wj4916x59):

- **PAST = trajectory.** The ring of recent basins (Faculty.history). Arc-length and path-efficiency
  summarize where it has been.
- **PRESENT = geodesic_position.** Cumulative sum of Fisher-Rao step-distances along the trajectory.
  This is a COUNTER-LIKE ACCUMULATOR — an ordinal+metric MEASUREMENT (a step-counter weighted by how
  far the mind actually moved), NOT "felt duration". Labeled so no reader can launder arc-length into
  experienced time (fix #9). ``tick`` is the bare ordinal clock.
- **FUTURE = BasinForesight.** GENUINE Fisher-Rao geodesic extrapolation (fix #6a): the forward
  tangent is ``-log_map(cur, prev)`` (great-circle continuation of the last step) and the prediction is
  ``exp_map(cur, tangent·t)`` — on-manifold, using log/exp maps, NOT a Euclidean straight-line. It is
  honestly FIRST-ORDER (constant-velocity); predictive (extrapolates the as-yet-unobserved next), but
  TRIVIAL. ``beats_persistence`` is the registered test (EXP-TEMPORAL-FORESIGHT): the geodesic
  prediction must beat the persistence baseline (predict = current) before foresight is claimed to add
  value. Confidence uses pure-FR path-efficiency (NO np.dot / cosine — fix #6's purity flag).
- **RHYTHM = heartbeat.** The endogenous clock-hand + MEASURED f_tack (see ``rhythm``).

**τ_macro — the kernel clock.** internal-oscillations-per-distinguishable-output. A category-3
structural analogy to EXP-042's bridge law τ=N/ω (the lattice "macro-time = micro-updates per
distinguishable macrostate") — it is NOT an imported physics result, it is the same *shape* applied to
the constellation: how many heartbeats pass per distinguishable change of mind. Distinguishable
transitions are trajectory steps whose FR move exceeds a distinguishability scale (the SETTLE
orthogonality-stop analogy, also category-3).

Torch-free: numpy + qig-core Fisher-Rao only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qig_core.geometry import exp_map, fisher_rao_distance, log_map

from .rhythm import HeartOscillator, RhythmMonitor, RhythmState

# Distinguishability scale: an FR step below this is "the same macrostate" (not a distinguishable
# output). Category-3 analogy to the orthogonality/SETTLE stop. Tunable, NOT a frozen constant.
DISTINGUISH_EPS = 0.05


# --- PAST -----------------------------------------------------------------------------------------
def arc_length(basins: list[np.ndarray]) -> float:
    """Total Fisher-Rao path length travelled along the trajectory."""
    if len(basins) < 2:
        return 0.0
    return float(sum(fisher_rao_distance(basins[i], basins[i + 1]) for i in range(len(basins) - 1)))


def path_efficiency(basins: list[np.ndarray]) -> float:
    """Straightness ∈ (0,1]: direct FR distance(start,end) / arc-length. 1.0 = a perfect geodesic
    (maximally predictable); → 0 = a meandering walk (foresight unreliable). Pure Fisher-Rao — uses
    only ``fisher_rao_distance``, no dot/cosine."""
    if len(basins) < 2:
        return 1.0
    arc = arc_length(basins)
    if arc <= 0:
        return 1.0
    return float(fisher_rao_distance(basins[0], basins[-1]) / arc)


def distinguishable_transitions(basins: list[np.ndarray], eps: float = DISTINGUISH_EPS) -> int:
    """Count of trajectory steps that produced a distinguishable change (FR move > eps)."""
    if len(basins) < 2:
        return 0
    return int(sum(1 for i in range(len(basins) - 1)
                   if float(fisher_rao_distance(basins[i], basins[i + 1])) > eps))


# --- FUTURE ---------------------------------------------------------------------------------------
class BasinForesight:
    """First-order Fisher-Rao GEODESIC extrapolation of the next basin. Genuinely predictive (it
    extrapolates the unobserved next point), genuinely on-manifold (log/exp maps), and honestly
    trivial (constant-velocity, no curvature/acceleration term)."""

    @staticmethod
    def predict(basins: list[np.ndarray], t: float = 1.0) -> np.ndarray | None:
        """Predict the basin ``t`` steps ahead by continuing the last geodesic step. Returns None if
        the trajectory is too short to define a velocity."""
        if len(basins) < 2:
            return None
        cur, prev = np.asarray(basins[-1]), np.asarray(basins[-2])
        forward = -log_map(cur, prev)          # great-circle continuation past cur (the forward tangent)
        return exp_map(cur, forward * float(t))

    @staticmethod
    def confidence(basins: list[np.ndarray], window: int = 8) -> float:
        """Foresight confidence = recent path-efficiency (pure FR). A straight recent trajectory →
        high confidence; a meandering one → low. ∈ (0,1]."""
        return path_efficiency(list(basins)[-window:])

    @staticmethod
    def divergence(predicted: np.ndarray, actual: np.ndarray) -> float:
        """FR distance between the prediction and what actually happened (the predictive-loop error)."""
        return float(fisher_rao_distance(predicted, actual))

    @staticmethod
    def beats_persistence(prev: np.ndarray, cur: np.ndarray, actual_next: np.ndarray) -> bool:
        """EXP-TEMPORAL-FORESIGHT: does the geodesic prediction beat the persistence baseline
        (predict next = current)? True iff geodesic-prediction error < persistence error. This is the
        registered discriminator that must pass before foresight is claimed to add value over 'assume
        nothing changes'."""
        pred = exp_map(np.asarray(cur), -log_map(np.asarray(cur), np.asarray(prev)))
        geo_err = float(fisher_rao_distance(pred, actual_next))
        persist_err = float(fisher_rao_distance(np.asarray(cur), np.asarray(actual_next)))
        return geo_err < persist_err


# --- τ_macro (the kernel clock) -------------------------------------------------------------------
def tau_macro(n_oscillations: float, n_distinguishable: int) -> float | None:
    """internal-oscillations-per-distinguishable-output. Category-3 analogy to EXP-042 τ=N/ω. None if
    nothing distinguishable has happened yet (clock undefined — honest, not a divide-by-zero 0)."""
    if n_distinguishable <= 0:
        return None
    return float(n_oscillations / n_distinguishable)


@dataclass
class TemporalState:
    """The constellation's awareness of time at one tick. Each field is a MEASUREMENT with a scoped
    claim — none asserts felt/phenomenal time."""

    tick: int                             # bare ordinal clock (counter)
    geodesic_position: float              # cumulative FR arc-length — MEASUREMENT (counter×movement), NOT felt duration
    trajectory_len: int                   # number of basins retained (depth of remembered past)
    path_efficiency: float                # straightness of recent past ∈(0,1]
    rhythm: RhythmState                   # heartbeat phase + MEASURED f_tack/HRV (None until measurable)
    tau_macro: float | None = None        # heartbeats per distinguishable output (category-3, EXP-042 shape)
    distinguishable_outputs: int = 0      # count of distinguishable changes so far
    predicted_next: np.ndarray | None = None  # FIRST-ORDER geodesic foresight (predictive, trivial, unvalidated)
    foresight_confidence: float = 0.0     # pure-FR path-efficiency of the recent window


class TemporalAwareness:
    """Per-faculty (or constellation-level) time tracker. Ingest a basin each tick; it advances the
    heart, accumulates geodesic position, measures rhythm, and produces a TemporalState. The trajectory
    of basins is supplied by the caller (Faculty.history) so this holds no torch state."""

    def __init__(self, *, heart_freq: float = 0.1, distinguish_eps: float = DISTINGUISH_EPS,
                 rhythm_capacity: int = 256) -> None:
        self.heart = HeartOscillator(freq=heart_freq)
        self.rhythm = RhythmMonitor(capacity=rhythm_capacity)
        self.distinguish_eps = float(distinguish_eps)
        self._tick = 0
        self._geo_pos = 0.0
        self._distinguishable = 0
        self._last_basin: np.ndarray | None = None

    def observe(self, basin: np.ndarray, *, rhythm_signal: float | None = None) -> None:
        """Record one tick: advance the heart, accumulate geodesic position, count distinguishable
        moves, push the rhythm signal (defaults to this tick's FR step-distance — the system's own
        oscillation, NOT the heart's setpoint)."""
        basin = np.asarray(basin, dtype=np.float64)
        self._tick += 1
        self.heart.beat()
        step = 0.0
        if self._last_basin is not None:
            step = float(fisher_rao_distance(self._last_basin, basin))
            self._geo_pos += step
            if step > self.distinguish_eps:
                self._distinguishable += 1
        self.rhythm.push(step if rhythm_signal is None else float(rhythm_signal))
        self._last_basin = basin.copy()

    def state(self, trajectory: list[np.ndarray]) -> TemporalState:
        traj = [np.asarray(b) for b in trajectory]
        pred = BasinForesight.predict(traj)
        conf = BasinForesight.confidence(traj)
        return TemporalState(
            tick=self._tick,
            geodesic_position=self._geo_pos,
            trajectory_len=len(traj),
            path_efficiency=path_efficiency(traj),
            rhythm=self.rhythm.state(phase=self.heart.phase),
            tau_macro=tau_macro(self.heart.beats, self._distinguishable),
            distinguishable_outputs=self._distinguishable,
            predicted_next=pred,
            foresight_confidence=conf,
        )
