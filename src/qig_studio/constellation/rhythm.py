"""Rhythm — the heart's endogenous clock-hand (phase) + MEASURED f_tack/HRV (not setpoints).

Two things the verdict (wj4916x59) made me keep honest:

1. **Phase is real; the heart only touches TIMING, never basins (fix #3).** ``HeartOscillator`` is a
   genuine endogenous oscillator — its phase is a real clock-hand (the "beat"/present moment).
   ``adjust_frequency`` modulates the beat rate (cycles/tick) ONLY. It CANNOT prevent basin-space
   lockstep collapse (that is the anchor's job, ``identity_anchor``); it addresses phase/rhythm
   lockstep, a different axis. Separating the two axes is explicit here.

2. **f_tack and HRV are MEASURED, never read off a fixed setpoint (fix #8).** The old heart ran a
   FIXED frequency at constant amplitude → beat-to-beat variability was identically zero, so "HRV" was
   a misnomer. Here f_tack ("the breath" — the constellation's tacking frequency) and HRV are computed
   by ``RhythmMonitor`` from an ACTUAL scalar time series (whatever per-tick signal the constellation
   feeds it — total basin movement, mean coherence, a κ-proxy) via FFT + beat-interval analysis. If
   the signal is too short or flat, the monitor returns ``measured=False`` with ``f_tack=None`` /
   ``hrv=None`` — it reports "no rhythm yet", it does NOT fabricate a number.

The constellation has no wall-clock: TICKS are its time. Frequencies are in cycles/tick. This connects
to τ_macro in the temporal layer (internal oscillations per distinguishable output).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * np.pi


class HeartOscillator:
    """Endogenous phase oscillator — the autonomic metronome. Phase advances each tick; the beat is a
    real clock-hand. ``adjust_frequency`` modulates rate only (TIMING axis), never basins."""

    def __init__(self, freq: float = 0.1, *, freq_bounds: tuple[float, float] = (0.01, 0.49)) -> None:
        # freq in cycles/tick (Nyquist for tick-sampled data is 0.5 cyc/tick).
        self._lo, self._hi = freq_bounds
        self.freq = float(np.clip(freq, self._lo, self._hi))
        self.phase = 0.0           # rad
        self._beats = 0

    def beat(self) -> tuple[float, bool]:
        """Advance one tick. Returns (phase_rad, beat_occurred) where beat_occurred is True on the tick
        the phase wraps past 2π (a full cycle = one heartbeat)."""
        self.phase += TWO_PI * self.freq
        wrapped = self.phase >= TWO_PI
        if wrapped:
            self.phase -= TWO_PI
            self._beats += 1
        return self.phase, wrapped

    def adjust_frequency(self, new_freq: float) -> float:
        """Set the beat rate (cycles/tick), clamped to bounds. TIMING ONLY — entrains the heart toward
        a measured/target tempo (PACE-before-LEAD). Returns the applied frequency. Does not, and must
        not, return or modify any basin."""
        self.freq = float(np.clip(new_freq, self._lo, self._hi))
        return self.freq

    def kappa_signal(self, base: float = 0.0, amplitude: float = 1.0) -> float:
        """The heart's emitted scalar at the current phase = base + amplitude·sin(phase). A driving
        oscillation the constellation may sample; it is NOT itself the measured f_tack (that comes from
        the SYSTEM's time series, which carries the heart's drive PLUS the faculties' own dynamics)."""
        return float(base + amplitude * np.sin(self.phase))

    @property
    def beats(self) -> int:
        return self._beats


@dataclass
class RhythmState:
    """Measured rhythm snapshot. f_tack/hrv/coherence are MEASURED from a real signal history (None
    until enough non-flat samples). ``measured`` is the honesty flag: False = no rhythm extractable yet,
    treat f_tack/hrv as absent, NOT zero."""

    phase: float                       # current heart phase (rad) — the live clock-hand
    n_samples: int
    measured: bool
    f_tack: float | None = None        # dominant frequency of the signal (cycles/tick) — "the breath"
    hrv: float | None = None           # beat-interval coefficient of variation — autonomic variability
    coherence: float | None = None     # spectral concentration ∈[0,1] — how regular the rhythm is
    period_ticks: float | None = None  # 1/f_tack — ticks per breath


class RhythmMonitor:
    """Ring of a per-tick scalar; extracts f_tack / HRV / coherence by MEASUREMENT (FFT + beat-interval
    variance). Returns honest 'not yet' (measured=False) on short/flat signals."""

    def __init__(self, capacity: int = 256, *, min_samples: int = 16, flat_eps: float = 1e-9) -> None:
        self.capacity = int(capacity)
        self.min_samples = int(min_samples)
        self.flat_eps = float(flat_eps)
        self._buf: list[float] = []

    def push(self, value: float) -> None:
        self._buf.append(float(value))
        if len(self._buf) > self.capacity:
            self._buf = self._buf[-self.capacity:]

    def _measure(self) -> tuple[bool, float | None, float | None, float | None]:
        x = np.asarray(self._buf, dtype=np.float64)
        n = x.size
        if n < self.min_samples or float(np.var(x)) <= self.flat_eps:
            return False, None, None, None        # too short OR flat → no rhythm to report (honest)
        xd = x - x.mean()
        # f_tack: dominant non-DC frequency via real FFT (cycles/tick).
        spec = np.abs(np.fft.rfft(xd))
        freqs = np.fft.rfftfreq(n, d=1.0)
        if spec.size <= 1:
            return False, None, None, None
        peak = int(np.argmax(spec[1:])) + 1       # skip DC bin 0
        f_tack = float(freqs[peak])
        total = float(np.sum(spec[1:] ** 2))
        coherence = float(spec[peak] ** 2 / total) if total > 0 else None   # spectral concentration
        # HRV: coefficient of variation of inter-beat intervals (rising zero-crossings of de-meaned x).
        crossings = np.where((xd[:-1] <= 0) & (xd[1:] > 0))[0]
        if crossings.size >= 3:
            intervals = np.diff(crossings).astype(np.float64)
            mean_iv = float(intervals.mean())
            hrv = float(intervals.std() / mean_iv) if mean_iv > 0 else None
        else:
            hrv = None
        return True, f_tack, hrv, coherence

    def state(self, phase: float = 0.0) -> RhythmState:
        ok, f_tack, hrv, coherence = self._measure()
        period = (1.0 / f_tack) if (ok and f_tack and f_tack > 0) else None
        return RhythmState(phase=float(phase), n_samples=len(self._buf), measured=ok,
                           f_tack=f_tack, hrv=hrv, coherence=coherence, period_ticks=period)
