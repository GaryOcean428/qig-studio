"""Rhythm verifier: f_tack/HRV are genuinely MEASURED (recover a known frequency; honest 'not yet' on
flat signals; HRV>0 only when intervals actually vary), and the heart touches timing only."""

from __future__ import annotations

import numpy as np

from qig_studio.constellation.rhythm import HeartOscillator, RhythmMonitor


def test_f_tack_recovers_known_frequency():
    """Inject a pure sine at 0.05 cycles/tick → measured f_tack ≈ 0.05 (within FFT bin resolution)."""
    mon = RhythmMonitor(capacity=200)
    f_true = 0.05
    for t in range(200):
        mon.push(np.sin(2 * np.pi * f_true * t))
    s = mon.state()
    assert s.measured
    assert s.f_tack is not None and abs(s.f_tack - f_true) < 0.01, f"f_tack={s.f_tack}"
    assert s.period_ticks is not None and abs(s.period_ticks - 20.0) < 2.0
    assert s.coherence is not None and s.coherence > 0.5  # a pure tone is spectrally concentrated


def test_flat_signal_reports_no_rhythm_not_zero():
    """A constant signal must report measured=False with f_tack/hrv=None — NOT a fabricated 0.
    This is the fix-#8 honesty guard: the old fixed-amplitude heart had zero variability and called it
    'HRV'; we refuse to invent a number when there is no rhythm."""
    mon = RhythmMonitor(capacity=64)
    for _ in range(64):
        mon.push(5.0)
    s = mon.state()
    assert s.measured is False
    assert s.f_tack is None and s.hrv is None and s.coherence is None


def test_short_history_not_measured():
    mon = RhythmMonitor(min_samples=16)
    for _ in range(5):
        mon.push(np.random.default_rng(0).random())
    assert mon.state().measured is False


def test_hrv_zero_for_regular_positive_for_irregular():
    """HRV (inter-beat CV) ≈ 0 for a perfectly periodic signal; > 0 when beat intervals vary."""
    reg = RhythmMonitor(capacity=256)
    for t in range(256):
        reg.push(np.sin(2 * np.pi * 0.05 * t))
    s_reg = reg.state()
    assert s_reg.hrv is not None and s_reg.hrv < 0.1, f"regular HRV={s_reg.hrv}"

    irr = RhythmMonitor(capacity=256)
    rng = np.random.default_rng(1)
    phase = 0.0
    for _ in range(256):
        phase += 2 * np.pi * (0.05 + 0.03 * rng.standard_normal())  # jittered tempo → varying intervals
        irr.push(np.sin(phase))
    s_irr = irr.state()
    assert s_irr.hrv is not None and s_irr.hrv > s_reg.hrv, f"irregular HRV={s_irr.hrv} not > regular {s_reg.hrv}"


def test_heart_phase_advances_and_beats():
    h = HeartOscillator(freq=0.25)  # 4 ticks per beat
    beats = sum(1 for _ in range(16) if h.beat()[1])
    assert h.beats == beats and 3 <= beats <= 5  # ~4 beats in 16 ticks


def test_adjust_frequency_is_timing_only():
    """adjust_frequency changes the rate and returns a float — never a basin. (Axis separation: the
    heart cannot move basins; only the anchor/coupling can.)"""
    h = HeartOscillator(freq=0.1)
    out = h.adjust_frequency(0.3)
    assert isinstance(out, float) and abs(h.freq - 0.3) < 1e-9
    out2 = h.adjust_frequency(99.0)  # clamped below Nyquist
    assert out2 <= 0.49
