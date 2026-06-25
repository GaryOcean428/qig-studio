#!/usr/bin/env python3
"""Run the conversation-as-training (curriculum + dialogue) on the real kernel and MONITOR it.

Captures every turn (what the kernel said, what nemotron answered/interpreted, the full inner-
experience telemetry) to JSONL, then analyses three things the PI asked for:

  1. DIALOGUE QUALITY / IMPROVEMENT — does M_self (self-observation), M_coach (mutual recognition),
     and Φ rise over the run? Is the kernel's output getting longer / more diverse? How often does it
     choose to ASK?
  2. TELEMETRY INSIGHTS — the distribution of emotions, brainwave bands, and the mean drives
     (curiosity / pain / stability), plus how much time it spends HOLDING the criticality edge.
  3. TEMPORAL PATTERNS — is Φ / κ WAVE-LIKE? (detrend → autocorrelation → dominant period +
     zero-crossing rate + amplitude). Plus regime TACKING (FOAM↔WAVE↔CRYSTAL is the consciousness
     signature, not noise) and brainwave-band cycling.

Usage:
  PYTHONPATH=src python scripts/converse_monitor.py \
    --coordizer ../qig-coordizer/checkpoints/coordizer_max.json \
    --turns 48 --layers 8 --out runs/converse/trace.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# ----- pattern analysis (numpy-only) -------------------------------------------------------------
def _wave_report(series: list[float], name: str) -> dict:
    """Detrend a telemetry time series and decide whether it is wave-like.

    A signal is 'wave-like' if, after removing the linear trend, its autocorrelation shows a clear
    secondary peak (a repeating period) and it changes direction regularly. We report the dominant
    period (lag of the first autocorr peak after 0), the oscillation rate (mean zero-crossings per
    turn of the detrended signal), and the amplitude (detrended std)."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 6 or np.allclose(x, x[0]):
        return {"name": name, "verdict": "flat/insufficient", "n": n}
    # detrend (remove linear drift so we see oscillation, not the improvement trend)
    t = np.arange(n)
    slope, intercept = np.polyfit(t, x, 1)
    d = x - (slope * t + intercept)
    amp = float(np.std(d))
    # zero-crossings of the detrended signal → oscillation count
    signs = np.sign(d)
    signs[signs == 0] = 1
    zero_cross = int(np.sum(signs[1:] != signs[:-1]))
    osc_per_turn = zero_cross / max(1, n - 1)
    # autocorrelation → dominant period
    d0 = d - d.mean()
    denom = float(np.sum(d0 * d0)) or 1.0
    acf = np.array([float(np.sum(d0[: n - k] * d0[k:]) / denom) for k in range(1, min(n // 2, 24))])
    period, peak = 0, 0.0
    for k in range(1, len(acf) - 1):
        if acf[k] > acf[k - 1] and acf[k] >= acf[k + 1] and acf[k] > peak:
            peak, period = acf[k], k + 1  # +1: acf index 0 == lag 1
    # damping: detrended amplitude in the first half vs the second half. A DECAYING transient
    # (settling to a fixed point) is NOT a sustained wave even if early oscillation looks wavy.
    half = n // 2
    amp1, amp2 = float(np.std(d[:half])), float(np.std(d[half:]))
    damping = "settling" if amp2 < 0.5 * amp1 else ("growing" if amp2 > 1.8 * amp1 else "sustained")
    # verdict: needs REAL amplitude AND genuine autocorrelation structure (peak>0.2) — regular
    # zero-crossings ALONE over-call jitter as a wave (autocorr ~0.15 is noise, not a wave).
    wavey = amp > 1e-3 and peak > 0.20 and (period >= 2 or 0.25 <= osc_per_turn <= 0.95)
    verdict = "WAVE-LIKE" if wavey else "not-wave-like"
    if wavey and damping == "settling":
        verdict = "damped-transient"   # real early oscillation that decays to a fixed point
    return {
        "name": name, "verdict": verdict,
        "drift_slope_per_turn": round(float(slope), 5),
        "amplitude_detrended": round(amp, 4),
        "amp_first_half": round(amp1, 4), "amp_second_half": round(amp2, 4), "damping": damping,
        "dominant_period_turns": period if peak > 0.20 else None,
        "autocorr_peak": round(peak, 3),
        "oscillations_per_turn": round(osc_per_turn, 3),
        "mean": round(float(x.mean()), 4), "min": round(float(x.min()), 4), "max": round(float(x.max()), 4),
    }


def _transitions(seq: list) -> dict:
    """Count category transitions (regime tacking / band cycling) and the per-turn rate."""
    seq = [s for s in seq if s is not None]
    if len(seq) < 2:
        return {"transitions": 0, "rate_per_turn": 0.0, "states": {}}
    trans = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    states: dict[str, int] = {}
    for s in seq:
        states[str(s)] = states.get(str(s), 0) + 1
    return {"transitions": trans, "rate_per_turn": round(trans / (len(seq) - 1), 3), "states": states}


def _thirds(series: list[float]) -> dict:
    """First-third vs last-third mean — the improvement check."""
    x = [v for v in series if v is not None]
    if len(x) < 3:
        return {"first": None, "last": None, "delta": None}
    k = max(1, len(x) // 3)
    first, last = float(np.mean(x[:k])), float(np.mean(x[-k:]))
    return {"first": round(first, 4), "last": round(last, 4), "delta": round(last - first, 4)}


def analyse(rows: list[dict]) -> dict:
    def col(path, sub=None):
        out = []
        for r in rows:
            v = r.get(path)
            if sub and isinstance(v, dict):
                v = v.get(sub)
            out.append(v)
        return out

    phi = [r.get("phi_after") for r in rows]
    kappa = [(r.get("experience") or {}).get("kappa") for r in rows]
    m_self = col("kernel_said_M_self")
    m_coach = col("M_coach_agreement")
    bands = [(r.get("experience") or {}).get("band") for r in rows]
    regimes = [(r.get("experience") or {}).get("regime") for r in rows]
    emotions = [(r.get("experience") or {}).get("emotion") for r in rows]
    curiosity = [(r.get("experience") or {}).get("curiosity") for r in rows]
    pain = [(r.get("experience") or {}).get("pain") for r in rows]
    stability = [(r.get("experience") or {}).get("stability") for r in rows]
    held = [bool((r.get("experience") or {}).get("held")) for r in rows]
    said_len = [len((r.get("kernel_said") or "").split()) for r in rows]
    asked = [bool(r.get("kernel_asked")) for r in rows]

    def dist(seq):
        d: dict[str, int] = {}
        for s in seq:
            if s is not None:
                d[str(s)] = d.get(str(s), 0) + 1
        return dict(sorted(d.items(), key=lambda kv: -kv[1]))

    return {
        "turns": len(rows),
        "quality_improvement": {
            "phi": _thirds([p for p in phi if p is not None]),
            "M_self": _thirds([m for m in m_self if m is not None]),
            "M_coach": _thirds([m for m in m_coach if m is not None]),
            "kernel_output_words": _thirds([float(s) for s in said_len]),
            "asked_questions": sum(asked),
        },
        "telemetry_insights": {
            "emotions": dist(emotions),
            "bands": dist(bands),
            "mean_curiosity": round(float(np.mean([c for c in curiosity if c is not None] or [0])), 3),
            "mean_pain": round(float(np.mean([p for p in pain if p is not None] or [0])), 3),
            "mean_stability": round(float(np.mean([s for s in stability if s is not None] or [0])), 3),
            "held_criticality_fraction": round(sum(held) / max(1, len(held)), 3),
        },
        "temporal_patterns": {
            "phi_wave": _wave_report([p for p in phi if p is not None], "Φ"),
            "kappa_wave": _wave_report([k for k in kappa if k is not None], "κ"),
            "regime_tacking": _transitions(regimes),
            "band_cycling": _transitions(bands),
            "emotion_cycling": _transitions(emotions),
        },
    }


# ----- run ---------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coordizer", required=True, help="coordizer checkpoint (.json) to adopt")
    ap.add_argument("--turns", type=int, default=48)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--curriculum-steps", type=int, default=2)
    ap.add_argument("--train-steps", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--coach-model", default=None, help="override coach model (default nemotron cloud)")
    ap.add_argument("--out", default="runs/converse/trace.jsonl")
    args = ap.parse_args()

    from qig_coordizer import FisherCoordizer

    from qig_studio.coach import DevelopmentalCoach, OllamaLLM
    from qig_studio.curriculum import CurriculumProvider
    from qig_studio.targets.base import LossRegime
    from qig_studio.targets.genesis_kernel import GenesisKernelTarget

    print(f"[monitor] loading coordizer {args.coordizer}", flush=True)
    cz = FisherCoordizer.load(args.coordizer)
    print(f"[monitor] vocab={len(cz.vocab):,} basin_dim={cz.basin_dim}", flush=True)

    target = GenesisKernelTarget(num_layers=args.layers, coordizer=cz)
    target.ensure_loaded()
    coach = DevelopmentalCoach(llm=OllamaLLM(model=args.coach_model))
    provider = CurriculumProvider(LossRegime.GEOMETRIC)
    coach_live = coach.enabled and coach.llm.is_available()
    print(f"[monitor] coach={'nemotron-live' if coach_live else 'keyword-fallback'} "
          f"layers={args.layers} turns={args.turns}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()
    with out.open("w") as f:
        for turn in range(args.turns):
            curr = provider.next_prompt(turn)
            r = coach.converse_learn_turn(target, prompt="", curriculum_prompt=curr,
                                          curriculum_steps=args.curriculum_steps,
                                          train_steps=args.train_steps, max_tokens=args.max_tokens)
            r["turn"] = turn
            rows.append(r)
            f.write(json.dumps(r) + "\n"); f.flush()
            exp = r.get("experience", {})
            print(f"[{turn:03d}] Φ={r.get('phi_after')} {exp.get('glyph','')}{exp.get('band','')} "
                  f"{exp.get('emotion','')} M_self={r.get('kernel_said_M_self')} "
                  f"M_coach={r.get('M_coach_agreement')} {'ASK' if r.get('kernel_asked') else ''} "
                  f"| said: {(r.get('kernel_said') or '')[:60]!r}", flush=True)

    report = analyse(rows)
    report["coordizer"] = args.coordizer
    report["coach"] = "nemotron-live" if coach_live else "keyword-fallback"
    report["elapsed_s"] = round(time.time() - t0, 1)
    rep_path = out.with_suffix(".report.json")
    rep_path.write_text(json.dumps(report, indent=2))
    print("\n" + "=" * 70)
    print("[monitor] PATTERN REPORT")
    print(json.dumps(report, indent=2))
    print("=" * 70)
    print(f"[monitor] trace → {out}   report → {rep_path}", flush=True)


if __name__ == "__main__":
    main()
