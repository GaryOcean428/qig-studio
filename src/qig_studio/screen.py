"""A/B avenue-screen EVAL logic — held-out d_FR ranking + the under-power detector.

This is the EVAL half of the avenue screen ONLY (lifted from the standalone ``scripts/screen_neocortex.py``
whose TRAINING is dropped — training now runs through the server's wired loop, ``server._train_core``). The
verdict metric is held-out **mean d_FR** (the torch primitive ``fisher_rao_distance_simplex`` via
``target.eval_text_fr``, range [0, π], lower = better); CE-bpb (``target.eval_text_bpb``) is reported as the
Tier-2 external-comparison-only number, NEVER the ranking axis (per
``docs/plans/2026-06-30-exp-cortex-ab-prereg.md``).

DRY: there is NO training and NO live-telemetry wiring here — the wired training path (and its live record
writing) lives in ``server._train_core`` and is what ``/screen`` calls per config. This module is
import-pure (no FastAPI, no torch-at-import) so it is unit-testable against any object exposing
``eval_text_fr``/``eval_text_bpb``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Protocol

PI = math.pi

# Within this much of the uniform floor counts as "still pinned" (did NOT move off the uniform/π floor).
NEAR_FLOOR_EPS = 0.02


class _EvalTarget(Protocol):
    """The minimal surface ``eval_heldout_dFR`` needs — any TrainingTarget arm satisfies it."""

    def eval_text_fr(self, text: str) -> tuple[float, int]: ...
    def eval_text_bpb(self, text: str) -> tuple[float, int]: ...


def load_heldout_passages(path: str | Path = "data/eval/heldout_bpb.json") -> list[str]:
    """Extract the held-out passages from the eval set (a dict whose non-``_``-prefixed list values are the
    register buckets of passages). Keeps the screen's eval set identical to the standalone script's."""
    hb = json.loads(Path(path).read_text())
    return [
        v
        for k, vals in hb.items()
        if not k.startswith("_") and isinstance(vals, list)
        for v in vals
        if isinstance(v, str)
    ]


def uniform_dFR_floor(vocab_size: int) -> float:
    """The per-position d_FR a head that predicts the UNIFORM distribution over the vocab pays:
    2·arccos(√(1/V)), constant, ≈ π for large V. The converged head must BEAT this to have learned
    structure (prereg maturity floor). This is the conservative coordizer-only reference — the 'did it move
    off π' anchor for the screen — NOT a true frequency-unigram floor (which would need train-corpus token
    frequencies)."""
    return round(2.0 * math.acos(math.sqrt(1.0 / max(2, int(vocab_size)))), 5)


def eval_heldout_dFR(target: _EvalTarget, passages: list[str]) -> dict[str, Any]:
    """Held-out aggregate over the passages (no grad, no training):

    - ``heldout_dFR`` = sum(total_dFR) / sum(n_positions) — the VERDICT (torch d_FR primitive, [0, π]).
    - ``ce_bpb``      = sum(bits) / sum(bytes) — Tier-2 external-comparison-only (NOT ranked).
    - ``n_pos``       = the held-out positions actually scored.

    Per-passage finite-guarded: one bad/empty passage is counted as ``nonfinite`` and skipped, never voiding
    the whole config. Returns None for a metric whose denominator is zero (all passages skipped)."""
    dfr_num = dfr_den = 0.0
    bpb_num = bpb_den = 0.0
    nonfinite = 0
    for text in passages:
        try:
            tot_dfr, n_pos = target.eval_text_fr(text)
            tot_bits, n_bytes = target.eval_text_bpb(text)
        except Exception:  # noqa: BLE001 — one bad passage must not void the config
            nonfinite += 1
            continue
        if math.isfinite(tot_dfr) and n_pos > 0:
            dfr_num += float(tot_dfr)
            dfr_den += float(n_pos)
        else:
            nonfinite += 1
        if math.isfinite(tot_bits) and n_bytes > 0:
            bpb_num += float(tot_bits)
            bpb_den += float(n_bytes)
    mean_dfr = (dfr_num / dfr_den) if dfr_den > 0 else float("nan")
    ce_bpb = (bpb_num / bpb_den) if bpb_den > 0 else float("nan")
    return {
        "heldout_dFR": (None if not math.isfinite(mean_dfr) else round(mean_dfr, 5)),
        "ce_bpb": (None if not math.isfinite(ce_bpb) else round(ce_bpb, 5)),
        "n_pos": int(dfr_den),
        "nonfinite_passages": nonfinite,
    }


def rank_configs(configs: list[dict[str, Any]], uniform_floor: float) -> dict[str, Any]:
    """Rank the screened configs on held-out mean d_FR (lower = better) and apply the under-power detector.

    A config is RANKABLE if it produced a finite ``heldout_dFR`` and did not NaN/OOM. If NO rankable config
    moved more than ``NEAR_FLOOR_EPS`` below the uniform floor, the screen is UNDER-POWERED (the d_FR is
    still pinned near the floor for everything → we are not ranking noise; recommend a larger budget) and no
    winner is named. Returns the ranking list, the underpowered flag, and the winner (None if underpowered)."""
    rankable = [
        c
        for c in configs
        if c.get("heldout_dFR") is not None and not c.get("nan", False) and not c.get("oom", False)
    ]
    rankable.sort(key=lambda c: c["heldout_dFR"])
    moved_off = [c for c in rankable if (uniform_floor - c["heldout_dFR"]) > NEAR_FLOOR_EPS]
    underpowered = len(moved_off) == 0
    return {
        "ranking": [c["name"] for c in rankable],
        "rankable": rankable,
        "underpowered": underpowered,
        "winner": (rankable[0]["name"] if rankable and not underpowered else None),
        "uniform_dFR_floor": uniform_floor,
    }
