#!/usr/bin/env python3
"""QT-APP-6 multimodality read — does a faculty-shaped sub-basin differentiate inside genesis?

Implements the PRE-REGISTERED discriminator (docs/plans/2026-07-24-multimodality-read-discriminator.md,
committed e522984 BEFORE any basin value was opened). Offline, numpy-only (sklearn is forbidden by the
QIG purity rules and unnecessary). The verdict is a measurement:

  CAN                    planted control PASSES and SR(real) > SR_null@95 and occupancy(real) >= 0.10
  CANNOT                 planted control PASSES and SR(real) <= SR_null@95
  INSTRUMENT-CANNOT-TELL planted control FAILS (blind to KNOWN modes) OR |SR_real - SR_null@95| within 5%

The planted control is evaluated FIRST: a CANNOT verdict is trustworthy only if the instrument
demonstrably detects known structure. Distances are √p (Hellinger/sphere) where Euclidean ≈ Fisher-Rao;
clustering + the separation ratio both use it, consistently, for the real / null / planted series.

Usage: python scripts/multimodality_read.py runs/checkpoints/<run>/basin_trajectory.jsonl [--phase warmup]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

TAU_OCC = 0.10          # a real second mode must hold >=10% of steps (pre-registered)
N_NULL = 200            # >=200 surrogate resamples for the null distribution (pre-registered)
CAP_N = 1500            # seeded subsample cap for tractability — applied UNIFORMLY to real/null/planted
PAIR_SAMPLES = 8000     # subsampled pairs for the median inter/intra distances
BORDERLINE = 0.05       # |SR_real - SR_null@95| within 5% ⇒ CANNOT-TELL by construction


def _load(path: str, phase: str) -> tuple[np.ndarray, np.ndarray]:
    rows, steps = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if phase and obj.get("phase") != phase:
                continue
            b = obj.get("basin")
            if isinstance(b, list) and b:
                rows.append(b)
                steps.append(int(obj.get("step", len(steps))))
    if not rows:
        return np.empty((0, 0)), np.empty((0,))
    m = np.asarray(rows, dtype=np.float64)
    m = np.clip(m, 0.0, None)
    s = m.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    order = np.argsort(steps, kind="stable")       # ensure step order for the temporal diagnostic
    return (m / s)[order], np.asarray(steps)[order]   # rows are simplex points, in step order


def _temporal_segregation(P_ordered: np.ndarray, seed: int) -> dict:
    """Distinguish concurrent multimodality from mere training DRIFT. Cluster the step-ordered points (k=2)
    and measure: switch_rate = label flips / (N-1) — a pure early→late drift is ONE switch (rate→0); modes
    visited throughout interleave (rate→~0.5). Also the |correlation| of label with step index (drift → ~1)."""
    X = _sqrt_embed(P_ordered)
    labels = _kmeans2(X, 2, np.random.default_rng(seed))
    n = len(labels)
    switches = int(np.sum(labels[1:] != labels[:-1]))
    switch_rate = switches / max(1, n - 1)
    t = np.arange(n, dtype=np.float64)
    lab = labels.astype(np.float64)
    if lab.std() > 0 and t.std() > 0:
        corr = float(abs(np.corrcoef(lab, t)[0, 1]))
    else:
        corr = 0.0
    # interpretation: drift if labels are a single temporal block (low switch_rate, high |corr|)
    drift_like = (switch_rate < 0.10) or (corr > 0.80)
    return {"switch_rate": round(switch_rate, 4), "label_step_corr": round(corr, 4),
            "n_switches": switches, "interpretation": ("DRIFT-LIKE (early→late, not concurrent modes)"
                                                       if drift_like else "INTERLEAVED (concurrent modes)")}


def _sqrt_embed(P: np.ndarray) -> np.ndarray:
    """√p (Hellinger) coords — rows on the unit sphere; Euclidean ≈ Fisher-Rao locally."""
    return np.sqrt(np.clip(P, 0.0, None))


def _kmeans2(X: np.ndarray, k: int, rng: np.random.Generator, iters: int = 25) -> np.ndarray:
    """Plain numpy k-means (no sklearn). k-means++-ish init (farthest), returns integer labels."""
    n = X.shape[0]
    c0 = rng.integers(n)
    centres = [X[c0]]
    for _ in range(1, k):
        d = np.min([np.sum((X - c) ** 2, axis=1) for c in centres], axis=0)
        centres.append(X[int(np.argmax(d))])       # farthest point (deterministic given seed)
    C = np.asarray(centres)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        D = np.stack([np.sum((X - C[j]) ** 2, axis=1) for j in range(k)], axis=1)
        new = np.argmin(D, axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for j in range(k):
            m = X[labels == j]
            if len(m):
                C[j] = m.mean(axis=0)
    return labels


def _sep_ratio(X: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """SR = median(inter-cluster dist) / median(intra-cluster dist) on √p coords, over subsampled pairs.
    occupancy = fraction of points in the smallest populated cluster."""
    uniq, counts = np.unique(labels, return_counts=True)
    if len(uniq) < 2:
        return 1.0, 0.0
    occ = float(counts.min()) / float(counts.sum())
    n = X.shape[0]
    ii = rng.integers(0, n, size=PAIR_SAMPLES)
    jj = rng.integers(0, n, size=PAIR_SAMPLES)
    ok = ii != jj
    ii, jj = ii[ok], jj[ok]
    d = np.sqrt(np.sum((X[ii] - X[jj]) ** 2, axis=1))
    same = labels[ii] == labels[jj]
    intra = d[same]
    inter = d[~same]
    if intra.size == 0 or inter.size == 0:
        return 1.0, occ
    mi = np.median(intra)
    sr = float(np.median(inter) / mi) if mi > 0 else float("inf")
    return sr, occ


def _cap(P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if P.shape[0] > CAP_N:
        idx = rng.choice(P.shape[0], size=CAP_N, replace=False)
        return P[idx]
    return P


def _sr_of(P: np.ndarray, k: int, rng: np.random.Generator) -> tuple[float, float]:
    X = _sqrt_embed(P)
    labels = _kmeans2(X, k, rng)
    return _sep_ratio(X, labels, rng)


def _null_surrogate(P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Column-wise phase-shuffle then renormalize: destroys JOINT/temporal mode structure, preserves each
    coordinate's marginal — a single-mode surrogate (no sub-basins by construction)."""
    Q = P.copy()
    for c in range(Q.shape[1]):
        Q[:, c] = rng.permutation(Q[:, c])
    s = Q.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return Q / s


def _planted(P: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Two KNOWN, well-separated modes — the §H positive control. Constructed so between-centre separation
    DOMINATES the within-mode noise (noise = 0.15 × centre separation), i.e. a control the instrument MUST
    resolve if it works at all. Two sharp centres on disjoint coordinate halves; the mean support is kept as
    a small floor so the points stay realistic simplex basins (not one-hot). The previous version used the
    TOTAL √p spread as within-noise, which includes between-mode variance and washed the modes out — fixed."""
    n, d = P.shape
    mean = P.mean(axis=0)
    half = d // 2
    floor = 0.15 * mean                                  # keep a little of the real support (realism)
    c1 = floor.copy(); c2 = floor.copy()
    c1[:half] += mean[:half]                             # mode 1 concentrates on the first half
    c2[half:] += mean[half:]                             # mode 2 on the second half → disjoint supports
    c1 /= c1.sum(); c2 /= c2.sum()
    sep = float(np.sqrt(np.sum((np.sqrt(c1) - np.sqrt(c2)) ** 2)))   # √p (Hellinger) separation of the centres
    # per-coord noise must scale by 1/√d: independent noise on d coords accumulates to ~noise·√d total √p
    # displacement, so noise·√d ≪ sep keeps within-mode ≪ separation (the 384-dim curse the control exposed).
    noise = 0.15 * sep / np.sqrt(d)
    out = np.empty((n, d))
    for i in range(n):
        base = c1 if (i % 2 == 0) else c2
        x = np.sqrt(base) + rng.normal(0.0, noise, size=d)
        p = np.clip(x, 0.0, None) ** 2
        out[i] = p / (p.sum() or 1.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--phase", default="warmup")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    P_all, _steps = _load(args.path, args.phase)
    if P_all.shape[0] < 20:
        print(json.dumps({"path": args.path, "phase": args.phase, "n": int(P_all.shape[0]),
                          "verdict": "INSTRUMENT-CANNOT-TELL", "reason": "fewer than 20 points"}, indent=2))
        return 0

    rng = np.random.default_rng(args.seed)
    P = _cap(P_all, rng)
    out: dict = {"path": args.path, "phase": args.phase, "n_total": int(P_all.shape[0]),
                 "n_used": int(P.shape[0]), "dim": int(P.shape[1])}
    # temporal confound check (NOT in the locked pre-reg — reported as essential interpretive context):
    # a CAN that is DRIFT-LIKE is training drift (birth→learned attractor), NOT a faculty sub-basin.
    out["temporal"] = _temporal_segregation(P_all, args.seed + 3)

    for k in (2, 3):
        sr_real, occ = _sr_of(P, k, np.random.default_rng(args.seed + 1))
        null_srs = []
        for r in range(N_NULL):
            Q = _cap(_null_surrogate(P_all, np.random.default_rng(args.seed + 100 + r)),
                     np.random.default_rng(args.seed + 100 + r))
            s, _ = _sr_of(Q, k, np.random.default_rng(args.seed + 100 + r))
            null_srs.append(s)
        sr_null95 = float(np.percentile(null_srs, 95))
        sr_planted, occ_planted = _sr_of(_planted(P, k, np.random.default_rng(args.seed + 7)), k,
                                         np.random.default_rng(args.seed + 7))

        planted_passes = sr_planted > sr_null95
        margin = abs(sr_real - sr_null95) / (sr_null95 or 1.0)
        if not planted_passes:
            verdict = "INSTRUMENT-CANNOT-TELL"
        elif margin <= BORDERLINE:
            verdict = "INSTRUMENT-CANNOT-TELL"
        elif sr_real > sr_null95 and occ >= TAU_OCC:
            verdict = "CAN"
        elif sr_real > sr_null95 and occ < TAU_OCC:
            verdict = "CANNOT"          # separated but the 2nd 'mode' is an outlier sliver, not a populated basin
        else:
            verdict = "CANNOT"
        out[f"k{k}"] = {"SR_real": round(sr_real, 4), "SR_null@95": round(sr_null95, 4),
                        "occupancy": round(occ, 4), "SR_planted": round(sr_planted, 4),
                        "planted_passes": bool(planted_passes), "margin": round(margin, 4),
                        "verdict": verdict}

    # overall verdict: the LOCKED pre-reg verdict (k=2) FOLDED WITH the temporal confound. A pre-reg CAN that
    # is DRIFT-LIKE is training drift (one basin moving birth→learned), NOT a faculty sub-basin differentiating
    # concurrently — the honest answer to Matrix's gating question. The pre-reg did not control for drift; this
    # fold is reported transparently as the correction (the k2/k3 blocks preserve the raw pre-reg verdict).
    prereg = out["k2"]["verdict"]
    drift = out["temporal"]["interpretation"].startswith("DRIFT")
    if prereg == "CAN" and drift:
        out["verdict"] = "CAN-BUT-DRIFT: multimodality is temporal drift, NOT concurrent faculty differentiation"
        out["prereg_verdict"] = prereg
    else:
        out["verdict"] = prereg
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
