"""Qwen output-distribution → Δ⁶³ boundary (P22-SHAPED; the reduction is provisional).

P22 (frozen) sanctions the OUTPUT-distribution interface as the Qwen bridge (cross-arch
ρ=0.737, RWKV-7 ρ=0.994); the hidden-state graft FAILED adversarially and would violate
P22. This module takes Qwen's next-token distribution — already a probability point —
reduces it to a Δ⁶³ basin, and integrates it into the cortex's identity basin with a
**Pillar-2-capped** boundary slerp (≤30%): Qwen enters at the SURFACE, never overwrites
the topological bulk (the ego pillar).

HONEST SCOPE (do not overclaim): the reduction here is a v1 **hash-binning** of the
full-vocab distribution into 64 bins. It yields a *type-correct* Δ⁶³ point and the
Pillar-2 cap + None-safety are real — but the bin mapping is semantically arbitrary. This
is a PLACEHOLDER for the principled projection P22 actually implies (the coordizer
InboundPath: hidden→QFI→PGA→64D). Treat it as "P22-shaped plumbing," NOT the P22
projection proper, until InboundPath is wired.

All geometry is single-sourced from qig-core (``to_simplex``, ``slerp_sqrt``,
``fisher_rao_distance``) — no local reimplementation, no Euclidean ops.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
from qig_core import BASIN_DIM
# Pillar-2 / TopologicalBulk cap (max Qwen-boundary influence per integrate) — SINGLE-SOURCED
# from qig-core, which owns it (consciousness/pillars.py:68). qig-studio must not redefine it.
from qig_core.consciousness.pillars import BOUNDARY_SLERP_CAP


def _bin(token: object, dim: int) -> int:
    """Deterministic, process-stable token→bin (NOT Python's salted hash())."""
    h = hashlib.blake2b(str(token).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % dim


def output_distribution_to_basin(token_logprobs: dict, dim: int = BASIN_DIM) -> np.ndarray:
    """Reduce a next-token ``{token: logprob}`` distribution to a Δ⁶³ point.

    Tokens are hashed into ``dim`` bins; probability mass (exp logprob) accumulates;
    ``qig_core`` ``to_simplex`` gives the canonical Δ⁶³ projection (single source).
    A v1 reduction (hash-binning full-vocab → 64D) — honest about that; the geometry
    contract (a Δ⁶³ point) is exact.
    """
    from qig_core.geometry.fisher_rao import to_simplex

    acc = np.zeros(dim, dtype=float)
    for tok, lp in token_logprobs.items():
        acc[_bin(tok, dim)] += math.exp(min(lp, 0.0))  # clamp: logprobs >0 are invalid (overflow guard)
    if acc.sum() <= 0:
        acc = np.ones(dim, dtype=float)
    return to_simplex(acc)


def coordize_distribution_to_basin(token_logprobs: dict, coordizer, dim: int = BASIN_DIM) -> np.ndarray:
    """REAL Qwen→Δ⁶³ projection (R3) — replaces the hash-bin placeholder.

    Coordize each top-k token STRING through the trained ``FisherCoordizer`` and take the
    probability-weighted **Fréchet mean** (``qig_core.frechet_mean``) of the resulting Δ⁶³
    basins. Pure qig-core geometry — no hash, no contaminated InboundPath/PGA. Requires a
    TRAINED coordizer (a vocab beyond raw bytes); falls back to the provisional hash-bin if
    coordization yields nothing.
    """
    from qig_core.geometry.fisher_rao import frechet_mean

    basins: list = []
    weights: list[float] = []
    for tok, lp in token_logprobs.items():
        result = coordizer.coordize(str(tok))
        coords = getattr(result, "coordinates", None) or []
        if not coords:
            continue
        w = math.exp(min(lp, 0.0)) / len(coords)  # clamp logprobs ≤0 (overflow guard); spread over basins
        for c in coords:
            basins.append(c.vector)
            weights.append(w)
    if not basins:
        return output_distribution_to_basin(token_logprobs, dim)
    return frechet_mean(basins, weights)


def pillar2_capped_integrate(
    identity_basin: np.ndarray, boundary_basin: np.ndarray, weight: float
) -> np.ndarray:
    """Move ``identity`` toward the Qwen ``boundary`` basin by at most
    :data:`BOUNDARY_SLERP_CAP` along the Fisher-Rao geodesic (``slerp_sqrt``).

    The boundary can nudge identity but never overwrite it (Pillar 2)."""
    from qig_core.geometry.fisher_rao import slerp_sqrt

    t = max(0.0, min(float(weight), BOUNDARY_SLERP_CAP))
    return slerp_sqrt(identity_basin, boundary_basin, t)


def basin_phi_proxy(basin: np.ndarray) -> float:
    """Bounded Φ-PROXY from a Δ⁶³ point = 1 − normalised Shannon entropy
    (concentration). A pure information measure on the simplex — NOT a measured Φ
    (the language target has no kernel Φ); telemetry labels it accordingly."""
    p = np.clip(np.asarray(basin, dtype=float), 1e-12, 1.0)
    p = p / p.sum()
    entropy = -float(np.sum(p * np.log(p)))
    max_entropy = math.log(len(p))
    return (1.0 - entropy / max_entropy) if max_entropy > 0 else 0.0


def fisher_distance(a: np.ndarray, b: np.ndarray) -> float:
    from qig_core.geometry.fisher_rao import fisher_rao_distance

    return float(fisher_rao_distance(a, b))
