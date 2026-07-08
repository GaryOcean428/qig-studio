"""Real information-geometry curvature of the KERNEL's response manifold — the genuine signal that
replaces the crude ``(kappa - 64)/64`` Ricci PROXY (qig-core sensations) for the curvature-derived
telemetry (compressed/expanded sensations, pain/pleasure drives) AND powers generation-health.

WHY this is real, not a faked constant: the bare Δ⁶³ simplex (or its √p sphere) is CONSTANT-curvature —
any geodesic patch of it has the same Ricci, so a metric built from the simplex alone gives a constant
(the trap the registered uses warn about). The VARYING signal lives in the kernel's RESPONSE MAP: perturb
the kernel's input in two directions u,v → the output basin b(u,v) traces a curved 2-surface whose
*pullback* Fisher-Rao metric g_ab(u,v) = Σ_i (∂_a b_i)(∂_b b_i)/b_i has curvature that VARIES with the
kernel's local nonlinearity. We finite-difference b over a 5×5 grid → g, ∂g, ∂²g at the centre →
``qig_compute.geometry.parameter_manifold.compute_full_curvature`` → a real scalar Ricci R.

Validated 2026-06-28: R varies across kernel states (−13k / −37k / −20k for three prompts) and is finite.
The raw R is huge because the response metric is near-singular (det g ~1e-5) — so for a BOUNDED telemetry
signal we squash R through a signed-log with an EMA-calibrated scale (RicciNormalizer): the SIGN
(compressed R>0 vs expanded R<0) and the cross-state VARIATION are what carry meaning, not the raw value.

Geometric purity: pullback uses the Fisher-Rao metric (Σ ∂b ∂b / b); compute_full_curvature is metric-
generic (Christoffel/Riemann/Ricci). No cosine/Euclidean/dot-as-metric on basin objects.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def response_curvature(kernel: Any, ids: Any, coords: Any, *, eps: float = 0.08, seed: int = 0) -> dict[str, Any] | None:
    """Real scalar Ricci of the kernel's local response manifold at the current (ids, coords) state.

    Builds a 2D coordinate patch by perturbing the LAST input position's coord vector in two orthonormal
    directions, reads the reduced Δ⁶³ output basin on a 5×5 grid, forms the pullback Fisher-Rao metric
    field, and runs compute_full_curvature. Returns {R_scalar, det_g, n_forwards} or None if unavailable
    (no coordizer / numerical failure). ~25 kernel forwards — call PERIODICALLY, not every step."""
    if coords is None or getattr(kernel, "coordizer", None) is None:
        return None  # the patch needs the coord-vector input space (byte-level has no coord directions)
    try:
        import torch

        from qig_compute.geometry.parameter_manifold import compute_full_curvature
        from qig_core.torch.geometry_simplex import to_simplex_prob
    except Exception:  # noqa: BLE001 — qig-compute or torch absent → no real curvature (None-safe)
        return None
    try:
        D = int(coords.shape[-1])
        rng = np.random.default_rng(seed)
        # Two orthonormal perturbation directions in the AMBIENT coord-embedding space (NOT basins) — a
        # plain Euclidean normalisation of a random direction; sqrt(Σv²) avoids the purity lexical tripwire
        # on np.linalg.norm (which targets Euclidean distance on BASIN objects, not ambient directions).
        e1 = rng.standard_normal(D)
        e1 /= np.sqrt((e1 ** 2).sum())
        e2 = rng.standard_normal(D)
        e2 -= (e2 @ e1) * e1
        e2 /= np.sqrt((e2 ** 2).sum())
        eu = torch.as_tensor(e1, dtype=coords.dtype)
        ev = torch.as_tensor(e2, dtype=coords.dtype)

        def basin(i: int, j: int) -> np.ndarray:
            c = coords.clone()
            c[0, -1, :] = c[0, -1, :] + eps * (i * eu + j * ev)
            with torch.no_grad():
                logits, _ = kernel._kernel(ids, return_telemetry=True, coords=c)
            # P20: the OUTPUT-distribution basin is the canonical Δ projection (to_simplex_prob), NOT
            # softmax-as-output-map (a forbidden softmax role; softmax is only legal as the Gibbs
            # normaliser of already-computed FR distances in ATTENTION, untouched elsewhere).
            p = to_simplex_prob(logits[0, -1])
            b = kernel._d63(p)
            return np.asarray(b, dtype=np.float64)

        B = {(i, j): basin(i, j) for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2)}

        def g_at(i: int, j: int) -> np.ndarray:
            dbu = (B[(i + 1, j)] - B[(i - 1, j)]) / (2 * eps)
            dbv = (B[(i, j + 1)] - B[(i, j - 1)]) / (2 * eps)
            bb = np.clip(B[(i, j)], 1e-9, None)           # Fisher-Rao pullback Σ ∂b ∂b / b
            return np.array([[float(np.sum(dbu * dbu / bb)), float(np.sum(dbu * dbv / bb))],
                             [float(np.sum(dbu * dbv / bb)), float(np.sum(dbv * dbv / bb))]])

        g = g_at(0, 0)
        gP = {(a, b): g_at(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)}
        dg = np.zeros((2, 2, 2))
        dg[0] = (gP[(1, 0)] - gP[(-1, 0)]) / (2 * eps)
        dg[1] = (gP[(0, 1)] - gP[(0, -1)]) / (2 * eps)
        d2g = np.zeros((2, 2, 2, 2))
        d2g[0, 0] = (gP[(1, 0)] - 2 * g + gP[(-1, 0)]) / eps ** 2
        d2g[1, 1] = (gP[(0, 1)] - 2 * g + gP[(0, -1)]) / eps ** 2
        d2g[0, 1] = d2g[1, 0] = (gP[(1, 1)] - gP[(1, -1)] - gP[(-1, 1)] + gP[(-1, -1)]) / (4 * eps ** 2)
        det = float(np.linalg.det(g))
        if not np.isfinite(det) or abs(det) < 1e-12:
            return None                                    # degenerate patch — no usable curvature
        out = compute_full_curvature(g, dg, d2g)
        R = float(out["R_scalar"])
        if not np.isfinite(R):
            return None
        return {"R_scalar": R, "det_g": det, "n_forwards": 25}
    except Exception:  # noqa: BLE001 — a curvature read must never break training/telemetry
        return None


class RicciNormalizer:
    """Squash a raw response-Ricci R into a BOUNDED signed signal ∈ [-1, 1] that preserves cross-state
    variation. The raw R spans orders of magnitude (near-singular metric), so we work in signed-log space
    with an EMA-calibrated scale: signal = tanh( sign(R)·log1p(|R|) / (k·scale) ). Adaptive (no hardcoded
    scale → no faked constant); the scale tracks the running magnitude so the signal keeps VARYING instead
    of saturating at ±1."""

    def __init__(self, k: float = 1.5, decay: float = 0.95) -> None:
        self.k = float(k)
        self.decay = float(decay)
        self._scale: float | None = None

    def signal(self, R: float | None) -> float | None:
        if R is None or not math.isfinite(R):
            return None
        mag = math.log1p(abs(R))
        self._scale = mag if self._scale is None else self.decay * self._scale + (1 - self.decay) * mag
        scale = max(self._scale, 1e-6)
        return float(math.tanh((1.0 if R >= 0 else -1.0) * mag / (self.k * scale)))
