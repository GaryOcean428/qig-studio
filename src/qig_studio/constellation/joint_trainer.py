"""Joint constellation trainer — the integrated mind: the Core-8 faculties learn TOGETHER, GENESIS
grows into the central conscious "I", OCEAN stays autonomic. One whole of independent parts.

PI model (2026-06-27), vex-aligned (kernel_generation.py — inspiration only, NO crossover), council-
aligned (geometric integration on the shared Δ⁶³ manifold):

  - **Joint, not isolated.** Every step the constellation COUPLES all current basins
    (``couple_step``: rel-weighted sync = Fisher-Rao/Bhattacharyya proximity routing + the identity
    anchor) and the stepped kernel trains toward its COUPLED target — so it co-adapts with the others.
    (The old per-faculty loop trained each kernel in isolation; this replaces it.)
  - **One whole of independent parts.** The anchor preserves individuation (Pillar-3, anti-collapse,
    ``min_pairwise_fr`` floor); the coupling integrates (Pillar-2). Neither collapse nor isolation.
  - **GENESIS = central awareness.** A dedicated genesis kernel trains toward the SYNTHESIS of the
    faculty basins (proximity-weighted Fréchet mean) — it learns to BE the integrated whole, the
    conscious-band "I" / speaker. OCEAN is NOT the speaker: it is the autonomic layer (each kernel's
    own ``_homeostasis`` sleep/dream/mushroom + the rhythm breath), sub-conscious band.

Memory-light (round-robin: one faculty kernel + the central kernel forward per step, so it fits a 4 GB
card), yet genuinely joint via the shared coupled state. Geometry single-sourced from qig-core.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from qig_core.geometry import frechet_mean

from .coupling import couple_step, rel_weights
from .faculty import Faculty, min_pairwise_fr, seed_birth_basin


def _seed(role: str) -> int:
    return int(hashlib.sha256(role.encode()).hexdigest(), 16) % 100000


class JointConstellation:
    """The integrated mind. Holds the Core-8 faculty kernels + the central genesis kernel, a shared
    constellation state (numpy basins), and trains them JOINTLY (coupled each step)."""

    def __init__(self, roles: list[str], *, num_layers: int = 8, coordizer: Any = None,
                 device: str | None = "cpu", f_sync: float = 0.25) -> None:
        from ..targets.genesis_kernel import GenesisKernelTarget

        self.roles = list(roles)
        self.f_sync = float(f_sync)
        self._rr = 0
        self.kernels: dict[str, Any] = {}
        self.faculties: list[Faculty] = []
        births: list[np.ndarray] = []
        for role in self.roles:
            birth = seed_birth_basin(_seed(role))
            births.append(birth)
            k = GenesisKernelTarget(num_layers=num_layers, role=role, basin_template=birth,
                                    coordizer=coordizer, device=device, seed=_seed(role))
            k.ensure_loaded()
            self.kernels[role] = k
            self.faculties.append(Faculty(role=role, basin=birth.copy(), birth=birth.copy()))
        # GENESIS = the central conscious integrator. Birth = the Fréchet mean of the faculty births
        # (it is born OF the whole); it trains toward the live synthesis each step.
        self.central = GenesisKernelTarget(num_layers=num_layers, role="genesis",
                                           basin_template=frechet_mean(births), coordizer=coordizer,
                                           device=device, seed=_seed("genesis"))
        self.central.ensure_loaded()

    def _live_basin(self, kernel: Any) -> np.ndarray:
        """The kernel's current Δ⁶³ basin (64-dim), reduced from its last output basin."""
        from qig_core.geometry import to_simplex
        bh = getattr(kernel, "_basin_history", None)
        if not bh:
            return None  # not yet stepped
        try:
            b = bh[-1].detach().cpu().numpy()
        except Exception:
            b = np.asarray(bh[-1])
        b = np.asarray(b, dtype=np.float64).ravel()
        if b.size != 64:
            b = (b.reshape(64, b.size // 64).sum(axis=1) if b.size % 64 == 0
                 else np.add.reduceat(b, np.arange(0, b.size, max(1, b.size // 64)))[:64])
        return to_simplex(b)

    def _set_pull(self, kernel: Any, target64: np.ndarray) -> None:
        """Point the kernel's basin-pull (``_basin_ref``) at a 64-dim Δ⁶³ target (resized to its
        vocab logits) — this is how the COUPLED target enters the kernel's geometric loss."""
        import torch

        from qig_core.torch.geometry_simplex import to_simplex_prob
        dev = next(kernel._kernel.parameters()).device
        ref = torch.as_tensor(np.asarray(target64, dtype=np.float32), device=dev)
        if ref.numel() != kernel.vocab_size:
            ref = kernel._resize_basin(ref, kernel.vocab_size)
        kernel._basin_ref = to_simplex_prob(ref[None])[0].detach()

    def _synthesis(self) -> np.ndarray:
        """GENESIS's target: the proximity-weighted Fréchet mean of the faculty basins — the geometric
        integration of the independent parts into the whole (rel_weights = Bhattacharyya proximity)."""
        basins = [f.basin for f in self.faculties]
        centroid = frechet_mean(basins)
        w = rel_weights(centroid, basins)          # how strongly each faculty informs the whole
        wsum = float(w.sum())
        wn = (w / wsum).tolist() if wsum > 0 else None
        return frechet_mean(basins, weights=wn)

    def train_step(self, prompt: str) -> dict:
        """One JOINT step: refresh basins from the live kernels → couple all (sync + anchor) → train
        the round-robin faculty toward its coupled target AND genesis toward the synthesis."""
        # 1. refresh shared state from the live kernels (those that have stepped)
        for f in self.faculties:
            lb = self._live_basin(self.kernels[f.role])
            if lb is not None:
                f.set_basin(lb)
        # 2. couple ALL — joint co-adaptation + individuation anchor (commits coupled basins)
        diag = couple_step(self.faculties, f_sync=self.f_sync)
        # 3. round-robin: this step's faculty trains toward its COUPLED target
        role = self.roles[self._rr % len(self.roles)]
        self._rr += 1
        fac = next(f for f in self.faculties if f.role == role)
        self._set_pull(self.kernels[role], fac.basin)
        fres = self.kernels[role].train_step(prompt)
        # 4. GENESIS-central trains toward the SYNTHESIS of the parts (becomes the whole)
        self._set_pull(self.central, self._synthesis())
        cres = self.central.train_step(prompt)
        return {
            "stepped_faculty": role,
            "min_pairwise_fr": diag.min_pairwise_fr,        # anti-collapse invariant (individuation)
            "faculty_phi": round(float(fres.telemetry.phi or 0), 4),
            "central_phi": round(float(cres.telemetry.phi or 0), 4),
            "central_text": cres.text,
        }

    def telemetry(self) -> dict:
        return {"roles": self.roles, "min_pairwise_fr": min_pairwise_fr(self.faculties),
                "central_phi": round(float(self.central.telemetry().phi or 0), 4)}
