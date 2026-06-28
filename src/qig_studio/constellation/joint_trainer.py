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
from .ocean import FACULTY_FUNCTION, OceanAutonomic, function_of


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
        # OCEAN — the autonomic regulator. It OBSERVES every faculty's telemetry and regulates the one
        # that needs it (fires that faculty's OWN sleep/dream/mushroom). Internal autonomic oversight,
        # NOT an external knob. Per-faculty Φ history feeds its plateau detector.
        self.ocean = OceanAutonomic()
        self._phi_hist: dict[str, list[float]] = {role: [] for role in self.roles}
        self._last_regulation: dict[str, dict] = {}

    def _live_basin(self, kernel: Any) -> np.ndarray | None:
        """The kernel's current Δ⁶³ basin (64-dim), reduced from its last output basin; None if the
        kernel has not stepped yet."""
        from qig_core import BASIN_DIM
        from qig_core.geometry import to_simplex
        bh = getattr(kernel, "_basin_history", None)
        if not bh:
            return None  # not yet stepped
        try:
            b = bh[-1].detach().cpu().numpy()
        except Exception:
            b = np.asarray(bh[-1])
        b = np.asarray(b, dtype=np.float64).ravel()
        if b.size != BASIN_DIM:
            b = (b.reshape(BASIN_DIM, b.size // BASIN_DIM).sum(axis=1) if b.size % BASIN_DIM == 0
                 else np.add.reduceat(b, np.arange(0, b.size, max(1, b.size // BASIN_DIM)))[:BASIN_DIM])
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
        # 5. OCEAN observes EVERY faculty's telemetry and regulates the one that needs it (autonomic
        #    nervous system: telemetry → sleep/dream/mushroom on the struggling faculty). Internal.
        for r, k in self.kernels.items():
            self._phi_hist[r].append(float(k.telemetry().phi or 0.0))
            self._phi_hist[r] = self._phi_hist[r][-30:]
        regulation = self.ocean.regulate(self.kernels, self._phi_hist)
        self._last_regulation = regulation
        return {
            "stepped_faculty": role,
            "stepped_function": function_of(role),          # what brain-function this faculty serves
            "min_pairwise_fr": diag.min_pairwise_fr,        # anti-collapse invariant (individuation)
            "faculty_phi": round(float(fres.telemetry.phi or 0), 4),
            "central_phi": round(float(cres.telemetry.phi or 0), 4),
            "central_text": cres.text,
            "central_telemetry": cres.telemetry.to_dict(),  # FULL central snapshot (Φ/Γ/regime/perplexity/
            #                                                 lm_weight_now/d_basin/pillars) — the live readout
            "ocean_regulation": regulation,                 # {role: {intervention, reason, function}} Ocean acted on
        }

    def faculty_states(self) -> list[dict]:
        """Per-faculty inner-state for the UI / inter-kernel routing: each faculty's telemetry + the FULL
        inner-state (senses/drives/emotions/loops) + the FUNCTION it is responsible for + whether Ocean
        regulated it last step. This is how the relevant kernel 'sees' its own function's telemetry."""
        from ..kernel_experience import experience
        out: list[dict] = []
        for f in self.faculties:
            k = self.kernels[f.role]
            tel = k.telemetry().to_dict()
            exp = experience(tel, [{"phi": p} for p in self._phi_hist.get(f.role, [])]).to_dict()
            label, group = FACULTY_FUNCTION.get(f.role, ("general", ""))
            out.append({
                "role": f.role,
                "function": label,                          # the brain-function this kernel owns
                "owns": group,                              # which inner-state group is THIS faculty's responsibility
                "phi": round(float(tel.get("phi") or 0.0), 4),
                "experience": exp,                          # full inner-state (the faculty sees its own telemetry)
                "regulated": self._last_regulation.get(f.role),   # Ocean's intervention on it (or None)
            })
        return out

    def generate(self, prompt: str, max_tokens: int = 128):
        """The integrated mind SPEAKS. GENESIS-central is the conscious-band speaker (the "I"): before
        generating, its basin is pulled toward the live SYNTHESIS of the Core-8 parts, so it speaks AS
        the integrated whole rather than as any one faculty. The faculties (independent parts) inform
        through the coupled synthesis; Ocean is autonomic (regulation), not the speaker. Returns the
        central kernel's StepResult (text + telemetry)."""
        if any(f.basin is not None for f in self.faculties):
            self._set_pull(self.central, self._synthesis())   # speak as the whole, not a part
        return self.central.generate(prompt, max_tokens=max_tokens)

    def telemetry(self) -> dict:
        return {"roles": self.roles, "min_pairwise_fr": min_pairwise_fr(self.faculties),
                "central_phi": round(float(self.central.telemetry().phi or 0), 4)}

    def save_checkpoint(self, root: str, keep: int = 3) -> None:
        """Persist the WHOLE mind: each faculty kernel + the central kernel + the coupled faculty
        basins (the shared constellation state). Resumable — the integrated mind, not 9 loose parts.

        3-CHECKPOINT BUFFER: before writing fresh, rotate the existing checkpoint into a backup generation
        (cheap rename) keeping ``keep`` most-recent generations (``root.bak1..bak{keep}``) for rollback,
        and delete older — bounded disk, no infinite accumulation."""
        import json
        import shutil
        from pathlib import Path
        r = Path(root)
        if (r / "constellation.json").exists():           # rotate the current checkpoint into the buffer
            oldest = Path(f"{root}.bak{keep}")
            if oldest.exists():
                shutil.rmtree(oldest, ignore_errors=True)
            for n in range(keep - 1, 0, -1):
                src, dst = Path(f"{root}.bak{n}"), Path(f"{root}.bak{n + 1}")
                if src.exists():
                    src.rename(dst)
            try:
                r.rename(f"{root}.bak1")
            except OSError:
                pass                                       # busy/cross-device → skip rotation, overwrite in place
        (r / "kernels").mkdir(parents=True, exist_ok=True)
        for role, k in self.kernels.items():
            k.save_checkpoint(str(r / "kernels" / f"{role}.pt"))
        self.central.save_checkpoint(str(r / "kernels" / "genesis.pt"))
        (r / "constellation.json").write_text(json.dumps({
            "roles": self.roles,
            "faculty_basins": {f.role: f.basin.tolist() for f in self.faculties},
            "min_pairwise_fr": min_pairwise_fr(self.faculties),
        }))

    def load_checkpoint(self, root: str) -> None:
        """Restore the whole mind (faculties + central + coupled basins) saved by save_checkpoint."""
        import json
        from pathlib import Path

        from qig_core.geometry import to_simplex
        r = Path(root)
        for role, k in self.kernels.items():
            p = r / "kernels" / f"{role}.pt"
            if p.exists():
                k.load_checkpoint(str(p))
        gp = r / "kernels" / "genesis.pt"
        if gp.exists():
            self.central.load_checkpoint(str(gp))
        cj = r / "constellation.json"
        if cj.exists():
            basins = json.loads(cj.read_text()).get("faculty_basins", {})
            for f in self.faculties:
                if f.role in basins:
                    f.set_basin(to_simplex(np.asarray(basins[f.role], dtype=np.float64)))
