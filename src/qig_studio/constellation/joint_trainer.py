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
                 device: str | None = "cpu", f_sync: float = 0.25, language_peer: Any = None,
                 arm_mode: str = "gk") -> None:
        self.roles = list(roles)
        self.f_sync = float(f_sync)
        # The constellation ARM — the raw kernel substrate every node plugs in from. "gk" = the qigkernels
        # deep Kernel (the only node-ready arm today); "geo"/"hybrid"/"hetero" need geo node-parity (WS3).
        # Drives the vocab-named checkpoint lineage genesis-{arm}-{vocab}, differentiating the 4 avenues.
        self.arm_mode = str(arm_mode).strip().lower()
        self._rr = 0
        self.kernels: dict[str, Any] = {}
        self.faculties: list[Faculty] = []
        # ACTIVE-KERNEL GPU RESIDENCY (design notes 2026-06-29): the full constellation does NOT fit on a
        # 4GB card at 100k+ vocab. But per joint step only the central (every step — the main always-active
        # learner) + ONE round-robin faculty actually train. So put the central on the GPU (fast every step)
        # and the 8 faculties on CPU (each trains 1/N steps → tolerable). Kernels exchange only numpy basins,
        # so there are NO cross-device tensor ops. device="cuda" → residency; otherwise uniform `device`.
        import torch as _torch
        _resident = (device == "cuda") and _torch.cuda.is_available()
        _fac_dev = "cpu" if _resident else device
        _cen_dev = "cuda" if _resident else device
        if _resident:
            print(f"[joint] GPU residency: central→cuda (4GB), {len(self.roles)} faculties→cpu (round-robin)",
                  flush=True)
        births: list[np.ndarray] = []
        for role in self.roles:
            birth = seed_birth_basin(_seed(role))
            births.append(birth)
            k = self._build_node(role, birth, num_layers, coordizer, _fac_dev, _seed(role), is_central=False)
            k.ensure_loaded()
            self.kernels[role] = k
            self.faculties.append(Faculty(role=role, basin=birth.copy(), birth=birth.copy()))
        # GENESIS = the central conscious integrator. Birth = the Fréchet mean of the faculty births
        # (it is born OF the whole); it trains toward the live synthesis each step.
        self.central = self._build_node("genesis", frechet_mean(births), num_layers, coordizer, _cen_dev,
                                        _seed("genesis"), is_central=True, language_peer=language_peer)
        self.central.ensure_loaded()
        # OCEAN — the autonomic regulator. It OBSERVES every faculty's telemetry and regulates the one
        # that needs it (fires that faculty's OWN sleep/dream/mushroom). Internal autonomic oversight,
        # NOT an external knob. Per-faculty Φ history feeds its plateau detector.
        self.ocean = OceanAutonomic()
        self._phi_hist: dict[str, list[float]] = {role: [] for role in self.roles}
        self._last_regulation: dict[str, dict] = {}
        self._step_count: int = 0
        self._coordizer_path: str | None = None
        self.coordizer = coordizer

    def _build_node(self, role: str, birth: "np.ndarray", num_layers: int, coordizer: Any,
                    device: str | None, seed: int, *, is_central: bool, language_peer: Any = None) -> Any:
        """Build ONE constellation node from the selected ARM. ``gk`` → the qigkernels deep Kernel
        (``GenesisKernelTarget``, node-ready: it already exposes run_protocol + basin coupling hooks + M).
        ``geo``/``hybrid`` plug a different substrate into the SAME node slot, but that substrate must first
        gain the constellation-node contract (geo node-parity, WS3) — until then they raise a clear,
        user-facing error rather than silently building a node Ocean can't regulate and ``couple_step``
        can't read. ``hetero`` = gk central + geo faculties."""
        arm = self.arm_mode
        sub = ("gk" if is_central else "geo") if arm == "hetero" else arm
        if sub == "gk":
            from ..targets.genesis_kernel import GenesisKernelTarget
            return GenesisKernelTarget(num_layers=num_layers, role=role, basin_template=birth,
                                       coordizer=coordizer, device=device, seed=seed,
                                       language_peer=language_peer if is_central else None)
        if sub in ("geo", "hybrid"):
            raise NotImplementedError(
                f"constellation arm {arm!r} needs the {sub!r} node-parity build (WS3/WS4): the geo/hybrid "
                f"substrate must expose the node contract (run_protocol + _basin_history + _basin_ref) before "
                f"it can couple + be Ocean-regulated. Only 'gk' is node-ready today.")
        raise ValueError(f"unknown constellation arm {arm!r} (expected gk|geo|hybrid|hetero)")

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
        self._step_count += 1
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
        import hashlib as _hl
        import json
        import shutil
        import subprocess
        from datetime import datetime, timezone
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

        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            git_commit = None

        coordizer_path = getattr(self, "_coordizer_path", None)
        coordizer_hash = None
        if coordizer_path:
            try:
                with open(coordizer_path, "rb") as _f:
                    coordizer_hash = _hl.sha256(_f.read()).hexdigest()[:8]
            except Exception:
                pass

        (r / "constellation.json").write_text(json.dumps({
            "roles": self.roles,
            "faculty_basins": {f.role: f.basin.tolist() for f in self.faculties},
            "min_pairwise_fr": min_pairwise_fr(self.faculties),
            "metadata": {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "training_step": getattr(self, "_step_count", 0),
                "coordizer_path": coordizer_path,
                "coordizer_vocab": getattr(self.coordizer, "vocab_size", None) if hasattr(self, "coordizer") else None,
                "coordizer_hash": coordizer_hash,
                "central_phi": round(float(self.central.telemetry().phi or 0), 4),
                "min_pairwise_fr": min_pairwise_fr(self.faculties),
                "git_commit": git_commit,
                "num_layers": getattr(self.central, "num_layers", None),
                "arm_mode": self.arm_mode,     # the raw-kernel arm → load_checkpoint refuses a different-arm restore
            },
        }))

    def load_checkpoint(self, root: str) -> None:
        """Restore the whole mind (faculties + central + coupled basins) saved by save_checkpoint."""
        import json
        from pathlib import Path

        from qig_core.geometry import to_simplex
        r = Path(root)
        # ARM GUARD: never load a DIFFERENT-arm checkpoint over this constellation (a geo checkpoint into a gk
        # mind, etc.) — the substrates differ. If the checkpoint records a mismatching arm, keep the fresh
        # build. (Pre-arm_mode checkpoints have no tag → load as before, treated as the legacy gk arm.)
        cj0 = r / "constellation.json"
        if cj0.exists():
            try:
                _ckpt_arm = (json.loads(cj0.read_text()).get("metadata", {}) or {}).get("arm_mode")
            except Exception:  # noqa: BLE001
                _ckpt_arm = None
            if _ckpt_arm and _ckpt_arm != self.arm_mode:
                print(f"⚠️  checkpoint arm {_ckpt_arm!r} != constellation arm {self.arm_mode!r} — NOT restoring "
                      f"a different-arm checkpoint; keeping the fresh {self.arm_mode} build", flush=True)
                return
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
