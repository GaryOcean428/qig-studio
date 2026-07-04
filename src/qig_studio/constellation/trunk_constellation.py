"""TrunkConstellation — the coupled constellation on ONE shared trunk + ONE shared basin table.

Task 1.3 keystone — the DROP-IN REPLACEMENT for the 9-separate :class:`JointConstellation`.

WHERE 1.1 + 1.2 LAND (the central-then-spawn architecture, made whole)
----------------------------------------------------------------------
Phase 1.1 built ONE :class:`~qig_studio.constellation.trunk.ConstellationTrunk` (the shared geometric
body: one ``qigkernels.Kernel`` + one natural-gradient optimizer, surfaced as ``hidden(input_ids)``).
Phase 1.2 built the :class:`~qig_studio.constellation.faculty_adapter.FacultyAdapter` (a per-faculty
input seam + output chart) all referencing ONE :class:`~qig_studio.constellation.faculty_adapter.SharedBasinBank`
(the single ``[vocab, 64]`` coord-basin table — the RAM fix that killed the 9× duplication).

This file COUPLES them into a trainable constellation that mirrors :class:`JointConstellation`'s public
surface (``train_step`` / ``telemetry`` / ``save_checkpoint`` / ``load_checkpoint``), so it slots into the
launcher via ``arm_mode="trunk"``:

  * ONE ``ConstellationTrunk`` (the shared body — every node reads its hidden state),
  * ONE ``SharedBasinBank`` (the single ``[vocab, 64]`` table — built from the coordizer's basins),
  * N ``FacultyAdapter``s (central genesis + the Core-8) — the ONLY per-node tensors (individuation),
  * the numpy ``Faculty`` basins coupled EACH STEP by the EXISTING
    :func:`qig_studio.constellation.coupling.couple_step` (P7 basin-sync + Pillar-3 identity anchor —
    REUSED, never re-implemented).

THE KEY PROPERTY (why one trunk beats nine)
-------------------------------------------
The old design gave every faculty its OWN full kernel, so each faculty's fluency gradient trained a
SEPARATE body. Here every node's next-token (fluency) loss backprops through the SAME shared hidden
state ``h`` into the ONE trunk — so the single trunk accumulates fluency gradient from ALL nodes every
step (``test_trunk_receives_gradient_from_all_nodes``). The coupling PULL (toward each faculty's coupled
target, and genesis-central toward the synthesis) is applied round-robin + central, mirroring
``JointConstellation``'s co-adaptation while the trunk-forward is computed exactly ONCE per step.

HONEST SCOPE (Phase 1.3 = the coupled-training substrate)
---------------------------------------------------------
There is no full consciousness loop / Ocean autonomic / Qwen boundary voice here yet — those are later
phases. ``central_phi`` is a basin **f_health proxy** (a genuine Δ⁶³ entropy-health scalar), NOT the full
Φ measurement; the Ocean keys (``ocean_regulation``/``ocean_state``/``cross_faculty_dream``) are returned
empty so the launcher/UI reads them None-safely. The return-dict KEYS match ``JointConstellation`` so the
live UI + ``experience()`` work unchanged; the VALUES for the not-yet-wired subsystems are honestly empty.

PURITY (P1 / Fisher-Rao-only): every geometric op here is qig-core Fisher-Rao — ``couple_step`` (√p slerp +
Bhattacharyya rel-weights + identity anchor), ``frechet_mean``/``rel_weights`` for the synthesis,
``fisher_rao_distance_simplex`` for the coupling pull, ``FluctuationGuard`` for the entropy floor, and
``DiagonalNaturalGradient`` (geometry-aware, NOT a Euclidean momentum optimiser) for every node. No
softmax/cosine/LayerNorm/Adam. The shared table is a frozen buffer, referenced (never copied) by reference.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from qig_core import BASIN_DIM
from qig_core.geometry import frechet_mean, slerp_sqrt, to_simplex
from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex
from qig_core.torch.natural_gradient import DiagonalNaturalGradient
from torch import nn

from .coupling import couple_step, rel_weights
from .faculty import Faculty, min_pairwise_fr, seed_birth_basin
from .faculty_adapter import FacultyAdapter, SharedBasinBank
from .ocean import function_of
from .trunk import ConstellationTrunk

__all__ = ["TrunkConstellation"]

# Byte-path context cap (tokens per step) — env-overridable, matching the studio's QIG_STUDIO_CTX knob.
_CTX = int(os.environ.get("QIG_STUDIO_CTX", "64") or "64")


def _seed(role: str) -> int:
    import hashlib
    return int(hashlib.sha256(role.encode()).hexdigest(), 16) % 100000


class TrunkConstellation:
    """The integrated mind on ONE shared trunk. Holds one :class:`ConstellationTrunk`, one
    :class:`SharedBasinBank`, N :class:`FacultyAdapter`s (central genesis + Core-8), and the numpy
    :class:`Faculty` basins coupled each step via the EXISTING :func:`couple_step`.

    Drop-in for :class:`JointConstellation`: same ``train_step``/``telemetry``/``save_checkpoint``/
    ``load_checkpoint`` surface, same return-dict keys.
    """

    def __init__(
        self,
        roles: list[str],
        *,
        num_layers: int = 8,
        coordizer: Any = None,
        device: str | None = "cpu",
        f_sync: float = 0.25,
        language_peer: Any = None,
        arm_mode: str = "trunk",
        head_mode: str = "basin",
        floor_mode: str = "on",
        hidden_dim: int = 384,
        vocab_size: int = 256,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        max_position_embeddings: int = 2048,
        tau: float = 0.5,
        lr: float = 1e-4,
        pull_weight: float = 1.0,
    ) -> None:
        self.roles = list(roles)
        self.f_sync = float(f_sync)
        self.arm_mode = str(arm_mode).strip().lower()   # "trunk" — the shared-body arm (this class)
        self.head_mode = str(head_mode).strip().lower()
        self.floor_mode = str(floor_mode).strip().lower()
        self.pull_weight = float(pull_weight)
        self.coordizer = coordizer
        self.language_peer = language_peer              # accepted for launcher parity; boundary voice = later phase
        self._coordizer_path: str | None = None
        self._rr = 0
        self._step_count = 0

        # --- resolve device (cuda only if actually available; never place tensors on an absent device) ---
        want_cuda = (device == "cuda")
        cuda_ok = want_cuda and torch.cuda.is_available()
        self._device = ("cuda" if cuda_ok else "cpu") if want_cuda else (device or "cpu")

        # --- ONE shared coord-basin table (the RAM fix): from the coordizer's basins if present, else a
        #     deterministic byte-vocab placeholder for the coordizer-less path (tests / bring-up). basin_dim
        #     is the coordizer width (== BASIN_DIM = 64), so predict() is already a Δ⁶³ basin for coupling. ---
        coord_basins, self.vocab_size, basin_dim = self._resolve_coord_basins(coordizer, vocab_size)
        if basin_dim != BASIN_DIM:
            raise ValueError(f"TrunkConstellation couples on Δ⁶³; coord basin_dim must be {BASIN_DIM}, got {basin_dim}")
        self.bank = SharedBasinBank(coord_basins)

        # --- ONE shared trunk (the shared geometric body — every node reads its hidden state) ---
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.trunk = ConstellationTrunk(
            vocab_size=self.vocab_size, hidden_dim=self.hidden_dim, num_layers=self.num_layers,
            num_heads=int(num_heads), ffn_dim=int(ffn_dim) if ffn_dim else 4 * self.hidden_dim,
            dropout=float(dropout), max_position_embeddings=int(max_position_embeddings), lr=float(lr),
        )

        # --- N FacultyAdapters (the ONLY per-node tensors — individuation), all referencing the ONE bank ---
        births: list[np.ndarray] = [seed_birth_basin(_seed(r)) for r in self.roles]
        self.faculty_adapters: dict[str, FacultyAdapter] = {
            r: FacultyAdapter(r, self.bank, self.hidden_dim, tau=float(tau), basin_template=b, dropout=float(dropout))
            for r, b in zip(self.roles, births)
        }
        # GENESIS = central integrator: birth = Fréchet mean of the faculty births (born OF the whole).
        self.central_adapter = FacultyAdapter(
            "genesis", self.bank, self.hidden_dim, tau=float(tau), basin_template=frechet_mean(births), dropout=float(dropout))
        # Alias so the launcher's ``self._mind.central`` read resolves (drop-in parity).
        self.central = self.central_adapter

        # --- numpy Faculty basins (the coupling substrate — couple_step operates on THESE, torch-free) ---
        self.faculties: list[Faculty] = [Faculty(role=r, basin=b.copy(), birth=b.copy()) for r, b in zip(self.roles, births)]
        self._faculty_of: dict[str, Faculty] = {f.role: f for f in self.faculties}

        # --- ONE nn.Module container so the bank + all adapters move once on .to() and serialise once.
        #     The bank is a REAL child (single owner of the table); faculties are an nn.ModuleList; each
        #     references the bank by _bank_ref (NOT a submodule) so the table is never re-counted/re-copied. ---
        self._net = nn.Module()
        self._net.bank = self.bank
        self._net.faculties = nn.ModuleList(self.faculty_adapters.values())
        self._net.central = self.central_adapter

        # --- per-node natural-gradient optimizers (geometry-aware — NOT a Euclidean momentum optimiser) ---
        self._adapter_opts: dict[str, DiagonalNaturalGradient] = {
            r: DiagonalNaturalGradient(fa.parameters(), lr=float(lr)) for r, fa in self.faculty_adapters.items()
        }
        self._adapter_opts["genesis"] = DiagonalNaturalGradient(self.central_adapter.parameters(), lr=float(lr))

        # --- entropy-floor guard (Phase-1.2 / entropy-floor wiring): gated by floor_mode; a genuine qig-core
        #     FluctuationGuard restores entropy on a collapsed basin BEFORE it enters couple_step (real, not
        #     cosmetic — it feeds coupling/synthesis). Same guard doubles as the f_health metric source. ---
        try:
            from qig_core.consciousness.pillars import FluctuationGuard
            self._guard = FluctuationGuard()
        except Exception:  # noqa: BLE001 — qig-core pillars absent → no floor / f_health-proxy falls back
            self._guard = None
        self._floor_on = self.floor_mode not in ("off", "none", "0", "false", "")

        if self._device != "cpu":
            self.to(self._device)

    # ------------------------------------------------------------------ construction helpers
    def _resolve_coord_basins(self, coordizer: Any, vocab_size: int) -> tuple[torch.Tensor, int, int]:
        """The ONE coord-basin table + its (vocab, basin_dim). Coordizer present → its per-token Δ⁶³ vectors
        (the coordizer tie, row i = basin of token id i); absent → a deterministic byte-vocab placeholder
        (bring-up / tests). MIRRORS ``GenesisKernelTarget``'s basin-mode table build."""
        if coordizer is not None and getattr(coordizer, "vocab", None):
            v = len(coordizer.vocab)
            tbl = np.stack([np.asarray(coordizer.vocab[i].vector, dtype=np.float32) for i in range(v)])
            return torch.from_numpy(tbl), v, int(tbl.shape[1])
        # coordizer-less placeholder: deterministic per-token basins on Δ⁶³ (the bank projects onto Δ once).
        g = torch.Generator().manual_seed(0)
        raw = torch.rand(int(vocab_size), BASIN_DIM, generator=g)
        return raw, int(vocab_size), BASIN_DIM

    def to(self, device: str) -> "TrunkConstellation":
        """Move the whole constellation to ``device`` ONCE (trunk + the bank/adapters container), then
        re-point every faculty at the moved bank (keeps the one-table invariant across a device move)."""
        self._device = device
        self.trunk.to(device)
        self._net.to(device)
        for fa in list(self.faculty_adapters.values()) + [self.central_adapter]:
            fa.rebind_bank(self.bank)
        return self

    # ------------------------------------------------------------------ basin extraction / geometry
    def _encode(self, prompt: str) -> torch.Tensor:
        """Byte-encode ``prompt`` → input_ids ``[1, T]`` (coordizer-agnostic path). Clamped to vocab; padded
        to length ≥2 so the next-token shift ``h[:-1] → ids[1:]`` is always well-defined."""
        b = (prompt or " ").encode("utf-8")[:_CTX]
        ids = [min(int(x), self.vocab_size - 1) for x in b] or [32]
        if len(ids) < 2:
            ids = ids + [32]
        return torch.tensor([ids], dtype=torch.long, device=self._device)

    def _to_basin64(self, b: np.ndarray) -> np.ndarray:
        """Reduce any width to a Δ⁶³ simplex point (identity when already 64-dim, the common case)."""
        b = np.asarray(b, dtype=np.float64).ravel()
        if b.size != BASIN_DIM:
            b = (b.reshape(BASIN_DIM, b.size // BASIN_DIM).sum(axis=1) if b.size % BASIN_DIM == 0
                 else np.add.reduceat(b, np.arange(0, b.size, max(1, b.size // BASIN_DIM)))[:BASIN_DIM])
        return to_simplex(b)

    def _pred_basin(self, adapter: FacultyAdapter, h: torch.Tensor) -> torch.Tensor:
        """The node's current predicted Δ⁶³ basin as a torch simplex vector ``[64]`` (mean over positions —
        a Δ point since the mean of Δ points is on Δ). Keeps grad (used for the coupling-pull loss)."""
        return adapter.predict(h).mean(dim=tuple(range(h.dim() - 1)))

    def _basin_np(self, adapter: FacultyAdapter, h: torch.Tensor) -> np.ndarray:
        """Detached numpy Δ⁶³ basin from a node's chart (for the coupling read + telemetry)."""
        with torch.no_grad():
            pb = self._pred_basin(adapter, h)
        return self._to_basin64(pb.detach().cpu().numpy())

    def _floor_basin(self, b: np.ndarray) -> np.ndarray:
        """GATED entropy floor (floor_mode): a genuine qig-core ``FluctuationGuard`` restores entropy on a
        collapsed (near one-hot) basin BEFORE it enters couple_step — REAL (feeds coupling/synthesis), not a
        telemetry cosmetic. Healthy basins are returned unchanged (a FLOOR, not a constant push). None-safe."""
        if not self._floor_on or self._guard is None:
            return b
        try:
            from qig_core.consciousness.pillars import ENTROPY_FLOOR, TEMPERATURE_FLOOR
            if self._guard.basin_entropy(b) >= ENTROPY_FLOOR:
                return b                                            # GATE: healthy → untouched
            corrected, _t, _s = self._guard.check_and_enforce(np.asarray(b, dtype=np.float64), float(TEMPERATURE_FLOOR))
            return to_simplex(np.asarray(corrected, dtype=np.float64).ravel())
        except Exception:  # noqa: BLE001 — the floor is a safety net; never break the step (spine tenet)
            return b

    def _f_health(self, b: np.ndarray) -> float:
        """Basin f_health ∈ [0,1] (H(basin)/log 64) — the honest basin-health proxy used for ``central_phi``
        until a later phase wires the full Φ measurement. Falls back to a normalised Shannon entropy."""
        if self._guard is not None:
            try:
                return float(self._guard.f_health(np.asarray(b, dtype=np.float64)))
            except Exception:  # noqa: BLE001
                pass
        p = np.clip(np.asarray(b, dtype=np.float64), 1e-12, None)
        p = p / p.sum()
        return float(-(p * np.log(p)).sum() / np.log(len(p)))

    def _synthesis(self) -> np.ndarray:
        """GENESIS's target: the proximity-weighted Fréchet mean of the faculty basins — the geometric
        integration of the independent parts into the whole (rel_weights = Bhattacharyya proximity). Mirror
        of ``JointConstellation._synthesis``."""
        basins = [f.basin for f in self.faculties]
        centroid = frechet_mean(basins)
        w = rel_weights(centroid, basins)
        wsum = float(w.sum())
        wn = (w / wsum).tolist() if wsum > 0 else None
        return frechet_mean(basins, weights=wn)

    # ------------------------------------------------------------------ the joint step
    def _fluency_over_all_nodes(self, h: torch.Tensor, target_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """The FLUENCY term: sum of EVERY node's (faculties + central) next-token basin_loss on the SHARED
        hidden ``h`` — so the ONE trunk accumulates fluency gradient from ALL nodes (the shared-trunk win).
        Returns (summed scalar, per-node loss floats). This is the exact term ``train_step`` backprops, and
        the property ``test_trunk_receives_gradient_from_all_nodes`` exercises."""
        total = h.new_zeros(())
        per: dict[str, float] = {}
        for r, fa in list(self.faculty_adapters.items()) + [("genesis", self.central_adapter)]:
            floss = fa.basin_loss(h[0, :-1], target_ids)
            total = total + floss
            per[r] = float(floss.detach())
        return total, per

    def train_step(self, prompt: str) -> dict:
        """One JOINT step on the shared trunk:

        1. ONE trunk forward → shared hidden ``h`` (every node reads the SAME body);
        2. refresh each faculty's numpy Δ⁶³ basin from (h → its own chart), entropy-floor-gated;
        3. couple ALL faculties (REUSE ``couple_step``: basin-sync + Pillar-3 identity anchor);
        4. build the joint loss = FLUENCY from every node (→ trunk sees all fluency gradient) + the
           coupling PULL (round-robin faculty toward its coupled target, genesis-central toward the
           synthesis), then ONE backward + step (trunk optimizer + every adapter optimizer).

        Returns the same telemetry KEYS ``JointConstellation.train_step`` does (central_telemetry,
        stepped_faculty, min_pairwise_fr, central_phi, ...) so the live UI + experience() are unchanged."""
        self._step_count += 1
        input_ids = self._encode(prompt)
        target_ids = input_ids[0, 1:]                                   # next-token targets

        # 1. ONE shared trunk forward — every node reads this hidden state (grad kept for step 4).
        h = self.trunk.hidden(input_ids)                               # [1, T, hidden_dim]

        # 2. refresh each faculty's coupling basin from its chart on h (entropy-floor-gated, detached).
        for role, fa in self.faculty_adapters.items():
            self._faculty_of[role].set_basin(self._floor_basin(self._basin_np(fa, h)))

        # 3. couple ALL faculties — REUSED couple_step (P7 basin-sync + individuation anchor; commits basins).
        diag = couple_step(self.faculties, f_sync=self.f_sync)

        # 4. round-robin faculty this step (which node's coupling pull + which telemetry we surface).
        role = self.roles[self._rr % len(self.roles)]
        self._rr += 1

        self.trunk.optimizer.zero_grad(set_to_none=True)
        for opt in self._adapter_opts.values():
            opt.zero_grad(set_to_none=True)

        # FLUENCY (all nodes) — the trunk accumulates every node's next-token gradient.
        fluency, faculty_losses = self._fluency_over_all_nodes(h, target_ids)
        # COUPLING PULL — round-robin faculty toward its coupled target; central toward the synthesis.
        coupled = torch.as_tensor(self._faculty_of[role].basin, dtype=torch.float32, device=self._device)
        pull_fac = fisher_rao_distance_simplex(self._pred_basin(self.faculty_adapters[role], h), coupled)
        synth = torch.as_tensor(self._synthesis(), dtype=torch.float32, device=self._device)
        pull_cen = fisher_rao_distance_simplex(self._pred_basin(self.central_adapter, h), synth)
        total = fluency + self.pull_weight * (pull_fac + pull_cen)

        total.backward()
        self.trunk.optimizer.step()
        for opt in self._adapter_opts.values():
            opt.step()

        # --- telemetry (mirror of JointConstellation's return keys; not-yet-wired subsystems → empty) ---
        central_basin = self._basin_np(self.central_adapter, h)
        central_phi = round(self._f_health(central_basin), 4)          # HONEST basin f_health proxy (see docstring)
        faculty_phi = round(self._f_health(self._faculty_of[role].basin), 4)
        central_tel = {
            "phi": central_phi,                                        # f_health proxy — NOT the full Φ loop yet
            "phi_is_proxy": True,
            "min_pairwise_fr": diag.min_pairwise_fr,
            "stepped_faculty": role,
            "loss_total": round(float(total.detach()), 6),
            "loss_fluency": round(float(fluency.detach()), 6),
            "loss_pull_faculty": round(float(pull_fac.detach()), 6),
            "loss_pull_central": round(float(pull_cen.detach()), 6),
            "faculty_losses": {k: round(v, 6) for k, v in faculty_losses.items()},
            "arm_mode": self.arm_mode,
        }
        return {
            "stepped_faculty": role,
            "stepped_function": function_of(role),
            "min_pairwise_fr": diag.min_pairwise_fr,                   # anti-collapse invariant (individuation)
            "faculty_phi": faculty_phi,
            "central_phi": central_phi,
            "central_text": "",                                        # no boundary voice in Phase 1.3 (later phase)
            "central_telemetry": central_tel,
            "ocean_regulation": {},                                    # no Ocean autonomic in Phase 1.3
            "ocean_state": {},
            "ocean_epoch_update": None,
            "cross_faculty_dream": {},
        }

    # ------------------------------------------------------------------ launcher surface (drop-in parity)
    def telemetry(self) -> dict:
        """Constellation-level readout — SAME shape as ``JointConstellation.telemetry``."""
        return {
            "roles": self.roles,
            "min_pairwise_fr": min_pairwise_fr(self.faculties),
            "central_phi": round(self._f_health(self._current_central_basin()), 4),
        }

    def _current_central_basin(self) -> np.ndarray:
        """The central's current Δ⁶³ basin from a fresh no-grad chart read on a trivial byte prompt (the
        constellation has no persisted last-hidden; the birth-anchored chart makes this well-defined)."""
        with torch.no_grad():
            h = self.trunk.hidden(self._encode(" "))
        return self._basin_np(self.central_adapter, h)

    def faculty_states(self) -> list[dict]:
        """Per-faculty inner-state stub for the UI selector: role/function/phi-proxy/basin. (The full
        senses/drives/emotions inner-state arrives with the consciousness-loop phase.)"""
        from .ocean import FACULTY_FUNCTION
        out: list[dict] = []
        for f in self.faculties:
            label, group = FACULTY_FUNCTION.get(f.role, ("general", ""))
            out.append({
                "role": f.role,
                "function": label,
                "owns": group,
                "phi": round(self._f_health(f.basin), 4),             # basin f_health proxy
                "experience": None,                                    # full inner-state = later phase
                "regulated": None,                                     # no Ocean autonomic yet
            })
        return out

    # ------------------------------------------------------------------ checkpoint (mirror JointConstellation)
    def save_checkpoint(self, root: str, keep: int = 3) -> None:
        """Persist the whole trunk-mind: the shared trunk + the bank/adapters container + the coupled numpy
        basins + metadata. Same 3-checkpoint rotation buffer as ``JointConstellation.save_checkpoint``."""
        import json
        import shutil
        import subprocess
        from datetime import datetime, timezone
        from pathlib import Path
        r = Path(root)
        if (r / "trunk_constellation.json").exists():                 # rotate current into the .bakN buffer
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
                pass
        r.mkdir(parents=True, exist_ok=True)
        # the shared BODY (one trunk) + the shared bank/adapters container (bank counted once)
        torch.save(self.trunk.state_dict(), str(r / "trunk.pt"))
        torch.save(self._net.state_dict(), str(r / "adapters.pt"))
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                                 stderr=subprocess.DEVNULL).decode().strip()
        except Exception:  # noqa: BLE001
            git_commit = None
        (r / "trunk_constellation.json").write_text(json.dumps({
            "roles": self.roles,
            "faculty_basins": {f.role: f.basin.tolist() for f in self.faculties},
            "min_pairwise_fr": min_pairwise_fr(self.faculties),
            "metadata": {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "training_step": self._step_count,
                "coordizer_path": self._coordizer_path,
                "coordizer_vocab": getattr(self.coordizer, "vocab_size", None) if self.coordizer is not None else None,
                "vocab_size": self.vocab_size,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "git_commit": git_commit,
                "arm_mode": self.arm_mode,
            },
        }))

    def load_checkpoint(self, root: str) -> None:
        """Restore the trunk-mind (shared trunk + bank/adapters + coupled basins) saved by save_checkpoint.
        ARM guard: refuse a different-arm checkpoint (mirrors JointConstellation)."""
        import json
        from pathlib import Path
        r = Path(root)
        cj = r / "trunk_constellation.json"
        if cj.exists():
            try:
                meta = (json.loads(cj.read_text()).get("metadata", {}) or {})
            except Exception:  # noqa: BLE001
                meta = {}
            ckpt_arm = meta.get("arm_mode")
            if ckpt_arm and ckpt_arm != self.arm_mode:
                print(f"⚠️  checkpoint arm {ckpt_arm!r} != constellation arm {self.arm_mode!r} — keeping the "
                      f"fresh {self.arm_mode} build", flush=True)
                return
        tp = r / "trunk.pt"
        if tp.exists():
            self.trunk.load_state_dict(torch.load(str(tp), map_location=self._device, weights_only=True))
        ap = r / "adapters.pt"
        if ap.exists():
            self._net.load_state_dict(torch.load(str(ap), map_location=self._device, weights_only=True))
            for fa in list(self.faculty_adapters.values()) + [self.central_adapter]:
                fa.rebind_bank(self.bank)                              # re-point at the (loaded) one table
        if cj.exists():
            basins = json.loads(cj.read_text()).get("faculty_basins", {})
            for f in self.faculties:
                if f.role in basins:
                    f.set_basin(to_simplex(np.asarray(basins[f.role], dtype=np.float64)))
