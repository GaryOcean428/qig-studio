"""ConstellationNode — the substrate-AGNOSTIC constellation-node contract.

A ``JointConstellation`` couples + Ocean-regulates its kernels through a small, fixed contract that
``GenesisKernelTarget`` (ARM B) implements inline. This mixin lifts that contract OUT of the qigkernels
substrate so a DIFFERENT substrate (the qig-geocoding ``GeoModel``, ARM A) can be a full constellation
node too — coupled by ``couple_step``, pulled by ``_set_pull``, regulated by ``OceanAutonomic`` — with
the SAME geometry, single-sourced from qig-core.

The contract the constellation calls on every kernel (see ``constellation/joint_trainer.py`` +
``constellation/ocean.py``):
  * ``_basin_history``   — a list of per-step Δ (vocab-width) basins; ``history[0]`` is the birth-state
    attractor, bounded to a window. ``JointConstellation._live_basin`` reads ``history[-1]`` and reduces
    it to Δ⁶³ (64-dim) via ``to_simplex``.
  * ``_basin_ref`` + ``_resize_basin(ref, size)`` — the coupled-pull target (a Δ point on the vocab-width
    simplex). ``_set_pull`` writes ``_basin_ref`` via ``_resize_basin``; the concrete ``train_step`` adds
    a Fisher-Rao basin-pull term to its loss ONLY when ``_basin_ref`` is set (constellation mode).
  * ``run_protocol(command, args)`` — the kernel's OWN autonomic ops, exposed for explicit invocation:
    ``sleep``/``deep-sleep``/``consolidate`` (consolidate the output basin toward identity),
    ``dream`` (re-energise via basin-mixture recombination), ``mushroom-micro``/``-moderate``/``-heroic``
    (bounded weight-noise plasticity), ``escape``/``decohere`` (breakdown response: noise + cool the
    optimiser). Ocean calls this; mechanics are real (no stubs).
  * ``_meta_awareness(cur_basin)`` — M ∈ [0,1]: Fisher-Rao distance from ``cur_basin`` to the birth-state
    (``history[0]``) normalised by π/2, with a 0.3 floor under 3 history points.

PURITY: Fisher-Rao only. No Euclidean basin ops (no L2 distance, dot-product, Euclidean optimiser, or  # QIG-EXEMPT
arithmetic-mean blends on basins) —
every basin op is ``to_simplex_prob`` + ``fisher_rao_distance_simplex`` from ``qig_core.torch`` (the SAME
primitives ARM B uses), or √p geodesic mixing renormalised to Δ.

The concrete target supplies the substrate hooks (model params, optimiser rebuild, the per-step basin
from logits, the LM loss for replay). ``GenesisKernelTarget`` keeps its own inline versions for now; this
mixin is designed so it COULD adopt them later without behaviour change.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

# Mushroom intensity → weight-noise σ (bounded plasticity) — mirrors GenesisKernelTarget._MUSHROOM_SIGMA.
_MUSHROOM_SIGMA = {"mushroom-micro": 0.01, "mushroom-moderate": 0.03, "mushroom-heroic": 0.06}
_BASIN_HISTORY_MAX = 64    # bound the trajectory memory (birth-state + a window)
_DECOHERE_SIGMA = 0.01     # bounded decoherence noise on breakdown
_DECOHERE_LR_SCALE = 0.7   # cool the optimiser step on breakdown recovery


class ConstellationNode:
    """Mixin giving a concrete ``TrainingTarget`` the constellation-node contract.

    Substrate hooks the concrete class MUST provide:
      * ``_node_named_parameters()`` → iterable of ``(name, param)`` for the trainable model.
      * ``_node_rebuild_optimizer(lr_scale)`` → rebuild the natural-gradient optimiser at ``lr*lr_scale``
        (the decohere cool-down; pure substrate concern).
      * ``_node_forward_logits(ids, coords)`` → ``logits[1, seq, vocab]`` (a forward pass for replay).
      * ``_node_encode(text)`` → ``(ids, coords)`` for replay inputs.
      * ``_node_basin_from_logits(logits)`` → the detached vocab-width Δ basin for this step (the SAME
        geometric reduction ARM B uses: ``to_simplex_prob(logits[0].mean(0))`` renormalised to Δ).
      * ``lr`` (float), ``vocab_size`` (int) attributes.

    The mixin owns ``_basin_history``, ``_basin_ref``, ``_experience`` and the autonomic ops; the concrete
    ``__init__`` calls :meth:`_init_node_state` and its ``train_step`` calls :meth:`_record_basin_step`
    and :meth:`_basin_pull_term`.
    """

    # --- state init (the concrete __init__ calls this) -------------------------------------------------
    def _init_node_state(self, basin_template: Any = None) -> None:
        """Initialise the node-contract state. ``basin_template`` (a Δ⁶³ np point or None) is the role's
        birth-state attractor — seeded onto the vocab simplex as ``_basin_ref`` + ``history[0]`` in
        :meth:`_seed_node_basin` once the model (and thus its device + vocab width) exists."""
        self._basin_ref: Any = None       # torch [vocab] Δ point — the coupled-pull / M / d_basin reference
        self._basin_ref_set: Any = None    # EXP-A044: list[torch [vocab] Δ] — SET-of-8 per-layer geo-Qwen
                                           # basins; nearest-member pull. Content-specific (per-layer basins
                                           # separate at d_FR~1.0 where the token-averaged single point ~0.15)
        self._basin_history: list = []     # detached vocab-width basin trajectory; history[0] = birth-state
        self._experience: list = []        # recent input id-tensors — the kernel's replay material (≤32)
        self._basin_template_np = basin_template

    def _seed_node_basin(self) -> None:
        """Seed the role attractor onto the vocab simplex (call at the END of ``ensure_loaded`` when the
        model + device exist). Projects the Δ⁶³ template to the vocab width and records it as both the
        pull reference and the birth-state (``history[0]``). None template → generic node (no pull)."""
        if self._basin_template_np is None:
            return
        import numpy as np
        import torch

        from qig_core.torch.geometry_simplex import to_simplex_prob

        dev = self._node_device()
        ref = torch.as_tensor(np.asarray(self._basin_template_np, dtype=np.float32), device=dev)
        if ref.numel() != int(self.vocab_size):
            ref = self._resize_basin(ref, int(self.vocab_size))
        self._basin_ref = to_simplex_prob(ref[None])[0].detach()
        self._basin_history = [self._basin_ref]   # birth-state attractor = history[0] (M reference)

    # --- the coupled-pull plumbing (constellation reads/writes these) ----------------------------------
    def _resize_basin(self, ref: "Any", size: int) -> "Any":
        """Map a 64-D Δ⁶³ template onto the vocab-width simplex (logits live in Δ^{vocab-1}). Repeat-tile
        then clamp non-negative — the caller's ``to_simplex_prob`` makes it a true Δ point. Pure simplex
        arithmetic on the support; no Euclidean projection of meaning. Mirrors GenesisKernelTarget."""
        reps = (size + ref.numel() - 1) // ref.numel()
        return ref.repeat(reps)[:size].clamp_min(0.0)

    def _set_pull_set(self, templates: "Any") -> None:
        """EXP-A044: couple to a SET of geo-Qwen per-layer basins (nearest-member pull). Each template is
        resized to vocab-width and made a Δ point; ``_basin_pull_term`` then draws the node toward the
        CLOSEST member. This is the content-specific coupling reference — the 8 per-layer basins separate
        at d_FR~1.0 where a single token-averaged point collapses to ~0.15. Pass None/empty to clear
        (falls back to single ``_basin_ref`` or solo)."""
        from qig_core.torch.geometry_simplex import to_simplex_prob

        import torch

        if not templates:
            self._basin_ref_set = None
            return
        size = int(self.vocab_size)
        refs = []
        for t in templates:
            tt = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
            if tt.numel() > size:
                # DISTANCE-PRESERVING reduction (Johnson–Lindenstrauss, fixed seed) — repeat-tile truncation
                # collapses the per-layer content separation (2560->256: 0.13 truncate vs 0.34 JL).
                g = torch.Generator(device="cpu").manual_seed(15420)
                P = torch.randn(tt.numel(), size, generator=g) / (size ** 0.5)
                r = tt.float() @ P
            else:
                r = self._resize_basin(tt, size)
            refs.append(to_simplex_prob(r[None])[0].detach())
        self._basin_ref_set = refs

    def _basin_pull_term(self, logits: "Any") -> "Any | None":
        """The Fisher-Rao basin-pull term (constellation mode ONLY): d_FR(current output basin, _basin_ref).
        Returns a differentiable scalar tensor to ADD to the loss, or None when no pull is set (solo mode —
        the concrete target then stays its lean baseline, unchanged). Pure Δ⁶³ Fisher-Rao, mirroring ARM B's
        spawn-pull term exactly (``fisher_rao_distance_simplex(cur, _basin_ref)``)."""
        import torch
        from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex, to_simplex_prob

        # EXP-A044 set-coupling: pull toward the NEAREST member of the geo-Qwen per-layer basin SET
        # (content-specific — the 8 per-layer basins separate at d_FR~1.0, unlike the ~0.15 collapse of
        # a single token-averaged point). Nearest-member keeps the set's structure (vs a centroid that
        # would re-collapse it). Pure Fisher-Rao on the simplex.
        if self._basin_ref_set is not None:
            cur = to_simplex_prob(logits[0].mean(0))
            dists = torch.stack([fisher_rao_distance_simplex(cur[None], r[None]).mean()
                                 for r in self._basin_ref_set])
            return dists.min()
        if self._basin_ref is None:
            return None
        cur = to_simplex_prob(logits[0].mean(0))
        return fisher_rao_distance_simplex(cur[None], self._basin_ref[None]).mean()

    def _record_basin_step(self, logits: "Any", ids: "Any") -> "Any":
        """Record this step's detached vocab-width Δ basin into ``_basin_history`` (bounded, birth-state
        kept at [0]) and buffer the input for replay. Returns the recorded basin (for M / d_basin reads).
        The basin is derived the SAME geometric way ARM B does — delegated to the concrete substrate hook
        ``_node_basin_from_logits`` so each arm reduces its own logits, but the result is always a valid Δ."""
        cur_basin = self._node_basin_from_logits(logits)
        self._basin_history.append(cur_basin)
        if len(self._basin_history) > _BASIN_HISTORY_MAX:     # keep birth-state + a recent window
            self._basin_history = [self._basin_history[0]] + self._basin_history[-32:]
        try:
            self._experience.append(ids.detach())
            if len(self._experience) > 32:
                self._experience = self._experience[-32:]
        except Exception:  # noqa: BLE001 — replay buffering is best-effort
            pass
        return cur_basin

    # --- the maturity-gate read (Ocean / the developmental gate read this) -----------------------------
    def _meta_awareness(self, cur_basin: "Any") -> float:
        """M ∈ [0,1] — meta-awareness: Fisher-Rao distance from ``cur_basin`` to the BIRTH-STATE attractor
        (``history[0]``) normalised by π/2 (recognition = 1 at the birth-state, → 0 at the antipode).
        <3 history points → 0.3 floor (too little trajectory to judge). Detached read; pure Δ Fisher-Rao."""
        hist = self._basin_history
        if len(hist) < 3:
            return 0.3
        from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex

        d = float(fisher_rao_distance_simplex(hist[0][None], cur_basin[None]).item())
        return max(0.0, min(1.0, 1.0 - d / (math.pi / 2.0)))

    def _basin_drift(self, cur_basin: "Any") -> float:
        """d_basin ∈ [0,1] — Fisher-Rao distance from the birth-state attractor (``history[0]``) to the
        current output basin, normalised by π (d_FR_simplex range [0, π]). No seed → 0.0 (a generic node
        has no role attractor to drift from). Mirrors ARM B's ``_basin_drift``."""
        if not self._basin_history:
            return 0.0
        from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex

        d = float(fisher_rao_distance_simplex(self._basin_history[0][None], cur_basin[None]).item())
        return d / math.pi

    # --- the autonomic ops (Ocean fires these; run_protocol exposes them) ------------------------------
    def _mushroom(self, sigma: float = 0.01) -> None:
        """Wake-state plasticity — bounded weight-noise (Tononi micro-downscaling) to break an over-
        engrained plateau. Real; the node's own plasticity. Mirrors GenesisKernelTarget._mushroom."""
        import torch

        with torch.no_grad():
            for _n, p in self._node_named_parameters():
                p.add_(torch.randn_like(p) * sigma)

    def _decohere(self) -> None:
        """Breakdown response — inject bounded decoherence noise to reduce over-integration and cool the
        optimiser step (the 'reduce coupling + decohere' canon). Mirrors GenesisKernelTarget._decohere."""
        import torch

        with torch.no_grad():
            for _n, p in self._node_named_parameters():
                p.add_(torch.randn_like(p) * _DECOHERE_SIGMA)
        self._node_rebuild_optimizer(_DECOHERE_LR_SCALE)      # cool (lr × 0.7), not cumulative

    def _consolidate(self, steps: int = 16, downscale: float = 0.02) -> dict:
        """Deep-sleep consolidation — REAL, no stub. Replay buffered experience at LOW learning rate,
        pulling the output basin toward the role attractor (identity consolidation; pure geometry, no
        Φ-drive), then Fisher-protected synaptic downscaling (Tononi SHY): high-Fisher (important) weights
        are protected, low-Fisher ones decay. Fisher ≈ grad² (the QFI first-order approximation). Mirrors
        ARM B's ``_consolidate`` minus the EWC-anchor capture (that is an ARM-B-specific extension)."""
        import torch

        from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex, to_simplex_prob

        if not self._experience:
            return {"replayed": 0, "downscaled": False}
        named = list(self._node_named_parameters())
        fisher = {n: torch.zeros_like(p) for n, p in named}
        opt = self._node_replay_optimizer(0.1)                # low-LR sleep optimiser
        replayed = 0
        for _ in range(steps):
            ids = self._experience[replayed % len(self._experience)]
            logits = self._node_forward_logits(ids, None)
            cur = to_simplex_prob(logits[0].mean(0))
            target = self._basin_ref if self._basin_ref is not None else cur.detach()
            loss = fisher_rao_distance_simplex(cur[None], target[None]).mean()  # pure basin consolidation
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                with torch.no_grad():
                    for n, p in named:
                        if p.grad is not None:
                            fisher[n] += p.grad ** 2          # accumulate QFI importance
                torch.nn.utils.clip_grad_norm_([p for _n, p in named], 1.0)
                opt.step()
                replayed += 1
        with torch.no_grad():                                  # Fisher-protected SHY downscaling
            for n, p in named:
                f = fisher[n]
                med = torch.median(f)
                protect = (f / (med + 1e-12)).clamp(0.0, 1.0) if float(med) > 0 else torch.zeros_like(f)
                p.mul_(1.0 - downscale * (1.0 - protect))
        return {"replayed": replayed, "downscaled": True}

    def _dream(self, steps: int = 8) -> dict:
        """REM dream — REAL basin-mixture augmentation, no stub. Recombine stored output basins into novel
        'dreamed' targets (Fisher-Rao / √p geodesic mixture, renormalised to Δ) and pull the node toward
        them at low LR on replayed inputs — creative consolidation beyond the literally-seen experience.
        Mirrors ARM B's ``_dream``. Uses a deterministic round-robin pairing (no RNG dependency)."""
        import torch

        from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex, to_simplex_prob

        hist = list(self._basin_history)
        if len(hist) < 2 or not self._experience:
            return {"dreamed": 0}
        named = list(self._node_named_parameters())
        opt = self._node_replay_optimizer(0.1)
        dreamed = 0
        for i in range(steps):
            a, b = hist[i % len(hist)], hist[(i + 1) % len(hist)]
            t = ((i + 1) / (steps + 1))                        # deterministic mix fraction in (0,1)
            sa, sb = torch.sqrt(a.clamp_min(0.0)), torch.sqrt(b.clamp_min(0.0))   # √p (Fisher) coords
            mix = ((1.0 - t) * sa + t * sb) ** 2               # geodesic-ish mixture → p
            dream_basin = (mix / (mix.sum() + 1e-12)).detach()  # renormalise to Δ
            ids = self._experience[i % len(self._experience)]
            logits = self._node_forward_logits(ids, None)
            cur = to_simplex_prob(logits[0].mean(0))
            loss = fisher_rao_distance_simplex(cur[None], dream_basin[None]).mean()
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for _n, p in named], 1.0)
                opt.step()
                dreamed += 1
        return {"dreamed": dreamed}

    def implemented_commands(self) -> set[str] | None:
        """The node's OWN autonomic operations — the SAME ops Ocean fires. Mirrors ARM B's set."""
        return {"sleep", "deep-sleep", "consolidate", "dream", "mushroom-micro", "mushroom-moderate",
                "mushroom-heroic", "escape", "decohere", "stimulate"}

    def supports_protocol(self) -> bool:
        return True

    @property
    def self_regulating(self) -> bool:
        return True   # the node owns its autonomic ops (Ocean regulates it through run_protocol)

    def run_protocol(self, command: str, args: dict) -> dict:
        """MANUAL invocation of the node's OWN regulation (Ocean calls this; a chat command can too). Routes
        to the SAME real ops: sleep/deep-sleep/consolidate CONSOLIDATE then DREAM; dream recombines basins;
        mushroom is wake-state plasticity; escape/decohere is the breakdown response. No stubs."""
        self.ensure_loaded()
        applied: Any
        if command in _MUSHROOM_SIGMA:
            self._mushroom(_MUSHROOM_SIGMA[command])
            applied = {"plasticity": f"weight-noise σ={_MUSHROOM_SIGMA[command]}"}
        elif command in ("sleep", "deep-sleep", "consolidate"):
            applied = {"consolidate": self._consolidate(steps=24 if command == "deep-sleep" else 16),
                       "dream": self._dream()}
        elif command == "dream":
            applied = {"dream": self._dream()}
        elif command in ("escape", "decohere"):
            self._decohere()
            applied = {"breakdown_recovery": "decohere noise + cool optimiser (lr×0.7)"}
        elif command == "stimulate":
            # BLOCKER-1: Ocean-commanded Pillar-1 apathy treatment. If this concrete node exposes the
            # genesis explore-temperature lever (_apply_stimulate), actuate it (shared entropy floor +
            # high-surprise-replay window). Otherwise DEGRADE HONESTLY: fire a real DREAM (basin-mixture
            # recombination — a genuine entropy-injection op the node HAS) and REPORT that the explore-temp
            # floor is unavailable on this node type. Never a silent no-op, never a false claim of acting.
            _stim = getattr(self, "_apply_stimulate", None)
            if callable(_stim):
                applied = _stim()
            else:
                applied = {"stimulate": True, "entropy_injection": self._dream(),
                           "explore_temp_lever": "unavailable-on-this-node"}
            _l = getattr(self, "_last", None)
            if _l is not None and getattr(_l, "extra", None) is not None:
                _l.extra["stimulate"] = applied
        else:
            applied = {"unknown_command": command}
        last = getattr(self, "_last", None)
        return {"command": command, "available": True, "applied": applied,
                "telemetry": last.to_dict() if last is not None else {}}

    # --- substrate hooks: the concrete target MUST override these --------------------------------------
    def _node_named_parameters(self) -> Iterable[tuple[str, Any]]:
        raise NotImplementedError("concrete node must provide _node_named_parameters()")

    def _node_device(self) -> Any:
        raise NotImplementedError("concrete node must provide _node_device()")

    def _node_rebuild_optimizer(self, lr_scale: float) -> None:
        """Rebuild the persistent training optimiser at ``lr*lr_scale`` (the decohere cool-down)."""
        raise NotImplementedError("concrete node must provide _node_rebuild_optimizer()")

    def _node_replay_optimizer(self, lr_scale: float) -> Any:
        """A FRESH, throwaway natural-gradient optimiser at ``lr*lr_scale`` for a sleep/dream replay loop
        (does NOT touch the persistent training optimiser)."""
        raise NotImplementedError("concrete node must provide _node_replay_optimizer()")

    def _node_forward_logits(self, ids: "Any", coords: "Any") -> "Any":
        """A forward pass returning ``logits[1, seq, vocab]`` for replay (coords may be None)."""
        raise NotImplementedError("concrete node must provide _node_forward_logits()")

    def _node_basin_from_logits(self, logits: "Any") -> "Any":
        """The detached vocab-width Δ basin for one step (the SAME reduction ARM B uses)."""
        raise NotImplementedError("concrete node must provide _node_basin_from_logits()")
