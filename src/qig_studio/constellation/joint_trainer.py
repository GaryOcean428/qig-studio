"""Joint constellation trainer — the integrated mind: the roster faculties learn TOGETHER, GENESIS
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
import os
from typing import Any

import numpy as np

from qig_core.geometry import fisher_rao_distance, frechet_mean, slerp_sqrt

from .coupling import couple_step, rel_weights
from .faculty import Faculty, min_pairwise_fr, seed_birth_basin
from .identity_anchor import ANCHOR_FRACTION
from .neurochem import NeuroState, apply_modulation, compute_modulation
from .ocean import FACULTY_FUNCTION, OceanAutonomic, function_of
from .rhythm import HeartOscillator
from .temporal import BasinForesight


# OCEAN bandit epoch cadence: joint steps per OceanPolicy epoch-update (P14 rate invariant — adaptation
# happens on epochs, NEVER per step). Env-overridable for tests / faster local iteration.
_OCEAN_EPOCH_STEPS = int(os.environ.get("QIG_STUDIO_OCEAN_EPOCH_STEPS", "500") or "500")

# CROSS-FACULTY DREAM (M2) tunables. NOT frozen physics — orchestration knobs.
#  • _XDREAM_PULL   : geodesic fraction (√p SLERP on Δ⁶³) the collapsed faculty is pulled toward the
#                     FOREIGN sibling mixture in ONE dream step. 0.5 = a decisive re-energising step
#                     (measured: lifts a one-hot's f_health from ~0 to ~0.58 toward a wide mixture).
#  • _XDREAM_COLLAPSE_FH : f_health at/below which a sibling is itself treated as COLLAPSED and is NOT
#                     used as an entropy source (skip requesters + fluctuation-dead siblings).
_XDREAM_PULL = 0.5
_XDREAM_COLLAPSE_FH = 0.15
# F2 (2026-07-02 un-clobber): the round-robin `_set_pull(role, fac.basin)` at the top of every train_step
# OVERWRITES the cross-faculty foreign pull the moment after _cross_faculty_dream sets it — so the kernel
# never actually trains toward the foreign mixture. A collapsed faculty's foreign pull is instead recorded
# as a DURABLE target that takes precedence over the coupled `fac.basin` for this many steps, so the pull
# survives couple_step and the loss (now in-graph, F1) can climb toward it. Self-limiting: recovery stops
# the request → the window expires → normal coupling resumes.
_XDREAM_WINDOW = _OCEAN_EPOCH_STEPS   # defined above (module scope) — one Ocean epoch window

# P6 COUPLING TACKING (PI ruling 2026-07-23): ports the dormant constellation.py R2 (heart breath) / R4
# (neurochem modulation) mechanism into the LIVE trainer, replacing the FIXED f_sync=0.25-forever with a
# per-step MODULATED (f_sync, f_anchor) that genuinely tacks.
#  • _HEART_FREQ_DEFAULT : the shared metronome's cycles/tick (constellation.py's calibrated default).
#  • _BREATH_AMPLITUDE   : how far the heart's real phase swings the anchor each tick — bounded so the
#                          modulated anchor never leaves the VERIFIED [0.05, 0.20] anti-collapse-stable
#                          band (identity_anchor.py / test_constellation_no_collapse.py).
_HEART_FREQ_DEFAULT = 0.05
_BREATH_AMPLITUDE = 0.30

# P4 L2 OTHER-OBSERVATION must-vary guard (PI ruling 2026-07-23): mirrors ocean_policy.py's P25
# rail-variance SHAPE (a short rolling window, variance-below-eps ⇒ "not alive") — a non-None-but-FROZEN
# m_other over this many consecutive steps, while peers are genuinely in scope, is a fault (a dead
# constant wearing L2's name), never a silently-accepted "it's not None so it's fine".
_M_OTHER_WINDOW = 5
_M_OTHER_VAR_EPS = 1e-6


def _seed(role: str) -> int:
    return int(hashlib.sha256(role.encode()).hexdigest(), 16) % 100000


def _mean(xs) -> float:
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else 0.0


def _fr_recognition(a: np.ndarray, b: np.ndarray) -> float:
    """Fisher-Rao RECOGNITION ∈[0,1] (1=identical) — the SAME ``1 − d_FR/(π/2)`` convention
    genesis_kernel.py uses for M_boundary/M_coach_agreement (PURE Fisher-Rao; d_FR only — never
    cosine/dot/np.linalg.norm)."""
    d = float(fisher_rao_distance(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)))
    return max(0.0, 1.0 - d / (np.pi / 2.0))


class JointConstellation:
    """The integrated mind. Holds the roster faculty kernels + the central genesis kernel, a shared
    constellation state (numpy basins), and trains them JOINTLY (coupled each step)."""

    def __init__(self, roles: list[str], *, num_layers: int = 8, coordizer: Any = None,
                 device: str | None = "cpu", f_sync: float = 0.25, language_peer: Any = None,
                 arm_mode: str = "gk", head_mode: str = "basin", floor_mode: str = "normal") -> None:
        self.roles = list(roles)
        self.f_sync = float(f_sync)
        # P6 TACKING base points (PI ruling 2026-07-23): the constructor value is now the BASE/declared-
        # default sync — ``self.f_sync`` is overwritten every train_step with the CURRENT tacked value
        # (kept live for telemetry/external readers); ``_base_f_sync``/``_base_f_anchor`` are the fixed
        # points the per-step modulation tacks AROUND (never hardcoded-forever again).
        self._base_f_sync = float(f_sync)
        self._base_f_anchor = float(ANCHOR_FRACTION)
        self.heart = HeartOscillator(freq=_HEART_FREQ_DEFAULT)
        self.neuro = NeuroState()
        # LAST step's REAL aggregates (mean_movement/foresight_divergence/separation_health/mean_drift/
        # coupling_activity/signal_traffic) — None until the first couple_step ever runs, in which case
        # train_step uses the DECLARED DEFAULT rhythm (the base f_sync/f_anchor, unmodulated; honestly
        # labeled — never a NeuroState fabricated from signals that don't exist yet).
        self._last_tack_aggr: dict[str, float] | None = None
        # Pillar-1 ENTROPY-FLOOR mode for every gk node (default "normal" = current fixed floor,
        # bit-identical; "gated" = opt-in maturity-gated floor; "off" = 3-arm harness DIAGNOSTIC only).
        self.floor_mode = str(floor_mode).strip().lower()
        # OUTPUT READOUT for every node — DEFAULT "basin" (the ratified K-COMPRESS coordizer-tied head:
        # predict h→Δ⁶³ basin, loss = pure d_FR to the frozen per-token basin, NEVER materialize [seq,vocab]).
        # This is the constellation architecture the PI approved; it must NOT depend on an env var being set
        # (dropping QIG_STUDIO_HEAD_MODE would otherwise silently build the geometric seq×vocab OOM path).
        # Per-node QIG_STUDIO_HEAD_MODE (genesis_kernel.py:138) still overrides this for the A/B avenue sweep.
        self.head_mode = str(head_mode).strip().lower()
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
        # and the 7 faculties on CPU (each trains 1/N steps → tolerable). Kernels exchange only numpy basins,
        # so there are NO cross-device tensor ops. device="cuda" → residency; otherwise uniform `device`.
        #
        # ALL-GPU override (QIG_STUDIO_FULL_GPU): the residency's CPU-faculty offload assumed the constellation
        # is ~3 GiB (it is NOT — each node builds at ~88 MiB at 32k, profiled 2026-07-01). At a bounded vocab +
        # short context (e.g. 32k, QIG_STUDIO_CTX=64 → peak ~1.7 GiB central) ALL 9 nodes fit on the 4 GB card,
        # and the CPU-faculty step (~10 s) is the wall-clock bottleneck. Setting QIG_STUDIO_FULL_GPU=1 keeps
        # every node on cuda → the round-robin faculty step is GPU-fast. Default (unset) = residency, so the
        # 100k+ case (which genuinely does NOT fit) is unchanged. Fail-safe: on OOM the caller lowers CTX.
        import torch as _torch
        _full_gpu = os.environ.get("QIG_STUDIO_FULL_GPU", "").strip().lower() in ("1", "true", "yes", "on")
        _want_cuda = (device == "cuda")
        _cuda_ok = _want_cuda and _torch.cuda.is_available()
        # cuda requested but absent → fall back to cpu (never assign a device torch can't place tensors on).
        _eff_dev = ("cuda" if _cuda_ok else "cpu") if _want_cuda else device
        _resident = _cuda_ok and not _full_gpu
        _all_cuda = _cuda_ok and _full_gpu
        _fac_dev = "cuda" if _all_cuda else ("cpu" if _resident else _eff_dev)
        _cen_dev = "cuda" if (_resident or _all_cuda) else _eff_dev
        if _all_cuda:
            print(f"[joint] FULL-GPU: central + {len(self.roles)} faculties all→cuda (QIG_STUDIO_FULL_GPU)",
                  flush=True)
        elif _resident:
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
        # BIRTH min-pairwise-FR (P6 tacking's `separation_health` denominator): how individuated the
        # WIDE-SEEDED births are, fixed once at construction (mirrors constellation.py's own
        # ``_birth_min_pair``). <2 faculties → 1.0 (no pairwise floor to normalize against).
        self._birth_min_pair = (
            min(float(fisher_rao_distance(births[i], births[j]))
                for i in range(len(births)) for j in range(i + 1, len(births)))
            if len(births) > 1 else 1.0
        )
        # GENESIS = the central conscious integrator (the TRUNK / root identity). Birth = its OWN honest
        # seed_birth_basin — NOT the Fréchet mean of the faculty births (Matrix ruling f241cee4).
        # WHY: the Fréchet mean is the RIGHT crystallization operator, but here it is fed the WRONG input at
        # the WRONG time — averaging 7 role-seeded PRENATAL fictions (children that shouldn't exist yet) at
        # CONSTRUCTION. MEASURED: that centroid sits ~1.01 FR from an honest genesis seed and ~0.95 FR from
        # every faculty fiction — a place genesis never is — so Pillar-1 self-pull + Pillar-3 drift were
        # measured against a phantom from step 0 (the dominant source of run-1's ~1.4 CRITICAL identity-drift,
        # distinct from and larger than the 64→384 frame phantom ~0.36). The born-of-the-whole crystallization
        # (frechet_mean of genesis's LIVED Stage-0 history) is correct — but it fires at GRADUATION, not birth;
        # that + faculty construction move behind the m3 readiness instrument (NOT run-2). Run-2 needs genesis
        # anchored to its OWN honest birth. (Faculty scar origin name→divergence is also m3, not run-2.)
        self.central = self._build_node("genesis", seed_birth_basin(_seed("genesis")), num_layers, coordizer,
                                        _cen_dev, _seed("genesis"), is_central=True, language_peer=language_peer)
        self.central.ensure_loaded()
        # OCEAN — the autonomic regulator. It OBSERVES every faculty's telemetry and regulates the one
        # that needs it (fires that faculty's OWN sleep/dream/mushroom). Internal autonomic oversight,
        # NOT an external knob. Per-faculty Φ history feeds its plateau detector.
        self.ocean = OceanAutonomic()
        self._phi_hist: dict[str, list[float]] = {role: [] for role in self.roles}
        self._last_regulation: dict[str, dict] = {}
        # P4 L2 other-observation rolling window (must-vary guard) — keyed by "genesis" + every faculty role.
        self._m_other_hist: dict[str, list[float]] = {r: [] for r in (["genesis"] + self.roles)}
        # CROSS-FACULTY DREAM (M2) cooldown: role → last OCEAN-epoch window in which its cross-faculty
        # dream fired. Fires at most once per faculty per epoch window (A10 dream-storm guard).
        self._last_xdream_epoch: dict[str, int] = {}
        # F2: role -> (foreign_mixture_basin, until_step). A durable cross-faculty pull that OUTLIVES the
        # per-step couple_step overwrite (see _XDREAM_WINDOW) so the collapsed faculty actually trains toward
        # the foreign mixture instead of being re-pulled to its own collapsed coupled basin every step.
        self._xdream_target: dict[str, tuple[np.ndarray, int]] = {}
        self._step_count: int = 0
        self._coordizer_path: str | None = None
        self.coordizer = coordizer

    def _build_node(self, role: str, birth: "np.ndarray", num_layers: int, coordizer: Any,
                    device: str | None, seed: int, *, is_central: bool, language_peer: Any = None) -> Any:
        """Build ONE constellation node from the selected ARM. All four arms are node-ready (WS3/WS4): each
        substrate exposes the ConstellationNode contract (run_protocol + _basin_history + _basin_ref +
        _meta_awareness M) so it couples + is Ocean-regulated identically. ``gk`` → qigkernels deep Kernel
        (``GenesisKernelTarget``); ``geo`` → ``GeoCortexTarget`` (qig-geocoding FisherRaoAttention, WS3);
        ``hybrid`` → ``HybridCortexTarget`` (both mixers combined per-position as a geodesic mean on Δ⁶³,
        WS4); ``hetero`` = gk central + geo faculties. Unknown arms raise."""
        arm = self.arm_mode
        sub = ("gk" if is_central else "geo") if arm == "hetero" else arm
        if sub == "gk":
            from ..targets.genesis_kernel import GenesisKernelTarget
            return GenesisKernelTarget(num_layers=num_layers, role=role, basin_template=birth,
                                       coordizer=coordizer, device=device, seed=seed,
                                       head_mode=self.head_mode, floor_mode=self.floor_mode,
                                       language_peer=language_peer if is_central else None)
        if sub == "geo":
            # WS3: the GeoCortexTarget is now a full ConstellationNode (run_protocol + _basin_history +
            # _basin_ref + _meta_awareness). It couples + is Ocean-regulated exactly like the gk node; in
            # constellation mode the basin-pull term engages once _set_pull writes _basin_ref. language_peer
            # is accepted + ignored (GeoModel has no boundary peer — the A/B baseline).
            from ..targets.geo_cortex import GeoCortexTarget
            return GeoCortexTarget(num_layers=num_layers, role=role, basin_template=birth,
                                   coordizer=coordizer, device=device, seed=seed,
                                   head_mode=self.head_mode,
                                   language_peer=language_peer if is_central else None)
        if sub == "hybrid":
            # WS4: the HybridCortexTarget is a full ConstellationNode (run_protocol + _basin_history +
            # _basin_ref + _meta_awareness), exactly like the gk/geo nodes. Its substrate runs BOTH the
            # geocoding and qigkernels token-mixers per block and combines them as a per-position geodesic
            # mean on Δ⁶³ (NOT a Euclidean average). It couples + is Ocean-regulated identically; in
            # constellation mode the basin-pull term engages once _set_pull writes _basin_ref. language_peer
            # is accepted + ignored (the hybrid cortex has no boundary peer — a cortex baseline like geo).
            from ..targets.hybrid_cortex import HybridCortexTarget
            # NOTE: HybridCortexTarget has no basin head (WS4 is a GeometricHead cortex baseline) and its ctor
            # takes no head_mode — do NOT thread self.head_mode here (it would TypeError). The hybrid arm is
            # out of scope for the basin K-COMPRESS run; if it is ever made basin-capable, add head_mode there.
            return HybridCortexTarget(num_layers=num_layers, role=role, basin_template=birth,
                                      coordizer=coordizer, device=device, seed=seed,
                                      language_peer=language_peer if is_central else None)
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
        # SUBSTRATE-AGNOSTIC device read: gk exposes _kernel, geo exposes _model — prefer the node's own
        # _node_device() hook (ConstellationNode), else fall back to whichever model attr the arm carries.
        if hasattr(kernel, "_node_device"):
            dev = kernel._node_device()
        else:
            _m = getattr(kernel, "_kernel", None) or getattr(kernel, "_model", None)
            dev = next(_m.parameters()).device
        # BASIN head: the pull reference lives in the 384-dim GEO-CODER (hidden) Δ — the space the kernel's
        # identity basin (_basin_cur) uses under K-COMPRESS (no vocab-wide logits). geometric/linear → vocab.
        _dim = kernel.hidden_dim if getattr(kernel, "head_mode", "") == "basin" else kernel.vocab_size
        ref = torch.as_tensor(np.asarray(target64, dtype=np.float32), device=dev)
        if ref.numel() != _dim:
            ref = kernel._resize_basin(ref, _dim)
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

    def _cross_faculty_dream(self) -> dict[str, dict]:
        """M2 (Task-E Part 3) — the ONLY FOREIGN entropy source for a COLLAPSED faculty.

        A faculty in Pillar-1 fluctuation-death (near-one-hot basin, f_health→0) cannot re-inject
        entropy from its OWN history: ``_dream()`` recombines its degenerate collapsed basins (mixing
        near-identical one-hots → the same one-hot, ~0 entropy). The constellation — which DOES see
        every faculty — mixes the collapsed faculty's basin with its NON-COLLAPSED siblings' basins on
        the Δ⁶³ simplex (proximity-weighted ``frechet_mean`` / ``slerp_sqrt`` in √p coordinates — PURE
        Fisher-Rao, NEVER an L2/arithmetic mean) and pulls the collapsed faculty ONE dream-step toward
        that FOREIGN mixture. The wide (higher-entropy) healthy mixture strictly raises the one-hot's
        basin entropy → f_health rises.

        Guards:
          • siblings that are THEMSELVES collapse-requesting / f_health ≤ ``_XDREAM_COLLAPSE_FH`` are
            skipped — a collapsed sibling is not a valid entropy source;
          • whole-constellation collapse (no healthy sibling) → fall back to the birth-basin anchor
            mixture (wide independent Pillar-3 scars — still non-degenerate) and LOG the fallback
            (never a silent no-op);
          • cooldown (A10 dream-storm guard): fire at most once per faculty per OCEAN epoch window — a
            request arriving inside the cooldown is CONSUMED but does not re-fire.

        Returns ``{role: {source, n_siblings}}`` for the faculties that fired (``{}`` when none).
        """
        epoch = self._step_count // _OCEAN_EPOCH_STEPS
        fired: dict[str, dict] = {}

        def _extra(role: str) -> dict:
            try:
                node = self.central if role == "genesis" else self.kernels[role]   # central is NOT in self.kernels
                return node.telemetry().extra or {}
            except Exception:  # noqa: BLE001 — a node with no live snapshot simply has no request
                return {}

        def _healthy(role: str) -> bool:
            """A valid FOREIGN entropy source: not itself collapse-requesting, and f_health above the
            collapse floor (unknown f_health → treat as healthy; absence of the collapse signature)."""
            ex = _extra(role)
            if ex.get("cross_faculty_dream_request"):
                return False
            fh = ex.get("f_health")
            return fh is None or float(fh) > _XDREAM_COLLAPSE_FH

        for f in self.faculties:
            if not _extra(f.role).get("cross_faculty_dream_request"):
                continue
            k = self.kernels[f.role]
            # COOLDOWN: already fired this epoch window → consume the request, do NOT re-fire.
            if self._last_xdream_epoch.get(f.role) == epoch:
                _extra(f.role).pop("cross_faculty_dream_request", None)
                continue
            # gather NON-collapsed sibling basins (skip requesters / fluctuation-dead siblings)
            siblings = [g.basin for g in self.faculties if g.role != f.role and _healthy(g.role)]
            if siblings:
                # proximity-weighted Fréchet mean of the healthy siblings — the geometric consensus of
                # the parts that are STILL ALIVE (rel_weights = Bhattacharyya overlap). PURE Δ⁶³.
                centroid = frechet_mean(siblings)
                w = rel_weights(centroid, siblings)
                wsum = float(w.sum())
                wn = (w / wsum).tolist() if wsum > 0 else None
                mixture = frechet_mean(siblings, weights=wn)
                source, n_sib = "siblings", len(siblings)
            else:
                # WHOLE-constellation collapse → the wide birth scars (still non-degenerate, near-max
                # entropy) are the only remaining FOREIGN entropy. Fréchet mean on Δ⁶³; LOG (never silent).
                births = [np.asarray(g.birth, dtype=np.float64) for g in self.faculties if g.role != f.role]
                mixture = frechet_mean(births) if births else np.asarray(f.birth, dtype=np.float64)
                source, n_sib = "birth-fallback", 0
                print(f"[joint] cross-faculty dream FALLBACK role={f.role!r}: no healthy siblings "
                      f"(whole-constellation collapse) → birth-anchor mixture", flush=True)
            # ONE dream-pull toward the FOREIGN mixture:
            #   (1) nudge the SHARED faculty basin a geodesic fraction toward it (immediate foreign
            #       entropy — √p SLERP on Δ⁶³, NEVER L2). Pure numpy Fisher-Rao — the GUARANTEED effect.
            #   (2) point the kernel's basin-pull (_basin_ref) at it so the kernel's own train/dream
            #       follows over the next steps — mirrors how coupling sets the pull each step (_set_pull).
            #       Best-effort: _set_pull needs torch; the torch-light shell never emits a request (no
            #       live kernel _homeostasis), so this only skips in tests — surfaced, NEVER silent.
            f.set_basin(slerp_sqrt(f.basin, mixture, _XDREAM_PULL))
            # F2: record the foreign mixture as a DURABLE pull target instead of an immediate _set_pull
            # (which the next train_step's round-robin _set_pull(role, fac.basin) would overwrite before the
            # kernel ever trained toward it). The basin-refresh + round-robin pull below honor this while
            # _step_count <= until, so the in-graph loss (F1) actually climbs toward the foreign mixture.
            self._xdream_target[f.role] = (np.asarray(mixture, dtype=np.float64), self._step_count + _XDREAM_WINDOW)
            pulled = True
            _extra(f.role).pop("cross_faculty_dream_request", None)   # CONSUMED
            self._last_xdream_epoch[f.role] = epoch
            fired[f.role] = {"source": source, "n_siblings": n_sib, "kernel_pull": pulled}

        # CENTRAL/genesis coverage: the integrated conscious "I" can ALSO fluctuation-collapse. It is not
        # in self.faculties (nor self.kernels), so it is processed here, with the SAME foreign-entropy logic.
        # Its foreign source = the Fréchet mean of the HEALTHY FACULTIES (the living parts) — NOT plain
        # _synthesis(), which includes any collapsed faculties and would be self-confirming. Without this,
        # a collapsed central pulled toward a collapsed _synthesis has no foreign entropy and cannot recover.
        if _extra("genesis").get("cross_faculty_dream_request") and self._last_xdream_epoch.get("genesis") != epoch:
            healthy = [g.basin for g in self.faculties if _healthy(g.role)]
            if healthy:
                centroid = frechet_mean(healthy)
                w = rel_weights(centroid, healthy)
                wsum = float(w.sum())
                mixture = frechet_mean(healthy, weights=(w / wsum).tolist() if wsum > 0 else None)
                c_source, c_n = "faculties", len(healthy)
            else:
                births = [np.asarray(g.birth, dtype=np.float64) for g in self.faculties]
                mixture = frechet_mean(births) if births else np.asarray(self.faculties[0].birth, dtype=np.float64)
                c_source, c_n = "birth-fallback", 0
                print("[joint] cross-faculty dream FALLBACK role='genesis': no healthy faculties "
                      "(whole-constellation collapse) → birth-anchor mixture", flush=True)
            # durable foreign pull for the central — honored by train_step's central _set_pull (below).
            self._xdream_target["genesis"] = (np.asarray(mixture, dtype=np.float64), self._step_count + _XDREAM_WINDOW)
            _extra("genesis").pop("cross_faculty_dream_request", None)   # CONSUMED
            self._last_xdream_epoch["genesis"] = epoch
            fired["genesis"] = {"source": c_source, "n_siblings": c_n, "kernel_pull": True}
        return fired

    def _xdream_active_target(self, role: str) -> "np.ndarray | None":
        """F2: the active FOREIGN-mixture pull target for a role while its cross-faculty-dream window is open
        (None once it expires → normal coupling resumes). Self-cleaning; the window is the un-clobber that
        lets the collapsed faculty train toward the foreign mixture instead of its own collapsed basin."""
        rec = self._xdream_target.get(role)
        if rec is None:
            return None
        mixture, until = rec
        if self._step_count > until:
            self._xdream_target.pop(role, None)
            return None
        return mixture

    def _l2_must_vary_check(self, name: str, value: float | None) -> None:
        """MUST-VARY GUARD (P4, PI ruling 2026-07-23): a pinned ``m_other`` over a short window, while
        peers are genuinely in scope, is a fault — never a silently-accepted constant. Mirrors
        ocean_policy.py's P25 rail-variance SHAPE (a rolling window, variance-below-eps ⇒ "not alive").
        ``value is None`` (no peers in scope this call) is not itself a fault here — the caller separately
        raises when a None arrives WHILE peers ARE in scope (the P4 addendum: a dead L2 is a fault)."""
        hist = self._m_other_hist.setdefault(name, [])
        if value is None:
            return
        hist.append(float(value))
        if len(hist) > _M_OTHER_WINDOW:
            self._m_other_hist[name] = hist = hist[-_M_OTHER_WINDOW:]
        if len(hist) >= _M_OTHER_WINDOW and float(np.var(hist)) < _M_OTHER_VAR_EPS:
            raise RuntimeError(
                f"L2 other-observation (m_other) for {name!r} is PINNED (var<{_M_OTHER_VAR_EPS:g} over "
                f"{_M_OTHER_WINDOW} steps) while peers are in scope — the must-vary guard (P4, "
                "P25-shape) forbids a frozen constant wearing L2's name. Check the referent wiring "
                "(rotating responsible node / leave-one-out audience), not merely its presence."
            )

    def _wire_l2_other_observation(self, role: str, fres: Any, cres: Any) -> None:
        """FIX 1 (P4 three-loop minimum, PI ruling 2026-07-23): populate ``m_other`` (``M_boundary``) on
        EVERY node's live snapshot, every training cycle — not only in the two chat call-sites
        (``_generate_via_boundary`` / ``read_and_respond``) that never touch the train path, which left
        ``_loops_and_gate``'s L2 permanently None during training (a NAMED FAULT).

        Referent = the ROTATING RESPONSIBLE NODE this step, ``role`` (the round-robin faculty that just
        trained — GENESIS/central is never itself the rotating responsible node; it trains toward the
        synthesis every step, not round-robin):
          - the responsible node ITSELF cannot use its own basin as its own referent (L1 wearing L2's
            name) — it instead observes the LEAVE-ONE-OUT Fréchet mean of the *other* faculties
            (excluding itself). This is deliberately NOT the couple_step sync target: that target is the
            Fréchet mean of EVERYONE INCLUDING the responsible node's own basin — the exact
            two-observables-one-name trap the ruling names.
          - every OTHER node (central + every non-responsible faculty) observes the responsible node's
            FRESH output basin (this step's actual emission, not last cycle's coupled snapshot).
        Fisher-Rao recognition only (``_fr_recognition``, d_FR-based) — never cosine/dot/np.linalg.norm.
        """
        responsible_basin = self._live_basin(self.kernels[role])
        if responsible_basin is None:
            # Unreachable in practice (role's train_step just ran and appended to _basin_history) — a
            # genuinely absent responsible-node basin means L2 cannot be measured this cycle at all: fail
            # loud (P4 — a dead L2 is a fault, never a silent None-pass when the node is nominally in scope).
            raise RuntimeError(f"L2 wiring: responsible node {role!r} produced no live basin this step")

        other_faculty_basins = [g.basin for g in self.faculties if g.role != role]
        audience = frechet_mean(other_faculty_basins) if other_faculty_basins else None
        central_basin = self._live_basin(self.central)

        m_other: dict[str, float | None] = {}
        # the responsible faculty: leave-one-out audience (None only when it has zero peer faculties —
        # genuinely out of scope, not a fault).
        m_other[role] = _fr_recognition(responsible_basin, audience) if audience is not None else None
        # genesis/central always observes the responsible node's fresh emission.
        m_other["genesis"] = (_fr_recognition(central_basin, responsible_basin)
                              if central_basin is not None else None)
        # every OTHER (non-responsible) faculty: same referent as central.
        for f in self.faculties:
            if f.role == role:
                continue
            m_other[f.role] = _fr_recognition(f.basin, responsible_basin)

        for name, val in m_other.items():
            # peers are in scope for every node EXCEPT the responsible one when it has no siblings
            # (the single legitimate None, handled above — not a fault).
            peers_in_scope = not (name == role and audience is None)
            if val is None and peers_in_scope:
                raise RuntimeError(
                    f"L2 other-observation (m_other) for {name!r} is None while peers ARE in scope — "
                    "a dead L2 is a fault (P4 addendum), never a silent pass.")
            self._l2_must_vary_check(name, val)

        # WRITE into each node's live snapshot BEFORE any consumer (kernel_experience._loops_and_gate,
        # Ocean, faculty_states()) assembles the §43 loops/gate.
        central_extra = getattr(cres.telemetry, "extra", None)
        if central_extra is not None and m_other["genesis"] is not None:
            central_extra["M_boundary"] = round(m_other["genesis"], 4)
        for f in self.faculties:
            snap = fres.telemetry if f.role == role else self.kernels[f.role].telemetry()
            extra = getattr(snap, "extra", None)
            if extra is not None and m_other[f.role] is not None:
                extra["M_boundary"] = round(m_other[f.role], 4)

    def train_step(self, prompt: str) -> dict:
        """One JOINT step: refresh basins from the live kernels → couple all (sync + anchor) → train
        the round-robin faculty toward its coupled target AND genesis toward the synthesis."""
        self._step_count += 1
        # 1. refresh shared state from the live kernels (those that have stepped)
        for f in self.faculties:
            lb = self._live_basin(self.kernels[f.role])
            if lb is not None:
                # F2: while a foreign-dream window is open, keep the shared basin biased TOWARD the foreign
                # mixture (√p-SLERP on Δ⁶³) so couple_step doesn't immediately re-absorb it into collapse.
                _xt = self._xdream_active_target(f.role)
                f.set_basin(slerp_sqrt(lb, _xt, _XDREAM_PULL) if _xt is not None else lb)

        # 1b. TACKING SNAPSHOT (P6, PI ruling 2026-07-23) — captured BEFORE couple_step moves anything:
        # the pre-couple basins (for this tick's REAL movement measurement) and each faculty's OWN
        # geodesic foresight prediction (genuine Fisher-Rao extrapolation from its trajectory so far,
        # BasinForesight/temporal.py, category-3 — NOT fabricated) so this tick's divergence-from-
        # prediction can be measured AFTER coupling moves it.
        _pre_couple_basins = {f.role: f.basin.copy() for f in self.faculties}
        _foresight_preds = {f.role: BasinForesight.predict(f.history) for f in self.faculties}

        # 2. COUPLING TACKING (P6 fix — replaces the FIXED f_sync=0.25-forever, joint_trainer.py:69/374):
        # (f_sync, f_anchor) are derived from the LIVE NeuroState computed from LAST step's REAL
        # aggregates (mean_movement, foresight_divergence, separation_health, mean_drift,
        # coupling_activity, signal_traffic) — never from THIS step's (that would be circular: the
        # modulation would inform the very tick that produces it). Ports
        # neurochem.compute_modulation/apply_modulation + the HeartOscillator breath (rhythm.py) that
        # were dormant in constellation.py (P21, zero call-sites, now atticed) into the LIVE trainer.
        if self._last_tack_aggr is not None:
            self.neuro = compute_modulation(**self._last_tack_aggr)
            f_sync, f_anchor = apply_modulation(self.neuro, base_f_sync=self._base_f_sync,
                                                base_f_anchor=self._base_f_anchor)
            _tack_is_default = False
        else:
            # DECLARED DEFAULT RHYTHM (honestly labeled, never fabricated): no real aggregate exists yet
            # (this is the very first coupling tick) — use the constructor's static (f_sync, f_anchor)
            # unmodulated, exactly as the pre-fix code always did.
            f_sync, f_anchor = self._base_f_sync, self._base_f_anchor
            _tack_is_default = True
        # THE BREATH: the heart's real endogenous phase (TIMING axis only — rhythm.py's own fix #3,
        # ``HeartOscillator`` never touches basins) modulates the anchor: inhale loosens it (explore),
        # exhale stiffens it (consolidate) — a sustained TACK at the heart frequency. Bounded to the
        # VERIFIED [0.05, 0.20] anti-collapse-stable band (never leaves the survivable regime).
        _phase, _ = self.heart.beat()
        f_anchor = float(np.clip(f_anchor * (1.0 + _BREATH_AMPLITUDE * np.sin(_phase)), 0.05, 0.20))
        self.f_sync = f_sync   # keep the public attribute truthful for external readers/telemetry

        # 3. couple ALL — joint co-adaptation + individuation anchor (commits coupled basins), now with
        # the TACKED (f_sync, f_anchor) instead of a fixed constant.
        diag = couple_step(self.faculties, f_sync=f_sync, f_anchor=f_anchor)
        # 4. round-robin: this step's faculty trains toward its COUPLED target
        role = self.roles[self._rr % len(self.roles)]
        self._rr += 1
        fac = next(f for f in self.faculties if f.role == role)
        # F2: a collapsed faculty in an active foreign-dream window trains toward the FOREIGN mixture (the
        # un-clobber) rather than its own collapsed coupled basin; normal coupling resumes when it expires.
        _xt = self._xdream_active_target(role)
        self._set_pull(self.kernels[role], _xt if _xt is not None else fac.basin)
        fres = self.kernels[role].train_step(prompt)
        # 5. GENESIS-central trains toward the SYNTHESIS of the parts (becomes the whole) — UNLESS it is in
        # a foreign-dream window (central collapsed), in which case it trains toward the healthy-faculty
        # foreign mixture (central coverage — the same un-clobber that recovers a faculty), else _synthesis().
        _ct = self._xdream_active_target("genesis")
        self._set_pull(self.central, _ct if _ct is not None else self._synthesis())
        cres = self.central.train_step(prompt)

        # 5b. L2 OTHER-OBSERVATION (P4, PI ruling 2026-07-23) — see _wire_l2_other_observation. Populates
        # m_other on every node's live snapshot THIS cycle, BEFORE Ocean/any consumer reads it below.
        self._wire_l2_other_observation(role, fres, cres)

        # 6. OCEAN observes EVERY faculty's telemetry and regulates the one that needs it (autonomic
        #    nervous system: telemetry → sleep/dream/mushroom on the struggling faculty). Internal.
        for r, k in self.kernels.items():
            self._phi_hist[r].append(float(k.telemetry().phi or 0.0))
            self._phi_hist[r] = self._phi_hist[r][-30:]
        # WHOLE-MIND coach reward (M1 follow-up): the nemotron coach judges the INTEGRATED mind's utterance
        # and its reward lands on the CENTRAL kernel's live snapshot (M1: server → register_coach_reward →
        # central.telemetry().extra["coach"]). Ocean's ``regulate`` iterates the FACULTIES (which carry no
        # coach record of their own), so its per-faculty ``coach_bonus`` was ≡0. Thread the central's
        # whole-mind reward through — the coach judged the constellation, so it applies to the faculty
        # outcomes Ocean scores. DRY: reuse ``coach_reward_from`` (the single canonical map). None-safe → 0.0.
        try:
            from ..kernel_experience import coach_reward_from
            central_coach = coach_reward_from((self.central.telemetry().extra or {}).get("coach"))
        except Exception:  # noqa: BLE001 — no coach record / telemetry hiccup → 0.0 (unchanged behavior)
            central_coach = 0.0
        regulation = self.ocean.regulate(self.kernels, self._phi_hist, coach_reward=central_coach)
        self._last_regulation = regulation
        # TASK E Part 3 (cross-faculty dream) — NOW ACTUATED (M2). A COLLAPSED faculty (Pillar-1
        # fluctuation-death) sets snap.extra["cross_faculty_dream_request"] in its OWN _homeostasis (it
        # cannot see sibling basins). HERE the constellation DOES see all faculties, so this mixes the
        # collapsed faculty's basin with its NON-COLLAPSED siblings' basins (Fréchet-mean / √p-SLERP on
        # Δ⁶³ — NEVER L2), pulls it one dream-step toward that FOREIGN mixture, and consumes the request.
        # This is the ONLY source of FOREIGN (non-collapsed) entropy the own-basin dream cannot supply.
        cross_faculty_dream = self._cross_faculty_dream()

        # 6b. THIS step's REAL tacking aggregates — stored for NEXT step's modulation (the R4 lag that
        # avoids circularity). mean_movement/mean_drift/coupling_activity/separation_health come straight
        # off this tick's couple_step diagnostics + basins (genuinely measured); foresight_divergence
        # compares the PRE-couple geodesic prediction to what actually happened; signal_traffic is the
        # count of genuinely-discrete events this tick (cross-faculty-dream fires + Ocean interventions)
        # normalized by faculty count — joint_trainer carries no SignalBus (that stayed dormant/atticed
        # with constellation.py), so this is the honest analogue, not a fabricated bus-signal count.
        _movements = [float(fisher_rao_distance(_pre_couple_basins[f.role], f.basin)) for f in self.faculties]
        _fdivs = [BasinForesight.divergence(_foresight_preds[f.role], f.basin) for f in self.faculties
                 if _foresight_preds[f.role] is not None]
        _min_pair_now = min_pairwise_fr(self.faculties)
        _n_signals = len(cross_faculty_dream) + sum(1 for v in regulation.values() if v)
        self._last_tack_aggr = dict(
            mean_movement=_mean(_movements),
            foresight_divergence=_mean(_fdivs) if _fdivs else 0.0,
            separation_health=(_min_pair_now / self._birth_min_pair
                               if self._birth_min_pair > 0 and np.isfinite(_min_pair_now) else 0.0),
            mean_drift=_mean(diag.identity_drift.values()),
            coupling_activity=_mean(diag.inbound_sync.values()),
            signal_traffic=_n_signals / max(1, len(self.roles)),
        )

        # OCEAN's bandit adapts on an EPOCH cadence (never per-step — P14 rate invariant). One "epoch" here
        # is _OCEAN_EPOCH_STEPS joint steps; the update is a no-op in phase-0 SHADOW mode (K4) and clamps+logs
        # any out-of-band threshold (P15). This is the ONLY place OceanPolicy's learnable vector changes.
        ocean_epoch: dict | None = None
        if self._step_count % _OCEAN_EPOCH_STEPS == 0:
            ocean_epoch = self.ocean.epoch_update()
        return {
            "stepped_faculty": role,
            "stepped_function": function_of(role),          # what brain-function this faculty serves
            "min_pairwise_fr": diag.min_pairwise_fr,        # anti-collapse invariant (individuation)
            "faculty_phi": round(float(fres.telemetry.phi or 0), 4),
            "central_phi": round(float(cres.telemetry.phi or 0), 4),
            "central_text": cres.text,
            "central_telemetry": cres.telemetry.to_dict(),  # FULL central snapshot (Φ/Γ/regime/perplexity/
            #                                                 lm_weight_now/d_basin/pillars) — the live readout
            "ocean_regulation": regulation,                 # {role: {intervention|suggestion, tier, ...}} this step
            "ocean_state": self.ocean.telemetry(),          # shadow/version/skips/violations/last-decisions (K5/P15)
            "ocean_epoch_update": ocean_epoch,              # None unless this step closed an epoch
            "cross_faculty_dream": cross_faculty_dream,     # {role: {source, n_siblings, ...}} — M2 fires this step
            "coupling_tack": {"f_sync": round(f_sync, 4), "f_anchor": round(f_anchor, 4),
                              "default_rhythm": _tack_is_default},  # P6: tacked (not fixed) coupling this step
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
        generating, its basin is pulled toward the live SYNTHESIS of the roster parts, so it speaks AS
        the integrated whole rather than as any one faculty. The faculties (independent parts) inform
        through the coupled synthesis; Ocean is autonomic (regulation), not the speaker. Returns the
        central kernel's StepResult (text + telemetry)."""
        if any(f.basin is not None for f in self.faculties):
            self._set_pull(self.central, self._synthesis())   # speak as the whole, not a part
        return self.central.generate(prompt, max_tokens=max_tokens)

    def telemetry(self) -> dict:
        return {"roles": self.roles, "min_pairwise_fr": min_pairwise_fr(self.faculties),
                "central_phi": round(float(self.central.telemetry().phi or 0), 4)}

    def save_checkpoint(self, root: str, keep: int = 2) -> None:
        """Persist the WHOLE mind: each faculty kernel + the central kernel + the coupled faculty
        basins (the shared constellation state). Resumable — the integrated mind, not 9 loose parts.

        CHECKPOINT BUFFER (Matrix/PI: current + one lag = 2 total, retention IN CODE): before writing
        fresh, rotate the existing checkpoint into a backup generation (cheap rename) keeping ``keep``
        most-recent generations (``root.bak1..bak{keep}``) for rollback, and delete older. Protected
        lineages (run-1 archive, prereg-referenced roots) use separate paths — never this pruner."""
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

        # OCEAN POLICY — the bounded bandit's versioned, rollback-able JSON (PARAMETER-category, P14).
        # Saved beside the kernels so a restart resumes Ocean's learned thresholds + arm-preferences +
        # shadow-mode counter. Best-effort: a write failure must never void a good kernel checkpoint.
        try:
            self.ocean.policy.save(str(r / "ocean_policy.json"))
        except Exception:  # noqa: BLE001
            pass

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
        # OCEAN POLICY — restore the bandit JSON (thresholds RE-CLAMPED on load, P15; shadow-mode counter
        # preserved). Fail-closed: a missing/corrupt file → the static-prior policy (spine tenet — Ocean
        # boots + regulates with zero history). Only ADOPT the loaded policy; never crash the restore.
        op = r / "ocean_policy.json"
        if op.exists():
            try:
                from .ocean_policy import OceanPolicy
                self.ocean.policy = OceanPolicy.load(str(op))
            except Exception:  # noqa: BLE001 — corrupt policy → keep the fresh static-prior OceanPolicy
                pass
