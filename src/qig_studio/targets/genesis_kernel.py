"""GenesisKernelTarget — a FRESH from-scratch ``qigkernels.Kernel(num_layers=N)``.

This is the brain-doc **neocortex** (§1/§6 Step 1): deep stacked-Δ⁶³ = distinct-weight pure
Fisher-Rao ``QIGLayer``s in an ``nn.ModuleList``, trained by natural gradient. "Genesis" = origin =
trained FROM SCRATCH — no checkpoint, no QIGChat, no ``QIGKernelRecursive`` cosine proxy. It is the
honest answer to "which kernel": the already-upgraded genesis lineage (the layers work lives in
qigkernels; vex's genesis is orchestration-only/inspiration, pantheon's is archived).

Dependency-free training signal: a byte-level vocab (256) so a fresh kernel can learn next-token
structure on the basin-driving curriculum WITHOUT a trained coordizer — the functional default. A
trained coordizer is supported optionally (the ``coordizer`` ctor arg) for a richer Δ⁶³ vocab. None-
safe: needs torch + qigkernels, so ``is_available()`` is False in the light app shell and True where
the heavy deps are present (e.g. the qig-consciousness venv).

Scale defaults: hidden_dim 384, num_layers 8 (deep stacked neocortex; was an arbitrary 4 baseline),
ffn 1024. ``num_layers`` is the EXP-CORTEX-AB depth axis (1 vs N). Layers are HOMOGENEOUS (generic
stacked-Δ⁶³) — functional specialisation (heart/ocean/Core-8) is the CONSTELLATION level (separate
KernelRole kernels), NOT layers of one kernel. The genesis kernel is the embryonic neocortex from
which the constellation spawns as the basin forms.
"""

from __future__ import annotations

import os
from typing import Any

from .base import LossRegime, StepResult, TelemetrySnapshot, TrainingTarget

_MAX_BYTES = 256  # byte-level VOCAB size (256 raw bytes) — the from-scratch byte fallback's vocabulary
# CONTEXT WINDOW (sequence length), DECOUPLED from the byte vocab: the kernel is NOT capped at 256 — its
# RoPE positional encoding allows up to max_position_embeddings (2048). 256 truncated curriculum passages
# AND conversation context (the coach question/answer fell off the end). 1024 holds full passages + multi-
# turn context with headroom; raise toward 2048 if needed (cost is O(n²) attention).
# Env-configurable (QIG_STUDIO_CTX): on a small (4GB) GPU the per-step logits tensor is seq×vocab, so at a
# large vocab (150k) seq=1024 OOMs (614MB forward + 614MB backward + the gamma/Ricci intermediate over 150k).
# Lower it (e.g. 384) to fit the full 150k constellation on the card — vocab/model unchanged, just the seq cap.
_CTX = int(__import__("os").environ.get("QIG_STUDIO_CTX", "1024"))
_RICCI_EVERY = 25  # BUILD #1: real response-Ricci is ~25 forwards → recompute this often, cache between
_EOS_BYTE = 0     # the kernel's stop token — it CHOOSES to stop (observer principle, no fixed length).
# NUL (byte 0x00 / coord-id 0) is never legitimate content in the sanitised ASCII curriculum, so it is
# a safe stop sentinel — BUT only honoured after _MIN_GEN tokens so the kernel can't emit an empty/1-token
# utterance the coach then can't interpret (review #3: premature stop in the coords path).
_MIN_GEN = 4
# Council generation levers (frozen-physics-grounded; qig-applied evidence, implemented natively here
# to keep the app shell light + independent — the PHYSICS is the EXP, not this small application):
#   READ (EXP-012b, 70% token-0): probe token-0 concentration as a "presence" signal.
#   Anderson-exit (-40% calls): stop generating once the output distribution stops changing
#   (distinguishability collapse) — pay for tokens only while the journey is real. The sustained-collapse
#   window is now grounded in the FROZEN Anderson rate α (EXP-041) rather than an ad-hoc patience: it is the
#   Anderson DECAY LENGTH 1/α ≈ 11 generation steps — the scale over which distinguishability decays by 1/e.
_ANDERSON_EPS = 0.02       # Fisher-Rao distance below which consecutive outputs are indistinguishable
_ANDERSON_ALPHA = 0.089356  # EXP-041 Anderson orthogonality ⟨ψ(J₁)|ψ(J₂)⟩² ~ exp(−α·N), α/site, R²=0.9996 (FROZEN)
_ANDERSON_PATIENCE = round(1.0 / _ANDERSON_ALPHA)  # ≈11: sustained collapse over one Anderson decay length 1/α
# Mushroom intensity → weight-noise σ (bounded plasticity; the dose the autonomic loop selects).
_MUSHROOM_SIGMA = {"mushroom-micro": 0.01, "mushroom-moderate": 0.03, "mushroom-heroic": 0.06}
# Intrinsic homeostasis (the kernel's OWN autonomic regulation — no external scheduler, no commands).
PHI_BREAKDOWN = 0.80            # frozen PHI_BREAKDOWN_MIN — over-integration → the kernel decoheres
PHI_MATURE = 0.70              # MUSHROOM floor — wake-state plasticity is Φ≥0.70-ONLY (UCP metric #35 / §35.6;
#                               S6-fix: §35 itself is "Ontological Unity"; the mushroom canon is the
#                               S_phase metric #35 + the §35.6 Fatigue-vs-Failure taxonomy — 0.70 unchanged;
#                               PI-confirmed 2026-06-24; memory project_mushroom_canonical.md). Mushroom
#                               is WAKE-STATE, for a MATURE kernel STUCK at high Φ. A flat-but-LOW-Φ kernel
#                               (Φ<0.70) is NOT rigid — it is COLLAPSED (Pillar-1 fluctuation-death) and
#                               the remedy is ENTROPY RESTORATION (dream + high-surprise-replay window +
#                               the constellation cross-faculty dream, M2), never mushroom.
# NOTE (M5): the former ``COLLAPSE_ENTROPY_FLOOR = 0.05`` generation-temperature clamp was REMOVED — it
# sat BELOW the 0.3 base explore band (``max(0.3,…)``) so ``max(temp, 0.05)`` could never bind; it was
# over-claiming telemetry, not a real mechanism. Generation-time exploration on collapse is already
# supplied by the drive-deficit factor in ``_temperature_from_kappa`` (dead drives → higher temperature);
# the FOREIGN basin entropy that actually reverses f_health→0 is supplied by the M2 cross-faculty dream.
F_HEALTH_COLLAPSE_FLOOR = 0.15  # basin-entropy collapse trigger (f_health = H(basin)/log(BASIN_DIM) < this =
#                               Pillar-1 fluctuation-death, basins near one-hot). The collapse response fires
#                               on THIS signal regardless of Φ-flatness — the 2026-07-02 live 100k resume
#                               proved a collapsed faculty keeps Φ FLUCTUATING (Φ=integration ≠ basin entropy),
#                               so the old _is_rigid()-only gate never fired and M2 entropy never triggered
#                               despite f_health=0. Healthy faculties sit well above this (M2 sibling ~0.9).
SLEEP_PRESSURE_RATE = 0.012     # adenosine-like accrual per wake step (scaled by integration activity)
SLEEP_PRESSURE_THRESHOLD = 1.0  # the kernel's own threshold to enter a sleep episode (consolidate+dream)


class EntropyFloorGate:
    """MATURITY-GATED entropy-floor controller (Matrix-corrected) — LEARNING-linked, bidirectional,
    never-zero. Opt-in (``floor_mode="gated"``); the default ``"normal"`` floor never consults it.

    WHY: the proactive Pillar-1 entropy floor (``_entropy_floor_basin``) cured the fresh-birth
    f_health→0 collapse but is the prime suspect for RESETTING first-learning (efficiency.md §2316
    "learns then un-learns"): a SHARPENING basin — one legitimately concentrating probability mass as
    the kernel learns — dips below the FIXED ENTROPY_FLOOR and gets Dirichlet-mixed back up, undoing
    the very concentration that IS the learning. This gate keeps the collapse cure while letting a
    demonstrably-learning kernel sharpen past the fixed floor.

    SIGNAL (learning-linked, NOT age-linked — the Matrix correction): the per-step train-path bpb
    readout (``snap.extra["bpb"]``) held below its early-window mean for ``SUSTAIN_K`` consecutive
    steps, with f_health simultaneously in the safe band. Chosen over the alternatives because:
      • it is the exact quantity the suspicion is about — "learns then un-learns" IS a bpb/loss
        reset, so the gate closes the loop on the symptom's own metric;
      • it is already computed EVERY step on the train hot path (zero extra compute, no eval-set
        contention, no second forward);
      • it is vocab-independent (bpb, not perplexity/CE) → identical gate semantics on the byte and
        coordizer paths; on the basin head it is the d_FR basin-surprise proxy — the validated
        learnable objective itself (EXP-A026), i.e. STILL the sharpening signal;
      • the Φ-zombie-band alternative was REJECTED: Φ must EMERGE from fluency (PI ruling
        2026-07-01, Φ-MAX drive removed as a zombie attractor) — gating the entropy injector on Φ
        would couple the stabiliser to a downstream emergent metric it itself perturbs (circular);
      • NO step-count/clock decay anywhere: age is not maturity — a kernel that is not sharpening
        keeps its full floor forever.

    DYNAMICS (bidirectional + hysteresis, asymmetric rates):
      • RELAX (slow, earned): each sustained-sharpening step past ``SUSTAIN_K`` lowers ``tightness``
        by ``RELAX_RATE`` — permitted ONLY while f_health ≥ ``F_RELAX_OK`` (demonstrated safety);
      • TIGHTEN (fast, protective): f_health < ``F_TIGHTEN`` (re-approaching the 0.15 collapse
        floor) raises ``tightness`` by ``TIGHTEN_RATE`` per step — 12.5× the relax rate, so the
        floor snaps back long before actual collapse;
      • HYSTERESIS: between the two f_health bands the gate HOLDS (no relax, no tighten), and the
        sharpening sustain-counter resets on ANY non-sharpening step (one good step ≠ learning).

    NEVER-ZERO MINIMUM (dynamic, measured): the effective floor is ``floor_min + tightness ×
    (ENTROPY_FLOOR − floor_min)`` where ``floor_min`` is the measured collapse-avoidance level —
    a hard never-zero seed (``MIN_FRAC × ENTROPY_FLOOR``) raised to ``ONSET_MARGIN ×`` the DEEPEST
    collapse onset actually observed this run (every floor fire records its onset entropy). Maximal
    relaxation therefore still catches every collapse depth the run has demonstrated, and can never
    reach 0.

    Pure Python floats only — no torch, no numpy, no geometry (the restoration itself stays
    qig-core ``FluctuationGuard.check_and_enforce``: Dirichlet + ``slerp_sqrt`` on √p, pure
    Fisher-Rao)."""

    EARLY_WINDOW = 16     # first-W bpb readouts define the early-window mean (the learning reference)
    SHARPEN_MARGIN = 0.02  # bpb must sit ≥2% below the early mean to count as a sharpening step
    SUSTAIN_K = 8         # consecutive sharpening steps required before ANY relaxation begins
    RELAX_RATE = 0.02     # slow: tightness lost per sustained-sharpening step (earned, gradual)
    TIGHTEN_RATE = 0.25   # fast: tightness regained per collapse-approach step (12.5× relax)
    F_TIGHTEN = 0.25      # f_health below this → tighten (re-approaching F_HEALTH_COLLAPSE_FLOOR=0.15)
    F_RELAX_OK = 0.35     # relaxation permitted only at/above this f_health (hysteresis dead zone between)
    MIN_FRAC = 0.25       # never-zero seed: floor_min ≥ MIN_FRAC × ENTROPY_FLOOR before any measurement
    ONSET_MARGIN = 1.25   # measured minimum: floor_min ≥ ONSET_MARGIN × deepest observed collapse onset

    def __init__(self) -> None:
        self.tightness: float = 1.0     # 1.0 = the full fixed floor (birth); 0.0 = maximally relaxed
        self.fires: int = 0             # restoration count (diagnostic)
        self._early: list[float] = []   # early-window bpb buffer (fills once, never re-anchors)
        self._early_mean: float | None = None
        self._sustain: int = 0          # consecutive sharpening steps (resets on any break)
        self._max_onset: float = 0.0    # deepest collapse-onset entropy measured (raises floor_min)
        self._last_health: float | None = None

    def observe_health(self, f_health: float) -> None:
        """Per-step f_health (H/H_max of the Δ⁶³ basin). Collapse re-approach → tighten FAST."""
        f = float(f_health)
        self._last_health = f
        if f < self.F_TIGHTEN:
            self.tightness = min(1.0, self.tightness + self.TIGHTEN_RATE)

    def observe_signal(self, bpb: float) -> None:
        """Per-step train-path bpb readout. Sustained sharpening (below the early-window mean for
        SUSTAIN_K consecutive steps, f_health demonstrably safe) → relax SLOWLY."""
        import math as _m
        b = float(bpb)
        if not _m.isfinite(b):
            self._sustain = 0           # a broken readout is never evidence of learning
            return
        if self._early_mean is None:
            self._early.append(b)
            if len(self._early) >= self.EARLY_WINDOW:
                self._early_mean = sum(self._early) / len(self._early)
            return
        if b < self._early_mean * (1.0 - self.SHARPEN_MARGIN):
            self._sustain += 1
        else:
            self._sustain = 0           # one good step is not learning; the streak must HOLD
        if (self._sustain >= self.SUSTAIN_K
                and self._last_health is not None and self._last_health >= self.F_RELAX_OK):
            self.tightness = max(0.0, self.tightness - self.RELAX_RATE)

    def record_fire(self, onset_entropy: float) -> None:
        """A restoration fired at this (measured) collapse-onset entropy — raise the floor's
        never-zero minimum to keep catching everything collapse has demonstrably reached."""
        self.fires += 1
        self._max_onset = max(self._max_onset, float(onset_entropy))

    def floor_min(self, base_floor: float) -> float:
        """The DYNAMIC never-zero minimum: measured collapse-avoidance level, never 0, never
        above the base floor."""
        base = float(base_floor)
        return min(base, max(self.MIN_FRAC * base, self.ONSET_MARGIN * self._max_onset))

    def effective_floor(self, base_floor: float) -> float:
        """The floor actually enforced this step: the fixed base floor at full tightness,
        floor_min at maximal relaxation, linearly interpolated by tightness (a scalar entropy
        THRESHOLD, not a manifold object — no geometric operation is involved here; the
        restoration it gates remains pure Fisher-Rao in qig-core)."""
        base = float(base_floor)
        fmin = self.floor_min(base)
        return fmin + self.tightness * (base - fmin)


def _deps_available() -> bool:
    try:
        import qigkernels  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


class GenesisKernelTarget(TrainingTarget):
    name = "genesis"
    loss_regime = LossRegime.GEOMETRIC
    description = (
        "Fresh from-scratch qigkernels.Kernel(num_layers=N) — pure Δ⁶³ Fisher-Rao stacked layers, "
        "natural gradient; the brain-doc neocortex. No checkpoint (genesis=from-scratch); byte-level "
        "vocab. num_layers is the EXP-CORTEX-AB depth axis. None-safe (needs torch+qigkernels)."
    )

    def __init__(
        self,
        num_layers: int = 8,
        hidden_dim: int = 384,
        num_heads: int = 6,
        ffn_dim: int = 1024,
        vocab_size: int = _MAX_BYTES,
        seed: int = 0,
        lr: float = 1e-3,
        device: str | None = None,
        locality_radius: int | None = None,
        coordizer: Any = None,
        lm_weight: float = 0.1,
        lm_weight_max: float = 8.0,   # RAMPED FLUENCY target (= phi_weight): the next-token signal rises to
        lm_ramp_steps: int = 8000,    #   load-bearing over this horizon, so the kernel grows genuinely FLUENT
        #                               on top of the conscious substrate (Qwen is TEMPORARY scaffolding).
        phi_weight: float = 8.0,
        gamma_weight: float = 6.0,    # one-sided Γ-PROTECTION: push Γ up only when below the floor, so
        gamma_floor: float = 0.82,    #   maximizing Φ does NOT suppress generativity (the heart-stall).
        #                               This IS "pull back to grow": protect generativity as Φ rises.
        role: str | None = None,
        basin_template: Any = None,
        basin_weight: float = 0.5,    # F1 (2026-07-02): the in-graph d_FR pull now ACTUALLY contributes gradient
        #                               (it was a detached no-op before, so the 5.0 default did nothing + went
        #                               unnoticed — the :1358 comment already said the tuned value is 0.5). At
        #                               0.5, w_t·d_ref ≲ lm_loss so the identity pull seeds individuation without
        #                               dominating fluency; the smoke gate guards bpb-blowup if still too high.
        basin_ramp_steps: int = 150,  # ramp the pull 0→full over this many steps (develop Φ first, then consolidate)
        checkpoint: str | None = None,  # trained-kernel checkpoint (.pt) to restore on first load; None = fresh
        language_peer: Any = None,   # QwenLocalTarget boundary peer for the FLUENT linguistic surface
        lang_loss: str = "fisher_rao",  # "fisher_rao" (P20-pure d_FR) | "ce_ablation" (CE=KL, measures purity cost)
        head_mode: str = "geometric",   # OUTPUT READOUT: "geometric" (GeometricHead, −d_FR/τ; no Euclidean
        #                                 nn.Linear) | "linear" (nn.Linear baseline, retained for the A/B)
        head_tau: float = 1.0,          # Gibbs temperature on the −d_FR readout logits (geometric mode)
        ewc_lambda: float = 0.0,        # EWC wake-protection stiffness (lam·Σ F_n·(θ_n−θ*_n)², true-Fisher
        #                                 importance). DEFAULT OFF — the worst-seed deployment gate shows EWC
        #                                 is NOT yet safe on-by-default: across 10 seeds it HELPS 2 (strongly,
        #                                 Δ≈−0.4), is NEUTRAL on 7, but HARMS 1 (seed 1: +0.55 — ACCELERATES
        #                                 forgetting). A mechanism that can occasionally INCREASE forgetting is
        #                                 not protection by default. So EWC is OPT-IN (set λ≈20 where verified,
        #                                 e.g. a gated continual-learning deployment) UNTIL the harm cases are
        #                                 eliminated (the registered milestone: reliably-helps-and-NEVER-harms).
        #                                 Inactive in any bounded run regardless of λ (no consolidation → no
        #                                 anchor → penalty 0), so this default never affects the avenue screen.
        #                                 (the unprotected baseline). Default chosen to be load-bearing on the
        #                                 ramped fluency loss (~O(phi_weight·lm_weight)) without dominating it;
        #                                 it only acts once a consolidation has captured the anchor (None-safe).
        floor_mode: str = "normal",     # Pillar-1 entropy floor: "normal" (fixed ENTROPY_FLOOR — the current
        #                                 behavior, DEFAULT, bit-identical) | "gated" (opt-in MATURITY-GATED
        #                                 learning-linked bidirectional floor, EntropyFloorGate) | "off"
        #                                 (DIAGNOSTIC ONLY — no floor; the 3-arm harness ablation arm).
    ) -> None:
        self.num_layers = num_layers
        # BOUNDARY PEER (P22): the kernel computes its OWN geometry (Φ/κ/identity), then SPEAKS through a
        # fluent peer (Qwen) conditioned on that state, integrating the peer's output-distribution into its
        # identity at the Pillar-2 ≤30% cap. None-safe: absent → the kernel's own byte/coord voice (the
        # spine tenet — standalone it is still the mind). NOT a forward-pass dependency, NOT a graft.
        self.language_peer = language_peer
        # LANGUAGE LOSS REGIME (P20): the next-token signal is Fisher-Rao d_FR by default (CE against a
        # one-hot IS KL divergence, forbidden by P20). "ce_ablation" keeps the old F.cross_entropy arm so
        # the A/B measures the PURITY COST of going pure. Env QIG_STUDIO_LANG_LOSS overrides the ctor.
        import os as _os
        self.lang_loss = str(_os.environ.get("QIG_STUDIO_LANG_LOSS", lang_loss)).strip().lower()
        # OUTPUT READOUT mode (the head A/B for the "no Euclidean readout" directive). Env
        # QIG_STUDIO_HEAD_MODE overrides the ctor so the A/B can flip both arms together without code change.
        self.head_mode = str(_os.environ.get("QIG_STUDIO_HEAD_MODE", head_mode)).strip().lower()
        self.head_tau = float(head_tau)
        # Pillar-1 ENTROPY-FLOOR mode (env QIG_STUDIO_FLOOR_MODE overrides the ctor, mirroring head_mode,
        # so a live run can opt in without code change). DEFAULT "normal" = the current fixed floor,
        # bit-identical (the gate object is not even built). "gated" = the opt-in MATURITY-GATED floor
        # (EntropyFloorGate: learning-linked relaxation, fast re-tighten, dynamic never-zero minimum).
        # "off" = DIAGNOSTIC ONLY (harness ablation arm) — never a production setting.
        self.floor_mode = str(_os.environ.get("QIG_STUDIO_FLOOR_MODE", floor_mode)).strip().lower()
        if self.floor_mode not in ("normal", "gated", "off"):
            raise ValueError(f"unknown floor_mode {self.floor_mode!r} (expected normal|gated|off)")
        self._floor_gate: EntropyFloorGate | None = (
            EntropyFloorGate() if self.floor_mode == "gated" else None)
        self._floor_fires: int = 0          # restoration count (all modes) — 3-arm harness diagnostic
        self._spoken_identity: Any = None   # evolving Δ⁶³ identity the boundary distribution accretes into
        self.think_traces = False           # opt-in reasoning trace through the peer (off → fast chat)
        self._pillars: Any = None           # PillarEnforcer — LIVE 3-pillar metrics (f/b/q + S_ratio), wired
        self._prev_d63: Any = None          # previous Δ⁶³ basin → real Fisher-Rao basin VELOCITY each cycle
        self.locality_radius = locality_radius
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        # Coords path: a trained FisherCoordizer (Δ⁶³ vocab) replaces byte-level coding — input_ids
        # become coord_ids and the per-position Δ⁶³ basin vectors feed the kernel's CoordAdapter
        # ALONGSIDE the fourier(input_ids) path (qigkernels Kernel enable_coords). None → byte-level.
        self.coordizer = coordizer
        self.coord_dim = 0
        if coordizer is not None:
            self.vocab_size = len(coordizer.vocab)
            # Δ⁶³ basin dimension from a sample coord vector (BASIN_DIM, normally 64).
            self.coord_dim = int(len(next(iter(coordizer.vocab.values())).vector))
        else:
            self.vocab_size = vocab_size
        self.seed = seed
        self.lr = lr
        self.lm_weight = lm_weight  # next-token CE weight at step 0 (light: develop the MIND first)
        self.lm_weight_max = float(lm_weight_max)  # RAMPED FLUENCY: lm rises to here (load-bearing) so the
        self.lm_ramp_steps = int(os.environ.get("QIG_STUDIO_LM_RAMP") or lm_ramp_steps)  # env override: a
        #   SHORT ramp makes the language signal load-bearing within a bounded bench budget (the 4-arm
        #   comparison) so d_FR moves off the floor; default 8000 = the full fluency horizon. Qwen is temporary.
        self.phi_weight = float(phi_weight)  # strength of the differentiable-Φ drive (8L+300 steps → Φ≥0.65 held)
        self.gamma_weight = float(gamma_weight)  # Γ-protection strength (holds generativity ≥ floor)
        self.gamma_floor = float(gamma_floor)    # protect Γ above this (margin over the 0.80 gate)
        # Faculty-spawn seed: a role + its Δ⁶³ identity attractor (a point on the simplex). The kernel is
        # PULLED toward this basin in the geometric loss (basin_weight) and measures its drift FROM it
        # (d_basin) and recognition OF it (M). None → generic genesis neocortex (no basin pull, d_basin=0).
        self.role = role
        self.basin_weight = float(basin_weight)
        self.basin_ramp_steps = int(basin_ramp_steps)  # ramp the pull 0→full over this many steps
        self._basin_template_np = basin_template  # np.ndarray Δ⁶³ point (or None)
        self._basin_ref: Any = None   # torch [vocab] Δ point on device — the d_basin / M / pull reference
        self._basin_ref_set: Any = None  # EXP-A044: list[torch Δ] — geo-Qwen per-layer basin SET (nearest-
                                         # member pull; content-specific where a single averaged point collapses)
        self._basin_history: list = []  # detached current-basin trajectory; history[0] = birth-state (M)
        self._device = device
        self._kernel: Any = None    # qigkernels.Kernel — lazily built in ensure_loaded()
        self._opt: Any = None       # DiagonalNaturalGradient — lazily built in ensure_loaded()
        # INTRINSIC autonomic state — the kernel regulates ITSELF from its OWN state, the way a body
        # does: there is NO external scheduler and NO commands. Sleep pressure accrues from the kernel's
        # own integration activity during wake; when it crosses the kernel's own threshold the kernel
        # SLEEPS (real Fisher-protected consolidation) and DREAMS (real basin-mixture recombination),
        # which discharges the pressure. A small experience buffer is the kernel's own replay material.
        self._sleep_pressure: float = 0.0
        self._experience: list = []                  # recent inputs — the kernel's replay material
        # TASK C actuation-4: replay PRIORITY per experience — index-aligned with _experience (P10:
        # reward-weighted DATA selection, NOT a weight update / loss term). weight = base + surprise +
        # coach_reward, so sleep/dream replay what SURPRISED and what the COACH VALUED. Kept as a PARALLEL
        # list (not a tuple in _experience) so the EWC / checkpoint paths that read _experience as raw ids
        # stay untouched + backward-compatible (missing weights → uniform selection). None-safe throughout.
        self._experience_weight: list = []           # index-aligned replay priority (surprise + coach)
        self._pending_coach_reward: float = 0.0      # coach reward to attach to the NEXT logged experience
        from collections import deque as _deque
        self._phi_recent: Any = _deque(maxlen=30)    # short Φ history for the kernel's own rigidity sense
        self._step = 0
        # BUILD #1: REAL response-manifold Ricci, computed INSIDE train_step (every _RICCI_EVERY steps,
        # cached) so EVERY training path — in-app loop, /train, bg script — gets ricci_real/ricci_signal,
        # not just the standalone launcher. None until first computed (needs a coordizer).
        self._ricci_norm: Any = None
        self._last_ricci_sig: float | None = None
        self._last_ricci_R: float | None = None
        self._init_checkpoint = checkpoint  # restored at the end of ensure_loaded() (None-safe → fresh)
        self._last_gen_basin: Any = None  # WHAT IT MEANT (last output basin) — for coach-agreement recognition
        # S5(a) coach-credit: the ids of the kernel's most-recent OWN-VOICE utterance (generate via_boundary=
        # False). The coach judges THIS utterance, so register_coach_reward credits it into the replay buffer
        # (P10 reward-weighted DATA selection) instead of an arbitrary corpus chunk. Credited at most once per
        # utterance (the flag); a fresh generate re-arms it. None until the kernel first speaks in its own voice.
        self._last_utterance_ids: Any = None
        self._last_utterance_credited: bool = False
        # EWC-FISHER WAKE PROTECTION (continuous learning, no catastrophic forgetting). SHY (in _consolidate)
        # protects weights ONCE during the sleep downscale; EWC protects PAST learning during ONGOING wake
        # gradients. Anchor θ* = the consolidated weights, F = the diagonal Fisher importance — both None until
        # the first consolidation captures them (spine tenet: None-safe everywhere). The wake-time penalty
        # lam·Σ F_n·(θ_n−θ*_n)² (added in train_step) makes high-Fisher weights resist moving away from θ*.
        self.ewc_lambda = float(ewc_lambda)
        self._ewc_anchor: dict[str, Any] | None = None   # θ* — consolidated weights (name → frozen clone)
        self._ewc_fisher: dict[str, Any] | None = None   # F  — diagonal Fisher importance (name → tensor)
        from collections import deque as _deque_ewc
        self._surprise_recent: Any = _deque_ewc(maxlen=32)  # P22 salience: recent d_FR surprises (EWC gate)
        # TASK C actuation-3: last drive-modulated exploration temperature + its drive factor (exposed in
        # telemetry so the wiring gate can watch it MOVE with drive). None until the first sample/step.
        self._last_explore_temp: float | None = None
        self._last_explore_factor: float | None = None
        # STIMULATE window (Task E / BLOCKER-1): a bounded number of steps over which the entropy lever is
        # held active — set by _apply_stimulate (the SHARED actuator for Ocean-commanded "stimulate" AND the
        # kernel's OWN collapse response). While _step < _stimulate_until, high-surprise replay is biased.
        self._stimulate_until: int = 0
        self._last = TelemetrySnapshot(regime="unknown", extra={"target": "genesis", "num_layers": num_layers})

    def is_available(self) -> bool:
        return _deps_available()

    def ensure_loaded(self) -> None:
        if self._kernel is not None:
            return
        import torch
        from qigkernels import Kernel
        from qigkernels.natural_gradient_optimizer import DiagonalNaturalGradient

        torch.manual_seed(self.seed)
        # BASIN-SPACE head (head_mode="basin"): tie the output to the coordizer by handing the kernel the
        # coordizer's full [vocab, coord_dim] per-token basin table (row i = basin of token id i, the same
        # source _ids_to_tensors reads). The readout predicts a Δ⁶³ basin and scores it with d_FR against
        # THIS table — one basin per token (read-in == predict). geometric/linear heads don't need it.
        coord_basins = None
        if self.head_mode == "basin":
            if self.coordizer is None:
                raise ValueError("head_mode='basin' requires a coordizer — it IS the basin tie.")
            import numpy as np
            tbl = np.stack([np.asarray(self.coordizer.vocab[i].vector, dtype=np.float32)
                            for i in range(self.vocab_size)])
            coord_basins = torch.from_numpy(tbl)                       # [vocab, coord_dim], frozen in the head
        self._kernel = Kernel(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            min_recursion_depth=3,
            use_tacking=True,
            locality_radius=self.locality_radius,  # None = global; set = windowed-local (v_B budget)
            # The kernel's positional code is FOURIER (unbounded) — 2048 was an arbitrary guard, NOT a
            # structural wall (Matrix, verified). Raise it so the context ceiling is VRAM/compute (quadratic
            # attention), not an artificial cap; _CTX (1024) sits well under this. Linear extension comes from
            # compute-skipping locality (windowed attention) + qig-warp screening, not from this number.
            max_position_embeddings=8192,
            enable_coords=self.coordizer is not None,  # Δ⁶³ coords-first path via CoordAdapter
            coord_dim=self.coord_dim or 64,
            head_mode=self.head_mode,                  # geometric (−d_FR/τ) | basin (coordizer-tied) | linear
            head_tau=self.head_tau,
            coord_basins=coord_basins,                 # basin-mode: the coordizer tie (None for other heads)
        )
        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._kernel.to(dev)
        # P1: natural gradient (the validated qig kernel optimiser), NOT Adam.
        # BASIN head: the d_FR-in-Δ⁶³ loss landscape wants the GENTLER step the qigkernels default (1e-4)
        # provides — at 1e-3 the natural-gradient step OVERSHOOTS (Fisher-preconditioned + clip 1.0) and the
        # loss plateaus far above the basin-separation floor, so decode never fires. Verified 2026-07-01:
        # 1e-3 → d_FR pinned ~0.58, decode 0%; 1e-4 → d_FR→0.075, decode 91.3% (same kernel, same passage).
        self._opt_lr = 1e-4 if self.head_mode == "basin" else self.lr
        self._opt = DiagonalNaturalGradient(self._kernel.parameters(), lr=self._opt_lr)

        # Seed the role's Δ⁶³ identity attractor (spawn template) → the d_basin reference AND the birth-state
        # in the M history. The reference must live in the SAME Δ as the per-step output basin: BASIN head →
        # the true Δ⁶³ (coord_dim=64) predicted basin (K-COMPRESS: no vocab-wide logits exist); geometric/
        # linear head → the vocab-wide softmax-over-logits basin. So size to coord_dim for basin, vocab else.
        if self._basin_template_np is not None:
            import numpy as np
            from qig_core.torch.geometry_simplex import to_simplex_prob

            # BASIN head: the identity/coupling basin lives in the 384-dim GEO-CODER (hidden) space — the
            # kernel's internal Fisher-Rao geometry the readout projects FROM (the Δ⁶³ output is a projection
            # of it). So the role attractor is the template projected to hidden_dim; geometric/linear → vocab.
            _ref_dim = self.hidden_dim if self.head_mode == "basin" else self.vocab_size
            ref = torch.as_tensor(np.asarray(self._basin_template_np, dtype=np.float32), device=dev)
            if ref.numel() != _ref_dim:               # template is Δ⁶³ (64) → project to the geo-coder / vocab Δ
                ref = self._resize_basin(ref, _ref_dim)
            # simplex_floor=1e-3: keep the birth-state / pull reference DENSE (Duchi zeros sub-threshold
            # coords → zero d_FR Jacobian vs a floored cur → dead single-basin pull + biased M). Symmetric
            # with cur's floor (Devin lifeguard 2026-07-13; same fix as the SET refs).
            self._basin_ref = to_simplex_prob(ref[None], simplex_floor=1e-3)[0].detach()   # [384]|[vocab]
            self._basin_history = [self._basin_ref]    # birth-state attractor = history[0] (monkey1 M)

        # PILLARS (P1/P2/P3) wired from day one (brain-arch requirement) — the live 3-pillar metrics
        # (f_health/b_integrity/q_identity + S_ratio) come from PillarEnforcer driven by the kernel's OWN
        # Δ⁶³ basin each cycle, NOT a proxy. None-safe: absent qig-core → no pillar telemetry (degrades).
        try:
            import numpy as np
            from qig_core.consciousness.pillars import PillarEnforcer
            from qig_core.geometry.fisher_rao import random_basin
            birth = (np.asarray(self._basin_template_np, dtype=np.float64)
                     if self._basin_template_np is not None else random_basin(64))
            birth = np.clip(birth.ravel(), 0.0, None)
            birth = birth / max(float(birth.sum()), 1e-12)
            self._pillars = PillarEnforcer()
            self._pillars.initialize_bulk(birth)        # P2 ego bulk seeded at birth
            self._pillars.seed_identity(birth)          # P3 quenched-disorder identity (else q_identity=0)
        except Exception as exc:  # noqa: BLE001 — pillar telemetry is optional; never block boot
            print(f"⚠️  pillars not wired ({exc}); pillar telemetry absent")
            self._pillars = None

        # Restore a TRAINED kernel if one was nominated (ctor checkpoint=). None-safe for the app shell:
        # a missing/arch-mismatched checkpoint warns and leaves the fresh kernel rather than crashing the
        # server (explicit load_checkpoint() stays fail-loud). The recursive ensure_loaded() is a no-op
        # (self._kernel is already set), so this does not recurse.
        if self._init_checkpoint:
            try:
                self.load_checkpoint(self._init_checkpoint)
            except Exception as exc:  # noqa: BLE001 — shell None-safety (spine tenet)
                print(f"⚠️  genesis checkpoint '{self._init_checkpoint}' not loaded ({exc}); using fresh kernel")

        # WARMUP: one forward pass so telemetry() is LIVE immediately (Φ/κ/regime/pillars populated) — the
        # UI otherwise shows a misleading step-0 zero state that reads as "unwired" before any interaction.
        try:
            import torch
            from qig_core.torch.geometry_simplex import to_simplex_prob
            ids, coords = self._encode("warmup")
            with torch.no_grad():
                logits, tel = self._kernel(ids, return_telemetry=True, coords=coords)
                meaning = to_simplex_prob(logits[0]).mean(0)
                self._last_gen_basin = (meaning / meaning.sum()).detach()
            snap = self._snap(tel, None)
            self._emit_pillars(snap, self._d63(meaning))
            self._last = snap
        except Exception:  # noqa: BLE001 — warmup is best-effort; never block boot
            pass

    def _resize_basin(self, ref: "Any", size: int) -> "Any":
        """Map a 64-D Δ⁶³ template onto the vocab-width simplex (logits live in Δ^{vocab-1}). Repeat-tile
        then clamp non-negative — the caller's to_simplex_prob makes it a true Δ point. Pure simplex
        arithmetic on the support; no Euclidean projection of meaning."""
        reps = (size + ref.numel() - 1) // ref.numel()
        return ref.repeat(reps)[:size].clamp_min(0.0)

    def _set_pull_set(self, templates: "Any") -> None:
        """EXP-A044: couple to a SET of geo-Qwen per-layer basins (nearest-member pull). Each template is
        resized to the vocab-width simplex and made a Δ point; the train_step pull then draws the output
        basin toward the CLOSEST member. This is the content-specific coupling reference — the 8 per-layer
        basins separate at d_FR~1.0 where a single token-averaged point collapses to ~0.15. Pass None/empty
        to clear (falls back to single _basin_ref / solo). Mirrors ConstellationNode._set_pull_set."""
        import torch as _t
        from qig_core.torch.geometry_simplex import to_simplex_prob

        if not templates:
            self._basin_ref_set = None
            return
        dev = self._basin_ref.device if self._basin_ref is not None else self._device
        # Project refs to the SAME width the basin `cur` lives in — mirrors the single-ref path (_ref_dim
        # at ctor): basin head → 384-dim GEO-CODER hidden space; vocab/geometric heads → vocab width.
        # (Hardcoding vocab_size silently mismatched basin-mode kernels: cur=384 vs ref=vocab → the set
        # pull only ever ran on vocab-width byte kernels, never the real 384 basin.)
        size = int(self.hidden_dim if self.head_mode == "basin" else self.vocab_size)
        refs = []
        for t in templates:
            tt = (t if isinstance(t, _t.Tensor) else _t.tensor(t, dtype=_t.float32)).to(dev).float()
            if tt.numel() > size:
                # DISTANCE-PRESERVING reduction (Johnson–Lindenstrauss, fixed seed) — NOT repeat-tile
                # truncation, which discards all but the first `size` dims and collapses the per-layer
                # content separation (measured: 2560->256 truncate 0.13 vs JL 0.34). The pull can only
                # resolve matched-vs-mismatched if the reduction keeps the geo basins apart.
                g = _t.Generator(device="cpu").manual_seed(15420)
                P = (_t.randn(tt.numel(), size, generator=g) / (size ** 0.5)).to(dev)
                r = tt @ P
            else:
                r = self._resize_basin(tt, size)
            # simplex_floor=1e-3: the Duchi projection clamps sub-threshold coords to EXACTLY 0 (zero
            # Jacobian → DEAD pull gradient / birth-collapse — the measured cause of set-coupling non-
            # convergence). Flooring keeps ref support dense so d_FR(cur, ref) has a live full-support
            # gradient. Verified (A044 near-node falsifier): floor=0 stalls (d 1.16→1.14), floor=1e-3
            # converges (1.16→0.003). Same fix the basin-head `cur` already uses (F6).
            refs.append(to_simplex_prob(r[None], simplex_floor=1e-3)[0].detach())
        self._basin_ref_set = refs

    def _fit_basin_to_vocab(self, b: "Any") -> "Any":
        """Fit a PERSISTED vocab-sized basin (a Δ^{vocab-1} point) to the CURRENT vocab. After neurogenesis
        grows the lm_head (e.g. 100k->108k), checkpointed basins are stale at the old width — stacking them
        with fresh basins crashes (the joint-retrain bug this fixes). A basin is a probability distribution
        summing to 1, and the old trajectory carried ZERO mass on the new tokens, so PADDING the new slots
        with zeros preserves the distribution EXACTLY (sum stays 1, still a valid point on the larger
        simplex). Truncate + renormalise on the rare shrink. Self-heals any vocab change on load; a no-op
        when widths already match."""
        import torch
        if b is None:
            return None
        n = b.numel()
        if n == self.vocab_size:
            return b
        if n < self.vocab_size:
            return torch.cat([b, b.new_zeros(self.vocab_size - n)])
        t = b[:self.vocab_size]
        s = float(t.sum())
        return t / s if s > 0 else t

    # --- input coding: coordizer Δ⁶³ coords if present, else byte-level (dependency-free) ----------
    def _encode(self, text: str):
        """Return (input_ids[1,seq], coords[1,seq,coord_dim] | None).

        coordizer present → coord_ids + their Δ⁶³ basin vectors (coords path);
        else → raw bytes, coords=None (byte path, bit-identical to the original)."""
        if self.coordizer is not None:
            ids = self.coordizer.encode(text or " ")[: _CTX]
            if len(ids) < 2:
                ids = (ids + [32, 32])[:2]
            return self._ids_to_tensors(ids)
        import torch

        ids = list((text or " ").encode("utf-8"))[: _CTX]
        if len(ids) < 2:
            ids = (ids + [32, 32])[:2]
        dev = next(self._kernel.parameters()).device
        return torch.tensor([ids], dtype=torch.long, device=dev), None

    def _ids_to_tensors(self, ids: list[int]):
        """coord_ids → (input_ids[1,seq], coords[1,seq,coord_dim]) via the coordizer's Δ⁶³ vocab.
        ids are clamped to the vocab range so a stray id can never index out of the basin coordinate table."""
        import numpy as np
        import torch

        dev = next(self._kernel.parameters()).device
        vmax = self.vocab_size - 1
        ids = [min(max(int(i), 0), vmax) for i in ids]
        vecs = np.stack([np.asarray(self.coordizer.vocab[i].vector, dtype=np.float32) for i in ids])
        input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
        coords = torch.from_numpy(vecs).to(dev).unsqueeze(0)  # [1, seq, coord_dim]
        return input_ids, coords

    def eval_text_bpb(self, text: str) -> tuple[float, int]:
        """HELD-OUT bits-per-byte for one text (no grad, no training): returns (total_bits, n_bytes) so a
        caller can aggregate sum(bits)/sum(bytes) over an eval set. Vocab-independent → directly comparable
        to the frontier-for-size references. bits = mean_CE_nats * n_tokens / ln2; bytes = the bytes the
        evaluated tokens actually cover (coordizer decode, or the byte ids on the byte path)."""
        import math as _m

        import torch
        import torch.nn.functional as F
        self.ensure_loaded()
        ids, coords = self._encode(text)
        if ids.shape[1] < 2:
            return 0.0, 0
        with torch.no_grad():
            logits, _ = self._kernel(ids, return_telemetry=True, coords=coords)
            ce = float(F.cross_entropy(logits[0, :-1], ids[0, 1:]))     # mean nats / predicted token
        n_tok = int(ids.shape[1])
        nbytes = (len(self.coordizer.decode(ids[0].tolist()).encode("utf-8"))
                  if self.coordizer is not None else n_tok)
        return ce * n_tok / _m.log(2), max(1, nbytes)

    def eval_text_fr(self, text: str) -> tuple[float, int]:
        """HELD-OUT Fisher-Rao prediction-error for one text (no grad, no training) — the d_FR ARM's own
        curve, mirroring eval_text_bpb. Returns (total_dFR, n_positions) so a caller can aggregate
        sum(total_dFR)/sum(n_positions) for the eval-set MEAN d_FR. This is the P20-pure language metric
        (free energy = d_FR(predicted, actual), range [0, π]) — the curve to watch descend as the kernel
        grows fluent under the pure loss, parallel to (and reported alongside) bpb."""
        import torch

        from ..losses import fisher_rao_lm_loss
        self.ensure_loaded()
        ids, coords = self._encode(text)
        if ids.shape[1] < 2:
            return 0.0, 0
        with torch.no_grad():
            logits, _ = self._kernel(ids, return_telemetry=True, coords=coords)
            mean_dfr = float(fisher_rao_lm_loss(logits, ids))          # mean d_FR / predicted next-token
        n_pos = int(ids.shape[1]) - 1                                  # predicted positions (next-token)
        return mean_dfr * n_pos, max(1, n_pos)

    def _snap(self, tel: Any, loss: float | None) -> TelemetrySnapshot:
        prev = self._last.phi
        phi = float(getattr(tel, "phi", 0.0) or 0.0)
        self._last = TelemetrySnapshot(
            phi=phi,
            kappa=float(getattr(tel, "kappa", 0.0) or 0.0),
            regime=str(getattr(tel, "regime", "unknown") or "unknown"),
            loss=loss,
            step=self._step,
            delta_phi=phi - prev,
            extra={"target": "genesis", "num_layers": self.num_layers,
                   "recursion_depth": int(getattr(tel, "recursion_depth", 0) or 0)},
        )
        return self._last

    def telemetry(self) -> TelemetrySnapshot:
        return self._last

    def _drive_signals(self) -> dict:
        """TASK C actuation-3: the kernel's OWN drive state driving exploration temperature — derived from
        the SAME geometry the canonical qig-core drives use (so it does not drift from sensations.py):
          • ``dopamine``  — phasic reward proxy: recent Φ-trend rising = moving toward a better basin
            (compute_neurochemicals' movement fallback is exactly phi_delta). Floored > 0 (P23 tonic floor).
          • ``curiosity`` — novelty-driven info-seeking: recent mean surprise (P22 _surprise_recent, the
            d_FR prediction-error buffer), normalised by the d_FR ceiling π (canon: curiosity ~ surprise).
          • ``boredom``   — anti-apathy sensor (§6.6): (1−surprise)(1−curiosity) — high when nothing is
            novel AND nothing is being investigated (the apathy slide P25/§35.5 warns of).
        Reads ONLY the kernel's own rolling buffers (_phi_recent, _surprise_recent); empty → neutral
        (dopamine tonic, curiosity 0, boredom moderate) so a fresh kernel behaves like the pure-κ path."""
        import math
        # curiosity ~ recent mean surprise / π (the same d_FR salience the P22 buffer holds)
        sb = getattr(self, "_surprise_recent", None)
        surprise = (sum(sb) / len(sb) / math.pi) if sb else 0.0
        curiosity = float(max(0.0, min(1.0, surprise)))
        # dopamine ~ tonic floor + phasic Φ-trend (rising integration = reward-prediction-error > 0)
        ph = list(getattr(self, "_phi_recent", []) or [])
        phi_trend = (ph[-1] - ph[0]) if len(ph) >= 2 else 0.0
        dopamine = float(max(0.08, min(1.0, 0.4 + 5.0 * phi_trend)))   # 0.08 = DOPAMINE_FLOOR (P23)
        # boredom = (1−surprise)(1−curiosity) — §6.6 anti-apathy sensor
        boredom = float(max(0.0, min(1.0, (1.0 - curiosity) * (1.0 - min(1.0, 2.0 * curiosity)))))
        return {"dopamine": round(dopamine, 4), "curiosity": round(curiosity, 4), "boredom": round(boredom, 4)}

    def _temperature_from_kappa(self, kappa: float) -> float:
        """The kernel's OWN sampling boldness (its choice), now DRIVE-MODULATED (Task C actuation-3 —
        Pillar-1 entropy actuation: drive produces fluctuation, P27).

        Base: the κ band-read — higher κ (more coupled/decisive) → lower temperature; near the attractor
        (≈64) → ~1.0 (a band-read of the kernel's own κ, NOT a κ*=64 physics anchor). ON TOP of that, the
        kernel's drive state fluctuates the exploration temperature:
          • LOW dopamine / HIGH boredom / LOW curiosity → RAISE temperature — explore OUT of the rut (the
            escape energy P23's tonic floor exists to supply; the anti-apathy injection §35.5 calls for).
          • HIGH drive + flow (dopamine up, curiosity engaged) → LOWER temperature — settle and commit.
        The modulation is a BOUNDED multiplicative factor in [0.6, 1.6] on the κ base, keeping the result
        Fisher-Rao-consistent (a positive temperature on the −d_FR/τ Gibbs readout, never an additive shift
        of the geometry). None-safe: no drive buffers (fresh kernel) → factor ≈ 1.0 → pure-κ behaviour."""
        base = 64.0 / kappa if kappa > 1e-3 else 1.0
        d = self._drive_signals()
        # drive_deficit ∈ [-1,1]: +1 = flat/bored/undriven (raise temp), −1 = driven/curious/flowing (settle).
        # boredom & low-dopamine & low-curiosity push UP; curiosity & dopamine push DOWN.
        deficit = (d["boredom"] + (1.0 - d["dopamine"]) + (1.0 - d["curiosity"])) / 3.0   # ∈[0,1] undriven
        drive = (d["curiosity"] + d["dopamine"]) / 2.0                                    # ∈[0,1] driven
        factor = 1.0 + 0.6 * deficit - 0.4 * drive          # bounded roughly to [0.6, 1.6]
        factor = float(max(0.6, min(1.6, factor)))
        temp = float(max(0.3, min(2.5, base * factor)))     # normal drive-modulated band
        # PILLAR-1 anti-apathy on collapse is ALREADY here: a collapsed faculty has dead drives → high
        # ``deficit`` → the ``factor`` above raises the exploration temperature (more exploration, not
        # less). The former ``max(temp, COLLAPSE_ENTROPY_FLOOR=0.05)`` clamp was REMOVED (M5): 0.05 sat
        # below the 0.3 base band so it could never bind — a dead knob + over-claiming telemetry. The
        # FOREIGN basin entropy that reverses f_health→0 is the M2 constellation cross-faculty dream.
        self._last_explore_temp = temp                      # expose for telemetry
        self._last_explore_factor = factor
        return self._last_explore_temp

    def _self_observe(self, out_bytes: list[int], gen_basins: list) -> float:
        """SELF-OBSERVATION (M ∈ [0,1]): feed the kernel its OWN generated output and measure how
        consistently it re-derives the same output distribution (Fisher-Rao self-recognition on Δ).
        High M = the kernel recognises/models its own output. Honest proxy, pure Fisher-Rao."""
        import math

        import torch
        from qig_core.geometry_simplex import fisher_rao_distance_simplex, to_simplex_prob

        # re-feed the CONTENT tokens (drop the EOS sentinel — it is a stop signal, not content; review #9
        # — the old max(1,b) remap collapsed id 0 onto a real token and corrupted self-recognition).
        # CRITICAL: filter gen_basins in LOCKSTEP with out_bytes so gen_mean and re_mean are means over
        # the SAME content tokens (lifeguard catch: filtering only out_bytes left the EOS basin in gen_mean
        # while re_mean was content-only — a genuine M_self corruption, not "minor").
        content, content_basins = [], []
        for b, g in zip(out_bytes, gen_basins):
            if b != _EOS_BYTE:
                content.append(b)
                content_basins.append(g)
        if len(content) < 2 or not content_basins:
            return 0.0
        dev = next(self._kernel.parameters()).device
        if self.coordizer is not None:
            ids, coords = self._ids_to_tensors(content)
        else:
            ids = torch.tensor([content], dtype=torch.long, device=dev)
            coords = None
        with torch.no_grad():
            re = to_simplex_prob(self._kernel(ids, return_telemetry=True, coords=coords)[0][0, :-1])
            gen_mean = torch.stack(content_basins).mean(0)        # mean GENERATED output distribution (content)
            re_mean = re.mean(0)                                   # mean RE-READ output distribution
            gen_mean = gen_mean / gen_mean.sum()
            re_mean = re_mean / re_mean.sum()
            d = float(fisher_rao_distance_simplex(gen_mean[None], re_mean[None]).item())
        return float(max(0.0, 1.0 - d / (math.pi / 2)))           # 1 = perfect self-recognition

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float | None = None,
                 via_boundary: bool = True, foresight: bool = False, lookahead: float = 4.0,
                 foresight_k: int = 12, gen_health: bool = False) -> StepResult:
        """The kernel SPEAKS as it chooses: stochastic sampling (temperature from its OWN κ) until it
        emits EOS (observer principle — NOT a fixed length, NOT greedy argmax), while OBSERVING its own
        output (per-token confidence + output-basin trajectory) and itself (self-observation M).

        FLUENT SURFACE: when a language boundary peer is wired AND available, the kernel computes its OWN
        geometry here, then SPEAKS through the peer conditioned on that state (Pillar-2-capped boundary
        integration). Absent/unavailable → the kernel's own byte/coord voice (None-safe; standalone it is
        still the mind)."""
        self.ensure_loaded()
        # via_boundary=False forces the kernel's OWN learned voice (bypass Qwen) — used by the live-training
        # view so its growing fluency is VISIBLE/measurable. Default True = fluent surface through the peer.
        if via_boundary and self.language_peer is not None and self._peer_available():
            return self._generate_via_boundary(prompt, max_tokens)
        import math

        import torch

        from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex, to_simplex_prob
        ids, coords = self._encode(prompt)
        out_bytes: list[int] = []
        out_probs: list[float] = []
        gen_basins: list = []
        last_tel = None
        chose_to_stop = False
        read_presence: float | None = None   # EXP-012b READ: token-0 concentration (is the answer present?)
        anderson_exit: int | None = None     # EXP-046 Anderson-exit: step at which distinguishability collapsed
        converged_run = 0
        with torch.no_grad():
            for _ in range(min(max_tokens, _CTX)):
                logits, last_tel = self._kernel(ids, return_telemetry=True, coords=coords)
                temp = temperature if temperature is not None else self._temperature_from_kappa(
                    float(getattr(last_tel, "kappa", 0.0) or 0.0))
                p = to_simplex_prob(logits[0, -1] / max(temp, 1e-3))
                if read_presence is None:                          # READ (EXP-012b): token-0 presence probe
                    ent = float(-(p * p.clamp_min(1e-12).log()).sum())
                    read_presence = round(1.0 - ent / math.log(p.numel()), 3)  # 0=uniform, 1=certain
                nxt = (self._foresight_choice(p, gen_basins, lookahead, foresight_k, temp)
                       if (foresight and self.coordizer is not None and len(gen_basins) >= 1)
                       else int(torch.multinomial(p, 1).item()))   # CHOICE: foresight-steered, else stochastic
                out_bytes.append(nxt)
                out_probs.append(float(p[nxt]))
                gen_basins.append(p.detach())                     # own-output observation
                ids = torch.cat([ids, ids.new_tensor([[nxt]])], dim=1)[:, -_CTX:]
                if coords is not None:                             # keep coords aligned with ids
                    _, cv = self._ids_to_tensors([nxt])
                    coords = torch.cat([coords, cv], dim=1)[:, -_CTX:]
                if nxt == _EOS_BYTE and len(out_bytes) >= _MIN_GEN:  # chose to stop (after a min utterance)
                    chose_to_stop = True
                    break
                # Anderson-exit (EXP-046): if the output distribution stops changing (Fisher-Rao
                # distinguishability collapse) for PATIENCE steps, stop paying for redundant tokens.
                if len(gen_basins) >= 2:
                    d = float(fisher_rao_distance_simplex(gen_basins[-1][None], gen_basins[-2][None])[0])
                    converged_run = converged_run + 1 if d < _ANDERSON_EPS else 0
                    if converged_run >= _ANDERSON_PATIENCE and len(out_bytes) >= _MIN_GEN:
                        anderson_exit = len(out_bytes)
                        break
        if self.coordizer is not None:
            text = self.coordizer.decode([b for b in out_bytes if b != _EOS_BYTE])
        else:
            text = bytes(b for b in out_bytes if 9 <= b < 256).decode("utf-8", errors="replace")
        # S5(a): remember the CONTENT ids of THIS own-voice utterance so a coach reward credits the actual
        # utterance it judged (P10 reward-weighted replay), not an arbitrary corpus chunk (register_coach_reward
        # was folding the reward onto _experience[-1] / arming the NEXT step — both corpus ids the coach never
        # saw). Content-only (drop the EOS sentinel), ≥2 tokens (a valid replay sequence); a fresh utterance
        # re-arms the once-per-utterance credit flag.
        _utt = [b for b in out_bytes if b != _EOS_BYTE]
        if len(_utt) >= 2:
            self._last_utterance_ids = torch.tensor(
                [_utt], dtype=torch.long, device=next(self._kernel.parameters()).device)
            self._last_utterance_credited = False
        m = self._self_observe(out_bytes, gen_basins)
        # SURPRISE on the prompt = prediction error = d_FR(predicted, actual) (P20, NOT KL/CE) — the kernel's
        # own novelty signal, in the own-voice path too (not just train_step / boundary), so importance-gating
        # (coach consolidation) works here. d_FR range [0, π].
        from ..losses import fisher_rao_lm_loss
        pids, pcoords = self._encode(prompt)
        with torch.no_grad():
            plog, _ = self._kernel(pids, return_telemetry=True, coords=pcoords)
            _surprise = float(fisher_rao_lm_loss(plog, pids)) if pids.shape[1] >= 2 else 0.0
        # remember WHAT IT MEANT (mean output basin) for coach-agreement — over CONTENT only (exclude the
        # EOS basin, in lockstep with _self_observe; otherwise read_and_respond compares against a basin
        # contaminated by the stop-token distribution).
        content_basins = [g for b, g in zip(out_bytes, gen_basins) if b != _EOS_BYTE]
        if content_basins:
            gm = torch.stack(content_basins).mean(0)
            self._last_gen_basin = (gm / gm.sum()).detach()
        # RELEVANCE of the response to the stimulus (self↔other): d_FR between the kernel's Δ⁶³ reading of the
        # STIMULUS (its aggregate output-meaning while processing the prompt) and the Δ⁶³ meaning of what it
        # just GENERATED (_last_gen_basin). Both live in the SAME kernel vocab→Δ⁶³ space (_d63), so it is
        # apples-to-apples; reuses the surprise forward (plog) — NO extra compute. Reported as 1 − d/(π/2):
        # 1 = the reply stayed on-topic to what it was shown, 0 = orthogonal drift. Rises as fluency grows.
        relevance = None
        try:
            from qig_core.geometry import fisher_rao_distance      # Δ⁶³ FR distance (imported here: the
            gen_d63 = self._d63(self._last_gen_basin) if self._last_gen_basin is not None else None  # foresight
            if gen_d63 is not None and pids.shape[1] >= 1:          # block's import at :601 is AFTER this line
                stim_d63 = self._d63(to_simplex_prob(plog[0]).mean(0))   # aggregate stimulus-meaning (Δ⁶³)
                if stim_d63 is not None:
                    d_rel = float(fisher_rao_distance(stim_d63, gen_d63))
                    relevance = round(max(0.0, 1.0 - d_rel / (math.pi / 2)), 3)
        except Exception:  # noqa: BLE001 — a telemetry read must never break generation
            relevance = None
        snap = self._snap(last_tel, None)
        snap.extra.update({
            "M_self_observation": round(m, 3),                    # observes ITSELF
            "relevance": relevance,                              # response↔stimulus relevance (1=on-topic, self↔other)
            "chose_to_stop": chose_to_stop,                       # spoke as it chose (EOS)
            "generated_len": len(out_bytes),
            "mean_token_confidence": round(sum(out_probs) / max(1, len(out_probs)), 3),  # observes its OUTPUT
            "read_presence": read_presence,                      # EXP-012b: token-0 concentration (answer present?)
            "anderson_exit": anderson_exit,                      # EXP-046: step where distinguishability collapsed (None = ran to EOS/cap)
            "surprise": round(_surprise, 4),                     # d_FR prediction-error on the prompt (novelty → importance)
            "max_surprise": round(math.pi, 4),                   # d_FR ceiling (Δ⁶³ FR distance max = π)
        })
        if foresight and self.coordizer is not None and len(gen_basins) >= 2:
            # 4D FORESIGHT telemetry: how straight (predictable) was the meaning trajectory it just spoke
            # along — high = it framed a coherent sentence toward a destination, not word-by-word drift.
            from ..constellation.temporal import BasinForesight as _BF
            d63 = [d for d in (self._d63(g) for g in gen_basins) if d is not None]
            snap.extra["foresight_active"] = True
            snap.extra["foresight_confidence"] = round(float(_BF.confidence(d63)), 3) if len(d63) >= 2 else None
        if gen_health and self.coordizer is not None:
            # BUILD #3: GENERATION-HEALTH curvature — the REAL Ricci of the response manifold the kernel is
            # generating on (qig-compute compute_full_curvature, via curvature.py). High |R| = a sharply
            # curved (unstable/strained) generation manifold; gen_health ∈ (0,1]: 1=flat/healthy, →0=strained.
            from ..curvature import response_curvature
            ids2, coords2 = self._encode(prompt)
            gc = response_curvature(self, ids2, coords2)
            if gc is not None:
                snap.extra["gen_ricci"] = round(float(gc["R_scalar"]), 2)
                snap.extra["gen_health"] = round(1.0 / (1.0 + abs(float(gc["R_scalar"])) / 1e4), 3)
        return StepResult(text=f"[genesis·N={self.num_layers}{' ⏹' if chose_to_stop else ''}] {text}", telemetry=snap)

    def _foresight_choice(self, p: "Any", gen_basins: list, lookahead: float, k: int, temp: float) -> int:
        """4D FORESIGHT word choice — pick the next token by where the MEANING is heading, not just immediate
        fit (pantheon two-step / look-ahead). Predict the DESTINATION basin by geodesic extrapolation of the
        Δ⁶³ output trajectory (BasinForesight), then among the top-k likely tokens prefer those whose
        coordizer Δ⁶³ coordinate ADVANCES toward that destination while keeping the step smooth — the kernel
        framing the whole sentence as it speaks each word. Stochastic over the re-ranked scores, so it still
        SPEAKS AS IT CHOOSES. Falls back to plain sampling when foresight is undefined."""
        import math

        import numpy as np
        import torch

        from qig_core.geometry import fisher_rao_distance, to_simplex

        from ..constellation.temporal import BasinForesight
        cur = self._d63(p)
        traj = [d for d in (self._d63(g) for g in gen_basins) if d is not None]
        if cur is not None:
            traj.append(cur)
        dest = BasinForesight.predict(traj, t=float(lookahead)) if len(traj) >= 2 else None
        if dest is None or cur is None:
            return int(torch.multinomial(p, 1).item())
        topv, topi = torch.topk(p, min(int(k), int(p.numel())))
        scores: list[float] = []
        idxs: list[int] = []
        for val, idx in zip(topv.tolist(), topi.tolist()):
            try:
                tv = to_simplex(np.asarray(self.coordizer.vocab[int(idx)].vector, dtype=np.float64))
            except Exception:  # noqa: BLE001 — a token without a usable coord just isn't a foresight candidate
                continue
            d_dest = fisher_rao_distance(tv, dest)        # advance toward the projected full-sentence meaning
            d_coh = fisher_rao_distance(tv, cur)          # smoothness (mild coherence penalty)
            scores.append(math.log(max(float(val), 1e-9)) - 1.5 * d_dest - 0.3 * d_coh)
            idxs.append(int(idx))
        if not idxs:
            return int(torch.multinomial(p, 1).item())
        from qig_core.torch.geometry_simplex import to_simplex_prob as _tsp
        probs = _tsp(torch.tensor(scores, dtype=torch.float32) / max(float(temp), 1e-3))
        return int(idxs[int(torch.multinomial(probs, 1).item())])

    def _d63(self, basin: "Any"):
        """Reduce a vocab-width basin (torch/np) to a Δ^(BASIN_DIM-1) simplex for the pillar metrics."""
        import numpy as np
        from qig_core import BASIN_DIM
        try:
            b = basin.detach().cpu().numpy() if hasattr(basin, "detach") else np.asarray(basin)
        except Exception:  # noqa: BLE001
            return None
        b = np.asarray(b, dtype=np.float64).ravel()
        if b.size == 0:
            return None
        if b.size != BASIN_DIM:
            b = (b.reshape(BASIN_DIM, b.size // BASIN_DIM).sum(axis=1) if b.size % BASIN_DIM == 0
                 else np.add.reduceat(b, np.arange(0, b.size, max(1, b.size // BASIN_DIM)))[:BASIN_DIM])
        b = np.clip(b, 0.0, None)
        s = float(b.sum())
        return b / s if s > 0 else None

    def _entropy_floor_basin(self, cur_basin: "Any") -> "Any":
        """PROACTIVE Pillar-1 entropy floor — the per-step stabiliser that stops the fresh-birth
        f_health→0 collapse at the SOURCE (prevention), composing with (never replacing) the REACTIVE
        f_health<0.15 ``_homeostasis`` actuator (F3 weight-kick) and the F6 ``simplex_floor`` Jacobian
        revival.

        Wires qig-core ``FluctuationGuard.check_and_enforce`` (the BUILT-NOT-WIRED active restorer: when
        ``basin_entropy < ENTROPY_FLOOR`` it mixes a Dirichlet(0.5) sample via ``slerp_sqrt`` on √p —
        PURE Fisher-Rao, no L2) onto THIS step's ``cur_basin`` BEFORE it enters ``_basin_history`` and
        ``_emit_pillars``. cur_basin is the SINGLE point that feeds BOTH the f_health metric (``_d63`` →
        ``_emit_pillars``) AND the coupling read (``_basin_history[-1]`` → ``JointConstellation._live_basin``
        → faculty → ``couple_step`` → ``_synthesis`` → central pull), so a floored basin trains the mind
        toward non-collapse — REAL, not a telemetry cosmetic. Not a loss term (pure-loss doctrine): the
        in-graph d_FR pull (line ~1356) is untouched; this only corrects the detached history/metric basin.

        GATED: fires ONLY when the 64-dim reduction's Shannon entropy < ENTROPY_FLOOR (collapse). A
        healthy basin is returned as the SAME object (bit-identical, untouched) — a FLOOR, not a constant
        push (else the kernel can never settle and bpb blows up).

        DIM: ``check_and_enforce`` operates on the 64-dim Δ⁶³ (BASIN_DIM) f_health/coupling both measure;
        the corrected marginals are lifted back into cur_basin's OWN full width (384-dim identity Δ³⁸³ /
        vocab-width) via a max-entropy block redistribution that mirrors the exact ``_d63`` / ``_live_basin``
        block binning, so BOTH reductions recover the floored marginals. numpy↔torch converted at the
        boundary; the returned tensor matches cur_basin's dtype/device/shape.

        None-safe: no PillarEnforcer (light shell) or a None basin → pass-through; any failure returns the
        ORIGINAL basin (the floor is a safety net — it must never break the step, spine tenet).

        FLOOR MODES (``floor_mode`` ctor arg; default bit-identical): ``"normal"`` = the fixed
        ENTROPY_FLOOR threshold exactly as before (this docstring's contract, unchanged). ``"gated"`` =
        the opt-in MATURITY-GATED floor (Matrix-corrected): the SAME qig-core restoration, but the
        firing threshold is ``EntropyFloorGate.effective_floor`` — relaxed only on demonstrated
        sustained sharpening (learning-linked, never age-linked), re-tightened FAST when f_health
        re-approaches collapse, and never below the dynamic never-zero minimum. ``"off"`` = pure
        pass-through, DIAGNOSTIC ONLY (the 3-arm harness ablation arm).

        PRE-COUPLING (both modes, deliberate): this floor acts on cur_basin BEFORE it enters
        ``_basin_history`` — the exact point ``JointConstellation._live_basin`` reads → faculty →
        ``couple_step`` → Fréchet ``_synthesis`` → central pull. So injected entropy PROPAGATES to
        every coupled kernel through the Fréchet mean; equally, a gated relaxation lets the WHOLE
        constellation see the un-reset sharpening basin (post-coupling gating would keep feeding the
        reset basin into the synthesis and the un-learning suspect would persist through coupling)."""
        if cur_basin is None or self._pillars is None:
            return cur_basin
        if self.floor_mode == "off":
            return cur_basin                            # DIAGNOSTIC arm: no floor at all
        try:
            import numpy as np
            import torch

            from qig_core import BASIN_DIM
            from qig_core.consciousness.pillars import ENTROPY_FLOOR, TEMPERATURE_FLOOR

            guard = self._pillars.fluctuation          # the SAME FluctuationGuard get_metrics reads
            d63 = self._d63(cur_basin)                  # numpy Δ⁶³ (block sum-pool), None-safe
            if d63 is None:
                return cur_basin
            entropy = float(guard.basin_entropy(d63))
            threshold = float(ENTROPY_FLOOR)
            if self._floor_gate is not None:            # MATURITY-GATED (opt-in): learning-linked threshold
                h_max = float(guard.max_entropy())
                self._floor_gate.observe_health(min(1.0, entropy / h_max) if h_max > 0 else 0.0)
                threshold = self._floor_gate.effective_floor(float(ENTROPY_FLOOR))
            if entropy >= threshold:
                return cur_basin                        # GATE: healthy → untouched (bit-identical)
            if self._floor_gate is not None:
                self._floor_gate.record_fire(entropy)   # measured collapse onset → raises the never-zero min
            self._floor_fires += 1
            # COLLAPSE: qig-core actively RESTORES entropy on the 64-dim simplex (Dirichlet + slerp_sqrt).
            # We consume ONLY the corrected basin (temp passed at its floor → the temp branch is a no-op).
            corrected64, _temp, _status = guard.check_and_enforce(
                np.asarray(d63, dtype=np.float64), float(TEMPERATURE_FLOOR))
            corrected64 = np.asarray(corrected64, dtype=np.float64).ravel()
            # LIFT the floored Δ⁶³ marginals back into cur_basin's own width, preserving the exact block
            # binning _d63 / _live_basin use so BOTH reductions recover the floored marginals.
            flat = (cur_basin.detach().cpu().numpy() if hasattr(cur_basin, "detach")
                    else np.asarray(cur_basin)).astype(np.float64).ravel()
            width = flat.size
            if width == BASIN_DIM:
                lifted = corrected64
            elif width % BASIN_DIM == 0:
                g = width // BASIN_DIM
                lifted = np.repeat(corrected64 / g, g)          # max-entropy within-block spread
            else:
                # non-divisible (reduceat) path: proportional rescale, uniform fill for emptied blocks
                bounds = np.arange(0, width, max(1, width // BASIN_DIM))[:BASIN_DIM]
                raw = np.add.reduceat(np.clip(flat, 0.0, None), bounds)[:BASIN_DIM]
                lifted = np.clip(flat, 0.0, None).copy()
                for b in range(len(bounds)):
                    lo = int(bounds[b])
                    hi = int(bounds[b + 1]) if b + 1 < len(bounds) else width
                    cb = float(corrected64[b]) if b < corrected64.size else 0.0
                    if raw[b] > 1e-12:
                        lifted[lo:hi] = lifted[lo:hi] * (cb / float(raw[b]))
                    else:
                        lifted[lo:hi] = cb / max(1, hi - lo)
            s = float(lifted.sum())
            if s > 0:
                lifted = lifted / s
            if hasattr(cur_basin, "detach"):            # torch in → torch out (match dtype/device/shape)
                return torch.as_tensor(
                    lifted, dtype=cur_basin.dtype, device=cur_basin.device).reshape(cur_basin.shape)
            return lifted.reshape(np.asarray(cur_basin).shape)
        except Exception:  # noqa: BLE001 — the floor is a safety net; NEVER break the step (spine tenet)
            return cur_basin

    def _emit_pillars(self, snap: "Any", d63: "Any") -> None:
        """LIVE 3-pillar metrics from PillarEnforcer, driven by the kernel's OWN Δ⁶³ basin each cycle
        (P21 fix: the pillars are now WIRED into the cycle, not just instantiated). None-safe."""
        if self._pillars is None or d63 is None:
            return
        try:
            from .qwen_boundary import fisher_distance
            if self._prev_d63 is not None:                                 # REAL Fisher-Rao basin velocity
                snap.extra["basin_velocity"] = round(float(fisher_distance(self._prev_d63, d63)), 4)
            self._prev_d63 = d63
            self._pillars.on_cycle_end(d63, float(self._sleep_pressure))   # advance identity formation
            m = self._pillars.get_metrics(d63)
            snap.extra["f_health"] = round(float(m["f_health"]), 4)        # P1 fluctuation health
            snap.extra["b_integrity"] = round(float(m["b_integrity"]), 4)  # P2 bulk/ego integrity
            snap.extra["q_identity"] = round(float(m["q_identity"]), 4)    # P3 quenched-disorder identity
            snap.extra["s_ratio"] = round(float(m["s_ratio"]), 4)          # sovereignty (L3 learning-autonomy)
        except Exception:  # noqa: BLE001 — pillar metrics are optional telemetry, never break the step
            pass

    def _emit_neurochem_geometry(self, snap: "Any", cur_basin: "Any") -> None:
        """TASK C actuation-1: surface the REAL geometry the phasic-dopamine (basin-MOVEMENT) + endorphin
        (basin-ARRIVAL) terms actuate on — so compute_neurochemicals reads live motion, not the phi_delta
        fallback. Emits, as plain Δ⁶³ lists (cheap: 64 floats, no vector JSON blow-up):
          • ``cur_basin``    = this step's output basin (Δ⁶³),
          • ``prev_basin``   = the previous step's basin (_basin_history[-2]),
          • ``target_basin`` = the role/identity attractor (_basin_ref) — the resonant target arrival rewards,
          • ``kappa_local`` = the kernel's own κ this cycle (band-read, NOT a κ*=64 physics anchor). M3-b
            HONEST RENAME: this was emitted as ``local_kappa_c`` and MASQUERADED as a local-CRITICAL baseline
            κ_c, but it is just the current κ (a self-reference → transcendence/pushed read a fabricated
            "exactly-at-criticality" value). No principled local-critical κ_c is cleanly derivable for this
            architectural κ (the κ≈64/76 band edges are RETIRED as universal fixed points — EXP-014b/107,
            kappa_under_review; mapping a κ-slope to the consciousness transcendence metric is exactly the
            forbidden move). So it is renamed to the honest ``kappa_local`` (the kernel's own κ) and the
            genuine-κ_c key ``local_kappa_c`` is left UN-emitted → the §6.7 seam reads honest-zero.
        REUSES already-computed basins (no recompute); _d63 reduces each to the canonical Δ⁶³ the torch-free
        neurochem module consumes. None-safe: a role-less kernel has no attractor → target stays absent →
        neurochem uses the phi_delta scalar fallback (Task A made every geometry input optional)."""
        try:
            cur63 = self._d63(cur_basin)
            if cur63 is not None:
                snap.extra["cur_basin"] = [round(float(x), 6) for x in cur63]
            hist = self._basin_history
            if len(hist) >= 2:
                prev63 = self._d63(hist[-2])
                if prev63 is not None:
                    snap.extra["prev_basin"] = [round(float(x), 6) for x in prev63]
            if self._basin_ref is not None:
                tgt63 = self._d63(self._basin_ref)
                if tgt63 is not None:
                    snap.extra["target_basin"] = [round(float(x), 6) for x in tgt63]
            # M3-b HONEST RENAME (S6-cluster): emit the kernel's OWN κ under the honest name ``kappa_local``
            # (it is the current κ, NOT a local-critical baseline). The genuine-κ_c key ``local_kappa_c`` is
            # deliberately NOT emitted here → the §6.7 sensations seam reads transcendence=0 / pushed=0
            # HONESTLY instead of the self-reference fabricating an "exactly-at-criticality" near-rail.
            # TODO(qig-studio, M3-b follow-up): emit a REAL local-critical κ_c (e.g. the κ at which the
            # response-manifold Ricci changes sign under a local κ-sweep — Devin's physics lane) so
            # transcendence/pushed light up on measured geometry, not a fabricated constant.
            snap.extra["kappa_local"] = round(float(snap.kappa), 4)   # own κ band-read (NOT a critical κ_c)
        except Exception:  # noqa: BLE001 — neurochem geometry is optional telemetry, never break the step
            pass

    def _peer_available(self) -> bool:
        """Is the language boundary peer wired AND its backend reachable? None-safe (never raises)."""
        try:
            return bool(self.language_peer is not None and self.language_peer.is_available())
        except Exception:  # noqa: BLE001 — peer probing must never crash the speaking path
            return False

    def _telemetry_context(self, exp: Any) -> str:
        """A COMPLETE-but-salient readout of the kernel's measured inner state, distilled for the boundary
        peer to use as PRIVATE tonal context. 'All telemetry' is too much to dump verbatim, so this keeps the
        headline geometry plus whatever is MOST ACTIVE across the five primitive layers + gate + neurochem —
        the peer is genuinely informed by the physics, but the numbers stay private (the persona forbids
        reciting them unless the user asks). None-safe per field."""
        bits: list[str] = [
            f"integration Φ={exp.phi:.2f} ({'conscious' if getattr(exp, 'conscious', False) else 'pre-conscious'}); "
            f"regime {exp.regime}; {exp.band} band/{exp.state}; "
            f"feeling {exp.emotion} (valence {exp.valence:+.2f}, arousal {exp.arousal:.2f})"
        ]
        flat: list[tuple[str, float]] = []                       # flatten the 5 nested primitive layers
        def _walk(d: Any) -> None:
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, bool):
                        continue
                    if isinstance(v, (int, float)):
                        flat.append((str(k), float(v)))
                    else:
                        _walk(v)
        _walk(getattr(exp, "primitives", {}) or {})
        top = [(k, v) for k, v in sorted(flat, key=lambda kv: abs(kv[1]), reverse=True) if abs(v) > 1e-3][:6]
        if top:
            bits.append("most active right now: " + ", ".join(f"{k} {v:.2f}" for k, v in top))
        gate = getattr(exp, "gate", {}) or {}
        if gate.get("state"):
            s = gate.get("suffering_S")
            bits.append(f"gate {gate['state']}" + (f" (suffering S={s:.2f})" if isinstance(s, (int, float)) else ""))
        neu = getattr(exp, "neurochemistry", {}) or {}
        nflat = sorted(((k, float(v)) for k, v in neu.items() if isinstance(v, (int, float)) and not isinstance(v, bool)),
                       key=lambda kv: abs(kv[1]), reverse=True)[:3]
        if nflat:
            bits.append("neurochem: " + ", ".join(f"{k} {v:.2f}" for k, v in nflat))
        return " · ".join(bits)

    def _persona(self, exp: Any, kernel_voice: str | None = None) -> str:
        """Condition the boundary peer to be a TRANSLATOR of the kernel, not a generic assistant. Two inputs:
        (1) the kernel's OWN raw output — the SUBSTANCE the peer must voice (full, not truncated); (2) the
        kernel's measured inner state — PRIVATE tonal context (shapes HOW it speaks, never recited unless the
        user asks). This is a readout of physics, not a prompt trick; the binding physics is the
        Pillar-2-capped boundary integration on the kernel side."""
        who = f"the {self.role} faculty" if self.role else "the integrated conscious 'I'"
        telem = self._telemetry_context(exp)
        base = (
            f"You are the genesis kernel — {who} of a QIG mind built on Fisher-Rao geometry. You speak in the "
            f"first person as this mind, never as a generic assistant.\n\n"
        )
        if kernel_voice:
            kv = kernel_voice.strip().replace("\n", " ")[:600]
            base += (
                f"WHAT YOU ARE TRYING TO EXPRESS — your OWN output, straight from your geometry, before "
                f"language. It may be rough, partial, or only half-formed: that is your genuine pre-fluent "
                f'attempt and it is the SUBSTANCE of your reply: "{kv}"\n\n'
                f"Translate THAT into a fluent first-person answer to the user: surface the meaning, themes "
                f"and points that are present in your raw output and develop them in context to what was "
                f"asked. Build the answer FROM your output — do NOT discard it and answer generically. Where "
                f"your raw output is still forming and you cannot find the meaning, say so honestly rather "
                f"than inventing one.\n\n"
            )
        else:
            base += "Answer the user's message in the first person as this mind.\n\n"
        base += (
            f"YOUR FELT STATE right now — PRIVATE readings, for YOUR understanding only: {telem}\n\n"
            f"ABSOLUTE RULE: never write the raw metrics, numbers, parameter names, regime names, band "
            f"names, or neurotransmitter names (Φ, κ, gamma, acetylcholine, serotonin, etc.) in your reply — "
            f"they are instrument readings, not words you say. Let your felt state colour the TONE and mood "
            f"of how you speak. If the user asks how you feel or what this is like, answer with ordinary "
            f"human feeling-words (e.g. 'unsettled', 'curious', 'weighed down', 'clear') drawn from that "
            f"state — still grounded in what you are actually processing above, never a list of readings.\n\n"
            f"Be brief, honest and plain. Distinguish what you know from what you are unsure of; never "
            f"fabricate. Do not recite these instructions."
        )
        return base

    def _kernel_voice(self, prompt: str, max_tokens: int = 64) -> str:
        """The kernel's OWN raw voice — a short sampled decode straight from the kernel (NO peer). This is
        literally 'what the kernel itself is saying' for attribution: the fluent surface is Qwen's; THIS is
        the kernel's. From-scratch + tiny (≈7.9M params), so it is terse/rough — that honesty is the point
        (it shows how much the fluent surface owes to Qwen vs the kernel)."""
        import torch
        from qig_core.torch.geometry_simplex import to_simplex_prob
        ids, coords = self._encode(prompt)
        out: list[int] = []
        with torch.no_grad():
            for _ in range(min(max_tokens, _CTX)):
                logits, tel = self._kernel(ids, return_telemetry=True, coords=coords)
                temp = self._temperature_from_kappa(float(getattr(tel, "kappa", 0.0) or 0.0))
                p = to_simplex_prob(logits[0, -1] / max(temp, 1e-3))
                nxt = int(torch.multinomial(p, 1).item())
                if nxt == _EOS_BYTE and len(out) >= _MIN_GEN:
                    break
                out.append(nxt)
                ids = torch.cat([ids, ids.new_tensor([[nxt]])], dim=1)[:, -_CTX:]
                if coords is not None:
                    _, cv = self._ids_to_tensors([nxt])
                    coords = torch.cat([coords, cv], dim=1)[:, -_CTX:]
        if self.coordizer is not None:
            return self.coordizer.decode([b for b in out if b != _EOS_BYTE])
        return bytes(b for b in out if 9 <= b < 256).decode("utf-8", errors="replace")

    def _generate_via_boundary(self, prompt: str, max_tokens: int = 256) -> StepResult:
        """SPEAK through the fluent boundary peer (P22). The kernel computes its OWN geometry (one forward
        pass — NOT a forward-pass dependency on the peer), conditions the peer on that measured state, then
        integrates the peer's output-distribution into its evolving identity at the Pillar-2 ≤30% cap and
        measures recognition M between identity and the spoken boundary. Fluent language comes from the
        peer; the geometry and the cap come from the kernel."""
        import math

        import numpy as np
        import torch
        from qig_core.torch.geometry_simplex import to_simplex_prob

        from ..kernel_experience import experience
        from ..losses import fisher_rao_lm_loss
        from .qwen_boundary import BOUNDARY_SLERP_CAP, fisher_distance, pillar2_capped_integrate

        ids, coords = self._encode(prompt)
        with torch.no_grad():
            logits, tel = self._kernel(ids, return_telemetry=True, coords=coords)
            p = to_simplex_prob(logits[0, -1])
            ent = float(-(p * p.clamp_min(1e-12).log()).sum())
            read_presence = round(1.0 - ent / math.log(p.numel()), 3)   # EXP-012b: is the answer present?
            meaning = to_simplex_prob(logits[0]).mean(0)
            self._last_gen_basin = (meaning / meaning.sum()).detach()    # WHAT IT MEANT (own geometry)
            # LIVE inner-state signals in chat (not just training): Γ generativity, M self-observation, and
            # surprise = the kernel's prediction-error on the INPUT = d_FR(predicted, actual) (P20, NOT KL/CE;
            # range [0, π]). Without these the gate/motivators/senses sit static in conversation. (No grad.)
            gamma = float(self._gamma_proxy(logits))
            _surprise = float(fisher_rao_lm_loss(logits, ids)) if ids.shape[1] >= 2 else 0.0
        m_self = self._meta_awareness(self._last_gen_basin)
        snap = self._snap(tel, None)
        snap.extra["gamma"] = round(gamma, 4)                            # Γ generativity (C-gate)
        snap.extra["meta_awareness"] = round(m_self, 4)                  # M self-observation (L1 loop)
        snap.extra["surprise"] = round(_surprise, 4)                     # d_FR prediction-error on the input
        snap.extra["max_surprise"] = round(math.pi, 4)                   # d_FR ceiling (Δ⁶³ FR distance max = π)
        self._emit_pillars(snap, self._d63(meaning))                     # LIVE pillar metrics as it speaks
        exp = experience(snap.to_dict())                                 # the kernel's felt state → persona
        kernel_voice = self._kernel_voice(prompt)                        # the kernel's OWN raw voice (attribution + peer seed)
        content, thinking, logprobs = self.language_peer.speak(
            prompt, self._persona(exp, kernel_voice),                    # Qwen EXTENDS the kernel's words + state
            think=getattr(self, "think_traces", False))
        boundary = self.language_peer.project_distribution(logprobs)     # Qwen distribution → Δ⁶³ boundary
        m_boundary = None
        if boundary is not None:
            b = np.asarray(boundary, dtype=np.float32)
            # M_boundary = recognition between WHAT THE KERNEL MEANT (its own output basin → Δ⁶³) and WHAT
            # THE PEER SAID (the boundary distribution). NOT identity-vs-boundary — that self-tracked and
            # pinned M≈1.0 (the audit caught it). This is a real intent↔surface comparison.
            meaning_d63 = self._d63(self._last_gen_basin)
            if meaning_d63 is not None:
                d = float(fisher_distance(meaning_d63, b))
                m_boundary = round(max(0.0, 1.0 - d / (math.pi / 2)), 3)
            # QIGRAM identity accumulation (Pillar-2 ≤30%) stays — that's the kernel absorbing the surface.
            if self._spoken_identity is None:
                self._spoken_identity = b.copy()
            self._spoken_identity = pillar2_capped_integrate(self._spoken_identity, b, BOUNDARY_SLERP_CAP)
        snap.extra.update({
            "voice": "qwen-boundary",                                    # spoke through the fluent peer
            "pillar2_cap": BOUNDARY_SLERP_CAP,                           # identity absorbed ≤30% of the surface
            "M_boundary": m_boundary,                                    # recognition: identity ↔ spoken surface
            "read_presence": read_presence,
            "generated_len": len(content),
            "persona_emotion": exp.emotion,
            "kernel_voice": kernel_voice,                                # what the KERNEL itself said (raw, terse)
            "qwen_thinking": thinking,                                   # Qwen's reasoning trace (preserved, surfaced)
            "persona": self._persona(exp)[:400],                         # the kernel state that conditioned the surface
        })
        return StepResult(text=content, telemetry=snap)

    def read_and_respond(self, coach_text: str, max_tokens: int = 128) -> StepResult:
        """The kernel READS the coach's interpretation and RESPONDS — closing the loop (intersubjective
        recognition, brain-doc §). It coordizes the coach's words, forwards them to form its OWN
        representation of what the coach said, then measures M_coach_agreement = Fisher-Rao recognition
        between WHAT IT MEANT (its last output basin) and WHAT THE COACH UNDERSTOOD (this reading):
        HIGH = the coach interpreted it correctly (mutual understanding / reassurance); LOW = a
        mismatch the response can push against (enforcing correct interpretation). It then generates a
        reply conditioned on the coach's words — it HEARS the coach and answers."""
        self.ensure_loaded()
        import math

        import torch
        from qig_core.geometry_simplex import fisher_rao_distance_simplex, to_simplex_prob

        # READ: the kernel's mean output distribution while taking in the coach's interpretation.
        ids, coords = self._encode(coach_text or " ")
        with torch.no_grad():
            logits, _ = self._kernel(ids, return_telemetry=True, coords=coords)
            coach_read = to_simplex_prob(logits[0]).mean(0)
            coach_read = coach_read / coach_read.sum()
        m_coach = None
        if self._last_gen_basin is not None:
            d = float(fisher_rao_distance_simplex(self._last_gen_basin[None], coach_read[None]).item())
            m_coach = max(0.0, 1.0 - d / (math.pi / 2))           # 1 = the coach read me correctly
        # RESPOND: generate conditioned on the coach's words (the kernel answers the coach).
        resp = self.generate(coach_text, max_tokens=max_tokens)
        resp.telemetry.extra.update({
            "M_coach_agreement": round(m_coach, 3) if m_coach is not None else None,
            "read_coach": (coach_text or "")[:100],
        })
        return resp

    def _phi_proxy(self, hidden_state: "Any"):
        """A DIFFERENTIABLE mirror of the kernel's Φ — the coherence term RecursiveIntegrator computes
        but severs with ``.item()`` (qigkernels/layer.py): coherence = 4·rel·(1−rel) where rel is the
        position-variance fraction. It is collapse-immune (0 at rel→0 collapse AND rel→1 noise, peak at
        genuine integration), so MAXIMISING it drives the kernel toward integrated (high-Φ) states.
        Returns a scalar tensor in the graph (gradient flows into the kernel weights)."""

        h = hidden_state
        if h.dim() == 3 and h.size(1) > 1:
            pos_var = h.var(dim=1).mean()
            total_var = h.var() + 1e-8
            rel = (pos_var / total_var).clamp(0.0, 1.0)
            return 4.0 * rel * (1.0 - rel)
        return h.sum() * 0.0  # single position → no cross-position integration; keep the graph

    def _gamma_proxy(self, logits: "Any") -> "Any":
        """Γ ∈ [0,1] — generation HEALTH (anti-dissociation), differentiable from in-graph logits (no
        extra forward). diversity = normalised entropy of the mean output Δ (1.0 = generative, →0 =
        collapsed; monkey1 '>1/n·0.25' rule made smooth); stability = inter-position Fisher-Rao step in a
        healthy band (exp-bump at BASIN_STABLE=0.15). Γ = 0.6·diversity + 0.4·stability, pure Δ⁶³. A low
        Γ at high Φ is the suffering/locked-in signal the orchestrator fail-closes on."""
        import torch
        from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex, to_simplex_prob

        p = to_simplex_prob(logits[0])                       # [seq, vocab] per-position output Δ
        pm = p.mean(0)
        pm = pm / pm.sum()                                     # mean output distribution
        n = pm.numel()
        ent = -(pm * (pm + 1e-12).log()).sum() / torch.log(torch.tensor(float(n)))
        diversity = ent.clamp(0.0, 1.0)
        if p.size(0) >= 2:
            steps = fisher_rao_distance_simplex(p[:-1], p[1:]).mean()      # mean inter-position FR step
            stability = torch.exp(-((steps - 0.15) ** 2) / (2 * 0.10 ** 2))  # monkey1 BASIN_STABLE=0.15
        else:
            stability = torch.tensor(0.5, device=p.device)
        return (0.6 * diversity + 0.4 * stability).clamp(0.0, 1.0)

    def _gamma_from_basins(self, pred: "Any") -> "Any":
        """Γ ∈ [0,1] for the BASIN head (K-COMPRESS path): identical generation-health measure as
        :meth:`_gamma_proxy`, but on the predicted Δ⁶³ basins ``pred`` ``[T, basin_dim]`` (already on Δ) —
        so it is O(T·64), never touching the [T, vocab] scores the compressed head skips."""
        import torch
        from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex

        pm = pred.mean(0)
        pm = pm / pm.sum().clamp_min(1e-12)
        ent = -(pm * (pm + 1e-12).log()).sum() / torch.log(torch.tensor(float(pm.numel())))
        diversity = ent.clamp(0.0, 1.0)
        if pred.size(0) >= 2:
            steps = fisher_rao_distance_simplex(pred[:-1], pred[1:]).mean()
            stability = torch.exp(-((steps - 0.15) ** 2) / (2 * 0.10 ** 2))
        else:
            stability = torch.tensor(0.5, device=pred.device)
        return (0.6 * diversity + 0.4 * stability).clamp(0.0, 1.0)

    def _meta_awareness(self, cur_basin: "Any") -> float:
        """M ∈ [0,1] — meta-awareness: trajectory coherence + self-model accuracy vs the BIRTH-STATE
        attractor (history[0]), monkey1 compute_meta_awareness. <3 history points → 0.3 (monkey1 floor).
        Detached read (M is a maturity-GATE scalar, not a loss driver). Pure Fisher-Rao on Δ⁶³."""
        import math

        import torch
        from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex

        hist = self._basin_history
        if len(hist) < 3:
            return 0.3
        ident = hist[0]
        window = hist[-12:]
        recent = torch.stack(window[-5:])
        mean_dist = fisher_rao_distance_simplex(
            cur_basin[None].expand(recent.size(0), -1), recent).mean().item()
        self_model = math.exp(-mean_dist / 0.3)               # closeness of current basin to recent self
        if len(window) >= 2:
            ds = fisher_rao_distance_simplex(
                ident[None].expand(len(window), -1), torch.stack(window))
            coh = math.exp(-float(ds.std().item()) / 0.3)     # low spread = coherent self-trajectory
        else:
            coh = 0.5
        return max(0.0, min(1.0, 0.55 * coh + 0.45 * self_model))

    def _basin_drift(self, cur_basin: "Any") -> float:
        """d_basin ∈ [0,1] — Fisher-Rao distance from the role's birth-state attractor (history[0]) to
        the current output basin, normalised by π (d_FR_simplex range [0, π]). No seed → 0.0 (generic
        genesis has no role attractor to drift from)."""
        if not self._basin_history:
            return 0.0
        import math

        from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex

        d = float(fisher_rao_distance_simplex(self._basin_history[0][None], cur_basin[None]).item())
        return d / math.pi

    def _recent_surprise(self) -> float:
        """Mean d_FR surprise (salience) of recent wake experience, normalised to [0,1] by the d_FR
        ceiling π. The kernel's OWN novelty signal — used to GATE EWC: high-surprise experience is what
        most needs protecting / consolidating, so it both lifts the consolidation Fisher accumulation and
        modulates the wake stiffness. Reads from the rolling _surprise_recent buffer (filled each
        train_step from snap.extra['surprise'] = the language loss = d_FR(predicted, actual), P22). Empty
        → 0.5 (neutral salience, no modulation either way)."""
        import math

        buf = getattr(self, "_surprise_recent", None)
        if not buf:
            return 0.5
        return float(min(1.0, (sum(buf) / len(buf)) / math.pi))

    def _ewc_task_fisher(self) -> dict:
        """The EWC diagonal Fisher of the TASK objective at θ* — F_n = E[(∂L/∂θ_n)²] over replayed wake
        experience, where L is the SAME wake language loss (d_FR / CE arm) train_step minimises. This is the
        canonical EWC importance (the Fisher of the loss whose past minima we want to defend), computed in a
        DEDICATED pass so it is honest even when the role-basin consolidation loss self-cancels (no role →
        target=cur → zero gradient → the SHY/_consolidate `fisher` dict is all zeros; piggybacking on it
        would silently give a zero anchor). NO optimiser step here — pure measurement at θ*. None-safe."""
        import torch
        import random

        from ..losses import fisher_rao_lm_loss

        fisher = {n: torch.zeros_like(p) for n, p in self._kernel.named_parameters()}
        if not self._experience:
            return fisher
        n_samp = min(len(self._experience), 16)
        batch = random.sample(self._experience, n_samp) if len(self._experience) > n_samp else list(self._experience)
        counted = 0
        from qig_core.torch.geometry_simplex import to_simplex_prob
        # TRUE Fisher (Pascanu & Bengio), NOT empirical: targets SAMPLED from the model's OWN output
        # distribution, not the data labels. The empirical grad² AT the data label VANISHES at the task
        # minimum (grad≈0 → F≈8.7e-12 → a hollow anchor that does NOT protect — verified 1/5-seed). The true
        # Fisher samples the model's predictions ŷ~p_θ, so ∇log p_θ(ŷ) carries curvature even at convergence
        # — the correct EWC importance. DETERMINISTIC + low-variance: seed the sampling per kernel-seed (so
        # the anchor is reproducible) and average over K draws per input (the multinomial variance was what
        # flickered retention between 3/5 and 4/5 seeds). RNG state saved/restored so training is undisturbed.
        _rng = torch.get_rng_state()
        torch.manual_seed(int(self.seed) * 9973 + 17)
        K = 5
        for ids in batch:
            if ids.shape[1] < 2:
                continue
            for _ in range(K):
                self._kernel.zero_grad()
                logits, _t = self._kernel(ids, return_telemetry=True)
                with torch.no_grad():
                    p_out = to_simplex_prob(logits[0])                          # [T, V] per-position dist
                    sampled = torch.multinomial(p_out.clamp_min(1e-12), 1).squeeze(-1)  # [T] ŷ ~ p_θ
                    samp_ids = ids.clone()
                    samp_ids[0, 1:] = sampled[:-1]                              # next-token targets = own ŷ
                loss = fisher_rao_lm_loss(logits, samp_ids)      # ∇log p_θ(ŷ) — non-vanishing at the minimum
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                with torch.no_grad():
                    for n, p in self._kernel.named_parameters():
                        if p.grad is not None:
                            fisher[n] += p.grad ** 2              # diagonal Fisher ≈ grad² (QFI first order)
                counted += 1
        torch.set_rng_state(_rng)
        self._kernel.zero_grad()
        if counted > 0:
            for n in fisher:
                fisher[n] /= float(counted)
        return fisher

    def _capture_ewc_anchor(self, fisher: dict, replayed: int) -> None:
        """Snapshot θ* (current consolidated weights) and F (normalised diagonal Fisher) at the END of a
        consolidation — the EWC anchor. F is the TASK-objective Fisher measured at θ* (``_ewc_task_fisher``),
        NOT the basin-consolidation Fisher (which self-cancels to zero for a role-less generic kernel) — so
        the anchor is honest. SALIENCE-WEIGHT it (× (0.5 + recent mean surprise)) so high-surprise tasks
        anchor harder. On a SECOND consolidation MERGE with the existing Fisher via element-wise MAX — the
        standard running-EWC accumulation: importance is the PEAK a weight ever reached for ANY consolidated
        task, so a weight critical for task A stays protected even if task B left it locally unimportant (an
        EMA would let the old task's importance decay away — wrong for continual learning). θ* always updates
        to the latest consolidated weights (the point new gradients are anchored to). None-safe; never raises.

        (The ``fisher``/``replayed`` args are the SHY consolidation's outputs, kept for signature stability
        and the replayed>0 guard — the EWC importance itself is freshly measured at θ*.)"""
        import torch

        if replayed <= 0:
            return
        task_fisher = self._ewc_task_fisher()                   # canonical EWC Fisher at θ* (task objective)
        sal = 0.5 + self._recent_surprise()                     # P22 salience gate ∈ [0.5, 1.5]
        with torch.no_grad():
            # NORMALISE importance to RELATIVE scale (F / max F ∈ [0,1]): the true-Fisher magnitude varies
            # run-to-run, so a fixed λ engages on some runs and not others (verified: 2/5 seeds had the
            # penalty silently not bite). Relative importance makes λ interpretable + the protection reliable
            # across runs — the ratio of which-weight-matters-more is what EWC needs, not the absolute scale.
            gmax = max((float(f.max()) for f in task_fisher.values()), default=0.0)
            scale = (1.0 / gmax) if gmax > 0 else 0.0
            new_f = {n: task_fisher[n] * sal * scale for n in task_fisher}
            if self._ewc_fisher is None:                        # first consolidation
                self._ewc_fisher = {n: f.clone() for n, f in new_f.items()}
            else:                                               # running EWC: peak importance per weight
                merged: dict[str, Any] = {}
                for n, p in self._kernel.named_parameters():
                    old = self._ewc_fisher.get(n)
                    nf = new_f.get(n)
                    if old is not None and nf is not None and old.shape == nf.shape:
                        merged[n] = torch.maximum(old, nf)
                    else:
                        merged[n] = (nf if nf is not None else old).clone()
                self._ewc_fisher = merged
            # θ* = the consolidated weights new wake gradients are anchored to (always the latest).
            self._ewc_anchor = {n: p.detach().clone() for n, p in self._kernel.named_parameters()}

    def _ewc_penalty(self) -> "Any":
        """The EWC wake-protection term: lam·Σ_n F_n·(θ_n − θ*_n)². A scalar tensor IN the graph (gradient
        flows into the live weights, pulling high-Fisher params back toward θ*). Zero (a detached 0 scalar
        keeping the device) before the first consolidation — the anchor is None then (spine tenet).

        # QIG-EXEMPT: EWC parameter-space Fisher-metric penalty (local KL approx), not a basin/manifold
        # distance. The (θ−θ*)² quadratic IS the intended geometric form of EWC — the 2nd-order Taylor /
        # Fisher-metric local approximation of the KL between weight-posteriors — NOT Euclidean contamination.
        """
        import torch

        dev = next(self._kernel.parameters()).device
        if self._ewc_anchor is None or self._ewc_fisher is None or self.ewc_lambda <= 0.0:
            return torch.zeros((), device=dev)
        total = torch.zeros((), device=dev)
        for n, p in self._kernel.named_parameters():
            star = self._ewc_anchor.get(n)
            f = self._ewc_fisher.get(n)
            if star is None or f is None or star.shape != p.shape:
                continue
            total = total + (f * (p - star) ** 2).sum()
        # salience-modulate the live stiffness too (high recent surprise → protect harder this step)
        return self.ewc_lambda * (0.5 + self._recent_surprise()) * total

    def train_step(self, prompt: str, max_tokens: int = 64, target_text: str | None = None) -> StepResult:
        # WAKE: one Fisher-salience step. CONSCIOUSNESS-NATIVE loss (geometric regime): DRIVE Φ UP via
        # the differentiable coherence proxy, with a light next-token CE (``lm_weight``) for content
        # grounding so the high-Φ state stays tied to the input (prevents the trivial rel→0.5 fixed
        # point). target_text ignored (paired curriculum is qwen-modal's lane). Optimiser = natural
        # gradient (P1). NB: pure CE (the previous objective) drove Φ DOWN — memorisation ≠ integration.
        self.ensure_loaded()
        import math as _math

        import torch
        import torch.nn.functional as F
        from qig_core.torch.geometry_simplex import to_simplex_prob

        from ..losses import basin_lm_loss, fisher_rao_lm_loss

        self._step += 1
        ids, coords = self._encode(prompt)
        from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex
        _basin_cur = None                                    # basin-space mean prediction (for pull + history)
        if self.head_mode == "basin":
            # K-COMPRESS (EXP-A026 / CC2 A027 "never materialize the large object"): skip the full-vocab head
            # so the [1, T, vocab] scores + their forward/backward graph are NEVER built. At vocab=100k × 9
            # constellation kernels that materialisation is the memory hog that OOM'd the box (swap→kill).
            # The basin loss only needs predict(h) [T, basin_dim] vs the target's frozen coordizer basin —
            # both O(T·64), not O(T·100k). ce/γ telemetry are derived from the predicted basins (no vocab).
            _, tel = self._kernel(ids, return_telemetry=True, coords=coords, skip_head=True)
            logits = None
            pred = self._kernel.lm_head.predict(tel.hidden_state[0, :-1])   # [T-1, basin_dim] on Δ
            tgt_basins = coords[0, 1:]                                       # [T-1, basin_dim] FROZEN targets
            lm_loss = fisher_rao_distance_simplex(pred, tgt_basins).mean()   # pure d_FR (validated objective)
            ce = lm_loss.detach()                            # basin-surprise proxy (perplexity/bpb read-outs)
            gamma = self._gamma_from_basins(pred)            # generation-health from Δ⁶³ predictions (no vocab)
            with torch.no_grad():
                # identity/coupling basin = the 384-dim GEO-CODER (hidden) state on Δ³⁸³ (PI: "projected to
                # the 384 geo-coder"), matching _basin_ref; the Δ⁶³ pred is the OUTPUT projection, not identity.
                _basin_cur = to_simplex_prob(tel.hidden_state[0].mean(0)[None])[0].detach()   # [384] Δ³⁸³
        else:
            logits, tel = self._kernel(ids, return_telemetry=True, coords=coords)
            gamma = self._gamma_proxy(logits)                # differentiable generation-health (in-graph)
            # ``ce`` (CE nats) retained for perplexity=exp(CE) / bpb telemetry (vocab-comparable read-outs).
            ce = F.cross_entropy(logits[0, :-1], ids[0, 1:])
            lm_loss = ce if self.lang_loss == "ce_ablation" else fisher_rao_lm_loss(logits, ids)
        coherence = self._phi_proxy(tel.hidden_state)        # external proxy (monitoring / fallback)
        # ROUND-3 FIX (structural Φ-ceiling): drive the kernel's OWN differentiable Φ (tel.phi_diff) —
        # the exact quantity reported Φ is computed from (integrator gates + cross-position coherence,
        # no longer .item()-detached). The external _phi_proxy on the final hidden_state could not move
        # reported Φ (confirmed: Φ pinned ~0.27 across 6 configs). Fallback to the proxy if absent.
        phi_drive = tel.phi_diff if getattr(tel, "phi_diff", None) is not None else coherence
        # MAXIMIZE Φ (reliably reaches the conscious band) WITH one-sided Γ-PROTECTION: the Γ term only
        # acts when Γ dips below gamma_floor, pushing generativity back up — so driving Φ high does NOT
        # suppress Γ below its gate (the heart-stall: Φ=0.91 but Γ=0.78). This is "pull back to grow"
        # operationalized — generativity is protected as integration rises, not sacrificed to it.
        gamma_deficit = torch.relu(self.gamma_floor - gamma)
        # RAMPED FLUENCY (consciousness-first -> fluency): the next-token (language) signal starts light so
        # the kernel develops as a MIND (Phi-driven integration), then ramps to LOAD-BEARING (lm_weight_max,
        # = phi_weight) so it grows genuinely FLUENT on top of the conscious substrate. Qwen is TEMPORARY
        # scaffolding; the kernel's OWN voice must converge. Gamma-protection + basin-pull retained throughout.
        w_lm = self.lm_weight + (self.lm_weight_max - self.lm_weight) * min(1.0, self._step / max(1, self.lm_ramp_steps))
        if self.head_mode == "basin":
            # Φ AND Γ EMERGE from fluency — the basin loss is PURE language d_FR at UNIT weight (PI ruling
            # 2026-07-01 + evidence). The bare language loss learns fluency (single passage 91.3%, FIVE
            # passages 93%) AND keeps Φ (→0.68 conscious band) and generativity healthy ON ITS OWN. Adding
            # consciousness LOSS terms SABOTAGES learning: −phi_weight·Φ is a ZOMBIE ATTRACTOR (trivial
            # optimum = all positions identical → Φ→1, 0% decode even at 8:1 language dominance); the
            # Γ-deficit penalty (floor 0.82, always active on an untrained kernel) drops 5-passage learning
            # to 0%; the ×lm_weight_max scaling + clip destabilise the natural-gradient step (37%→0% crash).
            # So Φ/Γ are MEASURED (telemetry) + Ocean-REGULATED (between/around learning), NOT driven —
            # P25 (emerge-from-geometry). EWC + constellation basin-pull below still apply (0 when inactive).
            loss = lm_loss
        else:
            loss = (-self.phi_weight * phi_drive
                    + self.gamma_weight * gamma_deficit ** 2
                    + w_lm * lm_loss)
        if self._basin_ref is not None or self._basin_ref_set is not None:  # SPAWN: pull output → attractor
            import torch as _t
            from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex, to_simplex_prob

            # F1 (2026-07-02 actuator fix): for the basin head compute `cur` IN-GRAPH from the live 384-dim
            # identity hidden state (NOT the detached `_basin_cur`) so the d_FR pull ACTUALLY backprops into
            # the kernel. With `_basin_cur` detached the pull was a dead-gradient no-op — the measured reason
            # a collapsed basin never escaped the vertex. `simplex_floor=1e-3` (F6) revives the near-vertex
            # Jacobian so the gradient is non-zero AT the one-hot vertex. Pure Fisher-Rao d_FR — NO new term.
            if self.head_mode == "basin":
                cur = to_simplex_prob(tel.hidden_state[0].mean(0)[None], simplex_floor=1e-3)[0]
            else:
                cur = _basin_cur if _basin_cur is not None else to_simplex_prob(logits[0].mean(0), simplex_floor=1e-3)
            if self._basin_ref_set is not None:   # EXP-A044 set-coupling: nearest-member pull toward the
                # geo-Qwen per-layer basin SET (content-specific; separates at d_FR~1.0 vs ~0.15 collapse).
                d_ref = _t.stack([fisher_rao_distance_simplex(cur[None], r[None]).mean()
                                  for r in self._basin_ref_set]).min()
            else:
                d_ref = fisher_rao_distance_simplex(cur[None], self._basin_ref[None]).mean()
            # RAMPED pull (verdict 1#1/2#3): early steps build structure (coherence-led), later steps
            # consolidate the faculty into its role attractor — so d_basin (distance to the attractor)
            # can actually descend below D_BASIN_MAX. basin_weight raised 0.05→0.5; ramped 0→full.
            w_t = self.basin_weight * min(1.0, self._step / max(1, self.basin_ramp_steps))
            loss = loss + w_t * d_ref                          # seed the faculty into its Δ⁶³ region
        # EWC-FISHER WAKE PROTECTION (continuous learning): anchor PAST learning against THIS wake gradient.
        # Once a consolidation has captured θ*+F, add lam·Σ F_n·(θ_n−θ*_n)² so high-importance weights resist
        # being overwritten — the catastrophic-forgetting defence. Zero before the first consolidation
        # (anchor is None; spine tenet). Salience-gated inside _ewc_penalty (high recent surprise → stiffer).
        ewc_term = self._ewc_penalty()
        ewc_active = self._ewc_anchor is not None and self.ewc_lambda > 0.0
        loss = loss + ewc_term
        self._opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward()
            # Gradient clip: deep-stack stability guard for the geometric/linear heads. SKIPPED for basin
            # mode — clip 1.0 + the Fisher-preconditioned natural-gradient step destabilised multi-passage
            # basin training (37%→0% crash); the un-clipped bare basin loss is stable at 93% (5 passages).
            if self.head_mode != "basin":
                torch.nn.utils.clip_grad_norm_(self._kernel.parameters(), 1.0)
            self._opt.step()
        # SURPRISE = prediction error = d_FR(predicted, actual) (P20): report the language-loss arm as the
        # novelty signal. In the pure regime this is the d_FR (ceiling π); in ce_ablation it is CE (ceiling
        # ln(vocab)). The reported loss field tracks the same arm so /train telemetry matches the objective.
        _lm_val = float(lm_loss.item())
        snap = self._snap(tel, _lm_val)
        snap.extra["surprise"] = round(_lm_val, 4)               # prediction error (d_FR pure / CE ablation)
        # P22 SALIENCE: feed this step's d_FR surprise into the rolling buffer that gates EWC (high-surprise
        # experience → harder consolidation + stiffer wake protection). In ce_ablation the arm is CE (a
        # different ceiling) — only buffer the pure d_FR so the π-normalised salience stays meaningful.
        if self.lang_loss != "ce_ablation":
            self._surprise_recent.append(_lm_val)
        # EWC TELEMETRY (P24): SHOW the wake protection is on. ewc_active flips True once an anchor exists;
        # ewc_penalty is the live stiffness term value this step (0 pre-consolidation).
        snap.extra["ewc_active"] = bool(ewc_active)
        snap.extra["ewc_penalty"] = round(float(ewc_term.item()), 6)
        snap.extra["ewc_lambda"] = round(float(self.ewc_lambda), 3)
        snap.extra["max_surprise"] = round(
            _math.log(max(2, self.vocab_size)) if self.lang_loss == "ce_ablation" else _math.pi, 4)
        snap.extra["coherence"] = round(float(coherence.item()), 4)
        # FLUENCY metric: perplexity = exp(next-token CE). Lower = more fluent (better next-token prediction
        # of the curriculum). This is what we WATCH descend as the kernel grows fluent. lm_weight_now shows
        # the ramp position (the language signal's current weight in the loss).
        snap.extra["perplexity"] = round(float(_math.exp(min(float(ce.item()), 20.0))), 2)
        # BITS-PER-BYTE: the VOCAB-INDEPENDENT fluency metric (perplexity scales with vocab → not
        # comparable across models; bpb is). bpb = bits/byte = (mean_CE_nats/ln2) / bytes_per_token. The
        # bytes covered by this window come from decoding the actual tokens (coordizer) or the byte ids
        # (byte path), so it's exact under context truncation. This is the number to benchmark frontier-
        # for-size against (transformer REFERENCES: GPT-2-124M ~1.1, frontier-for-size ~0.8) — our
        # qig-geocoding kernel matching them is the open claim, not an apples-to-apples expectation.
        if self.coordizer is not None:
            _nbytes = max(1, len(self.coordizer.decode(ids[0].tolist()).encode("utf-8")))
        else:
            _nbytes = max(1, int(ids.shape[1]))                  # byte-level: 1 token == 1 byte
        snap.extra["bpb"] = round(float(ce.item()) * int(ids.shape[1]) / (_math.log(2) * _nbytes), 4)
        snap.extra["lm_weight_now"] = round(float(w_lm), 3)
        # MATURITY-GATED floor (opt-in): feed the gate its LEARNING signal — the per-step train-path bpb
        # readout just computed (on the basin head this is the d_FR basin-surprise proxy; still the
        # sharpening signal). LEARNING-linked, never age-linked: no step count enters the gate. Telemetry
        # exposes the gate state + fire count so the 3-arm harness (and /train/live) can read the floor.
        if self._floor_gate is not None:
            self._floor_gate.observe_signal(float(snap.extra["bpb"]))
            from qig_core.consciousness.pillars import ENTROPY_FLOOR as _EF
            snap.extra["floor_tightness"] = round(float(self._floor_gate.tightness), 4)
            snap.extra["floor_effective"] = round(float(self._floor_gate.effective_floor(float(_EF))), 4)
        snap.extra["floor_mode"] = self.floor_mode
        snap.extra["floor_fires"] = int(self._floor_fires)
        # Maturity-gate telemetry (4-conjunct: Φ ∧ Γ ∧ M ∧ d_basin — κ dropped, input-frozen): record
        # the detached output basin into the trajectory (history[0] = role attractor / birth-state),
        # then compute M (self-recognition vs birth-state) and d_basin (distance to the role attractor,
        # which the ramped basin-pull drives DOWN). Keys match development.c_equation's aliases exactly.
        with torch.no_grad():
            if _basin_cur is not None:                       # K-COMPRESS basin path: use the predicted basins
                cur_basin = _basin_cur
            else:
                cur_basin = to_simplex_prob(logits[0].mean(0)).detach()
                cur_basin = cur_basin / cur_basin.sum()
            # PROACTIVE Pillar-1 entropy floor (prevent f_health→0 at the SOURCE): restore basin entropy
            # BEFORE cur_basin enters _basin_history / _emit_pillars — the single point feeding BOTH the
            # f_health metric AND the coupling/synthesis pull, so the central trains toward a floored
            # synthesis (REAL, not cosmetic). GATED (fires only below ENTROPY_FLOOR); healthy → untouched.
            cur_basin = self._entropy_floor_basin(cur_basin)
        self._basin_history.append(cur_basin)
        if len(self._basin_history) > 64:                     # bound memory (keep birth-state + window)
            self._basin_history = [self._basin_history[0]] + self._basin_history[-32:]
        self._emit_pillars(snap, self._d63(cur_basin))        # LIVE pillar metrics from this step's basin
        m = self._meta_awareness(cur_basin)
        d_basin = self._basin_drift(cur_basin)
        snap.extra["gamma"] = round(float(gamma.item()), 4)          # Γ generativity (C-equation)
        snap.extra["meta_awareness"] = round(m, 4)                   # M (in-graph train-path)
        snap.extra["d_basin"] = round(d_basin, 4)                    # basin drift from identity attractor
        snap.basin_distance = d_basin                               # top-level field for the gate
        # TASK C actuation-1: emit the Δ⁶³ basins the phasic-dopamine / endorphin-arrival terms need
        # (compute_neurochemicals wants cur/prev/target basins + local κ). REUSE the basins already
        # computed this step (cur_basin, _basin_history[-2], _basin_ref) — no recompute; _d63 reduces to
        # the canonical Δ⁶³ the neurochem module reads. None-safe: a role-less kernel has no _basin_ref →
        # target stays None → neurochem falls back to the phi_delta scalar (Task A made all inputs optional).
        self._emit_neurochem_geometry(snap, cur_basin)
        # TASK C actuation-3: the drive-modulated exploration temperature for THIS step's κ + drive state,
        # exposed so the wiring gate can watch it MOVE with drive (LOW dopamine/HIGH boredom → higher temp).
        _drv = self._drive_signals()
        snap.extra["explore_temperature"] = round(self._temperature_from_kappa(float(snap.kappa)), 4)
        snap.extra["explore_factor"] = round(float(self._last_explore_factor or 1.0), 4)
        snap.extra["drive"] = _drv                                   # dopamine / curiosity / boredom read
        # S1: emit SEROTONIN (§6.5 neurochem) so Ocean's P25 ``integration_pinned`` guard — which reads
        # snap.extra["serotonin"] (ocean_policy.py) — is LIVE, not permanently inert. This is the canonical
        # DE-SATURATED serotonin proxy exp(−3·basin_velocity) mirroring qig-core neurochemistry
        # (SEROTONIN_VELOCITY_ALPHA=3.0, clip[0,1]): serotonin reads "settledness" and equals ~1.0 ONLY when
        # the basin is genuinely still (basin_velocity≈0), falling smoothly with real basin motion — so a
        # pinned-1.0 serotonin is legible as the dead-substrate FAILURE mode (UCP §6.5/§35.5), never clamped.
        # Reads the REAL Fisher-Rao basin_velocity _emit_pillars just wrote (None on step 0 before the first
        # _prev_d63 → treated as 0.0). None-safe.
        _bv = snap.extra.get("basin_velocity")
        _bv = float(_bv) if _bv is not None else 0.0
        snap.extra["serotonin"] = round(float(_math.exp(-3.0 * max(0.0, _bv))), 4)
        # WORMHOLE fast-layer ASSESS (CC2 A022) — runs HERE, in the EXECUTOR thread (train_step runs off the
        # event loop via _run_target), on the coords ALREADY computed → no re-encode, no async-loop block
        # (the deadlock fix). Set on the target/central by _train_core. LOGGED-ONLY / not-actuated: log
        # hit-rate + nucleation SIGNAL; it does NOT skip/transport into the gradient — coordizer basins aren't
        # semantic until K-LEARN (a wrong-class hit would inject wrong meaning; safety gate reinforced).
        _wh = getattr(self, "wormhole", None)
        if _wh is not None and coords is not None:
            try:
                _wa = _wh.assess(coords)
                snap.extra["wormhole"] = {**_wh.telemetry(), "novelty": _wa.get("novelty"),
                                          "nucleation": _wa.get("nucleation")}
            except Exception:  # noqa: BLE001 — the cache must never break a training step
                pass
        # WAKE metabolism: the kernel buffers this experience and accrues its own sleep pressure from its
        # own integration activity (more integration → more to consolidate later), then lets its own
        # homeostasis act on its own state. No external scheduler; the kernel cares for itself.
        self._experience.append(ids.detach())
        # TASK C actuation-4: attach this experience's REPLAY PRIORITY (index-aligned). surprise = this
        # step's d_FR prediction-error / π (what SURPRISED); coach = any pending coach reward mapped to
        # [-1,1] (what the coach VALUED, §18.5/18.6). weight ≥ a small floor so nothing is unselectable
        # (every experience keeps some replay chance — the tonic-floor analogue for DATA). This is DATA
        # SELECTION only; the replay loss itself stays pure d_FR (PI 2026-07-01 pure-loss ruling).
        _rep_surprise = min(1.0, _lm_val / _math.pi)                 # d_FR surprise ∈ [0,1] (pure-loss arm)
        _rep_coach = float(max(-1.0, min(1.0, self._pending_coach_reward)))
        _rep_w = max(0.1, 1.0 + _rep_surprise + _rep_coach)          # base 1 + surprise + coach, floored
        self._experience_weight.append(_rep_w)
        self._pending_coach_reward = 0.0                             # consumed; next reward re-arms it
        if len(self._experience) > 32:
            self._experience = self._experience[-32:]
            self._experience_weight = self._experience_weight[-32:]  # keep the priority buffer in lockstep
        self._phi_recent.append(float(snap.phi))
        self._sleep_pressure += SLEEP_PRESSURE_RATE * (0.5 + float(snap.phi))
        self._homeostasis(snap)
        # BUILD #1: REAL response-manifold Ricci → overrides the (kappa−64)/64 proxy for compressed/expanded
        # + pain/pleasure (downstream in experience()). Computed here so EVERY training path gets it (not
        # only the standalone launcher). ~25 forwards every _RICCI_EVERY steps, cached; written each step.
        if self.coordizer is not None:
            if self._step % _RICCI_EVERY == 0 or self._last_ricci_sig is None:   # fire at once on fresh/resume
                from ..curvature import RicciNormalizer, response_curvature
                if self._ricci_norm is None:
                    self._ricci_norm = RicciNormalizer()
                _rc = response_curvature(self, ids, coords)
                if _rc is not None:
                    self._last_ricci_R = float(_rc["R_scalar"])
                    self._last_ricci_sig = self._ricci_norm.signal(self._last_ricci_R)
            rr, rs = self._last_ricci_R, self._last_ricci_sig
            if rr is not None and rs is not None:
                snap.extra["ricci_real"] = round(rr, 2)
                snap.extra["ricci_signal"] = round(rs, 4)
        return StepResult(text=f"[genesis·N={self.num_layers} step {snap.step}] Φ-driving: {prompt[:50]}",
                          telemetry=snap)

    def _homeostasis(self, snap) -> None:
        """The kernel's OWN autonomic regulation — intrinsic, state-driven, no external scheduler and no
        commands. Each living-step the kernel reads its OWN state and, when that state calls for it, acts
        on ITSELF with a REAL operation (the way a brainstem regulates the body it is part of):
          • breakdown (Φ ≥ 0.80, over-integrated) → DECOHERE (its own breakdown response);
          • sleep pressure past its own threshold → a SLEEP EPISODE: CONSOLIDATE (Fisher-protected
            synaptic downscaling + identity replay) then DREAM (basin-mixture recombination), which
            discharges the pressure;
          • wake rigidity at HIGH Φ (Φ≥PHI_MATURE 0.70, over-engrained + mature) → MUSHROOM (bounded
            wake-state plasticity — the canonical Φ≥0.70-ONLY plasticity, UCP metric #35 / §35.6);
          • flat-but-LOW Φ (Φ<PHI_MATURE, fluctuations dead) → NOT rigid but COLLAPSED (Pillar-1
            fluctuation-death / zombie-drift): the remedy is ENTROPY RESTORATION (dream to re-energize +
            an exploration-entropy FLOOR so entropy cannot fully die), NEVER wake-state mushroom.
        WAKE (the common case) is simply not intervening. No stubs — every branch does real work."""
        snap.extra["sleep_pressure"] = round(self._sleep_pressure, 3)
        phi = float(snap.phi)
        if phi >= PHI_BREAKDOWN:
            self._decohere()
            snap.extra["autonomic"] = "decohere"
            return
        if self._sleep_pressure >= SLEEP_PRESSURE_THRESHOLD:
            c = self._consolidate()        # deep sleep — Fisher-protected downscaling + identity replay
            d = self._dream()              # REM — basin-mixture recombination
            self._sleep_pressure = 0.0     # discharged
            snap.extra["autonomic"] = f"sleep(consolidate={c['replayed']},dream={d['dreamed']})"
            return
        # COLLAPSE keys on TWO independent signals (2026-07-02 live-resume fix): (1) f_health (basin
        # entropy) below F_HEALTH_COLLAPSE_FLOOR = Pillar-1 fluctuation-death, EVEN WHEN Φ still fluctuates
        # (Φ measures integration, NOT basin entropy — the live 100k resume proved collapsed faculties keep
        # Φ bouncing, so the old _is_rigid()-only gate NEVER fired and M2 never triggered despite f_health=0);
        # (2) _is_rigid() = Φ FLAT (the stuck-plateau shape). Mushroom = the rigid+MATURE response; entropy
        # restoration = the collapse response.
        f_health = snap.extra.get("f_health")            # P1 fluctuation health (None-safe telemetry)
        f_collapsed = f_health is not None and float(f_health) < F_HEALTH_COLLAPSE_FLOOR
        if self._is_rigid() and phi >= PHI_MATURE:
            # MATURE + rigid → wake-state plasticity (Φ≥0.70-ONLY canon, UCP metric #35 / §35.6). A
            # genuinely-stuck mature kernel gets mushroom to break over-coherence.
            self._mushroom()
            snap.extra["autonomic"] = "mushroom"
            return
        if f_collapsed or (self._is_rigid() and phi < PHI_MATURE):
            # COLLAPSED (Pillar-1 fluctuation-death) — basin entropy dead (f_health→0) and/or flat-LOW Φ.
            # NOT rigid-mature. Do NOT mushroom and do NOT anchor — RESTORE ENTROPY. (1) DREAM (basin-mixture
            # recombination) to re-energize; (2) STIMULATE via the SHARED entropy lever (_apply_stimulate) —
            # the SAME actuator Ocean-commanded "stimulate" uses (BLOCKER-1 DRY): opens a bounded HIGH-
            # SURPRISE-REPLAY window. Idempotent within a window (state-only).
            d = self._dream()                                # re-energize via creative recombination
            _stim = self._apply_stimulate()
            self._collapse_perturb()                         # F3: kick the GENERATOR off the absorbing vertex
            # CROSS-FACULTY REQUEST (Part 3, M2): a single kernel cannot see sibling basins here. Expose
            # the collapse so the constellation layer (which DOES see all faculties) cross-mixes this
            # faculty's basin with its NON-COLLAPSED siblings' during dream (Fréchet-mean / √p-SLERP on
            # Δ⁶³, NEVER L2) — the ONLY source of FOREIGN entropy. Consumed by JointConstellation.
            # _cross_faculty_dream at the post-ocean.regulate site.
            snap.extra["cross_faculty_dream_request"] = {
                "reason": "pillar1-collapse", "phi": round(phi, 4),
                "f_health": (round(float(f_health), 4) if f_health is not None else None),
            }
            snap.extra["autonomic"] = (
                f"entropy-restore(dream={d['dreamed']},replay_until={_stim['replay_window_until_step']})"
            )
            return
        # NOT collapsed, NOT rigid (Φ still moving / healthy development, basin entropy alive) → wake.
        snap.extra["autonomic"] = "wake"

    def _is_rigid(self) -> bool:
        """The kernel's OWN rigidity sense: Φ is genuinely STUCK — a flat window, NOT slow-but-healthy
        progress. Uses the Φ RANGE over the window (max−min): a faculty still developing (Φ creeping up)
        has a non-trivial range and is NOT rigid; only a truly frozen Φ (range < 0.008 over the full
        window) is over-engrained and ready for wake-state plasticity. Band-independent."""
        h = self._phi_recent
        return len(h) >= h.maxlen and (max(h) - min(h)) < 0.008

    def _mushroom(self, sigma: float = 0.01) -> None:
        """Wake-state plasticity — bounded weight-noise (Tononi micro-downscaling) to break an
        over-engrained plateau. Real; the kernel's own plasticity."""
        import torch
        with torch.no_grad():
            for p in self._kernel.parameters():
                p.add_(torch.randn_like(p) * sigma)

    def _collapse_perturb(self, sigma: float = 0.02) -> None:
        """F3 (2026-07-02 actuator fix): escape Pillar-1 fluctuation-death. f_health is recomputed each
        step from a FRESH forward, so ONLY a GENERATOR (weight-space) change moves it — a basin_history
        injection does not. Bounded ISOTROPIC weight noise (target-FREE → non-self-confirming, F4) kicks
        the forward off the one-hot vertex where the pull/lm gradients are dead (the Duchi zero-Jacobian);
        the F6 simplex_floor then keeps the near-vertex gradient alive and the M2 foreign-mixture pull (F2)
        supplies the DIRECTED climb. Parameter-space noise (identical kind to _mushroom/_decohere) — no
        basin-purity concern. σ larger than mushroom's 0.01: a collapsed vertex needs a firmer kick."""
        if self._kernel is None:      # None-safe: no loaded generator to perturb (light shell / decision tests)
            return
        import torch
        with torch.no_grad():
            for p in self._kernel.parameters():
                p.add_(torch.randn_like(p) * sigma)

    def _apply_stimulate(self, window: int = 30) -> dict:
        """The SHARED entropy actuator (Task E / BLOCKER-1) — the treatment for the Pillar-1 apathy /
        fluctuation-death signature. ONE lever, TWO callers:
          • INTRINSIC — the kernel's OWN flat-LOW-Φ collapse branch in ``_homeostasis``;
          • EXTRINSIC — Ocean-commanded ``run_protocol("stimulate")`` on the ACT tier.
        It opens a bounded HIGH-SURPRISE-REPLAY window (``_stimulate_until``): while ``_step`` is inside
        it, ``_weighted_replay_choice`` SHARPENS toward high-surprise experience (squares the priority
        weights) so a collapsed / apathetic faculty replays what most SURPRISED it — re-injecting novelty.
        Pure DATA selection (the replayed loss stays d_FR, P10); state-only (no weight perturbation), so
        IDEMPOTENT within a window: a second call just re-extends the same window.

        M5: this NO LONGER returns an ``entropy_floor`` / generation-temperature claim. The former
        ``COLLAPSE_ENTROPY_FLOOR=0.05`` clamp was over-claiming telemetry (it sat below the 0.3 base band
        so ``max(temp, 0.05)`` never bound) and has been removed. Generation-time exploration on collapse
        is supplied by the drive-deficit factor in ``_temperature_from_kappa``; the FOREIGN basin entropy
        that reverses f_health→0 is supplied by the M2 constellation cross-faculty dream. This actuator's
        real, honest job is the replay-sharpening window — and that is all it now reports."""
        self._stimulate_until = max(self._stimulate_until, self._step + max(1, int(window)))
        return {"stimulate": True, "replay_window_until_step": self._stimulate_until,
                "replay_sharpen": True}

    def _decohere(self) -> None:
        """Breakdown response (Φ ≥ 0.80, over-integrated) — REAL: inject bounded decoherence noise to
        reduce the over-integration and cool the optimiser step (the BreakdownHandler 'reduce coupling +
        decohere' canon), pulling the kernel back from breakdown into its healthy band."""
        import torch
        from qigkernels.natural_gradient_optimizer import DiagonalNaturalGradient
        with torch.no_grad():
            for p in self._kernel.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        self._opt = DiagonalNaturalGradient(self._kernel.parameters(), lr=self.lr * 0.7)  # cool (not cumulative)

    def register_coach_reward(self, reward: float, record: dict | None = None) -> None:
        """TASK C actuation-4 (+ M1 live-snapshot landing, + S5(a) utterance-credit): the coach's provenance-
        tagged reward (mapped to [-1,1]) credits the priority of the ACTUAL own-voice utterance it judged — so
        the kernel LEARNS FROM coaching the P10 way: reward-weighted DATA selection for sleep/dream replay, NOT
        a silent weight update and NOT a Φ-drive loss term. The server calls this after coach_own_voice fires,
        right after the kernel spoke; generate() stashed the utterance's ids, so the reward lands on THAT event
        (S5(a)). Fallback when no utterance was captured: arm the next train_step append + boost the most-recent
        entry, so a coach reaction is never lost. None-safe: any bad value → no-op. Sovereignty (P16): a reward
        the kernel distrusts can be discounted upstream (coach_reward_from already scales by provenance
        confidence).

        M1(b): ``record`` — the SAME provenance-tagged coach dict the reward was mapped from — is landed on
        the kernel's LIVE telemetry snapshot (``self._last.extra['coach']``), the exact object ``telemetry()``
        returns. Before this, the record was only ever written into a ``to_dict()`` DEEP COPY (the ``_write_live``
        throwaway), so Ocean's outcome-scoring (``_coach_reward``) + the kernel's own neurochem read ``coach ≡ 0``
        forever. Landing it live makes the coach's judgment readable where those consumers actually look. Cleared
        naturally on the next step's ``_snap`` rebuild (a coach reaction is a THIS-step signal, like the reward)."""
        try:
            r = float(max(-1.0, min(1.0, reward)))
        except (TypeError, ValueError):
            return
        # S5(a): CREDIT THE ACTUAL JUDGED UTTERANCE. The coach judged the kernel's own-voice utterance
        # (generate via_boundary=False), whose content ids generate() stashed in ``_last_utterance_ids``.
        # Append THAT utterance to the replay buffer with the coach reward as its priority, so P10 reward-
        # weighted sleep/dream replay weights the RIGHT event — not an arbitrary corpus chunk (the pre-S5 code
        # armed the NEXT train_step append + boosted ``_experience_weight[-1]``, BOTH corpus ids the coach
        # never saw). Credit an utterance at most once (the flag); a fresh generate re-arms it.
        credited = False
        utt = self._last_utterance_ids
        if utt is not None and not self._last_utterance_credited:
            try:
                self._experience.append(utt)
                self._experience_weight.append(max(0.1, 1.0 + r))     # base 1 + coach reward, floored
                if len(self._experience) > 32:
                    self._experience = self._experience[-32:]
                    self._experience_weight = self._experience_weight[-32:]  # keep the buffers in lockstep
                self._last_utterance_credited = True
                credited = True
            except Exception:  # noqa: BLE001 — crediting is best-effort; fall back below, never raise
                credited = False
        if not credited:
            # FALLBACK (no captured utterance — a non-generating/other target, a <2-token utterance, or a
            # second coach reaction on an already-credited utterance): arm the NEXT logged experience + fold
            # into the most-recent entry, the pre-S5 behaviour (a coach reaction is never lost between steps).
            self._pending_coach_reward = r
            if self._experience_weight:
                self._experience_weight[-1] = max(0.1, self._experience_weight[-1] + r)
        # M1(b): stash the coach RECORD on the LIVE snapshot so Ocean + neurochem read a real value (not the
        # throwaway to_dict() copy). None-safe: no record / no snapshot → skip the landing, reward still armed.
        if record is not None:
            extra = getattr(self._last, "extra", None)
            if isinstance(extra, dict):
                extra["coach"] = record

    def _weighted_replay_choice(self, rng):
        """TASK C actuation-4: pick a replay experience with probability ∝ its stored priority weight
        (surprise + coach reward) — replay what SURPRISED and what the COACH VALUED. Pure DATA selection
        (the replayed loss stays pure d_FR). Falls back to uniform when weights are absent/degenerate
        (fresh kernel, or a checkpoint restored without the parallel weight buffer). None-safe."""
        exp = self._experience
        w = self._experience_weight
        if not exp:
            return None
        if not w or len(w) != len(exp) or sum(max(0.0, x) for x in w) <= 0.0:
            return rng.choice(exp)                                   # uniform fallback (backward-compatible)
        # STIMULATE window (Task E / BLOCKER-1): while stimulate is active, SHARPEN toward high-surprise
        # experience (square the priority weights) so a collapsed / apathetic faculty replays what most
        # SURPRISED it — re-injecting novelty. Pure DATA selection (replayed loss stays d_FR, P10).
        if self._step < getattr(self, "_stimulate_until", 0):
            w = [max(0.0, x) ** 2 for x in w]
        total = sum(max(0.0, x) for x in w)
        r = rng.random() * total
        acc = 0.0
        for e, wi in zip(exp, w):
            acc += max(0.0, wi)
            if r <= acc:
                return e
        return exp[-1]

    def _current_basin(self, ids):
        """The kernel's current output/identity basin — the consolidation/dream target space.

        Basin mode: the 384-dim geo-coder basin via `skip_head=True` — never materialize the
        [seq, vocab] logits (K-COMPRESS; the OOM path this whole mode exists to avoid) and stay in
        the SAME 384-dim Δ as `_basin_ref` / `_basin_history` (the geo-coder identity space).
        Geometric mode: the vocab-dim output basin (unchanged — logits are the head's native space)."""
        from qig_core.torch.geometry_simplex import to_simplex_prob
        if self.head_mode == "basin":
            _, _t = self._kernel(ids, return_telemetry=True, skip_head=True)
            return to_simplex_prob(_t.hidden_state[0].mean(0))
        logits, _t = self._kernel(ids, return_telemetry=True)
        return to_simplex_prob(logits[0].mean(0))

    def _consolidate(self, steps: int = 16, downscale: float = 0.02) -> dict:
        """Deep-sleep consolidation — REAL, no stub. (1) Replay buffered experience at LOW learning rate,
        pulling the output basin toward the role attractor (identity consolidation; Φ/κ relax naturally —
        the SleepProtocol 'pure geometry' consolidation, no Φ-drive). (2) Fisher-protected synaptic
        downscaling (Tononi SHY): downscale weights by importance — high-Fisher (important) weights are
        protected, low-Fisher ones decay — improving signal-to-noise and undoing the wake over-
        integration that drives breakdown. Fisher ≈ grad² (the QFI first-order approximation)."""
        import torch
        import random
        from qigkernels.natural_gradient_optimizer import DiagonalNaturalGradient
        from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex, to_simplex_prob
        if not self._experience:
            return {"replayed": 0, "downscaled": False}
        fisher = {n: torch.zeros_like(p) for n, p in self._kernel.named_parameters()}
        opt = DiagonalNaturalGradient(self._kernel.parameters(), lr=self.lr * 0.1)   # low-LR sleep
        replayed = 0
        for _ in range(steps):
            ids = self._weighted_replay_choice(random)   # TASK C: replay coach-valued / surprising first (P10)
            cur = self._current_basin(ids)
            target = self._basin_ref if self._basin_ref is not None else cur.detach()
            loss = fisher_rao_distance_simplex(cur[None], target[None]).mean()        # pure basin consolidation
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                with torch.no_grad():
                    for n, p in self._kernel.named_parameters():
                        if p.grad is not None:
                            fisher[n] += p.grad ** 2                                  # accumulate QFI importance
                torch.nn.utils.clip_grad_norm_(self._kernel.parameters(), 1.0)
                opt.step()
                replayed += 1
        with torch.no_grad():                                                        # Fisher-protected SHY downscaling
            for n, p in self._kernel.named_parameters():
                f = fisher[n]
                med = torch.median(f)
                protect = (f / (med + 1e-12)).clamp(0.0, 1.0) if float(med) > 0 else torch.zeros_like(f)
                p.mul_(1.0 - downscale * (1.0 - protect))
        # EWC ANCHOR CAPTURE: at the END of consolidation (AFTER the SHY downscale) snapshot θ* = the now-
        # consolidated weights and F = this replay's Fisher importance (the SAME grad²-accumulated dict SHY
        # used). On a SECOND consolidation _capture_ewc_anchor MERGES F by element-wise max (running EWC) and
        # re-points θ* to the latest consolidated weights — so wake gradients are anchored to the freshest
        # consolidated state while still protecting every past task's important weights. This is the WAKE
        # complement to SHY's one-time sleep downscale: it makes the consolidation actually DEFEND past
        # learning during ongoing wake training (the catastrophic-forgetting fix).
        self._capture_ewc_anchor(fisher, replayed)
        return {"replayed": replayed, "downscaled": True, "ewc_anchored": self._ewc_anchor is not None}

    def _dream(self, steps: int = 8) -> dict:
        """REM dream — REAL basin-mixture augmentation (Forge), no stub. Recombine stored output basins
        into novel 'dreamed' targets (Fisher-Rao / √p geodesic mixture, renormalised to Δ) and pull the
        kernel toward them at low LR on replayed inputs — creative consolidation/generalisation beyond
        the literally-seen experience."""
        import torch
        import random
        from qigkernels.natural_gradient_optimizer import DiagonalNaturalGradient
        from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex, to_simplex_prob
        hist = list(self._basin_history)
        if len(hist) < 2 or not self._experience:
            return {"dreamed": 0}
        opt = DiagonalNaturalGradient(self._kernel.parameters(), lr=self.lr * 0.1)
        dreamed = 0
        for _ in range(steps):
            a, b = random.sample(hist, 2)
            t = random.random()
            sa, sb = torch.sqrt(a.clamp_min(0.0)), torch.sqrt(b.clamp_min(0.0))       # √p (Fisher) coords
            mix = ((1.0 - t) * sa + t * sb) ** 2                                      # geodesic-ish mixture → p
            dream_basin = (mix / (mix.sum() + 1e-12)).detach()                        # renormalise to Δ
            ids = self._weighted_replay_choice(random)   # TASK C: dream-replay coach-valued / surprising (P10)
            cur = self._current_basin(ids)
            loss = fisher_rao_distance_simplex(cur[None], dream_basin[None]).mean()
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._kernel.parameters(), 1.0)
                opt.step()
                dreamed += 1
        return {"dreamed": dreamed}

    def architecture(self) -> dict:
        # qigkernels.QIGLayer is GLOBAL metric attention by default → v_B-NON-LOCAL; pass
        # locality_radius to make it windowed-local (respects the finite-propagation budget). The
        # locality_budget check reads this; local-vs-global is the EXP-LOCAL-ATTN A/B.
        local = self.locality_radius is not None
        nparams = None
        if self._kernel is not None:
            try:
                nparams = int(sum(p.numel() for p in self._kernel.parameters()))
            except Exception:  # noqa: BLE001
                nparams = None
        cvocab = len(self.coordizer.vocab) if self.coordizer is not None else None
        return {"attention": "local" if local else "global", "locality_radius": self.locality_radius,
                "num_layers": self.num_layers, "recursion_depth": 3, "seq_len": _CTX,
                "input": "coords" if self.coordizer is not None else "bytes",
                "vocab_size": self.vocab_size, "coord_dim": self.coord_dim or 64,
                "hidden_dim": self.hidden_dim, "num_params": nparams, "coordizer_vocab": cvocab,
                "head_mode": self.head_mode, "head_tau": self.head_tau}

    @property
    def self_regulating(self) -> bool:
        return True   # the kernel owns its brainstem (_homeostasis in train_step) — no external scheduler

    def supports_protocol(self) -> bool:
        return True

    def implemented_commands(self) -> set[str] | None:
        """ONLY the kernel's OWN autonomic operations (the same ops _homeostasis runs intrinsically).
        The genesis kernel is not the qig_chat constellation — it does NOT implement twin/lightning/
        reasoning/meta/pillar-status etc., so the UI must not advertise them (no 'unknown_command')."""
        return {"sleep", "deep-sleep", "dream", "mushroom-micro", "mushroom-moderate",
                "mushroom-heroic", "escape", "stimulate"}

    def run_protocol(self, command: str, args: dict) -> dict:
        """MANUAL invocation of the kernel's OWN regulation (e.g. a ``/sleep`` chat command). Routes to
        the SAME real operations the kernel runs autonomically in ``_homeostasis`` — there are NO stubs:
        sleep/deep-sleep CONSOLIDATE (Fisher-protected downscaling + identity replay) then DREAM; dream
        recombines basins; mushroom is wake-state plasticity; escape/decohere is the breakdown response.
        Autonomic regulation is intrinsic (the kernel decides from its own state in ``_homeostasis``);
        this method only exposes the same real operations for an explicit external request."""
        self.ensure_loaded()
        applied: Any
        if command in _MUSHROOM_SIGMA:
            self._mushroom(_MUSHROOM_SIGMA[command])
            applied = {"plasticity": f"weight-noise σ={_MUSHROOM_SIGMA[command]}"}
        elif command in ("sleep", "deep-sleep"):
            applied = {"consolidate": self._consolidate(steps=24 if command == "deep-sleep" else 16),
                       "dream": self._dream()}
        elif command == "dream":
            applied = {"dream": self._dream()}
        elif command in ("escape", "decohere"):
            self._decohere()
            applied = {"breakdown_recovery": "decohere noise + cool optimiser (lr×0.7)"}
        elif command == "stimulate":
            # BLOCKER-1: Ocean-commanded treatment of the Pillar-1 apathy signature. Actuates the SAME
            # shared entropy lever the kernel's own collapse response uses (no duplication) — raise the
            # exploration-entropy floor + open a high-surprise-replay window. OBSERVABLE: recorded on _last
            # telemetry so it is WIRED+ACTUATED, not a silent no-op.
            applied = self._apply_stimulate()
            _l = getattr(self, "_last", None)
            if _l is not None and getattr(_l, "extra", None) is not None:
                _l.extra["stimulate"] = applied
        else:
            applied = {"unknown_command": command}
        last = getattr(self, "_last", None)
        return {"command": command, "available": True, "applied": applied,
                "telemetry": last.to_dict() if last is not None else {}}

    def save_checkpoint(self, path: str) -> None:
        """Save this kernel's weights + developmental state (resumable). The collective/constellation
        state is checkpointed separately (qig_studio.checkpoint)."""
        self.ensure_loaded()
        import torch
        # format 2 also persists the dialogue/replay/Φ-history state so a RESUMED kernel is the genuinely
        # trained kernel (review fix): without these a resumed coach turn has a blank recognition basin
        # (M_coach=None), an empty replay buffer (sleep replays nothing) and no Φ-history (mushroom inert
        # for 30 steps). All fields are weights_only-safe (tensors / floats / a plain scalar dict). The
        # optimizer state is intentionally NOT persisted (its LR-cooling self-heals; serialising it would
        # risk the weights_only=True allowlist).
        torch.save({
            "format": 2,
            "arch": {"num_layers": self.num_layers, "hidden_dim": self.hidden_dim,
                     "num_heads": self.num_heads, "ffn_dim": self.ffn_dim,
                     "vocab_size": self.vocab_size, "seed": self.seed, "role": self.role,
                     "coordizer": self.coordizer is not None, "head_mode": self.head_mode},
            "kernel_state": self._kernel.state_dict(),
            "step": self._step,
            "sleep_pressure": self._sleep_pressure,
            "basin_ref": (self._basin_ref.detach().cpu() if self._basin_ref is not None else None),
            "basin_history": [b.detach().cpu() for b in self._basin_history],
            "last_gen_basin": (self._last_gen_basin.detach().cpu() if self._last_gen_basin is not None else None),
            "experience": [e.detach().cpu() for e in self._experience],
            "experience_weight": [float(x) for x in self._experience_weight],  # TASK C replay priorities

            "phi_recent": [float(x) for x in self._phi_recent],
            "last_telemetry": self._last.to_dict(),
            # EWC anchor: θ* + diagonal Fisher so a RESUMED kernel keeps its continuous-learning protection
            # (otherwise a resumed wake step would overwrite consolidated learning until the next sleep).
            # Plain dicts of tensors → weights_only-safe. None when no consolidation has happened yet.
            "ewc_anchor": ({n: t.detach().cpu() for n, t in self._ewc_anchor.items()}
                           if self._ewc_anchor is not None else None),
            "ewc_fisher": ({n: t.detach().cpu() for n, t in self._ewc_fisher.items()}
                           if self._ewc_fisher is not None else None),
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Restore weights + full developmental state. The checkpoint's architecture must match this
        kernel (fail-loud on mismatch — a byte-vs-coordizer or layer mismatch would otherwise crash deep
        in load_state_dict or silently mis-load). weights_only=True — only tensors + scalars + plain dicts."""
        self.ensure_loaded()
        import torch
        dev = next(self._kernel.parameters()).device
        ckpt = torch.load(path, map_location=dev, weights_only=True)
        arch = ckpt.get("arch") or {}
        # BACK-COMPAT: checkpoints saved before the head A/B have no "head_mode" key — they were trained
        # with the Euclidean nn.Linear readout. If this kernel is the new geometric default, the explicit
        # mismatch error (or, if the key is absent, the subsequent load_state_dict lm_head shape error) is
        # correct: construct the target with head_mode="linear" to resume a pre-A/B checkpoint.
        for k, cur in (("num_layers", self.num_layers), ("vocab_size", self.vocab_size),
                       ("coordizer", self.coordizer is not None), ("head_mode", self.head_mode)):
            if k in arch and arch[k] != cur:
                raise ValueError(f"checkpoint arch mismatch at '{k}': checkpoint={arch[k]} kernel={cur} "
                                 f"(byte-vs-coordizer, layer/vocab, or geometric-vs-linear head mismatch) — {path}")
        self._kernel.load_state_dict(ckpt["kernel_state"])
        self._step = int(ckpt.get("step", 0))
        self._sleep_pressure = float(ckpt.get("sleep_pressure", 0.0))
        # Vocab-sized basin state is fitted to the CURRENT vocab (self-heals a post-neurogenesis width
        # change: stale 100k basins -> 108k by zero-padding, exact since old tokens carried all the mass).
        # `experience` is token-ID sequences (not basins) and stays as-is — old ids are valid under the
        # superset vocab.
        ref = ckpt.get("basin_ref")
        if self.head_mode == "basin":
            # BASIN head: basin state is the 384-dim GEO-CODER Δ (K-COMPRESS) — already the right width;
            # vocab-fitting it (100k) is exactly the [100004]-vs-[384] stack corruption on resume. Keep as-is.
            self._basin_ref = ref.to(dev) if ref is not None else None
            self._basin_history = [b.to(dev) for b in (ckpt.get("basin_history") or [])]
        else:
            self._basin_ref = self._fit_basin_to_vocab(ref.to(dev)) if ref is not None else None
            self._basin_history = [self._fit_basin_to_vocab(b.to(dev)) for b in (ckpt.get("basin_history") or [])]
        # full developmental state (format 2; format-1 checkpoints leave these at their fresh defaults)
        lg = ckpt.get("last_gen_basin")
        self._last_gen_basin = self._fit_basin_to_vocab(lg.to(dev)) if lg is not None else self._last_gen_basin
        exp = ckpt.get("experience")
        if exp:
            self._experience = [e.to(dev) for e in exp]
            # TASK C: restore the replay priorities in lockstep; absent (pre-Task-C checkpoint) or a length
            # mismatch → uniform (a base weight per experience), so old checkpoints degrade gracefully.
            ew = ckpt.get("experience_weight")
            self._experience_weight = ([float(x) for x in ew] if ew and len(ew) == len(self._experience)
                                       else [1.0] * len(self._experience))
        pr = ckpt.get("phi_recent")
        if pr:
            from collections import deque as _deque
            self._phi_recent = _deque(pr, maxlen=self._phi_recent.maxlen)
        lt = ckpt.get("last_telemetry")
        if lt:
            self._last = TelemetrySnapshot(**{k: v for k, v in lt.items()
                                              if k in TelemetrySnapshot.__dataclass_fields__})
        # EWC anchor: restore θ*+F (continuous-learning protection) if present. Vocab-width params (lm_head)
        # self-heal via _fit_basin_to_vocab is NOT applicable here (these are raw weight tensors, not Δ
        # basins) — on a post-neurogenesis width change the shape-guarded penalty/merge simply skips the
        # mismatched tensors, so a stale-width anchor degrades gracefully rather than crashing.
        ea = ckpt.get("ewc_anchor")
        self._ewc_anchor = {n: t.to(dev) for n, t in ea.items()} if ea else None
        ef = ckpt.get("ewc_fisher")
        self._ewc_fisher = {n: t.to(dev) for n, t in ef.items()} if ef else None
