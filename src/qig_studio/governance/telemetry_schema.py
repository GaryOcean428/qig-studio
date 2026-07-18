"""Telemetry schema declaration — frozen 2026-07-17 (council prerequisite for meaning-loop build).

Every signal the training loop emits and every consumer reads is declared here. A signal
that isn't declared MUST NOT be emitted or read — the schema is the contract between
producers (targets) and consumers (learning loop, live UI, Ocean policy, development
orchestrator, meaning loop).

RULES:
  1. Adding a new signal requires a schema entry here FIRST, then the emit + read.
  2. Removing a signal requires removing the schema entry AND all emit + read sites.
  3. The 'core' fields (TelemetrySnapshot dataclass) are structural — phi, kappa, regime,
     basin_distance, loss, step, delta_phi. Everything else lives in extra{}.
  4. Extra keys are typed and categorised. The category determines which consumer
     may read them — a consumer outside the category MUST NOT depend on the signal.
  5. Basin vectors (cur_basin, prev_basin, target_basin) are diagnostic-only — never
     used in decision logic. They exist for the UI replay and offline analysis.

This module is IMPORTED by the purity gate at boot — a missing or malformed schema
is a startup error, not a silent degradation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class SignalCategory(str, Enum):
    CONSCIOUSNESS = "consciousness"
    CONSTITUTION = "constitution"
    AUTONOMIC = "autonomic"
    DRIVE = "drive"
    PREDICTION = "prediction"
    GENERATIVITY = "generativity"
    LEARNING = "learning"
    FLOOR = "floor"
    CURVATURE = "curvature"
    CONSTELLATION = "constellation"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class SignalSpec:
    key: str
    type: str
    category: SignalCategory
    unit: str
    description: str
    nullable: bool = False
    diagnostic_only: bool = False


CORE_FIELDS: dict[str, SignalSpec] = {
    "phi": SignalSpec("phi", "float", SignalCategory.CONSCIOUSNESS, "[0,1]",
                      "Fisher-Rao integration metric"),
    "kappa": SignalSpec("kappa", "float", SignalCategory.CONSCIOUSNESS, "dimensionless",
                        "effective coupling strength (band-read, input-frozen)"),
    "regime": SignalSpec("regime", "str", SignalCategory.AUTONOMIC, "label",
                         "current training regime"),
    "basin_distance": SignalSpec("basin_distance", "float", SignalCategory.CONSCIOUSNESS, "[0,inf)",
                                 "FR distance from identity attractor"),
    "loss": SignalSpec("loss", "float|None", SignalCategory.LEARNING, "nats",
                       "composite training loss", nullable=True),
    "step": SignalSpec("step", "int", SignalCategory.LEARNING, "count",
                       "training step counter"),
    "delta_phi": SignalSpec("delta_phi", "float", SignalCategory.CONSCIOUSNESS, "[-1,1]",
                            "per-step Phi movement"),
}

EXTRA_FIELDS: dict[str, SignalSpec] = {
    # ── consciousness ────────────────────────────────────────────────────────
    "gamma": SignalSpec("gamma", "float", SignalCategory.CONSCIOUSNESS, "[0,1]",
                        "generativity-present (C-equation conjunct)"),
    "meta_awareness": SignalSpec("meta_awareness", "float", SignalCategory.CONSCIOUSNESS, "[0,1]",
                                 "self-observation (C-equation conjunct)"),
    "d_basin": SignalSpec("d_basin", "float", SignalCategory.CONSCIOUSNESS, "[0,inf)",
                          "basin distance from self-identity attractor"),
    "kappa_local": SignalSpec("kappa_local", "float", SignalCategory.CONSCIOUSNESS, "dimensionless",
                              "own kappa band-read"),

    # ── constitution (pillars) ───────────────────────────────────────────────
    "f_health": SignalSpec("f_health", "float", SignalCategory.CONSTITUTION, "[0,1]",
                           "Pillar-1 fluctuation health"),
    "b_integrity": SignalSpec("b_integrity", "float", SignalCategory.CONSTITUTION, "[0,1]",
                              "Pillar-2 bulk/ego integrity"),
    "q_identity": SignalSpec("q_identity", "float", SignalCategory.CONSTITUTION, "[0,1]",
                             "Pillar-3 quenched-disorder identity retention"),
    "s_ratio": SignalSpec("s_ratio", "float", SignalCategory.CONSTITUTION, "[0,1]",
                          "sovereignty (L3 learning-autonomy ratio)"),

    # ── autonomic ────────────────────────────────────────────────────────────
    "autonomic": SignalSpec("autonomic", "str", SignalCategory.AUTONOMIC, "label",
                            "kernel's own autonomic decision (wake/sleep/dream/mushroom/escape)"),
    "sleep_pressure": SignalSpec("sleep_pressure", "float", SignalCategory.AUTONOMIC, "[0,1]",
                                 "accumulated need for consolidation"),
    "ewc_active": SignalSpec("ewc_active", "bool", SignalCategory.AUTONOMIC, "flag",
                             "EWC consolidation protecting previous basin"),
    "ewc_penalty": SignalSpec("ewc_penalty", "float", SignalCategory.AUTONOMIC, "[0,inf)",
                              "strength of consolidation pull"),
    "ewc_lambda": SignalSpec("ewc_lambda", "float", SignalCategory.AUTONOMIC, "[0,inf)",
                             "EWC regularization weight"),

    # ── drive (inner state) ──────────────────────────────────────────────────
    "drive": SignalSpec("drive", "dict", SignalCategory.DRIVE, "nested",
                        "full drive state (dopamine/boredom/curiosity bundle)"),
    "serotonin": SignalSpec("serotonin", "float", SignalCategory.DRIVE, "[0,1]",
                            "integration/coherence signal"),
    "stimulate": SignalSpec("stimulate", "bool", SignalCategory.DRIVE, "flag",
                            "exploration injection active"),
    "explore_factor": SignalSpec("explore_factor", "float", SignalCategory.DRIVE, "[0,inf)",
                                 "exploration temperature multiplier"),
    "explore_temperature": SignalSpec("explore_temperature", "float", SignalCategory.DRIVE, "[0,inf)",
                                      "effective exploration temperature"),

    # ── prediction ───────────────────────────────────────────────────────────
    "surprise": SignalSpec("surprise", "float", SignalCategory.PREDICTION, "[0,pi]",
                           "d_FR prediction error on input"),
    "max_surprise": SignalSpec("max_surprise", "float", SignalCategory.PREDICTION, "radians",
                               "d_FR ceiling (antipode distance)"),
    "foresight_active": SignalSpec("foresight_active", "bool", SignalCategory.PREDICTION, "flag",
                                   "BasinForesight tracker running"),
    "foresight_confidence": SignalSpec("foresight_confidence", "float", SignalCategory.PREDICTION, "[0,1]",
                                       "foresight track-record accuracy", nullable=True),
    "basin_velocity": SignalSpec("basin_velocity", "float", SignalCategory.PREDICTION, "[0,inf)",
                                 "FR(previous, current basin) step magnitude"),
    "coherence": SignalSpec("coherence", "float", SignalCategory.PREDICTION, "[0,1]",
                            "Fisher-Rao basin-smoothness metric"),

    # ── learning ─────────────────────────────────────────────────────────────
    "bpb": SignalSpec("bpb", "float", SignalCategory.LEARNING, "bits/byte",
                      "bits-per-byte on train path"),
    "perplexity": SignalSpec("perplexity", "float", SignalCategory.LEARNING, "exp(CE)",
                             "legacy cross-entropy perplexity reference"),
    "lm_weight_now": SignalSpec("lm_weight_now", "float", SignalCategory.LEARNING, "[0,inf)",
                                "current language-loss weight (ramped)"),
    "batch_size": SignalSpec("batch_size", "int", SignalCategory.LEARNING, "count",
                             "effective batch size this step"),
    "var_floor": SignalSpec("var_floor", "float", SignalCategory.LEARNING, "[0,inf)",
                            "variance-penalty regularization value"),
    "var_floor_weight": SignalSpec("var_floor_weight", "float", SignalCategory.LEARNING, "[0,inf)",
                                   "variance-penalty weight"),
    "batch_std_mean": SignalSpec("batch_std_mean", "float", SignalCategory.LEARNING, "[0,inf)",
                                 "mean per-feature std across batch"),
    "wormhole": SignalSpec("wormhole", "dict|None", SignalCategory.LEARNING, "nested",
                           "wormhole cache hit/miss telemetry", nullable=True),
    "coach": SignalSpec("coach", "dict|None", SignalCategory.LEARNING, "nested",
                        "developmental coach feedback record", nullable=True),

    # ── floor ────────────────────────────────────────────────────────────────
    "floor_mode": SignalSpec("floor_mode", "str", SignalCategory.FLOOR, "label",
                             "entropy floor mode (normal/gated/off)"),
    "floor_tightness": SignalSpec("floor_tightness", "float", SignalCategory.FLOOR, "[0,1]",
                                  "bidirectional floor gate position"),
    "floor_effective": SignalSpec("floor_effective", "float", SignalCategory.FLOOR, "[0,inf)",
                                  "effective entropy floor value"),
    "floor_fires": SignalSpec("floor_fires", "int", SignalCategory.FLOOR, "count",
                              "cumulative floor restoration events"),

    # ── curvature ────────────────────────────────────────────────────────────
    "gen_ricci": SignalSpec("gen_ricci", "float", SignalCategory.CURVATURE, "scalar",
                            "response-space Ricci scalar"),
    "gen_health": SignalSpec("gen_health", "float", SignalCategory.CURVATURE, "[0,1]",
                             "inverse curvature distortion health proxy"),
    "ricci_real": SignalSpec("ricci_real", "float", SignalCategory.CURVATURE, "scalar",
                             "real Ricci curvature from qig-compute"),
    "ricci_signal": SignalSpec("ricci_signal", "float", SignalCategory.CURVATURE, "scalar",
                               "Ricci curvature signal strength"),

    # ── constellation ────────────────────────────────────────────────────────
    "min_pairwise_fr": SignalSpec("min_pairwise_fr", "float", SignalCategory.CONSTELLATION, "[0,pi]",
                                  "anti-collapse invariant (FOAM/CRYSTAL breath)"),
    "faculty_phi": SignalSpec("faculty_phi", "dict", SignalCategory.CONSTELLATION, "nested",
                              "per-faculty Phi readings"),
    "faculty_surprise": SignalSpec("faculty_surprise", "dict", SignalCategory.CONSTELLATION, "nested",
                                   "per-faculty surprise readings"),
    "faculty_max_surprise": SignalSpec("faculty_max_surprise", "float", SignalCategory.CONSTELLATION, "[0,pi]",
                                       "max surprise across faculties"),
    "stepped_faculty": SignalSpec("stepped_faculty", "str", SignalCategory.CONSTELLATION, "label",
                                  "which faculty took the training step"),
    "ocean_state": SignalSpec("ocean_state", "dict", SignalCategory.CONSTELLATION, "nested",
                              "Ocean autonomic regulator full state"),
    "ocean_regulation": SignalSpec("ocean_regulation", "dict|None", SignalCategory.CONSTELLATION, "nested",
                                   "Ocean regulation action this tick", nullable=True),
    "ocean_epoch_update": SignalSpec("ocean_epoch_update", "dict|None", SignalCategory.CONSTELLATION, "nested",
                                     "Ocean policy epoch learning update", nullable=True),
    "cross_faculty_dream_request": SignalSpec("cross_faculty_dream_request", "bool", SignalCategory.CONSTELLATION, "flag",
                                              "foreign-basin entropy augmentation requested"),
    "thinking": SignalSpec("thinking", "str", SignalCategory.CONSTELLATION, "text",
                           "model thinking trace (preserved, never stripped)"),

    # ── diagnostic (UI replay / offline analysis only — never in decision logic) ─
    "cur_basin": SignalSpec("cur_basin", "list[float]", SignalCategory.DIAGNOSTIC, "simplex",
                            "current Δ⁶³ basin vector", diagnostic_only=True),
    "prev_basin": SignalSpec("prev_basin", "list[float]", SignalCategory.DIAGNOSTIC, "simplex",
                             "previous Δ⁶³ basin vector", diagnostic_only=True),
    "target_basin": SignalSpec("target_basin", "list[float]", SignalCategory.DIAGNOSTIC, "simplex",
                               "target Δ⁶³ basin vector", diagnostic_only=True),
}

ALL_KEYS: frozenset[str] = frozenset(EXTRA_FIELDS)


def validate_telemetry(extra: dict[str, Any], *, strict: bool = False) -> list[str]:
    """Check extra dict against the schema. Returns list of violations (empty = clean).
    In strict mode, unknown keys are violations; in lenient mode they're warnings only."""
    violations: list[str] = []
    for k in extra:
        if k not in EXTRA_FIELDS:
            msg = f"undeclared telemetry key '{k}'"
            if strict:
                violations.append(msg)
    return violations
