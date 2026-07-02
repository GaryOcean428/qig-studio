"""Constellation — spawned Core-8 faculties that couple, observe self+others, communicate, share a
rhythm/clock, and keep individuated identity (no collapse).

The load-bearing mechanism (verified, wj4916x59): wide-independent birth seeding (Pillar-3 scar) is a
PREREQUISITE for coupling stability — the identity anchor pulls each faculty back toward its own wide
birth, making the coupled fixed configuration DIVERSE rather than a collapsed centroid. Round 0
(seeding) and Round 1 (coupling+anchor) are one mechanism, not two subsystems.

Torch-free decision path: faculties carry numpy Δ⁶³ basins; qig-core Fisher-Rao primitives only.
"""

from __future__ import annotations

from .constellation import Constellation, ConstellationTelemetry
from .coupling import INBOUND_BUDGET, CoupleDiag, couple_step, rel_weights
from .faculty import (
    BIRTH_CONCENTRATION,
    Faculty,
    FacultyView,
    min_pairwise_fr,
    seed_birth_basin,
    seed_constellation,
)
from .identity_anchor import ANCHOR_FRACTION, apply_anchor, equilibrium_distance, identity_drift
from .neurochem import NeuroState, apply_modulation, compute_modulation
from .ocean import OceanAutonomic, context_from_telemetry, function_of
from .ocean_policy import (
    ARM_MASKS,
    BANDS,
    PRIOR_THRESHOLDS,
    Decision,
    OceanContext,
    OceanPolicy,
    OutcomeScore,
    classify_signature,
    score_outcome,
)
from .rhythm import HeartOscillator, RhythmMonitor, RhythmState
from .signal_bus import Signal, SignalBus
from .temporal import (
    BasinForesight,
    TemporalAwareness,
    TemporalState,
    arc_length,
    distinguishable_transitions,
    path_efficiency,
    tau_macro,
)

__all__ = [
    "ANCHOR_FRACTION",
    "ARM_MASKS",
    "BANDS",
    "BIRTH_CONCENTRATION",
    "INBOUND_BUDGET",
    "PRIOR_THRESHOLDS",
    "BasinForesight",
    "Constellation",
    "ConstellationTelemetry",
    "CoupleDiag",
    "Decision",
    "Faculty",
    "FacultyView",
    "HeartOscillator",
    "NeuroState",
    "OceanAutonomic",
    "OceanContext",
    "OceanPolicy",
    "OutcomeScore",
    "RhythmMonitor",
    "RhythmState",
    "Signal",
    "SignalBus",
    "TemporalAwareness",
    "TemporalState",
    "apply_anchor",
    "apply_modulation",
    "arc_length",
    "classify_signature",
    "compute_modulation",
    "context_from_telemetry",
    "couple_step",
    "distinguishable_transitions",
    "equilibrium_distance",
    "function_of",
    "identity_drift",
    "min_pairwise_fr",
    "path_efficiency",
    "rel_weights",
    "score_outcome",
    "seed_birth_basin",
    "seed_constellation",
    "tau_macro",
]
