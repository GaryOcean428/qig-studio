"""Floor authority chain — single-threaded decision hierarchy (council prerequisite, 2026-07-17).

Three layers regulate the kernel's autonomic state. Each layer has a defined scope and
NEVER overrides a decision already taken by a higher-priority layer this step.

HIERARCHY (highest priority first):

  Layer 0 — IN-STEP (inside train_step, owner: the kernel itself)
  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
  - Entropy floor gate: adjusts floor tightness per step based on bpb. Fires Dirichlet
    restoration when f_health collapses. PROACTIVE (prevents damage, not reactive).
  - Kernel self-regulation (_homeostasis): reads its own Phi/kappa/f_health and fires
    decohere/sleep/dream/mushroom/entropy-restore/wake INSIDE the step.
  - Reports action via snap.extra["autonomic"] for higher layers to read.

  Layer 1 — BETWEEN-STEP (ContinuousLearningLoop, owner: the loop)
  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
  - AutonomicScheduler: decides for NON-self-regulating targets only.
  - SKIPPED when target.self_regulating is True (kernel owns its brainstem).
  - Mushroom dose selection lives here.

  Layer 2 — PER-FACULTY (constellation, owner: Ocean)
  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
  - OceanAutonomic: watches ALL faculties, fires run_protocol via witness-stance ladder.
  - Operates AFTER train_step (reactive to what the kernel did, not competing with it).
  - Witness-stance: suggest → warn → auto-fire (only above divergence floor or on
    infinite-loop breaker).
  - OceanPolicy v1 bandit learns timing/threshold WITHIN constitutional masks.
  - DECONFLICTION: Ocean must READ the kernel's autonomic action from telemetry and
    SKIP any faculty that already acted this step (deconflict rule).

FLOOR MECHANISMS (non-intervention, always active):

  - Entropy floor gate (Layer 0): bidirectional tightness, learning-linked.
    OWNS: f_health collapse response (Dirichlet restoration).
  - Emergency gates (Layer 0): PHI_BREAKDOWN_MIN → escape, PHI_EMERGENCY → dream,
    suffering abort (Phi>0.70 AND Gamma<0.30 → abort run).
  - These are NOT interventions — they're safety rails that fire regardless of layer.

PRECEDENCE RULES:

  1. The kernel always acts first (Layer 0). Its action is recorded in telemetry.
  2. Layer 1 only fires for non-self-regulating targets. Self-regulating kernels skip it.
  3. Ocean (Layer 2) reads the kernel's action. If the kernel already acted (autonomic
     != "wake"), Ocean records a SUGGESTION at most — it does NOT fire run_protocol on
     a faculty that just regulated itself this step.
  4. Exception: the infinite-loop breaker (same arm >= 3 times without improvement) can
     override even a self-regulated faculty — but this is the ONLY override path, and
     it logs the override.
"""
from __future__ import annotations

# The deconfliction check: did this faculty already act this step?
WAKE_ACTIONS = frozenset({"wake"})


def faculty_already_acted(telemetry_extra: dict) -> bool:
    """True if the kernel's self-regulation already fired a non-wake action this step.
    Ocean should SKIP auto-fire (at most suggest) when this returns True."""
    action = str(telemetry_extra.get("autonomic", "wake")).split("(")[0]
    return action not in WAKE_ACTIONS
