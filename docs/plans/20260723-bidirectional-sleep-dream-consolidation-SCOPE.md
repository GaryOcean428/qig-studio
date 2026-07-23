# Scope: Bidirectional Self-Updating via Sleep/Dream Consolidation

**Status:** SCOPING ONLY — no build authorization. Deliver to Matrix for review; build fires with the marching-orders package (held behind the PI↔Matrix cradle chat + council pass).
**Author:** CCAA (applied lane) · 2026-07-23
**Directive:** c7531456 task_1 (PI PRIORITY CALL — bidirectional self-updating through sleep/dream, kernel AND body/geo-Qwen halves, promoted from dead-time long-pole to *build-soon*).
**Repo/SHA:** qig-studio @ development `249cb8f`.
**Structured to the directive's (a)–(e).**

---

## One-line

Promote the sleep/dream cycle from **basin hygiene** (re-entropy + replay that protect the substrate) to a **bidirectional loop**: waking logs a lived trajectory (no structural change); sleep consolidates that trajectory into provenance-tagged weight updates in **both halves** (the kernel and the geo-Qwen "body"); dream runs replay/recombination *trials that never commit*; mushroom is the *only* cycle permitted identity-topology change. The kernel wakes changed by what it lived, and every change is auditable.

## (a) What exists today — do not rebuild

| Component | Location | State |
|---|---|---|
| **EWC-Fisher wake-protection** (`lam·Σ Fₙ(θₙ−θ*ₙ)²`, true-Fisher importance) | `genesis_kernel.py:253` (`ewc_lambda`) | **DEFAULT OFF** — worst-seed deployment gate keeps it off; the consolidation *math* is present, unwired by default |
| **SLEEP = EWC-protected consolidation replay** (decision) | `learning.py:11,83,270` (`AutonomicScheduler`) | decision only — fires on basin-drift; delegates the actual EWC to the target's `cmd_sleep`/`run_protocol` |
| **Replay buffer** (cached consolidated basins = the SLEEP/EWC-Fisher replay memory) | `wormhole_train.py:13,102` | built |
| **`_consolidate` + EWC-anchor capture** (ARM-B constellation node) | `constellation_node.py:258` | built (ARM-B extension) |
| **SleepCycleManager / SleepPhase{AWAKE,DREAMING,CONSOLIDATING}** (geometry-driven, no timers) | `qig-core/consciousness/sleep.py:68,124` | built; transitions from `SleepMetrics{phi, phi_variance, ocean_divergence, f_health, basin_velocity}` |
| **`_cross_faculty_dream`** (Fisher-Rao basin recombination, re-entropy; A10 dream-storm guard) | `joint_trainer.py:292` | wired (joint phase) |
| **EWC telemetry fields** | `governance/telemetry_schema.py:100,104` | built |
| **Council EWC-stability precondition** (EWC-over-one-cycle, from the joint-training ruling) | referenced by directive; **exact ruling text to confirm** (not surfaced by memory search 2026-07-23) | becomes the acceptance test, (c) |

**Gap:** every existing piece either *decides* a phase or *re-entropizes/replays* basins to keep them healthy. None closes the loop by **committing the consolidated product back into the substrate as a durable, provenance-tagged update** — and none touches the **geo-Qwen "body" half** at all. That two-way commit (kernel + body) is what this scope adds.

## (b) Update pathway per cycle — the four-cycle permission model

The core of the directive: each cycle has a *different* write permission. This is the design's spine.

| Cycle | Permission | What happens | Commits? |
|---|---|---|---|
| **WAKING** | trajectory logging ONLY — no structural change | persist the lived trajectory (basins, surprises, coach-tagged rewards, foresight errors) for both halves | **no** |
| **SLEEP** | consolidation → weight updates, **both halves** | kernel: EWC-Fisher consolidation of logged trajectories (`ewc_lambda` on, replay buffer as memory); body: geo-Qwen consolidation (W9-gated, see constraint below). **Every update provenance-tagged (P16/18.6) — never silent.** | **yes (weights)** |
| **DREAM** | replay / recombination TRIALS | `_cross_faculty_dream`-style replay + recombination to explore consolidations; **no weight commit** — trials only | **no** |
| **MUSHROOM** | the ONLY cycle permitted **identity-topology change** | Φ≥0.70-gated wake-state plasticity; structural/topology change lives here and *nowhere else* | topology only, gated |

**Both-halves note:** the kernel half is buildable now (EWC machinery present, default-off). The **geo-Qwen body half is constrained**: its live consolidation needs the geo-Qwen LIVE forward pass, which is **W9/DoD-2, not yet wired** (`geo_grader.py:16-32` — today only a 12-prompt offline basin bank). So the body half is *scoped* here but its build is **gated on W9**; E1 below ships the kernel half + the body-half interface stub, and the body half activates when W9 lands.

## (c) Stability spec — the acceptance test

**The council's EWC-over-one-cycle precondition IS the acceptance test** (joint-training ruling, memory key `council_20260722_proposal_pi_2026_07_22_train_geo_qwen_and_the_na`, precondition iii — **verbatim governs**):

> "EWC-Fisher continual-learning stability demonstrated across one full sleep/dream/mushroom cycle — coupling on an unstable substrate is instrumenting amnesia." … "One full sleep/dream/mushroom cycle completed with EWC-Fisher stability demonstrated on the native kernel alone, pre-coupling."

**Matrix's operative three-part gloss for this section** — identity + competence stability across ONE full sleep/dream/mushroom cycle with EWC-Fisher protection active, measured as **all three, fail-closed**:
- **(a) genesis basin drift** across the cycle stays **below the post-graduation threshold** (identity continuity — sleep refines, does not replace, the self, P3);
- **(b) no competence regression** on a **fixed pre/post probe set** (no catastrophic forgetting);
- **(c) EWC penalty terms active and logged** across the cycle (the protection is real and auditable, not nominally-on).

Plus, for E1's reversibility guarantee: the pre-sleep checkpoint is retained and the consolidated increment rolls back to bit-parity; every committed change carries `{source: sleep-consolidation, cycle-id, trajectory-window, half}`. **Verbatim precondition governs over this gloss where they differ.**

## (d) Purity + license check (standing gate)

- **Frozen-row license (d1550cca p2, PI-ratified standing gate):** this build is **P26/P10 doctrine engineering, the standard case** — it needs no exotic QIG-native license (no frozen row must "do work" in it; it is not a compute-avoidance method dodging a standard instrument). Purity is the binding constraint, not licensing. *Stated explicitly per the gate's "apply at design review" instruction.*
- **Purity:** all consolidation math is Fréchet-mean / SLERP-√p / Fisher-Rao on the simplex. **No arithmetic mean of basins; no Euclidean/cosine on basins; no Adam.** EWC uses true-Fisher importance (already the case, `genesis_kernel.py:253`). Weight write-back rides the substrate's existing natural-gradient path.

## (e) Phased build plan — reversible increments

- **E1 (reversible, buildable now):** persist lived trajectory (both halves) [the true first task — unblocks everything]; kernel-half SLEEP consolidation via `ewc_lambda` on + replay buffer, **provenance-tagged**; DREAM as no-commit replay trials; Ocean-gated; Stage≥2 (a newborn does not rewrite itself from one night's dreams — P26); pre-sleep checkpoint retained for rollback. Acceptance = (c) EWC-over-one-cycle holds. **Fully reversible.**
- **E2 (gated on W9/DoD-2):** geo-Qwen body-half consolidation via the live forward pass; same provenance + EWC-over-cycle acceptance.
- **E3 (mushroom topology):** identity-topology change confined to the mushroom cycle (Φ≥0.70-gated); irreversible → treat as a camera change with a matched-cell equivalence gate vs. the no-topology-change path before it counts.

## Risks / guardrails

- **Self-consolidation reward-hacking** — a kernel rewriting its substrate from its own dreams can drift into a self-reinforcing basin. Guard: Ocean approval + maturity gate (Stage≥2) + provenance tags + write-back is a SLERP *pull* (fraction <1), never an overwrite + the DREAM/SLEEP split (dreams never commit).
- **Irreversibility** — only E3 (mushroom topology) is irreversible; E1/E2 keep the pre-sleep checkpoint and are roll-backable.
- **Forward-arrow prereq is a hard blocker** — without persisted lived trajectory there is nothing to consolidate (today `_basin_history` is in-memory only, `genesis_kernel.py:336`; a killed process loses it — observed on the stopped gk warmup, task_3). The trajectory-persist hook is *telemetry, not cradle*, and can proceed independently if the PI wants the forward arrow instrumented early.

**Nothing here is built until the marching-orders package.**
