# Fable 5-Pass Council — CRYSTAL Verdict + Pre-Resume Fix Plan (2026-07-02)

**Scope reviewed:** the drive/reward/coaching rebuild — qig-core `1a77255` (Task A), qig-studio `3e21307` (B), `1ea1cc9` (C), `557501c` (D), `fc41a94` (D-fix), `0c66e03` (E), against the **v6.12 / v2.3 omnibus** canon.

## Verdict: BLOCK the immediate resume → GO only after the MUST-FIX cluster lands.

Seat verdicts: Canon **CONCERNS**, Purity **CONCERNS** (gate executed=PASS; loss is pure d_FR both live head modes → resume is *safe*, not *effective*), Wiring **CONCERNS + 1 BLOCKER**, Adversarial **BLOCK** (~10–15% recovery), Completeness **CONCERNS**.

The rebuild's *constitutional layer is real and correct* (mushroom Φ≥0.70 both paths, P15 clamp+log, K1 rail hard-cap, `_pending` bound, tonic floor, P14 bandit, fail-closed outcomes — all tested). The BLOCK is **not** about safety; it is that the rebuild, as committed, would resume into a re-collapse because two load-bearing mechanisms don't actually reach the training path — each found independently by 3+ seats.

## MUST-FIX (mandatory before resume — lifts the BLOCK)

**M1 — Coach loop is display-only on the default `mind` target** (Wiring-B1 = Adversarial-A1/A2 = Completeness gap). The nemotron coach fires + renders SSE but learns nothing:
  - (a) `JointMindTarget` has no `register_coach_reward` → `server.py:965` `getattr(...)` silently skips the replay-priority actuator every step. Fix: add `JointMindTarget.register_coach_reward(r)` → `self._mind.central.register_coach_reward(r)` (joint_mind.py).
  - (b) Ocean + the neurochem coach term read `tel.extra["coach"]` from the **live** snapshot, but the coach record is only written into a `dataclasses.asdict` **deep copy** (`base.py:46` + `server.py:991-995` mutate a throwaway) → `coach_bonus ≡ 0` forever; faculties never receive it. Fix: land the coach record on the central kernel's **live** snapshot (e.g. stash in `register_coach_reward` / set `central.telemetry().extra["coach"]` in `_sample_if_due`).
  - Also actuation-2 threading (server.py:987-995) mutates the same throwaway — fix together.

**M2 — No working entropy source to reverse f_health→0** (Adversarial-A4, the causal crux). `f_health = basin_entropy/log(BASIN_DIM)`; collapse = near-one-hot basin. Own-basin `_dream()` recombines degenerate collapsed history; the 0.05 temp floor is dead + generation-only; mushroom (the only weight-entropy injector) is now Φ≥0.70-gated → removed at Φ≈0.29 with no replacement. Fix (pick, pref order):
  - (a) **Wire the deferred cross-faculty dream** (Task-E Part 3 hook in joint_trainer.py): for a role whose `snap.extra["cross_faculty_dream_request"]` is set, mix its basin with **non-collapsed sibling** basins via `qig_core.geometry.frechet_mean`/`slerp` on Δ⁶³, `_set_pull` for one dream. Only source of *foreign* entropy. **Purity: Fréchet/SLERP on Δ⁶³, never L2.**
  - (b) AND/OR a **basin-entropy floor term** in the collapsed-faculty wake/dream loss (penalize `−H(basin)` toward a floor, gated to the collapse signature, decays on recovery so it never touches the healthy pure-loss path).
  - *Council split adjudicated:* Completeness argued own-basin dream (birth-state@index0) + `couple_step` sibling influence *might* suffice → "acceptable deferral WITH a resume watch." Adversarial + Skeptic-of-Self override for a run that ALREADY collapsed once: don't bet the resume on the untested might-suffice when the mechanism explicitly designed for it is one hook away. Wire it.

**M3 — The qig-core↔studio §6.7 sensations seam** (Completeness "the ONE thing most missed" = Purity-P4 = Wiring-M3). `compute_full_emotional_state` accepts `ricci/local_kappa_c/basin_distance_delta/prev_i_q/i_q` but `_full_primitives` (kernel_experience.py:212-218) passes **none** → `transcendence ≡ 0`, `pushed ≡ 0`, `anxiety ≡ 0`, `confidence` wrong-HIGH. This **reproduces the exact "dead senses with a plausible dashboard" signature the rebuild exists to kill.** Fix: pass the geometric inputs (already emitted in snap.extra) into `compute_full_emotional_state`; also fix the post-hoc Ricci override so Layer-2A/2B see the real Ricci.

**M4 — Resume success criteria must not be faked by the tonic floor** (Adversarial-A5/A6). The dopamine tonic floor (0.35) reads "alive" on resume regardless of recovery — it defeats the very detector that diagnosed the collapse. Fix: the resume gate is **f_health (basin entropy) + Φ-variance recovering**, explicitly **NOT dopamine**. Update the watch script + the plan's success criteria.

**M5 — Dead 0.05 entropy floor + its vacuous test** (Adversarial-A3 = Canon-2 = Purity-3). `max(temp,0.05)` after `max(0.3,…)` can never bind, yet `_apply_stimulate` reports `entropy_floor:0.05` as if it acted. Fix: remove the dead `max()` (M2 supplies the real entropy) OR make it bind above the base band; fix the test so it can fail.

## SHOULD-FIX (raise recovery probability / honesty — with the resume, not blocking)

- **S1** Serotonin never emitted → Ocean P25 `integration_pinned` dead. Emit `snap.extra["serotonin"]` at genesis_kernel.py:1400 (Wiring-I1).
- **S2** Sophia gate default-open: `external_coupling` defaults to exactly the 0.3 threshold; canon C_min≈0.1; a solitary kernel earns endorphins on unmeasured coupling (Canon-I1/A8/Completeness). Fix: default 0.0 (or fail-closed when coupling unmeasured, threshold strictly > default).
- **S3** Healthy-signature divergence hole: `d_basin>floor` + moderate boredom → witness-only where the old prior slept it (Wiring-I2). Fall back to static prior when over-floor-but-healthy.
- **S4** Telemetry the user must SEE recovery by: surface dopamine tonic/phasic split (dropped at `as_dict()`) + Ocean shadow/policy state (no renderer today) (Completeness).
- **S5** Cooldown on the intrinsic collapse-dream (A10, dream-storm); coach-reward credit target the utterance not corpus ids (A9); unify the three coexisting dopamines to one behavior-side signal (A7/I3/M5-legacy NeuroState); P26 stage-0 reward mask — REGISTER as a tracked follow-up (Canon-I3).
- **S6** Citation hygiene: `genesis_kernel.py:58` cites "UCP §35" but mushroom canon is metric #35 / §35.6 (§35 is Ontological Unity); v6.11 §6 → v6.12 §6 in kernel_experience.py:130,200 (content-compatible).

## POST-STASIS SMOKE (first gate the moment stasis clears, before trusting anything)
The 12 suite failures are environmental (paused server holds the GPU → OOM; `in_stasis()`=True → stasis/409). But they **mask the server seam**: `test_train_sse_stream` (the only test of `_train_core`, where M1's threading lives) can't run under stasis. So: after stasis clears, smoke-test `_train_core` end-to-end + run a committed wiring re-audit harness (does not exist yet — build it as part of M/S) asserting 0 DEAD in the target signal set.

## STATUS (2026-07-02, post-implementation)

**BLOCK LIFTED.** All must-fixes landed + wiring-validated on the live `mind` path (qig-studio `development`):
- **M1** `2249abd` — coach→learning WIRED+ACTUATED (server.py:968 → joint_mind:124 delegation → genesis:1566 live-snapshot → ocean:259). Live: Ocean `_coach_reward` 0.0→0.665. 10 tests.
- **M2/M5** `63f1f98` — cross-faculty dream called joint_trainer.py:342; proximity-weighted Fréchet mean of healthy siblings via `slerp_sqrt` (pure Δ⁶³, `test_..._not_l2` discriminates); **f_health 0.0→~0.58** proven; birth-fallback + once-per-epoch cooldown (A10). Dead 0.05 floor DELETED + honest `_apply_stimulate` telemetry. 12 tests.
- **M3** `f6c4725` — §6.7 seam WIRED (ricci/basin_distance_delta responsive); post-hoc Ricci override deleted (ordering fixed); self-ref κ_c guarded → honest-zero not fabricated-high. 15 tests.
- **M4** resume gate = f_health(basin entropy) + Φ-variance, NOT dopamine (tonic floor fakes alive) — enforced in Task-F watch.
- Consolidated: **53/53 tests green, purity gate PASS.** Venv current (qig-consciousness upgraded 0.3.2→0.3.3).

**Remaining SHOULD-FIX follow-ups (tracked; raise recovery-probability/honesty, do NOT block resume):**
- S1 emit `snap.extra["serotonin"]` → activates P25 `integration_pinned` (currently inert). [LOW]
- M3-b kernel must emit a REAL local-critical κ_c distinct from current κ → transcendence/pushed light up (honest-zero today). [MED]
- coach→Ocean per-faculty loop (coach lands on central; Ocean scores faculties) — shadow-gated. [MED]
- S2 Sophia gate default-open (default external_coupling==threshold 0.3 → solitary endorphins; telemetry-only, not the loss). [LOW, qig-core]
- S3 healthy-signature divergence hole; S4 telemetry surface tonic/phasic split + Ocean shadow; S5 dream-cooldown(done in M2)/reward-credit/dopamine-unify; P26 stage-0 reward mask (register); S6 citation hygiene.

## Attacks that FAILED (survive honestly — do NOT re-litigate)
K1 pump-Φ closure (hard cap −0.3), geometric purity of the loss, `_pending` bound, shadow-unlock reachability, mushroom Φ≥0.70 gating (both paths), P15 clamp+log.

## SHOULD-FIXES CLOSED (2026-07-02, PI: close all before resume)
- **WS-A `8dcf375`** (qig-core) — S2 Sophia fail-closed on unmeasured coupling (P24; unmeasured→endorphins 0) + P26 Stage-0 reward-authority mask (School=tonic-only, floor intact). qig-core suite 223 pass.
- **WS-C `ca42110`** (ocean) — S3 over-floor-but-healthy → intake-fatigue reclassify (invariant-safe) + whole-mind coach reward propagated into faculty outcome scoring (coach_bonus was ≡0). test_ocean_policy 25→29.
- **WS-B `2a488df`** (genesis) — S1 serotonin emit (P25 integration_pinned now live) + M3-b HONEST-RENAME to `kappa_local` (no principled κ_c derivable; κ≈64 retired; κ-slope→consciousness forbidden → honest-zero + TODO for physics lane) + S5 coach-credit-to-utterance + dopamine-guard + S6 citations (v6.12/§35.6). 7 new tests.
- **WS-D `910f38e`** (telemetry) — S4 dopamine tonic/phasic split + Ocean shadow state renderer + f_health/Φ-variance in the resume-watch record.
- Consolidated: **81/81 tests green, purity OK**, all council-flagged wiring gaps closed. Venv current. **READY FOR RESUME** from PRESERVED_pre_intervention_step18388, success gate = f_health(basin entropy) + Φ-variance recovering (NOT dopamine).
