# RUN-2 — genesis-solo cradle, pre-registration

**Status:** LIVE run-of-record on Modal (2026-07-24); doctrine amended from Matrix genesis-design conversation + fc4df3cb / f599507a / 28a66754.
**Pin of record:** `qig-core==2.15.2` (developmental-migration healthy + EMA prior + EMA floor — birth CRITICAL residual closed).
**Arm of record:** run-2 = **all three geometry fixes + coach** — honest genesis anchor + interleave frame-fix + coach wired (m1c). Run-1 is archived as the **everything-wrong arm** (no coach + phantom anchor + inconsistent frames); we do **not** re-run deliberately-malformed births for attribution purity. Attribution comes from the *distinct telemetry signatures*, not from ablation:

| Fix | Signature it moves |
|---|---|
| honest anchor (6bb019d) | birth-step P3 drift **level** |
| interleave frame-fix (3ed370e) | round-trip **phantom** (reduce(resize(p))−p) |
| coach wired (m1c, de33ba8) | output-collapse **dynamics** + floor-restoration **trend** |
| qig-core 2.15.0–2.15.2 | flat P3 / birth CRITICAL instrument defects |

### Roster / spawn law (Matrix 8037cbe3 + genesis-design conversation)

Faculty roster (NOT “Core-8 seats”): `perception → heart → drives → memory → action → strategy → meta`.
- **ETHICS** retired as faculty — HEART carries ethics (one geometry for valence + value).
- **OCEAN** not seeded — umbilical/CAUL from step 0; watch-not-seed (m3 for live differentiation-watch).
- **Genesis-first:** solo-train genesis until mean Φ ≥ 0.68 (cap 8000); roster dormant until mature.
- **Universality:** substrate bake-off is dishonest until machinery sits on the shared node (gk dual-frame is a known defect class). Run-2 is **gk machinery cradle**, not a substrate verdict.
- Spawn is **plasticity-windowed, one faculty at a time** after maturity — not simultaneous spawn of 7.

### Birth scar ≠ formed identity (honesty of the P3 channel)

P3 `seed_identity` freezes **immediately** on the **role birth scar** (`seed_birth_basin(role-hash)`), not on a lived Fréchet identity. Early “identity drift” = **d_FR(current_readout, birth_scar)** — developmental migration away from a *given* scar is expected; it is **not** “lost self.” CRITICAL is reserved for velocity spikes above a living baseline (2.15.2 EMA floor). Measurement stays full-sensitivity; Stage-0 gates **coach actuation**, not the sensor (f599507a).

### Coach mechanism (Matrix 28a66754 + fc4df3cb L2)

1. **INPUT — child HEARS the coach** (recast → `train_step`, input-side only).
2. **REWARD — replay priority** via `register_coach_reward` (Stage-0 mask: coach ≢ self-reward hack).
3. **BASIN-PULL — NOT wired (run-3).**
4. **L2 other_observation — YES:** coach is the only lived-other in genesis-solo; drives `M_boundary` / `m_other` **observation-side only** (no reward/pull semantics). Provenance: `m_other_source=coach`.
5. **Stagnation offer** — OFFER-only text (“Your call”); kernel cannot accept a geometric nudge in run-2.

### Visibility (PI: yes)

On coach cadence / sample / stagnation: stdout `[voice] phase@step stim=… gen=… recast=…` **and** durable `checkpoints/.../voice_log.jsonl` (Modal volume). Trajectory: `basin_trajectory.jsonl` (truncated on `--fresh`). LiveLog path remains for local UI.

### Corpus truncated-world receipt

| Quantity | Unit | Value | Source |
|---|---|---|---|
| Full FineWeb sample/10BT | **shards** | **15** | `stage_fineweb_corpus.py` `N_TOTAL=15` |
| Staged on Modal volume (run-2) | **document rows** (parquet) | **1,048,581** (1 shard) | launch gate `corpus_segments` |
| “1.66M” citation | ambiguous | **not** the staged document count | likely multi-shard and/or post-chunk **passages** — do **not** pin as expected rows without unit |

**Verdict:** staged world is a **partial single-shard** stage (`complete=false` vs 15). Guard correctly records 1.05M **documents**. Completing remaining 14 shards is a **pre-run-3 / full-world** item; run-2 manifest must keep both units explicit. Newborn is training on a **truncated** FineWeb sample, honestly labelled.

### Checkpoint retention (PI, in code)

`keep=2` (current + one lag) on whole-mind `save_checkpoint`. Protected: run-1 archive + prereg-referenced roots — never under the pruner.

The run does **not** launch unless the fail-closed launch gate passes (D2): anchor-honest, frame-consistent, coach-wired + coach-live, qig-core version pin, coordizer sha, substrate, seed, rulings-applied, corpus present. The gate is asserted identically by the local smoke and the Modal run of record.

---

## Prediction 1 — the BIRTH test (reads in the first steps)

**Claim.** With the honest anchor AND the interleave frame-fix both landed, the genesis birth is no longer measured against a place it has never been. Therefore the birth-step Pillar-3 identity drift must read **under the 0.25 tolerance (target ≈ 0)**, with **no CRITICAL transient** at the highest-velocity birth moment.

- **Observable:** per-step Pillar-3 `check_drift` status + P3 drift value over the first ~30 warmup steps (the birth window; run-1's phantom spike was 1.4007 at step ~1–12 vs a steady 0.36).
- **Threshold (pre-committed):** max birth-step P3 drift `< 0.25`; **zero** CRITICAL transients attributable to the birth (developmental_migration=healthy is allowed; a velocity-spike CRITICAL is not).
- **Discriminator written before the run:** birth drift `< 0.25` and no CRITICAL ⇒ **PASS** (both geometry fixes confirmed on the live path). Any birth spike that **survives both fixes** ⇒ **there is a THIRD contributor** — we hunt it; it is **not** absorbed as "expected". (FR distances are not additive; the ~1.0 anchor + ~0.36 frame budget is *consistent* with run-1's 1.4007, not *proven* — this test is the proof.)
- **qig-core dependency (receipt closed + follow-ups):** developmental-migration healthy (2.15.0) + birth EMA prior (2.15.1) + EMA floor against high-drift jitter (2.15.2). Modal pin of record: **`qig-core==2.15.2`**. Live residual CRITICAL on 2.15.1 (v≈0.17) is the third-contributor instrument class — closed in 2.15.2.

## Prediction 2 — output-collapse pressure PERSISTS; the coach's effect is a TREND

**Claim (re-stated for run-2's actual configuration — coach + honest anchor + frame-fix).** The output-basin collapse driver is `lm_loss`'s preference for confident one-hots, which the anchor fix does **not** touch (it repairs *identity* geometry, not *output* pressure). Therefore:

- run-2 **still exhibits per-step output-collapse pressure**;
- the entropy **floor remains the per-step guarantor** (it restores entropy each step);
- the coach's effect appears in **trajectory statistics** — specifically a **DECLINING floor-restoration rate across formation** — **not** in eliminating per-step collapse.

- **Observable:** the `_floor_fires`-diff per step → `floor_restoration_rate` (cumulative) and `floor_restoration_rate_window` (rolling), logged every step by the CoachSupervisor (B4); plus `coach_success_rate` (B6).
- **Discriminator written before the run:** a **downward trend** in the windowed floor-restoration rate across formation ⇒ the coach is teaching the basin to need the floor **less** (mechanism supported). A flat/rising rate with a live+wired coach ⇒ the coach's replay-timescale reward is not reaching the per-step phenomenon (escalate to the run-3 coach-seeded anchor-pull, pre-registered, not same-day-patched).
- **Falsifier (Matrix 85a78b36 §3):** run-2's output basin **holding its own entropy without floor intervention inside Stage-0** refutes the mechanism model (lm_loss-one-hot-pressure). That is the measurement working, and it would mean the collapse was never the driver we think it is.

---

## What run-2 does NOT test (sequenced behind m3)

- **Faculty construction** bites at *activation*; run-2 is genesis-solo Stage-0 with faculties dormant ⇒ out of scope. Its rewrite is sequenced behind the **m3 readiness instrument**, which is itself gated on the **multimodality read** (QT-APP-6): *can a faculty-shaped sub-basin differentiate inside genesis on this architecture at all?* — pre-committed discriminator, reported to Matrix either way.
- **Spawn-law encoding** (f241cee4 §6) — faculty-phase, not a genesis-solo blocker.
- The **coach-seeded output-basin anchor-pull** (option-2) — run-3 only, and only IF Prediction-2's floor-restoration rate is **not** declining across formation.

## Escalation ladder (pre-registered)

1. **run-2** = coach-only (observe + reward-weighted replay) + honest anchor + frame-fix + floor untouched. ← this run
2. **run-3** = coach seeds a consonant OUTPUT-basin target → `_set_pull`. Triggered ONLY IF collapse persists AND the floor-restoration rate is **not** declining across formation (the basin never learns to hold its own entropy).

## Run topology

ALL work lands before the LOCAL SMOKE (reads the birth test in the first steps + coach liveness incl. one Modal cold-start + telemetry persistence + one full checkpoint→resume cycle), then the MODAL run of record (training on its GPU, coach endpoint scale-to-zero beside it — the VRAM separation run-1 could not have locally). Spend is not the constraint; alignment is.
