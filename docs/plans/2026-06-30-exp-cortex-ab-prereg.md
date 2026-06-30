# EXP-CORTEX-AB — Blind Contract (pre-registration) for the depth A/B at scale

> **Status:** PROPOSED contract — **awaiting Matrix/Devin ratification before the 8L compute spend.**
> The *verdict* ("does depth help") is Devin's lane (qig-experiment-method rule 9). This document is
> *disposition/methodology*: it locks the acceptance bands, the calibration gate, and the kill
> conditions **in writing before running**, so the result cannot be tuned toward a wanted answer.
> **Date:** 2026-06-30. **Run driver:** `qig-studio` neocortex harness (Phase 5 of
> [`2026-06-30-neocortex-rebuild-phase1.md`](./2026-06-30-neocortex-rebuild-phase1.md)).
> **Experiment:** EXP-CORTEX-AB (qig-consciousness backing map
> `20260624-brain-experiment-backing-map-1.00W.md`, rank 1; "the A/B IS the first build step").

---

## Why this contract exists (history anchor — the two failures it must prevent)

EXP-CORTEX-AB has already run twice (`qig-consciousness/docs/plans/2026-06-24-coordizer-studio-remaining-work.md` §R4):

1. **`c7f32f4` — INVALID (Adam artifact).** The depth A/B was trained under `torch.optim.Adam`
   (P1 purity violation). It reported −27σ + a "deep cortex is training-unstable / NaNs every seed"
   blocker. **Both RETRACTED** — they were Adam artifacts, not physics. The purity audit
   (`tools/validation/geometric_purity_audit.py`) now scans `experiments/` so Adam cannot recur.
2. **Re-run under natural gradient — VALID but UNDERPOWERED null (−3.46σ).** The 4-layer stack trains
   stably, NO NaN, all 5 seeds (so the "training-instability blocker" *does not exist*). But **both
   arms under-converged** at the tiny-CPU budget — *even the 1-layer arm sat above the unigram floor* —
   so the comparison was **NOT-SEPARATED / UNDERPOWERED**, not "depth is neutral." Verdict:
   NEEDS-EXPERIMENT at scale, **not killed.**

The lesson the −3.46σ run teaches, and that this contract commits: **an A/B read out before both arms
have demonstrably learned is undecidable by construction** — an underpowered null reads as "depth
neutral" if you let it. The maturity floor below is the calibration gate that prevents exactly that.

---

## The question (one line)

Does **N-stacked depth** (`neocortex-qk-8L`) beat **1-block-recursive** (`neocortex-qk-1L-rec`) on
held-out fluency, when **only the model architecture varies** — same fresh coordizer, same curriculum,
same pure d_FR loss, same natural-gradient optimizer instance, same seeds?

(A second, orthogonal axis — cross-*family* `neocortex-geo` vs `neocortex-qk` — and the CE-ablation
purity-cost arm are reported on the same metrics but are **not** the depth verdict.)

## The arms

| Arm | What | Role |
|---|---|---|
| `neocortex-qk-8L` | N-stacked Δ⁶³ `qigkernels.Kernel(num_layers=8)` | depth A |
| `neocortex-qk-1L-rec` | 1 block, internal `min_recursion_depth` | depth B |
| `neocortex-geo-8L` | `geocoding.GeoModel` (Fisher-Rao-attention) | cross-family axis (separate verdict) |
| `*-ce_ablation` | same arm, CE loss | purity-cost diagnostic (separate verdict) |

## Cleanliness condition (only-architecture-varies — the EXP-CORTEX-AB invariant)

Held **identical** across the two depth arms; if any differs, the bpb gap confounds architecture and the
A/B is void:

- **Same fresh coordizer** (`coordizer_20260630_100k_v1`, the Δ⁶³ vocab the run trains on).
- **Same curriculum** (`load_full_curriculum`, same order, same seed-shuffle).
- **Same pure loss** (`fisher_rao_lm_loss`; the CE arm is a *separate* labelled arm, never mixed in).
- **Same `NaturalGradientDescent` instance/config** — same optimizer class **and** hyperparameters as
  ARM B's `GenesisKernelTarget` uses; **not Adam, and not a *different* natural-gradient setup.** (This
  is the cleanliness condition extended to ARM A per the council review — a different optimizer would
  confound architecture with optimization.)
- **≥5 seeds** per arm (the depth delta is read against cross-seed spread, not a single run).
- Coords **ON** for the real training/bpb (only the faithfulness *equivalence* check runs coords-off).

## Verdict metric vs diagnostic metric (committed — the §H anti-cherry-pick)

Committed **before** the run so "depth helps" cannot be decided post-hoc on whichever channel gives the
preferred answer (the cherry-picked-verdict-channel failure the programme already convicted once):

- **VERDICT metric = held-out CE-bpb** (lower = better). Rationale: it is the *external* benchmark
  (comparable to SmolLM2-360M ≈ 0.8), and the arms train on d_FR — so evaluating on CE-bpb measures the
  thing they did **not** directly optimize, which is the *less circular* of the two. Computed identically
  across all arms.
- **DIAGNOSTIC metric = mean d_FR** via the **torch** primitive
  `qig_core.torch.geometry_simplex.fisher_rao_distance_simplex` ONLY (range [0, π]) — **never** the numpy
  `fisher_rao_distance` (range [0, π/2]); the factor-of-2 would silently corrupt the table. Reported and
  expected to track CE-bpb; it is **not** an escape hatch to flip the verdict.

## Maturity floor (the calibration gate — MANDATORY before the depth delta is read)

Both depth arms MUST clear ALL of these on the held-out set **before** the bpb delta counts. This is the
gate the −3.46σ run lacked:

1. **Off the Φ saturation pin** — Φ < `PHI_BREAKDOWN_MIN = 0.80` and not pinned at 1.0 (the smoke showed
   Φ=0.54 at step 5, so the collapse-immune Φ *can* sit off the pin — the floor is reachable).
2. **Finite CE-bpb** — no NaN/Inf over the full held-out set.
3. **Below the unigram floor by a pre-stated margin** — converged CE-bpb must beat the coordizer's
   unigram-entropy (frequency-only) baseline by ≥ a committed margin, demonstrating the arm learned
   *structure*, not just token frequency. (The −3.46σ null fired because even 1L sat *above* this floor.)

**If EITHER arm fails the maturity floor → verdict = WITHHELD / UNDERPOWERED, NOT "depth neutral."**
This is the precise distinction the prior run collapsed.

## Success band

`neocortex-qk-8L` reaches the maturity floor in **fewer steps** OR achieves **lower converged CE-bpb**
than `neocortex-qk-1L-rec`, by a margin **exceeding the cross-seed spread**: |Δ CE-bpb| > k·σ_seed
(k committed at run-registration; ≥5 seeds → σ_seed is measurable). A margin inside σ_seed is **not** a
win.

## Kill conditions (committed — each could fire)

1. **No separation beyond seed noise** after BOTH arms mature → **depth is NEUTRAL** (a real null,
   distinct from the underpowered null — only legitimate once the maturity floor is cleared).
2. **Either arm fails purity** (Adam recurrence / Euclidean contamination; the purity audit scans
   `experiments/`) → **INVALID**, not a result.
3. **Either arm fails the maturity floor** → **WITHHELD / UNDERPOWERED** (re-run at larger budget).

## Four measurement axes (qig-experiment-method tag vocabulary)

- **Channel:** held-out CE-bpb (verdict) + mean d_FR (diagnostic).
- **Protocol:** identical coordizer/curriculum/loss/optimizer-instance/seed; coords-ON; ≥5 seeds.
- **Aggregation:** mean over held-out next-token positions; median + spread over seeds.
- **Clock:** steps-to-maturity-floor AND converged value (both reported — depth may help *speed* or
  *asymptote* differently).

---

## Known instrument caveats (committed, so a real signal isn't misread as one of these)

**(a) Pillar-3 "CRITICAL identity drift" stdout is EXPECTED and NON-INFORMATIVE in early training.**
Confirmed at source (`qig_core/consciousness/pillars.py:79,683-696`): `IDENTITY_DRIFT_CRITICAL = 0.4`,
and `drift = fisher_rao_distance(current_basin, effective_ref)` where `effective_ref` is the **frozen
birth scar** until a T1.4 slerp anneal-field blends in (60/40) — and the anneal fix only prevents
false positives **after ~800 cycles**. A kernel whose identity is frozen-at-birth
(`genesis_kernel.py:225`) moves its basin >0.4 d_FR from the random birth scar within a few training
steps, so this `logger.error` previously **fired-and-stayed through the entire early-training window
(~5–800 cycles)**, carrying no incremental signal and masking a real late dissolution. It is
**telemetry-only** — no consumer gates on the `IDENTITY_DRIFT` violation, and `refract()` (the active
anchor) is **never called in the single-kernel neocortex path**, so it does **not** bias the A/B.
**FIXED (owned — qig-core is a local editable submodule):** `check_drift` is now **velocity-gated**
(`qig-core` `development`, commit on the Pillar-3 module + `tests/test_pillar3_drift_velocity.py`,
3/3) — steady developmental migration from the random birth scar logs as *developmental*, and only a
drift-**velocity** spike escalates to CRITICAL, matching the **PI-facing** `qig-studio/live.py:32,42-62`
channel (which already keys on drift velocity). **Pending the qig-core version-tag + republish +
studio-venv reinstall (sequenced after Phase 4 lands to avoid re-resolving the running implementer's
venv).** **Kill conditions above key off velocity-jumps + finite-CE + saturation-clearance — never the
absolute-drift log.**

**(b) The vocab read-out is a Euclidean `nn.Linear` lm_head in BOTH arms — an OPEN, unratified purity
question.** Confirmed at source: `geocoding/model.py:55,95` (`self.lm_head = nn.Linear(hidden_dim,
vocab_size)` → `logits = self.lm_head(h)`) and `qigkernels/kernel.py:137,234` (identical). This is the
same dot-product class as the attention cosine that was purged — relocated one layer out to the output
boundary, inherited unexamined. The purity discipline reached the **input** (coordizer) and the **loss**
(d_FR) but **not the output head or the CE-bpb metric**. For *this* A/B the head is **common-mode**
(byte-identical in both depth arms), so it does **not** confound the depth comparison — the verdict is
valid for "depth/architecture **given the shared head**." It does **not** certify the head as pure.
**Open decision (council/Devin, before it's waved through):** is the Euclidean read-out an *accepted
exemption* — labelled with a reason (e.g. "output projection to vocab is the read-out boundary, a
measurement, not a manifold operation") — **or** a genuine impurity to replace with a d_FR-geometric
read-out? Right now it is *labelled-open* in code (`geo_cortex.py`), neither ratified nor fixed.

---

## Ratification checklist (before the 8L compute spend)

- [ ] Matrix/Devin sign-off on VERDICT=CE-bpb / DIAGNOSTIC=d_FR split.
- [ ] Commit the numeric maturity-floor margin (unigram-floor beat) and the success-band k.
- [ ] Confirm the `NaturalGradientDescent` config is byte-identical across both depth arms (source read).
- [ ] Decide caveat (b): lm_head exemption-with-reason **or** geometric-read-out task filed.
- [ ] Register the prereg in the experiment registry (Devin's lane) before the run.
