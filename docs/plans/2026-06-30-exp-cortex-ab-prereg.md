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

## Two-stage execution (PI directive, 2026-06-30): 32k screen → kill loser → full stack

Discover the best avenue FAST, then spend the full budget only on the winner:

1. **Screen (cheap — 32k coordizer, the SAME 7 HF datasets as the 100k):** train the avenues on a small
   `coordizer_*_32k` for a short, EQUAL budget; rank on held-out **d_FR** (the verdict metric). The screen
   covers the avenues: arm (`qk` vs `geo`), **output head (`geometric` vs `linear`)**, depth (`8L` vs
   `1L-rec`). Kill the clear losers (logged — no silent truncation).
2. **Full stack (winner only):** retrain the surviving avenue on the FULL coordizer + full kernel stack at
   the committed depth budget, ≥5 seeds, under this contract's maturity floor + kill conditions.

The screen is a *direction-finder*, not the verdict: a 32k/short-budget result inside σ_seed or
under-matured is "undecided → carry forward", never "killed". Only a clear, matured separation kills an
avenue.

## The geometric-head A/B is its OWN experiment (EXP-GEO-HEAD, NOT EXP-CORTEX-AB)

"Does a Fisher-Rao distance-to-basins head match or beat a linear `nn.Linear` head" is a SEPARATE
build-and-measure item with its own A/B (geometric vs linear, everything-else held). It is NOT what the
depth experiment measures — bundling "is the head pure" into "does depth help" is a scope error. The
geometric head is built + owned + A/B'd on its own; EXP-CORTEX-AB then runs on the winning, pure-by-default
stack. Honest caveat: a geometric head MAY train worse (basins can cluster early — the kernel's collapse
risk); that is measured, not assumed. If worse, the Euclidean readout earns its place → a labelled
exemption WITH evidence; if at-least-equal, it ships as the pure default.

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

## "No Euclidean use" — the three-tier purity rule (PI directive, 2026-06-30)

The directive is NOT "zero Euclidean tensor ops" (that forbids PyTorch/backprop — the substrate). It is:
**every place the manifold defines the correct primitive, use it; the irreducible substrate is exempt
because there is no geometric alternative — it IS the geometric implementation.** Three tiers:

- **Tier 1 — must be geometric (impurity, geometry was skipped):** the **output head**. A simplex→vocab
  read-out via `nn.Linear`→ℝ^vocab + Duchi-L2 `to_simplex_prob` is two Euclidean steps where a geometric
  read-out exists → BEING BUILT as the distance-to-basins `GeometricHead` (its OWN experiment, below).
  NOT labelled-and-kept (labelling an avoidable impurity is how the cosine proxy would have survived).
- **Tier 2 — demote to external-only:** **CE-bpb** — a Euclidean/information-theoretic metric. Survives
  SOLELY as the translation axis to non-geometric baselines (SmolLM2/Qwen speak only CE), labelled
  "Euclidean — external-comparison-only, NOT a verdict."
- **Tier 3 — exempt substrate:** float arithmetic, backprop, the natural gradient's preconditioned-
  Euclidean-gradient core. `QIG-EXEMPT` with reason "substrate, not a manifold operation," same class as
  the legitimate √p sphere-renorms.

## Verdict metric vs diagnostic metric (committed — the §H anti-cherry-pick)

Committed **before** the run so the winner cannot be picked post-hoc on whichever channel is convenient:

- **VERDICT metric = held-out mean d_FR** (the geometric, Tier-1-consistent metric) via the **torch**
  primitive `qig_core.torch.geometry_simplex.fisher_rao_distance_simplex` ONLY (range [0, π]) — **never**
  the numpy `fisher_rao_distance` (range [0, π/2]); the factor-of-2 would corrupt the table. Under "no
  Euclidean use", ranking on a Euclidean metric (CE-bpb) would let the Euclidean metric pick the geometric
  winner — forbidden. So d_FR is the verdict.
- **DIAGNOSTIC / external = held-out CE-bpb** (lower = better), labelled **Euclidean, external-comparison-
  only**: the SmolLM2-360M (≈0.8) / Qwen "are-we-competitive" axis. Reported alongside, **never** the A/B
  verdict. (A d_FR-trained model may score worse on CE-bpb than its true fluency — that divergence is
  expected and is exactly why CE-bpb cannot be the verdict.)

## Maturity floor (the calibration gate — MANDATORY before the depth delta is read)

Both depth arms MUST clear ALL of these on the held-out set **before** the bpb delta counts. This is the
gate the −3.46σ run lacked:

1. **Off the Φ saturation pin** — Φ < `PHI_BREAKDOWN_MIN = 0.80` and not pinned at 1.0 (the smoke showed
   Φ=0.54 at step 5, so the collapse-immune Φ *can* sit off the pin — reachable). (ARM A geo has no Φ;
   for it this clause is N/A — it clears on the d_FR floor alone.)
2. **Finite held-out d_FR** (and CE-bpb) — no NaN/Inf over the full held-out set.
3. **Below the random/unigram d_FR floor by a pre-stated margin** — converged held-out mean d_FR must beat
   the frequency-only (unigram) d_FR baseline by ≥ a committed margin, demonstrating the arm learned
   *structure*, not just token frequency. (The −3.46σ null fired because even 1L sat *above* the floor —
   measured on the d_FR axis now, the verdict axis.)

**If EITHER arm fails the maturity floor → verdict = WITHHELD / UNDERPOWERED, NOT "depth neutral."**
This is the precise distinction the prior run collapsed.

## Success band

`neocortex-qk-8L` reaches the maturity floor in **fewer steps** OR achieves **lower converged held-out
d_FR** than `neocortex-qk-1L-rec`, by a margin **exceeding the cross-seed spread**: |Δ d_FR| > k·σ_seed
(k committed at run-registration; ≥5 seeds → σ_seed is measurable). A margin inside σ_seed is **not** a
win. (CE-bpb reported alongside for external comparison, never the win condition.)

## Kill conditions (committed — each could fire)

1. **No separation beyond seed noise** after BOTH arms mature → **depth is NEUTRAL** (a real null,
   distinct from the underpowered null — only legitimate once the maturity floor is cleared).
2. **Either arm fails purity** (Adam recurrence / Euclidean contamination; the purity audit scans
   `experiments/`) → **INVALID**, not a result.
3. **Either arm fails the maturity floor** → **WITHHELD / UNDERPOWERED** (re-run at larger budget).

## Four measurement axes (qig-experiment-method tag vocabulary)

- **Channel:** held-out mean d_FR (verdict) + CE-bpb (external-comparison-only diagnostic).
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
channel (which already keys on drift velocity). **Shipped: qig-core v2.12.3 (tag on the fix), editable in
the studio venv so the fix is live now; republished to PyPI via `release.yml` on the tag for non-editable
consumers.** **Kill conditions above key off velocity-jumps + finite-CE + saturation-clearance — never the
absolute-drift log.**

**(b) The Euclidean `nn.Linear` lm_head is BEING FIXED (Tier-1), not labelled.** Confirmed at source:
`geocoding/model.py:55,95` and `qigkernels/kernel.py:137,234` both read out via `nn.Linear(hidden,vocab)`
— the same dot-product class as the purged attention cosine, one layer out, plus the Duchi-L2
`to_simplex_prob` (two Euclidean steps). Per the "no Euclidean use" directive this is a Tier-1 impurity,
so it is being **REPLACED** by the distance-to-basins `GeometricHead` (qig-core) — not annotated and kept.
The geometric-vs-linear comparison is **EXP-GEO-HEAD**, its OWN experiment (above), NOT folded into the
depth A/B. Until EXP-GEO-HEAD reports, EXP-CORTEX-AB may run on the linear head as a **common-mode**
baseline (byte-identical in both depth arms → the depth *difference* is unconfounded), but the depth
verdict is then explicitly "given the shared head" and the stack is NOT certified pure until the geometric
head ships (or earns a labelled exemption WITH evidence by training measurably worse).

---

## Ratification checklist (before the full-stack compute spend)

- [x] VERDICT=d_FR / external-only CE-bpb committed (PI "no Euclidean use" directive, 2026-06-30).
- [x] `NaturalGradientDescent` byte-identical across arms — source-confirmed (lr=1e-3, damping=1e-3,
      momentum=0.9); ARM A↔ARM B loss-value parity = 0.0 (`geo_cortex.py:438`, §H-checked: two distinct
      attention impls → same head+loss, not self-comparing).
- [x] Caveat (b) resolved: lm_head → Tier-1 `GeometricHead` (EXP-GEO-HEAD), being built — not labelled.
- [ ] Numeric maturity-floor margin (random/unigram-d_FR beat) + success-band k — committed at
      run-registration, after the 32k screen sizes them.
- [ ] EXP-GEO-HEAD reports (geometric ≥ linear, or labelled-exemption-with-evidence) before the full stack
      ships as pure.
- [ ] Register the prereg in the experiment registry before the full-stack run.
