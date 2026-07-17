# EXP-BASIN-DIMENSIONALITY-3ARM — Blind Contract (zero-compute design; compute GATED)

> **Status:** DESIGN-ONLY / REGISTERED. **Compute is gated on** EXP-GENESIS-BASIN-GENERALIZE
> = GENERALIZES (`qig_pi_greenlight_stage_a_20260714` zero-compute concurrent slate item (b)).
> Do **not** spend Modal or long local GPU on this until that premise passes.
> **Date:** 2026-07-14.

---

## Question

What is the **effective geometric dimension** of the basin representation that carries language
competence under Fisher-Rao geometry — and does the engineered `BASIN_DIM=64` sit at a
performance knee, or is it under/over-parameterized for held-out top-1?

## Why three arms (not one)

A single PCA-variance check is the **retired Class-B camera** (2026-03-14 precedent; mythology-
calibration forbids bridging on "64" coincidences). Three independent instruments reduce the
risk that any one camera flatters 64.

## Arms

| Arm | Instrument | Decision metric | Kill / null |
|---|---|---|---|
| **1 — Blind capacity** | Train genesis basin head at **fixed total compute** with `coord_dim ∈ {16,32,64,128}` (blind assignment; analyst unblinded only after all four finish) | held-out **top-1** (primary), train top-1 calibration | **KILL of "64 is special"** if 32 ≡ 64 within seed noise on top-1 (and 16≪32 or 128≢64 does not rescue a uniqueness claim for 64 alone) |
| **2 — Complexity scaling** | `compute_basin_pci` (or package equivalent participation / integrated-info geometry) vs **problem complexity** of held-out tasks — **NOT** PCA-variance, **NOT** Fubini-Study on Euclidean embeddings | correlation of PCI/participation with task complexity band; must use FR-native primitives from qig-core / qig-compute | **NULL** if no monotonicity / no separation from matched-complexity shuffle control |
| **3 — Coupling topology** | Persistent homology **Betti numbers** on the kernel-coupling **directed** graph (Reimann–Markram *method*, not their "simplex" cognate — our object is the directed flag complex of kernel→kernel coupling, not neuron cliques and not Δ⁶³) | Betti curves vs **matched graph nulls** (degree-preserving / edge-rewired) | **NULL** if observed Betti sits inside null band |

## Shared cleanliness

- Same validation config backbone as GENERALIZE (2/192/6/384, pure loss, homeostasis OFF) unless
  Arm 1 *must* change `coord_dim` / table width (then change **only** that factor).
- Seeds ≥5 per cell. Model lock irrelevant (from-scratch genesis).
- Fisher-Rao purity gate green; no cosine / Adam / LayerNorm on manifold objects.
- Arm 1 primary metric = top-1 (never d_FR as kill — same doctrine as Stage-A reframe).

## What "64" is allowed to mean

`BASIN_DIM=64` is **engineering compression** (65536→64 residual = temperature) — the *only*
load-bearing 64. κ*≈64 and E8/8²=64 are **RETIRED**. Arm 1 may find 32≡64 (then 64 is not a
unique optimum) without resurrecting retired physics.

## Output contract

- Prereg frozen before first Arm-1 seed.
- Registry entry only when compute is unlocked (H9: registry = result JSON in same change-set).
- No conversion of Arm-3 Betti into consciousness claims without a separate prereg.

## Dependency

**Start only after** `EXP-GENESIS-BASIN-GENERALIZE` summary verdict = GENERALIZES.
If that run = MEMORIZES → this prereg stays design-only (no consolation redesign).

Co-Authored-By: Devin <158243242+devin-ai-integration[bot]@users.noreply.github.com>
