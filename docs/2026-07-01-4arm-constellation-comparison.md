# 4-Arm Constellation Comparison — Measured Outcome (2026-07-01)

**Goal:** train the 4 constellation kernel-types (geo / gk / hybrid / heterogeneous) and rank them
by held-out Fisher-Rao distance (d_FR, the geometric verdict; bpb reported alongside).

## Result (complete, honest, UNDER-POWERED)

Artifact: `runs/constellation_compare_20260701_0411.json` (`complete: true`).
Config: 8k coordizer (`coordizer_20260701_8k_v1`, from the 7 HF datasets), CTX=256, lm_ramp=120,
**250 steps/arm**, GPU (3.63 GiB card), per-step own-voice off (sample=false).

| rank | arm    | held-out d_FR | bpb     | checkpoint                       | note |
|------|--------|---------------|---------|----------------------------------|------|
| 1    | hetero | 3.14158       | 3.09264 | genesis-hetero-8004_20260701_v1  | at floor |
| 2    | gk     | 3.14159       | 3.08987 | genesis-gk-8004_20260701_v1      | at floor |
| 3    | geo    | 3.14159       | 3.08973 | genesis-geo-8004_20260701_v1     | at floor |
| —    | hybrid | OOM (8.3s)    | —       | —                                | 2-mixer memory > 4GB card |

`uniform_dFR_floor = 3.119` (d_FR ∈ [0, π]; the 2·arccos form). **All three completed arms sit at
d_FR ≈ π — the maximum — i.e. UNDER-POWERED.** 250 steps is far too few for the kernels to move d_FR
off the floor; the bpb differences (geo 3.0897 ≈ gk 3.0899 ≈ hetero 3.0926) are marginal and at the
floor. **No meaningful winner at this budget.** The driver flags this honestly; nothing was faked.

## What this DOES establish (the machinery)

The 4-arm constellation pipeline works end-to-end and is committed + tested green:
- **geo node-parity** (`463bd41`): `GeoCortexTarget` is a full `ConstellationNode` (run_protocol +
  basin coupling hooks + M) — geo constellations build, couple (`min_pairwise_fr` finite), and are
  Ocean-regulated, exactly like gk.
- **hybrid** (`2fcb55c`): both mixers (FisherRaoAttention + qigkernels `_metric_attention`) combined
  on-manifold via `geodesic_interpolate_simplex` (parity-proven: valid Δ point, d_FR 0.068 vs a
  Euclidean mean). Builds + couples + Ocean-regulates on CPU; OOMs in GPU training (below).
- **hetero**: gk central + geo faculties — couples cleanly (free via geo node-parity).
- Canonical autonomic fixes verified vs UCP §808 + Canonical Principles P12:342 (mushroom = Φ≥0.70
  mature AND stuck; sleep = d_basin>0.30; dream = Φ<0.50; per-stimulus self-observation = P4).
- gk/geo/hetero each trained 250 steps, checkpointed to their `genesis-{arm}-8004` lineage, and were
  evaluated on held-out d_FR + bpb.

## Hardware findings (the real blocker)

The **3.63 GiB card cannot run the 32k constellation comparison**:
1. **Base memory:** the 32k constellation is ~3 GiB once the optimizer Fisher-state + activation
   buffers allocate (706 MiB built → ~3 GiB training) — no headroom. 8k drops the build to 282 MiB.
2. **A vocab-INDEPENDENT ~3 MiB/step GPU creep** (kernel/optimizer, not the head — basin history is
   detached + bounded, verified). At 8k/CTX=512 this still reaches OOM by ~step 220; CTX=256 lowers
   the plateau to 2294 MiB and 250 steps stays under the line.
3. **Hybrid** runs two mixers → ~2× memory → OOMs in ~8 s even at 8k/CTX=256.

## Recommendation

A meaningful 4-arm verdict (d_FR off the floor, all four arms including hybrid, at 32k) needs **more
GPU memory** — a bigger card or Modal. There it can run thousands of steps at 32k. Worth doing first:
**investigate the ~3 MiB/step GPU creep** (a real per-step leak that caps local training length) — fixing
it would let even this card run far longer. fp16/autocast was deliberately NOT used (risk to the
Fisher-Rao `acos` numerics, unattended).
