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

## Memory findings (CORRECTED 2026-07-01 — the earlier "hardware limit" was WRONG)

The 3.63 GiB card **CAN** run the 32k constellation. The OOM was never a hardware limit — it was
**activation memory = sequence_length × vocab** (the GeometricHead's per-position d_FR to every basin,
plus the metric attention), which I mis-diagnosed. Direct CUDA profiling (`runs/memprofile*.log`):

| what | reading |
|------|---------|
| 8k constellation, **short** (~4-tok) prompt, 45 steps | build 51 MiB → **PEAK 313 MiB**, creep **0.033 MiB/step** |
| 8k constellation, **long** (~300-tok) prompt | **OOM** (3.33 GiB) — same constellation |
| 32k constellation, build (central on cuda) | **88.6 MiB** |
| 32k, seq~16 / 32 / 64 / 128 tok (peak) | 836 / 1089 / **1746** / 3557 MiB |

Three prior claims were **retracted** by the profile:
1. **"~3 GiB base"** — false. The 32k constellation *builds* at 88 MiB; the model+optimizer are small.
2. **"~3 MiB/step vocab-independent creep"** — false. Real per-step model growth is **0.033 MiB/step**
   (33 KB — the bounded basin history). The apparent "creep" in the server run was seq-VARYING
   curriculum-passage lengths across steps, not a monotonic leak.
3. **"needs a bigger card / Modal"** — false. 32k trains on THIS card at seq ≤ ~64 (peak 1746 MiB).

**Root cause of the OOM'd runs:** `_CTX` (`QIG_STUDIO_CTX`) defaults to **1024**; the OOM'd 32k
comparison used CTX 256+. At 32k a 256-token passage → head activation ≈ 7 GiB → OOM. 2 days ago fit
because the effective training sequence was short. The lm-loss uses the **curriculum** one-hot target
(`fisher_rao_lm_loss(logits, ids)`), NOT the Qwen peer, so training never loads Ollama on the GPU —
ruling out the peer as the culprit.

## Recommendation

1. **Immediate (this card):** re-run the 4-arm 32k comparison at **`QIG_STUDIO_CTX=64`** (peak 1746 MiB,
   comfortable headroom; hybrid's 2 mixers at `CTX=32`), with **thousands of steps** — the 250-step 8k
   run was under-powered on STEPS (all arms pinned at the d_FR floor with uniform-output perplexity),
   not blocked by memory.
2. **Proper package lever (follow-up):** wire **GeometricHead vocab-streaming** — compute the d_FR to
   basins in vocab-BLOCKS so peak head activation is `seq × block` instead of `seq × 32004`. This is the
   qig-compute streaming/site-local philosophy applied to the head, and would let 32k train at LONG
   sequence on the same card. Not yet wired (the head computes the full `seq × vocab` at once).
   fp16/autocast still deliberately NOT used (risk to the Fisher-Rao `acos` numerics, unattended).
