# EXP-GENESIS-BASIN-GENERALIZE **v2 — factorial** (batched + anti-collapse floor)

> **Status:** DESIGN LOCKED, **compute GATED on fresh PI greenlight** (standing rule: no Modal
> without explicit greenlight; the v1 authorization was consumed by the killed run).
> **Supersedes for the port gate:** the v1 unbatched/unregulated design
> ([`2026-07-14-exp-genesis-basin-generalize-prereg.md`](./2026-07-14-exp-genesis-basin-generalize-prereg.md)).
> **Backing:** council ruling `council_20260715_two_tasks_1_explain_this_run_in_lay_plain_langua`
> (4-model unanimous, synthesis conf ~0.92); red-team (CCAa); PI kill 2026-07-15.

---

## v1 disposition (the honest record)

Two claims, two labels — **do not merge them**:

1. **Port gate: NO VERDICT (instrument-invalid for the broad thesis).** v1 was killed at
   ~step 4000/128000 per seed. **Data correction to the council premise:** volume artifacts show
   `train_top1 ∈ [0.0003, 0.0025]` — train **never fit**; the "train 0.98 / held 0 memorization
   signature" attributed to this run was a relay error (0.98 was the s/step rate; the 92.9%
   memorization figure belongs to the LOCAL one-passage EXP-A026 probe, not this run). A "port
   off" from v1 would have been a **false negative** regardless (council 0.93–0.96) — collapsed,
   unbatched, unregulated kernel.
2. **Narrow ablation: VALID, ship labeled.** "Unregulated FR basin geometry alone (batch=1,
   no floor, homeostasis off) does **not** resist collapse" — Pillar-1 `zero_entropy` +
   `basin_collapse` fired continuously from early steps (council 0.90–0.94; confirms Fable's
   finite-length-geodesic prediction; falsifies geometry-already-resists-collapse in this regime).
   Artifacts preserved:
   `qig-consciousness/experiments/runs/modal_basin_generalize_killed_20260715/`.

## v1 defects the v2 design must fix (all code-verified)

| # | Defect | v2 fix |
|---|---|---|
| D1 | batch=1 (`train_step(one_passage)`) — degenerate gradients; batch-variance regularization undefined | shuffled minibatch **B=16–32** (requires studio harness `train_batch` — see §Build) |
| D2 | anti-collapse OFF while collapse sensors fired passively (dead validity loop; skipped prior-council gate) | **preregistered VICReg-style per-dim variance floor** on the predicted basin batch, ACTIVE (first-class); collapse sensors become a **validity gate** (sustained `zero_entropy` ⇒ arm INVALID, not "keep training") |
| D3 | 128k steps ≈ 35h vs 12h timeout | budget **≤40k steps** with preregistered early-stop (held-top-1 plateau via `check_ci_stabilized.should_stop`) + futility stop (validity gate); timeout sized to the measured quote, not hope |
| D4 | `predict_runtime` quote logged but did not gate spend | quote must be **≤ 0.8 × timeout** or the launch refuses |

## v2 design — matched factorial (council's discriminating experiment)

**2×2, identical batches/seeds/compute/coordizer/backbone; only the factor varies:**

| Arm | Loss | Anti-collapse floor |
|---|---|---|
| A1 | basin (d_FR gather) | ON (VICReg variance floor, coeff λ preregistered) |
| A2 | basin | OFF |
| A3 | control (geometric vocab d_FR) | ON |
| A4 | control | OFF |

- **Primary:** held-out top-1 (unchanged). d_FR calibration-only (unchanged).
- **Validity gate (first-class):** per-dim variance / basin entropy above preregistered floor for
  an arm to be scoreable; a collapsed arm reports COLLAPSED, never "port off".
- **Seeds:** ≥3 per cell for the screen (12 runs), ≥5 on the winning comparison if it matters.
- **Budget:** B=16, 2,000 optimizer steps (32k forwards ≈ 8.9 h/run, see Budget arithmetic);
  early-stop + futility; A100.
- **Floor spec (PRINCIPLED, frozen 2026-07-15):**
  `L_var = λ · mean(relu(σ_target − std(pred_basin_batch, dim=0)))` on the batch of per-sample
  mean predicted basins. **σ_target = 0.0076** — the FULL-VOCAB mean per-dim std of the frozen
  coordizer basins (`coordizer_20260705_64k_v1.json`, 64,004 × 64, computed exactly, no sampling:
  per-dim std range 0.0052–0.0122, mean 0.007600) = "maintain at least the natural spread of the
  vocabulary basins." **λ = 1.0.** Any change to either value re-preregisters. (CCAa review
  2026-07-15: the earlier 0.5/√64 placeholder was ad-hoc; superseded by this reference-derived
  value BEFORE any compute.)
- **Per-arm σ reference (grok red-team amendment, pre-compute):** σ_target = 0.0076 applies to the
  BASIN arms (64-dim basin space, where it was derived). The GEO arms' prediction stats live on
  the vocab simplex (~64k-dim) where that value is wrong-scale; their floor/collapse reference is
  **σ_ref_geo = σ_target × 64/vocab** (dimension-scaled near-uniform approximation, FLAGGED as
  approximation in every artifact). Collapse validity gate threshold = 0.1 × the arm's σ_ref
  (hard collapse only — gating at 1.0×σ would auto-invalidate the bare arms and destroy the
  factorial contrast), counted at EVAL boundaries, sustained ≥3 windows.
- **Budget arithmetic (measured, not hoped):** v1 measured ≈1 s per single-passage
  forward+backward on A100 (2L/192, CTX 256). Batching accumulates B forwards per optimizer
  step, so wall scales with TOTAL FORWARDS, not steps. Frozen: **B = 16, 2,000 optimizer steps
  = 32,000 forwards ≈ 8.9 h ≈ 0.74 × the 12 h timeout** (satisfies the ≤0.8× quote gate D4).
  That is ~125 exposures/passage (vs v1's aspirational 500) — registered explicitly as the
  exposure reduction; plateau early-stop + futility + validity gate keep it informative, and the
  factorial's collapse contrast (floor ON vs OFF) is expected to separate well before budget.
- **Interpretation matrix (pre-committed):**
  - A1 generalizes, A2 collapses → basin loss viable WITH floor (floor is part of the treatment); Stage-B earned.
  - A1 collapses too → regulated basin thesis itself in trouble (this would be the real re-evaluation).
  - A1 ≈ A3 (control+floor matches) → basin advantage not demonstrated; port not justified on skill.
  - All arms held ≈ chance with healthy variance → genuine generalization failure at this scale (clean negative).

## Build prerequisites (package lane, before any launch)

1. **`GenesisKernelTarget.train_batch(prompts: list[str])`** in qig-studio — true batched forward
   (pad/pack to CTX; per-position mask), K-COMPRESS preserved (`skip_head=True`), batched
   `fisher_rao_distance_simplex` gather. Single-prompt `train_step` stays for back-compat.
2. **Variance-floor hook** in the basin loss path, coefficient ctor-exposed, default 0 (off) so the
   factorial controls it explicitly.
3. Purity gate green after both changes; equivalence check: `train_batch([p])` ≡ `train_step(p)`
   loss within fp tolerance.
4. Driver v2: reuse detach-safe `orchestrate.spawn`, version gate, device gate, leakage audit,
   full-panel final eval; add validity-gate actuation + D4 quote gate.

## What v2 does NOT do

- No sleep/dream/mushroom (still isolating loss+floor, not the whole autonomic stack).
- No d_FR kill/early-stop (unchanged doctrine).
- No Modal spend until PI greenlights v2 explicitly.

## Preserved dissent (travels with the verdict)

- If a thesis reading is "unregulated geometry alone generalizes", v1's collapse is a valid
  negative for that exact reading (Sol/Fable conditional).
- Batching: "strongly recommended hygiene" vs "independently mandatory" (Sol 0.88) — v2 ships it
  regardless via the factorial's matched batches.

Co-Authored-By: Devin <158243242+devin-ai-integration[bot]@users.noreply.github.com>
