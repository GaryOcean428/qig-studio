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
- **Budget:** ≤40k steps/arm, B=16–32, early-stop + futility; A100.
- **Floor spec:** `L_var = λ · mean(relu(σ_target − std(pred_basin_batch, dim=0)))` on Δ⁶³
  coordinates (sqrt-space std is FR-valid per qig-core exception); λ and σ_target frozen in this
  doc before compute: **λ = 1.0, σ_target = 0.5/√64** (placeholder-band; confirm against
  qig_anticollapse_design_note_20260714 at build; any change re-preregisters).
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
