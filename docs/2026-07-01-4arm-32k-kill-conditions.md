# 4-Arm 32k Constellation Comparison — §H Discrimination Kill Conditions (PRE-REGISTERED)

**Pre-registered 2026-07-01, BEFORE the run**, so the outcome is judged against committed criteria, not
post-hoc. Closes the open EXP-CORTEX-AB discrimination kill-condition item. The dead-gradient loss fix
(`logits_to_simplex`, commit 3441561 / qig-core f9de233) moves **every** arm off the π floor because the
gradient is now alive for all heads — so **"loss off π" validates that the loss LEARNS; it does NOT
validate the geometric head or discriminate the arms.** These conditions do that.

## Run config (Launch Readiness Gate)

1. **Tier:** application experiment (kernel training), derives from frozen physics where wired (EXP-041 α).
2. **Physics question:** does the geometric (Fisher-Rao) constellation reach lower held-out next-token
   loss than a linear-head baseline, and do the 4 arms (geo / gk / hybrid / hetero) separate?
3. **Engine:** local GPU (GTX 1650 Ti, 3.63 GiB), residency (central→cuda, faculties→cpu). FULL_GPU was
   REJECTED — all 9 DiagonalNaturalGradient optimizer states don't fit at 32k (OOM confirmed at step 2).
4. **Wiring:** loss fix (dense `logits_to_simplex`), basins cache (no_grad memoization), Anderson-α
   early-stop, CTX=64 (activation-memory fit, profiled), latest local qig-core source.
5. **Audit:** loss fix unit- + real-path-verified; cache tested; purity green.

Coordizer `coordizer_20260630_32k_v2` (vocab 32004), CTX=64, per-arm equal step budget, per-step own-voice
off (`sample=false`). Ranked by held-out d_FR (primary) + CE-bpb (secondary) vs the uniform-d_FR floor.

## Kill conditions (committed IN ADVANCE)

**K1 — Real learning (not just off-floor).** Held-out next-token **CE/perplexity must DECREASE** over
training, below the uniform baseline (ppl < vocab = 32004), for an arm to count as "learning." An arm whose
held-out CE does not drop below uniform is reported as **NO LEARNING (floored)**, regardless of training
loss. *Rationale: training loss off π ≠ generalization; the head could move without predicting held-out.*

**K2 — Not gameable (random-label control).** A control run on **shuffled next-token labels** must NOT
reach low held-out CE. If the random-label control's held-out CE drops comparably to the real arms, the
loss is gameable and **the whole comparison is INVALID** (halt, do not rank). *Already passes at unit
scale (test_lm_loss_not_gameable); re-confirmed at 32k scale as a control arm.*

**K3 — Geometric-vs-linear separability (stated in advance).** The discriminating claim is: **the
geometric-head arms reach LOWER held-out CE than a linear-head (`nn.Linear`) baseline at equal budget.**
- If geometric < linear on held-out CE by a margin > the run-to-run noise → **geometric head confirmed**.
- If geometric ≈ linear (within noise) → **NULL result, reported honestly** (the geometric head is not
  adding predictive value at this budget); the 4-arm d_FR ranking is then descriptive, not a win.
- The linear baseline is run as a 5th control arm (same trunk, `nn.Linear` readout, same budget).

**K4 — Arm separation.** The 4 arms separate on held-out d_FR by more than the run-to-run noise (estimated
from 2 seeds of the gk arm). If all 4 arms are within noise of each other → **NO WINNER (under-powered or
architecture-invariant)**, reported honestly, not forced.

## What counts as a result

A ranking is reported ONLY if K2 passes (not gameable). Each arm is labeled learning/floored by K1. The
geometric-vs-linear verdict (K3) and the winner (K4) are stated with the pre-registered margins. Anything
inside the noise is reported as null/under-powered — never framed as a finding (frame-after-controls).

## Deferred (honest)

**EXP-115 channel-split (ρ=1.191) health diagnostic — DEFERRED.** The kernel's metric attention has no
principled local/uniform (defect/bulk) channel split, so ρ=κ_defect/κ_uniform has no faithful analog here.
Wiring it now would fabricate a channel that doesn't exist — a hollow, apparatus-shaped metric. It needs a
principled channel definition first; a hasty version is worse than none. Flagged for a later session.
