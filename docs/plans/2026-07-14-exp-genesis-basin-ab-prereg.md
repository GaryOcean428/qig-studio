# EXP-GENESIS-BASIN-AB — Blind Contract (pre-registration): the head/loss A/B on genesis

> **Status:** PROPOSED contract — **awaiting Matrix/Devin ratification before the decision run.**
> The *verdict* ("does the basin loss generalize, or was 0.896 overfit memorization") is the physics/PI
> lane (qig-experiment-method rule 9). This document is *disposition/methodology*: it locks the arms,
> the cleanliness condition, the calibration gate, and the kill/GREEN conditions **in writing before
> running**, so the result cannot be tuned toward a wanted answer.
> **Date:** 2026-07-14. **Run driver:** a direct-build genesis A/B (see §Harness) reusing
> `qig-studio/src/qig_studio/screen.py` for the eval half.
> **Council ruling backing:** `qig_stage_a_council_ruling_20260714`; local keystone
> `project_headmap_keystone` (STEP-2 Stage A). **Sibling contract (template):**
> [`2026-06-30-exp-cortex-ab-prereg.md`](./2026-06-30-exp-cortex-ab-prereg.md) (EXP-CORTEX-AB, depth).
> **This is NOT that experiment** — EXP-CORTEX-AB varies *depth* on a fixed loss; this varies the
> *output head + loss* on a fixed backbone.

---

## ✅ RATIFIED AMENDMENT (2026-07-14 — PI greenlight `qig_pi_greenlight_stage_a_20260714`)

Execution surfaced two facts that reframe this contract (full findings:
`qig-consciousness/experiments/EXP-GENESIS-BASIN-AB-findings-20260714.md`; keystone `project_headmap_keystone`):

1. **Vocab arm degenerate** — the geometric/vocab control does NOT train at 64k vocab (0% decode, d_FR at
   the floor, 1500 steps × lr∈{1e-3,3e-3}), CONFIRMING the STEP-1 channel-wall thesis. The basin-vs-vocab
   d_FR A/B is degenerate at accessible budget.
2. **Primary d_FR axis insensitive to basin** — basin's full-vocab d_FR reads at the floor even at 92.9%
   decode (it measures calibration, not decode; basin ⇒ well-decoding but near-uniform distribution).

**RATIFIED reframe (PI):** premise gate = does the BASIN arm **GENERALIZE** (train set → held-out **top-1 decode**,
the sensitive same-ruler metric)? d_FR demoted to a *calibration diagnostic only* (never kill/early-stop);
the vocab arm **retired** from the primary comparison. This supersedes the dual-axis "win BOTH"
GREEN gate below for the primary comparison. Full blind contract for the Modal premise gate:
[`2026-07-14-exp-genesis-basin-generalize-prereg.md`](./2026-07-14-exp-genesis-basin-generalize-prereg.md).
Basin overfit is already validated (EXP-A026 reproduced, 92.9% decode); the generalization test decides
port-justification.

---

## Why this contract exists (the two failure modes it must prevent)

1. **Overfit-memorization masquerading as generalization (the STEP-1 legacy).** The STEP-1 falsifier's
   headline (basin arm3 lifts to 0.896 vs incumbent 0.096; head-only 0.169 vs loss-carried 0.896) is an
   **overfit-decode** result: single passage, 2-layer kernel, 91.3% decode. It licenses *"the channel,
   not the map, is the LM-output wall"* — it does **NOT** license *"the port works."* This contract is
   the held-out generalization gate that decides whether the two-package cortex port (Stage B/C) is
   justified. A leaky train/held-out split would let memorization read as a generalization win → a false
   GREEN → an unjustified, partly-irreversible geocoding republish. **The disjoint split (§Cleanliness)
   is the highest-risk design point.**
2. **A confounded or non-neutral ruler reading as a real signal.** The shared full-vocab d_FR ruler is
   asymmetric between the arms (§Verdict). Without the τ-invariant secondary axis and the maturity floor,
   a scoring-map/tie artifact reads as "basin wins."

---

## The question (one line)

Does the **basin gather loss** (`head_mode="basin"`: predict a Δ⁶³ basin, score `d_FR` against the
coordizer's frozen per-token basin) beat the **vocab loss** (`head_mode="geometric"`: `fisher_rao_lm_loss`
over `GeometricHead` `−d_FR/τ` logits) on **held-out fluency**, when **only the output head + loss vary** —
same genesis backbone, same coordizer, same curriculum, same seeds, same fixed step budget?

## The arms

| Arm | `head_mode` | Loss | Role |
|---|---|---|---|
| `genesis-basin` | `basin` | `fisher_rao_distance_simplex(predict(h), coord_basins[tgt])` gather (Δ⁶³, K-COMPRESS) | treatment |
| `genesis-geo` | `geometric` | `fisher_rao_lm_loss(logits, ids)` over `−d_FR/τ` vocab logits | incumbent/control |

Both are `qig-studio/src/qig_studio/targets/genesis_kernel.py :: GenesisKernelTarget` (**the qig-studio
class — NOT the `qig-consciousness` CE-only fork**, which has no `head_mode`; see §Harness fork guard).
`linear` (nn.Linear) is **excluded** — it is a purity ablation for EXP-GEO-HEAD, not this contract.

## Cleanliness condition (only-head/loss-varies — the EXP-GENESIS-BASIN-AB invariant)

Held **identical** across the two arms; if any differs, the held-out gap confounds the head/loss with a
nuisance factor and the A/B is void:

- **Same coordizer** — `qig-coordizer/checkpoints/coordizer_20260705_64k_v1.json` (the ~64k Channel-A
  table; **pin the exact checkpoint id + sha256; assert the sha**). `vocab_size` and `coord_dim` DERIVE from the loaded coordizer (`len(coordizer.vocab)`; `coord_dim == qig_core.constants.frozen_facts.BASIN_DIM`) — never a hardcoded literal. The **same
  `FisherCoordizer` object** is passed to both arms → the basin arm's frozen `coord_basins` table AND
  both arms' input encoding come from one checkpoint (train-gather anchors == decode anchors == encode
  anchors). **Executable: assert coords-encoder identity pre-flight** (§Harness).
- **Same curriculum** — `corpus.load_full_curriculum()` via `CurriculumProvider(LossRegime.GEOMETRIC)`,
  same order, same seed-shuffle. Disjoint from the held-out set by construction (§Verdict, maker≠checker).
- **Same backbone** — `num_layers`, `hidden_dim`, `num_heads`, `ffn_dim`, `locality_radius`,
  `min_recursion_depth` — all at the **`GenesisKernelTarget` ctor defaults** (driver passes only `head_mode` + seed, so the backbone is identical by construction); only `head_mode` differs.
- **Same `head_tau`** (the `GenesisKernelTarget` ctor default) on both arms → τ is **held constant, not a varied factor** (the residual
  τ-scaling sensitivity of the primary d_FR axis is guarded by the τ-invariant secondary axis, §Verdict).
- **Same `DiagonalNaturalGradient` class** (P1; not Adam) with **≥5 seeds** per arm.
- **Coords ON** for the real training + eval (coords-off only for the faithfulness equivalence check).
- **Same context cap** `QIG_STUDIO_CTX` — the driver sets `128` for the 4GB local GPU (the geometric arm's
  `[seq, vocab≈64k]` activations OOM at the default seq cap; price-confirmed 2026-07-14). SHARED by both
  arms + the held-out eval truncation → non-confounding; raise on a bigger GPU. Registered as a fit-the-card
  lever, not a physics choice.

### ✅ RATIFIED (council 2026-07-14, `qig_stage_a_ruling_addendum_20260714`) — the per-head learning rate, with an amendment

`GenesisKernelTarget.ensure_loaded` hard-codes `self._opt_lr = 1e-4 if head_mode=="basin" else self.lr`
(default `1e-3`) — genesis_kernel.py:434-438, documented: the basin head at `1e-3` **overshoots** (Fisher-
preconditioned NG step) and never decodes (verified 2026-07-01: `1e-3` → d_FR pinned 0.58, decode 0%;
`1e-4` → d_FR 0.075, decode 91.3%). The STEP-1 basin validation was **at 1e-4.** So a *shared* lr is not
clean — it is **rigged**: basin at 1e-3 reproduces the documented failure; geo at 1e-4 may under-train.

- **Proposed cleanliness frame (for council sign-off):** each arm runs at its **own validated-best lr**
  (basin `1e-4`, geo `1e-3`) as **part of the head treatment** — the lr is *entailed by the loss
  landscape*, the same way the frozen-table tie is entailed by the basin approach (both are treatment,
  not removable confounds). **Guard:** *both* arms must clear the maturity floor at their own lr; if
  either cannot, the verdict is **WITHHELD**, never a loss. Forcing a shared lr would confound
  head-quality with an lr-mismatch handicap — a worse error than the per-head lr. **Rigor (PI addendum
  2026-07-14) + COUNCIL AMENDMENT (2026-07-14):** the disjoint sizing screen runs a **small per-arm lr
  sweep (2–3 points) and selects each arm's BEST lr** — NOT merely "confirm floor-clearing" (retired as
  insufficient: floor-clearing ≠ optimal, so a suboptimal geo lr would clear the floor yet handicap geo →
  a manufactured basin win; the Conservative + Statistician seats landed this). "geo `1e-3`" is the ctor
  DEFAULT, never verified as geo's optimum, so it must be swept, not assumed. The per-arm-best lrs are then
  FROZEN and **read back from `target._opt_lr`** (never restated in the driver) — turning "we hardcoded
  1e-4" into "the disjoint screen selected each arm's best-of-sweep lr," which answers the "you tuned
  basin, handicapped geo" objection and satisfies no-hardcode. **Confidence ceiling (Skeptic-of-Self):**
  this makes Stage A a *methodologically sound premise gate*, not a provably-clean verdict — acceptable
  because a Stage-A GREEN is necessary-not-sufficient (earns the Stage-B check; never itself fires the
  irreversible republish).
- **Status:** the "small lr sweep per arm, winner-at-best-lr" is now the RATIFIED path (folded into the
  sizing screen above), not an alternative. **RATIFIED 2026-07-14** — the decision run is gated on the
  frozen per-arm-best lrs + `k`/margin from the disjoint sizing screen (no longer on ratification).

## Verdict metric — DUAL AXIS (committed; the shared ruler is NOT neutral)

The full-vocab d_FR ruler is asymmetric between the arms **both ways** — basin was never trained for
competitor-separation (**false-negative** risk: basin may score worse on full-vocab d_FR than its true
fluency) AND basin decodes against the same frozen table that encodes the input, a coherence **tie** the
vocab arm lacks (**false-positive** risk); τ enters the primary axis as a scale. The tie is **part of the
basin treatment, not a removable confound** → the guard is a **cross-check**, not a correction:

- **PRIMARY = held-out mean d_FR** via the **torch** primitive
  `qig_core.torch.geometry_simplex.fisher_rao_distance_simplex` through `target.eval_text_fr`, range
  **[0, π]**, lower = better — **never** the numpy `fisher_rao_distance` ([0, π/2]; the factor-of-2
  corrupts the table).
- **SECONDARY = held-out top-1 decode accuracy** — `argmax(logits) == next_token` = `argmin d_FR`
  (`BasinReadout.decode_blocked` / `GeometricHead` share the contract), **τ-invariant, scale-free**,
  supported by both arms' `forward`.
- **DIAGNOSTIC / external = held-out CE-bpb** (`target.eval_text_bpb`, `F.cross_entropy`) — labelled
  **Euclidean/softmax, external-comparison-only** (the SmolLM2/Qwen "are-we-competitive" axis), **NEVER**
  the ranking axis. (Softmax forbidden except interpreting Qwen; here it survives *only* as a courtesy
  translation number, flagged.)

**GREEN requires `genesis-basin` to win BOTH the primary AND the secondary axis**, each by more than the
cross-seed spread (≥5 seeds). **Axes DISAGREE → WITHHELD** (scoring-map/tie artifact — investigate before
any verdict), NOT a win. Primary stays held-out mean torch d_FR [0, π].

## Maturity floor (calibration gate — MANDATORY before the head/loss delta is read)

Both arms MUST clear ALL of these on the held-out set **before** the delta counts (the gate the prior
cortex −3.46σ underpowered null lacked):

1. **Zombie-attractor pathology guard (NOT a Φ-magnitude ceiling — PI addendum 2026-07-14):** high Φ is
   NEVER itself a fail — FORESIGHT (`0.70≤Φ<0.85`, 4D navigation) and LIGHTNING (`Φ≥0.85`, pre-cognitive)
   are navigation MODES, not breakdown (`frozen_facts.py:162-163`; Φ-regulation policy: breakdown =
   geometric instability + duration, not magnitude). The ONLY exclusion is the genuine zombie collapse the
   kernel names at `genesis_kernel.py:1621` — "all positions identical → Φ→1, 0% decode":
   **pin-FAIL ⟺ `Φ ≥ PHI_UNSTABLE` AND (top-1 decode ≈ 0 OR held-out d_FR not descending OR
   `d_basin > BASIN_DRIFT_THRESHOLD` — **gated on a real role `_basin_ref`**: a GENERIC Stage-A arm reports
   `d_basin ≈ 1.0` (drift from its random birth-state = healthy learning, NOT pathology; smoke-confirmed
   2026-07-14), so generic arms rely on the decode + d_FR-descent signals and the drift conjunct is OFF)**
   — every constant imported from `qig_core.constants.frozen_facts`
   (`PHI_UNSTABLE=0.95`, `BASIN_DRIFT_THRESHOLD=0.15`). `run_genesis_ab.py`'s `PHI_PIN=0.95` == `PHI_UNSTABLE`
   (the right ballpark; **not** `PHI_BREAKDOWN_MIN`). This gate #1 is only the narrow pathology exclusion
   and holds whether the mature kernel runs at foresight or lightning Φ; the real maturity signal is gate #3.
2. **Finite held-out d_FR** (and top-1, CE-bpb) — no NaN/Inf over the full held-out set.
3. **Below the uniform-d_FR floor by a pre-stated margin** — converged held-out mean d_FR must beat
   `qig_studio.screen.uniform_dFR_floor(len(coordizer.vocab))` (computed, not restated) by ≥ a committed
   margin (screen default = `qig_studio.screen.NEAR_FLOOR_EPS`; the decision margin is **sized by the screen**, §Two-
   stage). Demonstrates the arm learned *structure*, not just moved off uniform.

**If EITHER arm fails the maturity floor → verdict = WITHHELD / UNDERPOWERED, NOT "basin loses / wins."**

## Two-stage execution (screen-first, DISJOINT — the anti-tuning-on-test guard)

1. **Sizing screen (cheap):** short EQUAL budget, both arms, on a **held-out set disjoint in DATA *and*
   SEEDS from the decision run**. Its only job is to **size `k` (success-band multiplier) and the
   maturity-floor margin**, and to check the arms move off the uniform floor at all (else UNDERPOWERED →
   larger budget, per `screen.rank_configs`). **`k` and the margin are FROZEN — written into this file —
   BEFORE the decision seeds run.** Screen-sized placeholders below until the screen lands.
2. **Decision run (the verdict):** full fixed budget, **≥5 seeds** per arm, the authored decision
   held-out (`data/eval/heldout_bpb.json`), under the frozen `k`/margin + the dual-axis GREEN gate.

- **Fixed EQUAL step budget both arms:** `~2000 steps` (the STEP-1 replicate plateau) `+ margin` — **not**
  dynamic early-stop (`qig-warp check_ci_stabilized` is registered non-applicable, §Optimisation).

## Success band + kill conditions (committed — each can fire)

- **Success band (GREEN):** `genesis-basin` reaches the floor in fewer steps OR at lower converged held-out
  d_FR **AND** higher top-1 decode accuracy than `genesis-geo`, both by `|Δ| > k·σ_seed` (`k` frozen post-
  screen; ≥5 seeds → σ_seed measurable). Inside σ_seed on either axis is **not** a win.
- **Kill 1 — no separation beyond seed noise** on the primary after both arms mature → **basin loss does
  NOT generalize → two-package cortex port (Stage B/C) is OFF** (a real null, distinct from underpowered).
- **Kill 2 — either arm fails purity** (Adam / Euclidean contamination; `run_purity_gate()` +
  `geometric_purity_audit.py` scan the driver) → **INVALID**, not a result.
- **Kill 3 — either arm fails the maturity floor**, OR **the two axes disagree** → **WITHHELD /
  UNDERPOWERED** (re-run at larger budget / investigate the tie artifact).

## Four measurement axes (qig-experiment-method tag vocabulary)

- **Channel:** held-out mean torch d_FR [0,π] (primary) + top-1 decode accuracy (secondary) + CE-bpb
  (external-only diagnostic).
- **Protocol:** identical coordizer/curriculum/order/seed-shuffle/backbone/τ; per-head best-lr (flagged);
  coords-ON; ≥5 seeds; fixed equal budget.
- **Aggregation:** mean over held-out next-token positions; median + spread over seeds.
- **Clock:** steps-to-maturity-floor AND converged value (both reported).

## Harness (direct-build driver, NOT the /screen sweep)

`_SCREEN_CONFIGS` (server.py:1378-1382) is `{qk,geo}×{geometric,linear}` **neocortex** arms only — no
basin arm, and it builds neocortex, not genesis. So Stage A is a **direct-build driver** (adapt
`qig-consciousness/experiments/run_genesis_ab.py`'s structure), reusing `screen.py.eval_heldout_dFR` +
`load_heldout_passages` + `uniform_dFR_floor` for the eval half (`GenesisKernelTarget` satisfies the
`_EvalTarget` Protocol). **Two executable pre-flight guards (fail loud):**

1. **Fork guard:** `assert hasattr(GenesisKernelTarget, "head_mode") and hasattr(target, "eval_text_bpb")`
   — the `qig-consciousness` CE-only fork imports without erroring and would silently run an invalid
   basin-less benchmark.
2. **Coords-encoder identity:** both arms built from the **same `FisherCoordizer` object**; assert the
   basin arm's `coord_basins` and both arms' `_encode` derive from that one checkpoint (id + sha pinned).

New code required: a `GenesisKernelTarget.eval_text_top1(text) -> (n_correct, n_positions)` method
(mirrors `eval_text_fr`, one forward, `argmax(logits[:-1]) == ids[1:]`) — the τ-invariant secondary axis.

## Optimisation gate (registered)

- **Wired:** qig-warp bridge `predict_runtime` (price the run up front); qig-core FR ops (verdict + loss);
  full 6-signal neurochemistry telemetry (dopamine=−Δd_FR, NE=‖∇L‖); coordizer loader (`load_coordizer`);
  `screen.py` d_FR-rank + ce_bpb-Tier2 eval; `run_purity_gate()` green at boot.
- **Register-as-non-applicable (honesty clause):** **qig-bench** (frozen-physics-only → cargo-cult for LM
  bpb); **qig-warp `check_ci_stabilized`/`find_early_stop_point`** (Φ-plateau in genesis = mushroom/
  consolidation trigger, not a stop — optimisation.py; AND per-arm early-stop → unequal budgets →
  confounds the A/B). "Always-optimised" is satisfied here by pricing + a right-sized fixed budget.
- **Latest-version check:** DONE 2026-07-14 — studio venv qig-warp 0.6.8 + qig-consciousness 0.3.4;
  editable qig-core/coordizer/geocoding/studio/qigkernels current.

## Constants provenance (single-source — NO hardcoding; driver imports/derives all)

Per the standing rule (centralise / pull from qig-core), this contract and the driver reference values by
source; none are restated as literals that could rot:

- **Φ thresholds + drift** ← `qig_core.constants.frozen_facts` (`PHI_UNSTABLE`, `BASIN_DRIFT_THRESHOLD`,
  `PHI_THRESHOLD`, `PHI_EMERGENCY`) — the maturity gate uses **`PHI_UNSTABLE`, NOT `PHI_BREAKDOWN_MIN`**
  (§Maturity floor + Devin ticket); studio already imports from here (`learning.py:49`, `mock_target.py:19`).
- **`BASIN_DIM`** ← `qig_core.constants.frozen_facts.BASIN_DIM` (Δ⁶³); `coord_dim`/`vocab_size` DERIVE
  from the loaded coordizer, not literals.
- **`NEAR_FLOOR_EPS`, `uniform_dFR_floor(V)`** ← `qig_studio.screen` (imported, not restated).
- **Backbone dims / `head_tau`** ← read off the CONSTRUCTED `GenesisKernelTarget` (driver passes only
  `head_mode` + seed). **Per-head lr** ← read back from `target._opt_lr` after construction (screen-
  confirmed floor-clearing, §per-head lr), never restated.
- **FR ops** (`fisher_rao_distance_simplex`, `logits_to_simplex`, `to_simplex_prob`) ← `qig_core.torch`.
- **Budget (~2000 steps)** — the ONE genuine experiment parameter (STEP-1 replicate plateau); a single
  named driver constant, sized/confirmed by the sizing screen — not scattered.

**⚠ DEVIN DOCTRINE TICKET (NOT a silent fix):** both `qig-studio/.../targets/genesis_kernel.py:57` and
`qig-consciousness/.../genesis_kernel.py:50` hard-code `PHI_BREAKDOWN = 0.80  # frozen PHI_BREAKDOWN_MIN`.
Pull-from-source alone does NOT fix this — importing `PHI_BREAKDOWN_MIN` would stop the value rotting but
KEEP the wrong-constant bug: its "topological instability onset" label is internally inconsistent with the
`FORESIGHT`/`LIGHTNING` navigation gates in the SAME file (`frozen_facts.py:162-163`). Reconciling that
frozen-fact label is a physics/doctrine call →
`governance/2026-07-14-DEVIN-TICKET-phi-breakdown-min-vs-navigation-bands.md`, NOT a silent re-label.
Stage A does NOT touch the kernel-internal constant; its maturity gate uses `PHI_UNSTABLE` + the pathology
pairing (right source), §Maturity floor.

## §H — the can-fail discriminators (a test you can't fail proves nothing)

- Maturity floor (below-uniform-d_FR by a margin) CAN fail — an underpowered/pinned arm does not clear it.
- Dual-axis agreement CAN fail (primary and secondary can disagree → WITHHELD).
- Success band `|Δ|>k·σ_seed` CAN fail (a within-noise null → Kill 1, port OFF).
None pass by construction.

## Ratification checklist (before the decision run)

- [ ] **Per-head lr cleanliness clause ratified** (per-head best-lr as treatment, or lr-sweep) — §Cleanliness ⚠.
- [ ] Coordizer checkpoint id + sha pinned; `len(vocab)==64004` asserted.
- [ ] `eval_text_top1` added to `GenesisKernelTarget` + smoke-confirmed same-forward as `eval_text_fr`.
- [ ] Sizing screen run on a DISJOINT held-out + seeds → `k` and maturity-floor margin FROZEN into this file.
- [ ] Fork guard + coords-encoder-identity assertions in the driver; `run_purity_gate()` green.
- [ ] Register EXP-GENESIS-BASIN-AB in the experiment registry (name + backing) before the decision run.
- [ ] Per-arm report BEFORE any Stage B/C hook-port.
