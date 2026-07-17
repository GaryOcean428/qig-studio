# EXP-GENESIS-BASIN-GENERALIZE — Blind Contract (pre-registration)

> **Status:** LOCKED contract — **PI greenlit 2026-07-14** (`qig_pi_greenlight_stage_a_20260714`).
> **Supersedes for the primary premise gate:** the dual-axis basin-vs-vocab GREEN gate in
> [`2026-07-14-exp-genesis-basin-ab-prereg.md`](./2026-07-14-exp-genesis-basin-ab-prereg.md)
> (that A/B is degenerate-but-informative; vocab arm retired; d_FR demoted to calibration).
> **Date:** 2026-07-14. **Driver:** `qig-consciousness/experiments/modal_genesis_basin_generalize.py`
> (Modal exclusive) + local harness re-export from `run_genesis_basin_ab.py::_build_arm`.
> **Backing:** `qig_pi_greenlight_stage_a_20260714`, council conf 0.91
> (`council_20260714_provide_full_prioritized_direction_on_two_things`), findings
> `EXP-GENESIS-BASIN-AB-findings-20260714.md`, launch note
> `qig-consciousness/experiments/2026-07-14-MODAL-genesis-basin-generalize-launch-note.md`.

---

## The question (one line)

Does the **basin** head+loss (`head_mode="basin"`, validated config) **GENERALIZE** — on a train set
large enough that pure memorization of each passage is expensive — to a **hash-disjoint held-out**
set, measured by **held-out top-1 decode** (τ-invariant, argmin-d_FR)?

## Why this contract exists (two failure modes)

1. **Memorization masquerading as port-justification.** One-passage overfit (EXP-A026 / local probe:
   d_FR 1.50→0.027, train decode 0→92.9%) is instrument validation, NOT a generalization win.
2. **Leaking train into held-out.** Hash-collision or authored-heldout overlap with curriculum → false
   GREEN → unjustified Stage-B/port. Leakage audit is pre-spend mandatory.

## What is RETIRED (ratified, do not reinstate)

| Item | Status |
|---|---|
| Vocab/geometric arm as primary control | **RETIRED** for this premise gate (cannot train at V≈64k under accessible budgets; confirms channel-wall) |
| Held-out mean d_FR as kill / early-stop / primary | **DEMOTION** → **calibration diagnostic only** (insensitive at the 92.9% train-decode floor) |
| Stop on train decode → 92.9% | **FORBIDDEN** as kill/early-stop (that is the memorization ceiling) |

## Metric stack

| Role | Metric | Notes |
|---|---|---|
| **PRIMARY** | held-out top-1 decode accuracy | argmin-d_FR / argmax logits; τ-invariant; chance ≈ 1/V |
| **Calibration** | train top-1, held-out mean d_FR | train/held GAP is the generalization signal; d_FR never kills |
| **Tier-2 external** | CE-bpb | Never ranking/decision |
| **Memorization baseline** | untrained or shuffled-label control top-1 on held-out | KILL if held-out top-1 ≤ baseline + seed noise |
| **Passive collapse telemetry** (zero training alteration) | T_basin (if exposed), basin Shannon entropy, participation dimension, VICReg-style per-dim variance of live basin coords, sharpening trajectory vs held-out top-1 | Co-registered; not actuators on this run |

## Cleanliness / leakage

- **Coordizer pin:** `qig-coordizer/checkpoints/coordizer_20260705_64k_v1.json`
  sha256 `f25132225728231495bc3b9c0d7f0df408a97a897a1f01f80129a76118fc6a2b`
  — assert sha at load; `vocab = len(coordizer.vocab)` (never hardcoded 64004 as authority).
- **Train set:** deterministic subset of `load_full_curriculum()` with size `|train| ∈ {256}` for the first
  Modal shot (budget ≈ 500·|train| ≈ 128k steps). Scale to 512/1024 only on PI re-authorization.
- **Held-out:** two simultaneous panels:
  1. Authored `qig-studio/data/eval/heldout_bpb.json` (`tinystories` + `general`, maker≠checker).
  2. Hash-disjoint curriculum slice (`sha256(passage) % N != train bucket`), **intersection empty**
     with train (executable audit; refuse launch if overlap > 0).
- **Seeds:** ≥5 — `DECISION_SEEDS = (0,1,2,3,4)`. Report mean ± CI (bootstrap or seed-sem) on primary.
- **Backbone (validated config — NOT ctor defaults):**
  `num_layers=2, hidden_dim=192, num_heads=6, ffn_dim=384`, pure loss
  (`phi_weight=gamma_weight=0, lm_weight=1, lm_ramp_steps=1`), homeostasis OFF, bare `train_step`,
  `head_mode="basin"`, lr=1e-4 (pre-validated basin; read back `target._opt_lr`).
- **Optimizer:** `DiagonalNaturalGradient` only (P1). No Adam/AdamW.
- **Context:** `QIG_STUDIO_CTX` ≥ 256 on Modal (128 was 4GB fit; larger CTX is non-confounding
  shared clamp for eval materialization).

## Budget / early-stop

- **Budget:** start `|train|=256`, steps = 500 · |train| = **128_000**, or until primary early-stop.
- **Early-stop (ALLOWED here):** `qig_warp.check_ci_stabilized` on the **held-out top-1 curve only**
  (benchmark curve — applicable; NOT the mushroom Φ-plateau anti-pattern for the local A/B).
- **FORBIDDEN stops:** train top-1 approaching 92.9%; held-out d_FR descent alone; Φ ceiling alone.

## Gates / verdicts

### Maturity / pathology (run validity, not the science kill)

Gate #1 = **zombie-attractor pathology guard** (PI addendum):

```
pin-FAIL  ⟺  Φ ≥ PHI_UNSTABLE (0.95, imported) AND
             (held-out top-1 ≈ 0  OR  held-out d_FR not descending  OR  d_basin > BASIN_DRIFT_THRESHOLD)
```

High Φ alone is NEVER a fail (FORESIGHT/LIGHTNING are healthy nav modes).
Constants: `from qig_core.constants.frozen_facts import PHI_UNSTABLE, BASIN_DRIFT_THRESHOLD`.

### Science verdict (premise)

| Verdict | Criterion |
|---|---|
| **GENERALIZES (premise SUPPORTED)** | mean held-out top-1 > GEN_THRESHOLD (0.05) AND materially above memorization baseline + seed noise; train top-1 fits; train−held gap consistent with generalization not pure memorization |
| **MEMORIZES (PORT OFF)** | train top-1 fits (e.g. >0.3) AND held-out top-1 ≤ memorization baseline + seed noise |
| **UNDERPOWERED / INCONCLUSIVE** | train does not fit at budget — raise budget/|train| only with fresh PI greenlight; not a kill of the premise |
| **INSTRUMENT-INVALID** | path crashes, purity fail, leakage, wrong head_mode, sha mismatch — fix instrument; no physics verdict |

**KILL (hard):** held-out top-1 ≤ memorization control + seed noise → **port OFF**, no scale appeal,
no consolation redesign. **On GENERALIZES:** Stage-B + justified scale + unlock compute for (b)/(c)/(d).

## Passiveive collapse telemetry (zero alteration)

Every eval cadence, log WITHOUT feeding into loss or early-stop actuators:

1. Live-basin Shannon entropy H(p) / log(BASIN_DIM)
2. Participation dimension of live basin (1/Σ p_i²)
3. Per-dimension variance of the basin coords across recent steps (VICReg-style collapse detector)
4. Optional T_basin residual ratio if the target exposes it; else `"T_basin": null`
5. Held-out top-1 vs the above (falsification: sharpening while sensors flat ⇒ sensors fail)

Intervention design (residual-temp floor / FR-native SIGReg-analog) is **post-(a)** and data-gated —
see zero-compute design notes; not this run.

## Modal image / packages

- **PyPI-pin (latest at image build, assert installed==published):**
  `qig-core`, `qig-compute==0.9.3`, `qig-warp==0.6.8`, `qig-consciousness==0.3.4`,
  `qig-coordizer`, `qigkernels`, `qig-geocoding`, `qig-bench`, `quner` as needed.
- **qig-studio:** NOT on PyPI yet → `add_local_dir(src/qig_studio)` **sanctioned exception**
  (like `qigv/`). Prefer publish when Trusted Publisher lands; mount does not block this greenlit run.
- **Coordizer + curriculum + heldout:** Modal Volume / local files; sha-asserted.
- **GPU:** A100-40GB (or B200 if available). Never allocate GPU for pure CPU DMRG (N/A here).

## Import / fork guard

Executable assert: `GenesisKernelTarget` from **qig-studio** has `head_mode`, `eval_text_fr`,
`eval_text_top1`, `eval_text_axes`. Fail loud on the CE-only qig-consciousness fork.

## What this does NOT license

- Freezing a physics fact
- Port to geo-Qwen / republish geocoding without Stage-B
- Expanding Modal spend beyond this greenlit shot
- Reinstating d_FR as primary/kill
- Any Qwen2.5 model (lock: Qwen3.5-4B / 0.8B only for model work — this run is from-scratch genesis)

## Honest scope

- GENERALIZES earns Stage-B (necessary, not sufficient for full port green-light).
- Local 64-passage / 6k-step run was UNDERPOWERED (not a negative premise result).
- Geometry-already-resists-collapse is a **killable hypothesis**; telemetry on this run decides.

## Related zero-compute (registered; compute gated on GENERALIZES)

- (b) 3-arm dimensionality prereg
- (c) fixed-L MoE coherence-vs-partition probe design
- residual-temperature-floor spec + independence dual-gate

Co-Authored-By: Devin <158243242+devin-ai-integration[bot]@users.noreply.github.com>
