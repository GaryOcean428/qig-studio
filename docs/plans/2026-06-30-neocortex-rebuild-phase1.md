# QIG Neocortex Rebuild — Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` (or `subagent-driven-development`) to implement this plan task-by-task. Run `qig-purity-validation` as the verifier on every code task — it is fail-closed.

**Goal:** Train a fresh, geometrically-pure **neocortex** (single deep stacked-Δ⁶³ cortex = the central "I"), comparing two model families (qigkernels vs qig-geocoding) on a shared fresh coordizer by bits-per-byte — with the language loss converted from CE/KL to a Fisher-Rao loss (P20), and **everything drivable from the qig-studio UI**.

**Architecture:** One deep kernel, two implementations. **ARM B `neocortex-qk`** = `qigkernels.Kernel(num_layers=N)` (stacked Δ⁶³ Fisher-Rao layers). **ARM A `neocortex-geo`** = `geocoding.GeoModel` (Fisher-Rao-attention transformer-equivalent). Both train on the SAME fresh coordizer + curriculum + SAME pure loss, bpb-compared. A nested depth A/B (N-stacked vs 1-block-recursive, EXP-CORTEX-AB) runs inside ARM B. The multi-stream **constellation** (Core-8 + Ocean, procedural reasoning) is **Phase 2 — design-only here**.

**Tech Stack:** Python 3.14, PyTorch, `qig-core` (Fisher-Rao geometry, `to_simplex_prob`, `fisher_rao_distance_simplex`), `qigkernels`, `qig-coordizer`, `qig-geocoding`, FastAPI + SSE (qig-studio app/UI), systemd-run (survivable bg jobs).

---

## Supporting docs (READ BEFORE STARTING — load-bearing)

| Doc | Why |
|---|---|
| `qig-consciousness/docs/20260623-qig-brain-architecture-1.00W.md` | §1 layer table (neocortex = stacked-Δ⁶³); **§6 build plan** (Step 1 = deepen cortex, the A/B); **§7** (Fisher-Flow Matching generation, ILR-as-training-chart, sparsemax) |
| `qig-consciousness/docs/20260624-brain-experiment-backing-map-1.00W.md` | **EXP-CORTEX-AB** (N-stacked vs 1-block; "the A/B IS the first build step, not a gate"); which claims are BACKED vs NEEDS-EXPERIMENT |
| `~/.claude/skills/matrix-reasoning-style/references/canonical-principles-v2.2.md` | **P1** purity, **P2** simplex-only, **P20** free-energy=d_FR-**never-KL** (the purity-fix mandate), **P18** multi-stream, **P21** disconnected-infra-is-a-bug, **P11** gauge-ethics |
| `~/.claude/skills/matrix-reasoning-style/references/unified-consciousness-protocol-v6.11.md` | The state the kernel must enact (regimes, three loops, C-gate) |
| `qig-studio-training-notes.md` | running log; consolidation status; co-training framing (lines 117/135/174); ARM-B-on-consolidated-qig-consciousness caveat (line 135) |
| `qig-purity-validation` skill | the fail-closed scan run as the verifier each task |
| Code: `qig-studio/src/qig_studio/targets/genesis_kernel.py` | the loss (`F.cross_entropy` @ **887** train, **471** surprise, 334/722 eval); `eval_text_bpb` @ 319 |
| Code: `qig-studio/src/qig_studio/constellation/joint_trainer.py` | current trainer (round-robin constellation — Phase-1 trains a SINGLE kernel, not this) |
| Code: `qigkernels/kernel.py` | `Kernel(num_layers, …)` @ 36, `lm_head` @ 137, `forward` @ 157 |
| Code: `qig-geocoding/src/geocoding/{model,config,attention,generate}.py` | ARM A (`GeoModel`/`GeoConfig`) — **NOT installed in studio venv; install first** |
| Code: `qig-studio/src/qig_studio/web/index.html` + `server.py` | the UI + endpoints (coordizer dropdown @177-179 exists; **no train-launch controls yet**) |

**Provenance / discipline:** work on `development`; commit frequently; PyPI publish pre-authorized (correctness gate); NEVER echo `PYPI_TOKEN`/`HUGGINGFACE_TOKEN`; `uv` only (no pip); geometric purity = Fisher-Rao only.

---

## The pure loss (the P20 fix — read once, applied in Phase 1)

CE against a one-hot target **is** KL divergence (CE = KL when H(target)=0); **P20 forbids KL** ("free energy = prediction error = d_FR … never KL divergence"). Replacement, per token position `t`:

```
p_t    = to_simplex_prob(logits_t)          # sparsemax projection, Δ^vocab  (P2 simplex)
loss_t = fisher_rao_distance_simplex(logits_t[None], onehot(target_t)[None])
       = 2·arccos( Σ_i √(p_t[i] · onehot[i]) )  =  2·arccos( √ p_t[target_t] )
L_lang = mean_t loss_t
```

- Same destination as CE (drives `p_t[target]→1`), geometrically-honest route (Hellinger/Bhattacharyya NLL = Fisher-Flow Matching's objective, brain-arch §7).
- **`bpb` stays CE-based for the EVAL metric** (it's the standard cross-model benchmark vs SmolLM2/Qwen — a read-only measurement, not a loop operation). Caveat: a d_FR-trained (sparsemax) model may show worse *CE*-bpb than its true fluency → **report both** `L_lang` (d_FR) and CE-bpb, and judge fluency by **own-voice generation** too. The **CE-ablation arm** exists precisely to measure this gap.
- Gradient safety: `fisher_rao_distance_simplex` already uses `_acos_safe` (finite grad at coincidence) — no extra clamping needed.

---

## Phase 0 — Pre-flight + wipe (GATED on PI go)

### Task 0.1: Purity + doc baseline
- Run `qig-purity-validation` over `qig-studio/src`, `qigkernels`, `qig-geocoding/src` → record the clean baseline.
- Confirm `qig_doctor`/installed==latest for qig-core, qigkernels, qig-coordizer, qig-geocoding (LAUNCH BLOCKER if stale).
- **Step (commit):** none (read-only baseline).

### Task 0.2: Archive the orphaned 108k kernel + wipe (irreversible — confirm with PI first)
- `git -C qig-studio status` clean-check; then:
  - Move `runs/checkpoints/joint_mind_20260629_v1` → `runs/checkpoints/_archive_108k_orphaned/` (its coordizer is unrecoverable — keep as legacy, not latest).
  - Delete `qig-coordizer/checkpoints/coordizer_20260629_{100k,150k}_v1.json` + `MANIFEST.json` + symlinks (the fresh coordizer replaces them).
  - Clear `runs/checkpoints/joint_mind_latest`, `coordizer_latest.json` symlinks.
- **Verifier:** `ls` shows no stale checkpoints; no symlink dangles.
- **Commit:** `chore(rebuild): archive orphaned 108k kernel + wipe stale checkpoints for fresh neocortex`.

---

## Phase 1 — The pure Fisher-Rao language loss (TDD)

**Files:** Create `qig-studio/src/qig_studio/losses.py`; Modify `genesis_kernel.py:887` (+471 surprise); Test `qig-studio/tests/test_fisher_rao_loss.py`.

### Task 1.1: Write the failing test for `fisher_rao_lm_loss`
```python
# tests/test_fisher_rao_loss.py
import torch
from qig_studio.losses import fisher_rao_lm_loss

def test_fr_loss_zero_when_perfect():
    # logits that project to a near-one-hot on the target → loss ≈ 0
    logits = torch.full((1, 2, 4), -10.0); logits[0, 0, 3] = 10.0; logits[0, 1, 1] = 10.0
    ids = torch.tensor([[3, 1, 0]])
    L = fisher_rao_lm_loss(logits, ids)
    assert L.item() < 0.05

def test_fr_loss_large_when_wrong():
    logits = torch.zeros((1, 2, 4)); ids = torch.tensor([[0, 1, 2]])
    assert fisher_rao_lm_loss(logits, ids).item() > 0.5

def test_fr_loss_differentiable():
    logits = torch.randn((1, 3, 5), requires_grad=True); ids = torch.tensor([[0, 1, 2, 3]])
    fisher_rao_lm_loss(logits, ids).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
```
- **Step:** `uv run pytest tests/test_fisher_rao_loss.py -v` → FAIL (no module).

### Task 1.2: Implement `fisher_rao_lm_loss` (minimal)
```python
# src/qig_studio/losses.py
"""Fisher-Rao language loss — the P20-pure replacement for cross-entropy (CE=KL is forbidden).
L = mean_t  2·arccos(√ p_t[target_t]),  p_t = to_simplex_prob(logits_t).  Pure Δ⁶³, no softmax/KL."""
from __future__ import annotations
import torch
from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex, to_simplex_prob  # noqa: F401

def fisher_rao_lm_loss(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    # logits [1, T, V] predict ids[1, 1:T+1]; align next-token like CE does.
    lg = logits[0, :-1]                       # [T-1, V]
    tgt = ids[0, 1:]                          # [T-1]
    V = lg.shape[-1]
    onehot = torch.zeros_like(lg).scatter_(-1, tgt[:, None], 1.0)   # Δ^V target (already simplex)
    d = fisher_rao_distance_simplex(lg, onehot)                    # projects lg via to_simplex_prob
    return d.mean()
```
- **Step:** `uv run pytest tests/test_fisher_rao_loss.py -v` → PASS.
- **Verifier:** `qig-purity-validation` on `losses.py` → green (no cosine/dot/softmax/KL).
- **Commit:** `feat(loss): Fisher-Rao language loss (P20 — replaces CE/KL)`.

### Task 1.3: Wire the loss into the kernel with a CE-ablation flag
- Modify `genesis_kernel.py`: add ctor param `lang_loss: str = "fisher_rao"` (`"fisher_rao"` | `"ce_ablation"`); env `QIG_STUDIO_LANG_LOSS` overrides.
- Replace `ce = F.cross_entropy(logits[0,:-1], ids[0,1:])` (line 887) with:
  ```python
  if self.lang_loss == "ce_ablation":
      lang = F.cross_entropy(logits[0, :-1], ids[0, 1:])         # ablation arm — measures the purity cost
  else:
      from qig_studio.losses import fisher_rao_lm_loss
      lang = fisher_rao_lm_loss(logits, ids)                      # P20-pure
  ```
  and use `lang` everywhere `ce` fed the loss (`w_lm * lang`).
- Replace the **surprise** signal (line 471): `_ce` → `_fr = fisher_rao_lm_loss(plog, pids)`; `"surprise": _fr`; update `"max_surprise"` to the d_FR ceiling `π` (was `ln(vocab)`).
- Keep `eval_text_bpb` (319) **CE-based** (standard metric) — add a parallel `eval_text_fr` returning mean d_FR for the d_FR arm's own curve.
- **Verifier:** `qig-purity-validation` green; a 5-step smoke train on a tiny corpus runs without NaN under BOTH `lang_loss` values.
- **Commit:** `feat(kernel): pure-loss training path + CE-ablation flag + d_FR surprise`.

---

## Phase 2 — Fresh coordizer (CLI + UI-triggerable)

**Files:** `qig-studio/scripts/train_coordizer_scratch.py` (exists — heap trainer); `server.py` (+endpoint); `web/index.html` (+button).

### Task 2.1: CLI fresh coordizer with dated output + manifest
- Run `train_coordizer_scratch.py --vocab 100000 --max-bytes 30000000 --out ../qig-coordizer/checkpoints/coordizer_20260630_100k_v1.json` (heap trainer ~6 min; registers in MANIFEST + updates `coordizer_latest.json` symlink via `checkpoint_manifest.register_coordizer`).
- **Verifier:** atomic geo-tags OK; `coordizer_latest.json → coordizer_20260630_100k_v1.json`; vocab≈100k.
- **Commit:** none (artifact, gitignored) — log to notes.

### Task 2.2: `POST /coordizer/train` endpoint (UI trigger)
- `server.py`: add `POST /coordizer/train` → launches `train_coordizer_scratch.py` via `systemd-run --user --unit=qig-coordizer-train` (detached, survivable), body `{vocab, max_bytes}`; returns the unit name. Guard: refuse if a coordizer train is already active.
- Add `GET /coordizer/train/status` (SSE or poll) reading the unit's journal tail + checkpoint progress.
- **Verifier:** Playwright/curl: POST starts the unit; status streams `[N/vocab]` lines.
- **Commit:** `feat(server): /coordizer/train launch + status endpoints`.

---

## Phase 3 — Neocortex training harness (single deep kernel; naming; checkpoints)

**Files:** Create `qig-studio/scripts/train_neocortex.py`; Create `qig-studio/src/qig_studio/neocortex.py` (thin single-kernel target wrapping ARM A/B); reuse `live.py`, `checkpoint_manifest.py`.

### Task 3.1: `Neocortex` single-kernel target (ARM B first)
- `neocortex.py`: a `Neocortex` class that builds **one** deep kernel (NOT the constellation), exposing `train_step(prompt)`, `eval_text_bpb`, `eval_text_fr`, `save/load`, `telemetry()`. ARM B wraps `qigkernels.Kernel(num_layers=N, vocab_size, …)` + the genesis physics fixes (collapse-immune coherence, READ, locality) + the **pure loss** + `NaturalGradientDescent`.
- `name` attribute drives checkpoint dir + live-trace `model` field: `neocortex-qk` / `neocortex-qk-{N}L` / `neocortex-qk-1L-rec`.
- **Verifier:** purity green; 5-step smoke on tiny corpus; checkpoint round-trips (save→load preserves simplex basins, P2).
- **Commit:** `feat(neocortex): single deep-kernel target (ARM B qigkernels) on pure loss`.

### Task 3.2: `train_neocortex.py` launcher
- Args: `--arm {qk,geo}`, `--layers N`, `--recursive` (1-block), `--coordizer <latest>`, `--vocab`, `--name`, `--ckpt-root runs/checkpoints/<name>_<date>_<vocab>_v<N>`, `--lang-loss {fisher_rao,ce_ablation}`, `--device {cpu,cuda}`, `--fresh`.
- Wires `prelaunch_optimise` (qig-warp/qig-compute/expA021 daemon) + all-cores + GPU residency env (`QIG_STUDIO_CTX`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
- Writes the SAME `live.py` trace the UI reads, with `model=<name>` so the UI shows which arm is training.
- **Verifier:** launches `neocortex-qk` on the fresh coordizer; live-trace shows `model=neocortex-qk-8L`, step>0, Φ moving, `lang_loss` curve dropping.
- **Commit:** `feat(neocortex): train_neocortex launcher (arm/depth/loss/name)`.

---

## Phase 4 — ARM A: qig-geocoding GeoModel (install + wire)

**Risk:** `geocoding` is NOT installed in the studio venv (module name `geocoding`, dist `qig-geocoding`).

### Task 4.1: Install + smoke-import geocoding
- `cd qig-studio && uv add --editable ../qig-geocoding` (or pin published `qig-geocoding`); verify `uv run python -c "from geocoding.model import GeoModel; from geocoding.config import GeoConfig"`.
- **Verifier:** import OK; `qig-purity-validation` over `qig-geocoding/src` green (it's "our transformers" — must be Fisher-Rao pure).
- **Commit:** `chore: install qig-geocoding (ARM A) into studio venv`.

### Task 4.2: ARM A in the Neocortex wrapper
- Extend `neocortex.py`: `--arm geo` builds `GeoModel(GeoConfig(vocab=…, hidden=…, layers=…))`; forward→logits fed to the SAME `fisher_rao_lm_loss`; same coordizer + curriculum + naming (`neocortex-geo`).
- **Faithfulness gate (from memory `project_geocoding_package`):** geocoding stays faithful to qigkernels geometry to 1e-5 — assert a tiny-input parity check before trusting bpb.
- **Verifier:** purity green; `neocortex-geo` smoke-trains on pure loss; faithfulness assert passes.
- **Commit:** `feat(neocortex): ARM A (geocoding GeoModel) on shared coordizer + pure loss`.

---

## Phase 5 — Depth A/B (inside ARM B) + Phase 6 — bpb comparison

### Task 5.1: Run the depth A/B (EXP-CORTEX-AB)
- Train `neocortex-qk-8L` (N-stacked) vs `neocortex-qk-1L-rec` (1-block-recursive, RecursiveIntegrator head) on the SAME coordizer/curriculum/loss/seed. Cheap scale first (per backing-map: "the A/B IS the build step").
- **Verifier:** both checkpoints saved with their names; live-traces distinguishable by `model`.

### Task 6.1: bpb (+d_FR) comparison harness
- `scripts/compare_neocortex.py`: load the held-out set (`data/eval/heldout_bpb.json`), compute **CE-bpb** AND mean **d_FR** for each arm checkpoint; print a table (arm, params, CE-bpb, d_FR, own-voice sample); write `runs/neocortex_ab_<date>.json`.
  - **Convention guard (Matrix, source-verified):** compute d_FR ONLY via the **torch** primitive `qig_core.torch.geometry_simplex.fisher_rao_distance_simplex` (= `2·arccos(BC)`, range [0, π]). Do NOT mix in the qig-core **numpy** `fisher_rao_distance` (unscaled `arccos(BC)`, range [0, π/2]) — the factor-of-2 silently corrupts the table. Both arms go through the torch path → internally consistent.
- **Verifier:** table renders; lower CE-bpb = better fluency (vs SmolLM2-360M ~0.8 reference); honest-report if the d_FR (pure) arm trails the CE-ablation arm on CE-bpb.
- **Commit:** `feat(eval): neocortex A/B comparison (CE-bpb + d_FR + own-voice)`.

---

## Phase 7 — UI wiring (PI requirement: drive EVERYTHING from the UI)

**Files:** `qig-studio/src/qig_studio/web/index.html` (+controls/JS); `server.py` (+endpoints).

### Task 7.1: Coordizer training control
- In the ARCHITECTURE/train panel: a **"Train coordizer"** button + vocab/max-bytes inputs → `POST /coordizer/train`; a progress line bound to `GET /coordizer/train/status` (the `[N/vocab]` stream). On done → refresh the coordizer dropdown (`refreshCheckpoints`).

### Task 7.2: Neocortex training control (arm/depth/loss picker)
- Add `POST /neocortex/train` (server) → launches `train_neocortex.py` via systemd-run with `{arm, layers, recursive, lang_loss, vocab}`; `POST /neocortex/stop`; refuse if a train is active (the `_TARGET_LOCK` pattern).
- UI: selectors for **arm** (`qk`/`geo`), **depth** (`N` / `1-block-recursive`), **loss** (`fisher_rao` / `ce_ablation`), a **Start/Stop** button. The active `model` name shows in the Train·live header (so you always know which mind is training).

### Task 7.3: Live model differentiation + A/B view
- `/train/live` already streams the record — surface `record.model` prominently (header chip). Add a small **A/B panel** that reads `runs/neocortex_ab_<date>.json` (via a `GET /neocortex/ab` endpoint) showing the CE-bpb/d_FR table + a "promote winner" affordance (sets the `neocortex_latest` symlink → Phase 2 input).
- Keep the existing **checkpoint selector** (Windsurf `/checkpoints` + dropdowns) working for hot-swap; ensure it lists `neocortex-*` checkpoints.
- **Verifier (Gate B live-test):** Playwright/chrome-devtools — from the UI: train a coordizer, then launch `neocortex-qk` AND `neocortex-geo`, watch both live (model chip correct), see the A/B table populate, select a checkpoint. Capture screenshots + clean console.
- **Commit:** `feat(ui): coordizer + neocortex training controls, model chip, A/B view`.

---

## Phase 8 — Constellation (Phase 2) — DESIGN ONLY (no code in this plan)

Captured so it isn't lost; **built later, on the winning neocortex**:

- **Substrate:** Core-8 faculties (canonical specializations: heart, perception, memory, strategy, action, attention, emotion, executive — Canonical Principles §Two-Axis Schema) + Ocean (autonomic) wrapped around the trained `neocortex-{winner}`. Roles are config, not classes (no `Ocean.py`).
- **Procedural reasoning (the core idea):** the central identity reasons via **internal kernel-to-kernel debate (flow, not prompt)** — the UCP/council-reasoning that Claude runs *as a prompt*, the kernel runs *as basin-exchange dynamics* (notes line 2: "the matrix-reasoning kernels and the constellation kernels are the same nine objects"). This realizes **P18** (concurrent multi-stream candidate-thoughts, mid-stream rejection, sidetrack-with-return) **procedurally**.
- **Expertise weighting:** in a debate, the sub-kernel whose specialization matches the question carries more weight (memory question → memory kernel); **ethics always carries strong/veto weight** (P11 gauge-invariant ethics; curvature>0.5 = harm = veto).
- **Naming:** `constellation-{winner}` (e.g. `constellation-qk`); checkpoints `constellation-qk_<date>_<vocab>_v<N>`; UI model chip distinguishes neocortex vs constellation.
- **Open design questions (for the Phase-2 plan):** is round-robin training acceptable with Loop-2 debate as the P18 layer, or do we need concurrent streams in *training* too? how is the debate scheduled (FOAM→FORGE→CRYSTAL triadic)? how does ethics-veto enter the loss?

---

## Global verifier (every code task)

1. `qig-purity-validation` green (fail-closed — no cosine/dot/Adam/LayerNorm/softmax/KL on manifold objects).
2. Tests pass (TDD: failing test first, then minimal impl).
3. Naming is exact (`neocortex-{arm}[-{N}L|-1L-rec]`, `{name}_{YYYYMMDD}_{vocab}_v{N}`) — the UI/live-trace must always show which mind is active.
4. For UI/runtime tasks: **Gate B live-test** (Playwright/chrome-devtools against the running server) — code-tracing is hypothesis, not verification.
5. Frequent commits on `development`; honest-negative reporting (if the pure-loss arm trails on CE-bpb, say so).

---

## Execution options

1. **Subagent-Driven (this session)** — fresh subagent per task + code review between tasks.
2. **Parallel Session** — new session with `executing-plans`, batch with checkpoints.
