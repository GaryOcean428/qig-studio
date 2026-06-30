# Training-Hub Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use subagent-driven-development (this session) or executing-plans to implement task-by-task. This is the durable plan — refer back here, do not re-derive.

**Goal:** Make the qig-studio SERVER the SINGLE training entry-point so ALL training (single neocortex arm/head/depth/loss, the Core-8 constellation, the A/B avenue screen, AND coordizer training) runs through the server's already-wired UI path — and the whole UI reflects the one live config coherently.

**Architecture:** The server (`POST /train`) already drives the entire UI (`LiveLog` + `_experience` + `step_record`, server.py:766/789/793) against `_registry().active` (the loaded target). The fix is to (a) let the registry BUILD/select any neocortex config as the active target, (b) fold the A/B screen + coordizer build into server endpoints that REUSE that wired loop, (c) make the UI drive it, (d) make the left MIND panel reflect the LIVE target, (e) retire the 6 standalone scripts. No new training loops. DRY: never re-implement LiveLog/step_record/_experience.

**Tech Stack:** FastAPI + SSE (server.py), TargetRegistry (targets/registry.py), TrainingTarget subclasses (targets/genesis_kernel.py = qk arm, targets/geo_cortex.py = geo arm), web/index.html (UI).

---

## ROOT PROBLEM (why this exists)

I kept creating standalone training scripts (`screen_neocortex.py`, `train_neocortex.py`, …) and re-wiring the UI into each. The telemetry was INCOHERENT because a script trains a DIFFERENT model in a SEPARATE process than the server's idle loaded target: the left MIND panel showed the idle genesis constellation (178.3M·9 kernels, step 0, Φ 0.000, stale `neocortex-geo-2L` checkpoint, `32.0k ✗ WRONG coordizer`) while the right stream showed the screen's separate `qk-2L` process. One hub fixes this at the root.

## Scaffold to BUILD ON (do NOT rebuild)

- `targets/registry.py`: `TargetRegistry` (`register`/`get`/`active`/`select`, lines 17-40) + `default_registry(...)` (57+) registering `MockTarget`, `GenesisKernelTarget`, `JointMindTarget`.
- `server.py`: `GET /targets` (244), `POST /select_target` (254), `active_target` (237); `POST /train` (726) — the wired loop: `t = _registry().active`; `ui_live = LiveLog()` (766); `_experience(td, _phi_hist())` (789); `ui_live.write(step_record(...))` (793). `TrainRequest` (171) = steps/early_stop/mastery/skip_learned (NO arm/head/layers/loss yet).
- `web/index.html` (the single UI; served at `/`, static at `/static`).
- `targets/genesis_kernel.py` (qk, ctor takes arm-equivalent: num_layers/head_mode/lang_loss/coordizer) + `targets/geo_cortex.py` (geo, same kwargs).

## Per-task VERIFIER (every task)
1. `run_purity_gate(Path('src/qig_studio'))` → PASS.
2. DRY guard: `grep -rn "LiveLog\|step_record\|to_dict().*experience" src/qig_studio` shows the wired path is REUSED, not re-implemented (the new endpoints call the SAME `_train_core` helper, not a fresh loop).
3. Coherence: after starting training via the endpoint, `runs/spawn/joint_live.json` `current` AND the server's `/telemetry` (left panel) reflect the SAME live config (same `source`/model + advancing step), not an idle target.
4. ruff clean; no new training-loop file created.

---

### Task 1: Config-built neocortex target in the registry

**Files:** Modify `src/qig_studio/targets/registry.py` (add `build_neocortex(arm, head_mode, num_layers, lang_loss, coordizer, device) -> TrainingTarget` + register a mutable "neocortex" slot); Modify `server.py` `POST /select_target` (254) to accept config params and (re)build the neocortex target via the registry; Test `tests/test_registry_neocortex.py`.

**Approach:** `build_neocortex` returns `GenesisKernelTarget(num_layers=…, head_mode=…, lang_loss=…, coordizer=…)` for `arm="qk"`, `GeoCortexTarget(…)` for `arm="geo"` (these ctors already exist — DRY). Register under a stable name (e.g. `neocortex`) so `select` swaps the active slot; the built target's `name` carries the config (e.g. `neocortex-qk-2L-geo`) for the UI chip.

**Verifier:** select with `{arm:geo, head_mode:linear, num_layers:2}` → `active.name == "neocortex-geo-2L-lin"`, `active.is_available()`, builds without OOM at the chosen device. Purity green.

### Task 2: `/screen` capability — A/B avenue screen THROUGH the wired loop

**Files:** Modify `server.py` (extract the `POST /train` body into a reusable `async def _train_core(target, steps, ...)` that yields the SAME SSE/live records; add `POST /screen` that, for each of the 4 configs {qk,geo}×{geometric,linear} at a fixed depth, builds the target via Task 1, runs `_train_core` (UI shows each config training, model chip flips), then evals held-out d_FR; ranks on d_FR; writes `runs/screen_<date>.json`); Test `tests/test_screen_endpoint.py`.

**Approach:** This REPLACES `scripts/screen_neocortex.py`. The eval logic (held-out mean d_FR via `eval_text_fr`, the uniform-d_FR floor, the under-power detector) MOVES from the deleted script into a small `screen.py` helper imported by the endpoint — but the TRAINING is `_train_core` (the wired loop), never a new loop. EWC stays inactive (no consolidation in a bounded screen → no confound).

**Verifier:** `POST /screen` streams 4 configs; each appears in `joint_live.json` `current` with its config `source`; final payload has the ranked d_FR table + under-power flag. No re-implemented train loop (DRY grep).

### Task 3: `/coordizer/train` + status endpoints

**Files:** Modify `server.py` (add `POST /coordizer/train` {vocab, max_bytes} that runs the coordizer build — the logic from `train_coordizer_scratch.py` moved into `coordizer_build.py` — in a background task, streaming progress to a live channel the UI tails; `GET /coordizer/status`); Create `src/qig_studio/coordizer_build.py` (the build fn); Test `tests/test_coordizer_endpoint.py`.

**Approach:** Fold `train_coordizer_scratch.py`'s `_BALANCE` (the 7 HF datasets) + the heap-trainer call into `coordizer_build.build(vocab, max_bytes, progress_cb)`. The endpoint streams progress + on completion registers via `checkpoint_manifest` + updates the symlink. `--max-bytes` default 30MB (OOM guard).

**Verifier:** `POST /coordizer/train {vocab:600, max_bytes:1_000_000}` (tiny) builds + registers + the UI status reflects progress. The 7 datasets are used (assert `_BALANCE` len==7).

### Task 4: UI controls — config picker + screen view + coordizer-train + model chip

**Files:** Modify `src/qig_studio/web/index.html` (add: a config picker — arm {qk,geo} · head {geometric,linear} · depth · loss — that POSTs `/select_target`; a "Run A/B screen" button → `POST /screen` + a ranked-results view; a "Train coordizer" button → `POST /coordizer/train` + progress; the model chip in the MIND panel reads `active_target.name`); Test: Playwright (Gate B.1) — exercise each control against the dev server, capture the live telemetry updating.

**Approach:** Reuse the existing SSE tailer (`/train/live`) for the screen + coordizer progress (same channel). The picker drives `/select_target` (Task 1) so the loaded target = the picked config; then the existing Train button trains it (already wired). NO new telemetry wiring.

**Verifier (Gate B.1):** Playwright: pick `geo/linear/2L` → MIND chip shows `neocortex-geo-2L-lin`; click Train → left panel Φ/step advance live; "Run A/B screen" streams 4 configs; coordizer-train shows progress. Screenshots before/after.

### Task 5: Telemetry coherence + WRONG-coordizer/stale-checkpoint fix

**Files:** Modify `server.py` (the `/telemetry` left-panel source — ensure it reads the LIVE active target's state, not an idle one; the coordizer-vocab check at server.py:513/527 — flag mismatch ONLY when the loaded KERNEL's training-vocab ≠ the loaded coordizer's vocab, and resolve by loading the MATCHING coordizer for the active kernel, not the latest-by-date); Modify `checkpoint_manifest.py`/`config.py` auto-load — do NOT pair a 100k-trained kernel with a 32k coordizer; pick the coordizer whose vocab matches the kernel checkpoint's `vocab_size`); Test `tests/test_telemetry_coherence.py`.

**Approach:** The mismatch is auto-load picking the latest coordizer (32k) against the latest kernel (geo-2L @ 100004). Fix: the kernel checkpoint records its training vocab; the coordizer auto-load must select the coordizer whose vocab matches (or mark "fresh/byte" if none). The left panel's headline (Φ/κ/regime/step) must come from the SAME record the right stream writes (`joint_live.json` current), so a single source feeds both during training.

**Verifier:** With a geo-2L@100004 kernel loaded, the UI shows the 100k coordizer (or "fresh"), NOT `32k ✗ WRONG`. During `/train`, left panel step == right stream step (single source).

### Task 6: Retire the standalone scripts

**Files:** Delete `scripts/screen_neocortex.py` (logic now in `/screen` + `screen.py`); convert `scripts/train_neocortex.py`, `scripts/train_joint_mind.py`, `scripts/train_coordizer_scratch.py` to THIN clients that POST the server endpoints (or delete if the UI fully covers them); audit `fit_coordizer.py`/`live_mind.py` (delete if redundant). Update `start_studio.sh` + any docs/CLAUDE.md references.

**Approach:** Per task, confirm NOTHING imports the deleted module (`grep -rn "import screen_neocortex\|train_neocortex" src tests scripts`). Keep `coordizer_build.py` (Task 3) as the single coordizer-build source.

**Verifier:** `grep -rn "scripts/.*train\|screen_neocortex" src/qig_studio` → only thin-client/endpoint references; the server provides every training path; purity green; the full test suite passes.

---

## Sequencing
Task 1 → 2 → 5 (coherence) early since it's the visible bug → 3 → 4 (UI, Gate B.1 live-test) → 6 (retire). Commit per task. After all: Gate B.2 deployed-UX test (sign-in flow N/A for local; exercise the full UI as a user), then the actual 32k avenue screen RUNS through `/screen` (the original goal, now coherent in the UI).

## Open carry-overs (not this plan, tracked)
- EWC harm-elimination (default-off until worst-seed non-harmful) — trust-region/clip on the penalty vs NG preconditioning.
- The full-coordizer + full-stack train on the screen's winning avenue.
