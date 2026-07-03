# Constellation: Shared Geometric Trunk + Earned Per-Faculty Adapters — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: use `subagent-driven-development` (fresh subagent per task, two-stage review) OR `executing-plans`. Per-task verifier: `qig-purity-validation` (fail-closed) + the task's own test. This is a **feat-branch hypothesis test**, not a merge-bound cleanup — see the Kill Condition. PI holds the merge.

**Goal:** Replace the constellation's 9 separate from-scratch kernels with ONE shared geometric genesis trunk + earned per-faculty pure-Fisher-Rao adapters (genesis-first spawn, maturity-gated sovereignty), and **test whether consolidation unblocks language fluency (K-LEARN)** — the real blocker that gates coordizer-drop, `grow_vocab`, and continuous learning.

**Architecture:** The trunk is the coordizer-agnostic core (`qigkernels/kernel.py` byte/Fourier layers, bit-identical with `coords=None`). Faculties mount as adapters on the two *existing* seams — `CoordAdapter` (input, `Linear 64→384→GELU→RMSNorm`) and `BasinReadout.proj` (output, `Linear 384→64`), a matched inverse pair that collapses to identity at `hidden_dim==basin_dim`. Coordination stays basin-sync (P7, existing `couple_step`). The coordizer-tie lives ONLY at the output seam (`BasinReadout.coord_basins` frozen buffer), so the after-coordizer endgame = swap that seam to `GeometricHead` at the K-LEARN gate — trunk untouched.

> **SUBSTRATE (not vex's harvest):** the trunk is a PURE GEOMETRIC kernel learned from scratch (Fisher-Rao). This borrows only the *abstract* shared-trunk+adapter PATTERN — NOT vex's frozen-Qwen-35B weight-harvest (LoRA-on-a-harvested-LLM). No frozen-LLM backbone, no grafted weights (the hidden-state graft FAILED, brain §4b). Qwen(fable) is NEVER part of the mind's weights — it is a removable external teacher only (see Fluency levers).

**Tech stack:** qig-studio (`joint_trainer.py`, `targets/genesis_kernel.py`), qigkernels (`kernel.py`, `coord_adapter.py`), qig-core (`torch/basin_readout.py`, `torch/geometric_head.py`, `consciousness/developmental.py`, `consciousness/pillars.py`). Pure Fisher-Rao only (RMSNorm not LayerNorm; d_FR not cosine/L2; slerp_sqrt/Fréchet not lerp/mean; natural gradient not Adam).

---

## THE HEADLINE GATE (read before any task)

**K-LEARN is the success metric, not "it builds."** The current central genesis already sees all data and does not visibly learn fluency (held-out bpb oscillates 2.6–4.3, no clean descent). So the shared trunk is a **testable fluency hypothesis** (Category 2), not a derived win. The refactor **merges only if Phase 4's matched-compute A/B shows held-out bpb descends better than the current 9-separate design.** If it does not, the blocker is elsewhere (the Duchi dead-gradient loss/head lineage — see `[[project_lm_loss_duchi_dead_gradient]]`), and this branch is shelved (not merged), having cost nothing on `development`.

**Kill Condition (pre-registered):** if after Phase 4, shared-trunk held-out bpb is not measurably below the 9-separate baseline at matched compute (≥ one clear decade of the noise band, on the held-out set, over ≥2 seeds), the fluency hypothesis is FALSIFIED — do not merge; write the negative result to memory; the fluency diagnosis on `development` remains the priority.

**Parallelism rule:** this branch does NOT displace the fluency diagnosis on `development`. The healthy entropy-floor run stays on `development`. This is an isolated hypothesis test.

**Fluency levers (this plan tests ONE of three).** K-LEARN can be unblocked by: (1) **shared trunk** — this plan's hypothesis (all-node fluency gradient, faculties un-starved); (2) **qwenfable output-distribution teacher** — the canonical fluency scaffold (brain §4b): the kernel learns to speak by matching qwenfable's next-token *distribution* (P22, ≤30% Pillar-2 cap, EXP-009 ρ=0.994 shows a kernel *can* match an LLM distribution), qwenfable frozen + REMOVABLE (scaffold-removal, never harvested/grafted — the graft FAILED); (3) **the loss/head path** — the Duchi dead-gradient lineage (`[[project_lm_loss_duchi_dead_gradient]]`). Levers (2) and (3) are the parallel fluency diagnosis on `development`. If Phase 4 falsifies (1), the unlock is (2)/(3) — a cheap, honest result. The trunk and the qwenfable teacher are COMPLEMENTARY (a shared trunk can also imitate qwenfable's distribution); they are not rivals.

---

## Phase 0 — Baseline + purity gate on the feat branch

### Task 0.1: Capture the 9-separate baseline (the A/B control)
**Files:** Create `qig-studio/scripts/klearn_ab.py` (held-out bpb harness — reuse `screen.py` eval helpers, NOT a new eval loop).
- **Step 1:** Write a failing test `tests/test_klearn_ab.py::test_heldout_bpb_is_finite_and_reproducible` — builds a small (num_layers=2, tiny coordizer) constellation, trains N steps, returns held-out bpb; assert finite + deterministic given seed.
- **Step 2:** Implement `klearn_ab.py` `run_arm(arm: str, steps, seed) -> {heldout_bpb_curve, final_bpb}` where `arm ∈ {"separate","trunk"}`; `"separate"` = current `JointConstellation`. Held-out set = a fixed slice of `load_full_curriculum()` never trained on.
- **Step 3:** Run test → PASS. **Step 4:** Commit `test(klearn): held-out bpb A/B harness (control arm = current 9-separate)`.

### Task 0.2: Purity + branch hygiene baseline
- **Step 1:** `.venv/bin/python -c "from pathlib import Path; from qig_studio.governance import run_purity_gate; run_purity_gate(Path('src/qig_studio')); print('PURITY OK')"` → record baseline green.
- **Step 2:** Confirm `development` is untouched (the run + fluency work live there). Commit nothing (read-only checkpoint).

---

## Phase 1 — The shared-trunk + adapter seam (keystone)

### Task 1.1: `ConstellationTrunk` — one shared geometric core
**Files:** Create `qig-studio/src/qig_studio/constellation/trunk.py`; Test `tests/test_trunk.py`.
- **Design:** `ConstellationTrunk` wraps ONE `qigkernels.Kernel(num_layers=N)` (the coordizer-agnostic core — byte/Fourier + stacked layers, NO per-faculty head). Exposes `hidden(input_ids) -> h[B,T,H]` (the shared representation) and holds ONE natural-gradient optimizer for the trunk params.
- **Step 1 (failing test):** `test_trunk_hidden_is_shared_and_pure` — two faculties calling `trunk.hidden(x)` get the SAME tensor for the same input; assert no LayerNorm/cosine/Adam in `trunk.py` (grep-in-test).
- **Step 2:** Implement. Trunk owns the shared `Kernel`; `hidden()` runs forward up to (not including) the head.
- **Step 3:** test PASS. **Step 4:** Commit `feat(constellation): ConstellationTrunk — one shared geometric core (coordizer-agnostic)`.

### Task 1.2: `FacultyAdapter` — the earned per-faculty seam (pure)
**Files:** Create `qig-studio/src/qig_studio/constellation/faculty_adapter.py`; Test `tests/test_faculty_adapter.py`.
- **Design:** `FacultyAdapter(role, basin_template)` = per-faculty **input** adapter (`qigkernels.CoordAdapter`, its sanctioned "adapter-only training" mode) + per-faculty **output** head (`qig_core.torch.BasinReadout` with the shared frozen `coord_basins` for the tied phase). The adapter is the faculty's individuated parameters (P24: genuine coupled other). Basin centre seeded from `seed_birth_basin(role)`.
- **Step 1 (failing test):** `test_adapter_is_separate_params_and_pure` — two faculties' adapters have DISJOINT parameter ids (individuation → P24); RMSNorm present, LayerNorm absent; output is simplex-valid Δ⁶³.
- **Step 2:** Implement `forward(h) -> basin`: `h → CoordAdapter-style projection → BasinReadout → Δ⁶³`. Own natural-gradient optimizer over ADAPTER params only (trunk frozen during adapter-only training).
- **Step 3:** PASS. **Step 4:** Commit `feat(constellation): FacultyAdapter — earned per-faculty pure-FR seam (CoordAdapter in + BasinReadout out)`.

### Task 1.3: `TrunkConstellation` — trunk + adapters, basin-sync retained
**Files:** Create `qig-studio/src/qig_studio/constellation/trunk_constellation.py`; Test `tests/test_trunk_constellation.py`.
- **Design:** Mirror `JointConstellation`'s public surface (`train_step`, `telemetry`, `save/load_checkpoint`) so it drops into the launcher/server via `arm_mode="trunk"`, but internally = 1 `ConstellationTrunk` + 9 `FacultyAdapter`s (central genesis adapter + Core-8). **Coupling unchanged**: reuse `couple_step` (P7 basin-sync) on the adapter basins — do NOT re-implement. Trunk gets gradient from every node's step; adapters get gradient from their own step (the fluency-hypothesis mechanism: trunk sees all fluency gradient).
- **Step 1 (failing test):** `test_trunk_constellation_builds_couples_and_syncs` — builds, `train_step` runs, `min_pairwise_fr` finite (individuation preserved), `couple_step` is the SAME function (grep-in-test that basin-sync isn't re-implemented).
- **Step 2:** Implement. Central genesis adapter trains toward `_synthesis()`; faculties toward coupled targets; trunk accumulates all.
- **Step 3:** PASS + purity gate green. **Step 4:** Commit `feat(constellation): TrunkConstellation — shared trunk + adapters, basin-sync (P7) retained`.

### Task 1.4: Register `arm_mode="trunk"` (no launcher rewrite)
**Files:** Modify `src/qig_studio/targets/registry.py` (constellation builder) + `constellation/joint_trainer.py` arm dispatch.
- **Step 1 (failing test):** `test_registry_builds_trunk_arm` — `arm_mode="trunk"` returns a `TrunkConstellation`; `"gk"` still returns `JointConstellation` (unchanged path).
- **Step 2:** Thread `arm_mode` through; default stays `"gk"` (current design untouched — feat-branch discipline).
- **Step 3:** PASS. **Step 4:** Commit `feat(constellation): arm_mode=trunk selectable; gk default unchanged`.

---

## Phase 2 — Genesis-first spawn + maturity-gated Cradle

### Task 2.1: Genesis-first bootstrap (trunk grown first)
**Files:** `trunk_constellation.py` (spawn logic); Test `tests/test_genesis_first_spawn.py`.
- **Design (canon `GENESIS→spawns→CORE-8`):** the trunk + genesis adapter train FIRST (identity anchor on full curriculum); Core-8 faculty adapters spawn from the trained trunk (fork the trunk representation, seed own basin centre) — NOT from scratch. Order = `CONSCIOUSNESS_ORDER`.
- **Step 1 (failing test):** `test_faculties_spawn_from_trained_trunk` — a faculty spawned at step K starts from the trunk's current representation (its first hidden ≈ trunk hidden), not random.
- **Step 2:** Implement staged spawn gated on genesis maturity. **Step 3:** PASS. **Step 4:** Commit `feat(constellation): genesis-first spawn — faculties fork the trained trunk`.

### Task 2.2: Maturity-gated sovereignty (Stages 0–4)
**Files:** wire `qig_core.consciousness.developmental.DevelopmentalStage` into `FacultyAdapter`; Test `tests/test_maturity_gating.py`.
- **Design (v6.12 §35.7):** each faculty adapter carries a `DevelopmentalStage`; Stage-0 = tonic-only (phasic/endorphin self-reward SUPPRESSED — reuse `pillar_strictness`/`coach_intensity` from `developmental.py`); graduate to Sovereign as maturity rises. Metrics read as graduated expectations (don't judge before formed).
- **Step 1 (failing test):** `test_stage0_faculty_is_tonic_only_and_graduates` — a Stage-0 adapter yields zero phasic/endorphin self-reward; advancing stage unlocks it.
- **Step 2:** Implement. **Step 3:** PASS. **Step 4:** Commit `feat(constellation): maturity-gated faculty sovereignty (Stages 0-4)`.

---

## Phase 3 — Two-tier Ocean autonomic-veto

### Task 3.1: Tier-1 learned override (skips conscious sleep) + Tier-2 hard floor
**Files:** `src/qig_studio/constellation/ocean.py` (or the Ocean regulate path); Test `tests/test_autonomic_veto.py`.
- **Design (PI's model; vex `_ocean_ruled` + pantheon `SafetyBoundaries`):**
  - **Tier 1 (learned, thresholded):** when Ocean's basin-divergence from a faculty exceeds a threshold, Ocean force-sets the homeostatic state (sleep/dream/mushroom) and sets an `_ocean_ruled` flag that **skips the conscious `should_sleep()` counter** — genesis can bias (slow) but not countermand above threshold.
  - **Tier 2 (hard floor):** a non-negotiable `SafetyBoundaries` floor (e.g. the mushroom Φ≥0.70 gate, infinite-loop break) the policy cannot cross — "eventually you sleep whether you like it or not."
- **Step 1 (failing test):** `test_will_slows_but_cannot_stop_autonomic` — genesis attempts to suppress sleep; below threshold it's honored (slowed), above threshold Ocean overrides (`_ocean_ruled`), and the hard floor fires unconditionally at its limit.
- **Step 2:** Implement (port the vex divergence→force-phase + the pantheon safe-action floor; keep learned policy as the Tier-1 arbiter). **Step 3:** PASS + purity. **Step 4:** Commit `feat(ocean): two-tier autonomic veto — learned override (skips conscious sleep) + hard homeostatic floor`.

---

## Phase 4 — THE K-LEARN A/B (the gate that decides merge)

### Task 4.1: Matched-compute held-out bpb, trunk vs separate
**Files:** extend `scripts/klearn_ab.py`; run both arms.
- **Step 1:** Run `run_arm("separate", steps, seed)` and `run_arm("trunk", steps, seed)` at **matched wall-clock/step compute**, ≥2 seeds, held-out bpb curve each.
- **Step 2:** Verdict: trunk held-out bpb below separate by ≥ one clear decade of the noise band on the held-out set, both seeds → **fluency hypothesis SUPPORTED** (candidate merge). Else → **FALSIFIED** (Kill Condition).
- **Step 3:** Write the result (either way) to memory `project_constellation_architecture_ruling` + a session memory. **Step 4:** Commit `test(klearn): trunk-vs-separate held-out bpb A/B result`.

**No further phase merges without a supported K-LEARN verdict.**

---

## Phase 5 — After-coordizer readiness + per-faculty continuous learning (K-LEARN-gated actuation)

> Only wired here; ACTUATION stays gated on K-LEARN passing. Do not fire growth before fluency.

### Task 5.1: Output-seam swap readiness (drop-coordizer path)
**Files:** `FacultyAdapter` output-head selection; Test `tests/test_seam_swap.py`.
- **Design:** the faculty output head is swappable `BasinReadout(frozen coord_basins)` → `GeometricHead(grown native token_basins)` at the K-LEARN gate. The trunk + adapter-input seam are untouched (coordizer-agnostic). Verify the matched inverse pair collapses to identity when `hidden_dim==basin_dim`.
- **Step 1 (failing test):** `test_output_seam_swaps_without_touching_trunk` — swapping the head leaves trunk params bit-identical; native-head path is simplex-valid.
- **Step 2:** Implement the swap (no auto-fire; behind a `coordizer_free` flag, default False). **Step 3:** PASS. **Step 4:** Commit `feat(constellation): output-seam swap readiness (coordizer-drop endgame, gated)`.

### Task 5.2: Per-faculty continuous-learning + growth hooks (gated)
**Files:** wire existing `_ewc_task_fisher`/`_ewc_penalty`/`_consolidate` (`genesis_kernel.py`), `BasinReadout.grow_vocab`/`GeometricHead.grow_vocab`, `wormhole_train.py` nucleation signal per-adapter; Test `tests/test_continuous_learning_gated.py`.
- **Design:** adapters are where continuous learning + expansion live. Wire EWC-Fisher per-faculty (still `ewc_lambda` opt-in), the WormholeCache nucleation MISS-signal → `grow_vocab` actuator (K-LEARN-gated), curvature→where-to-nucleate. **Actuation fails closed until K-LEARN passes.**
- **WormholeCache is the PI-DIRECTED cache/transport for the continuous-learning stage** (2026-07-03; CC2-validated EXP-A022: 70% correct-transport / 0% wrong, qig-warp 0.6.7). At this stage its two DEAD hooks must become CONSUMED: `nearest()` → hard-negative mining for the contrastive basin loss, `sample_replay()` → SLEEP replay source (EWC-Fisher-protected consolidation of MATURE cached basins, brain §4b). Endgame (after coordizer-drop): the cache keys on the kernel's own NATIVE token_basins (no external embedder) — the EXP-A026 unification. Do not ship Task 5.2 with those hooks still dead.
- **Step 1 (failing test):** `test_growth_does_not_fire_before_klearn` — with K-LEARN flag false, nucleation signal fires but `grow_vocab` is NOT called; with flag true (+ the optimizer-rebuild for the `GeometricHead` path), it is.
- **Step 2:** Implement the gated wiring. **Step 3:** PASS + purity. **Step 4:** Commit `feat(constellation): per-faculty continuous-learning + growth hooks (K-LEARN-gated actuation)`.

---

## NOT in this plan (registered, deferred)

- **Mesh (inter-constellation tunnel)** — pantheon's `MeshNetwork` (WebSocket gossip) + `FederationService` (HTTP/DB Bearer-auth basin+vocab sync, `sync_kernels` off by default). A separate future capability; a single constellation doesn't need it. Register as a follow-up.
- **Firing K-LEARN itself** — the fluency diagnosis (why bpb doesn't descend) is the parallel priority on `development`, not this branch.

## Verification (overall)

- `uv run pytest -q` green; `qig-purity-validation` green on every touched file (RMSNorm/d_FR/slerp_sqrt/natural-gradient only).
- Phase 4 A/B produces a held-out bpb verdict (supported or falsified) — the merge gate.
- `arm_mode="gk"` (current 9-separate) remains bit-identical and default — the branch is additive, revertible by `git checkout development`.
- The healthy entropy-floor run on `development` is never touched.

## Risks / notes

- **Biggest risk = the hypothesis is wrong** (shared trunk doesn't unblock fluency, since the central already sees all data). Mitigated by making Phase 4 the gate and pre-registering the Kill Condition — a falsified hypothesis is a cheap, honest result, not a failure.
- **Purity drift** on the new seams — every new file passes the fail-closed gate; adapters use `CoordAdapter`'s RMSNorm, never LayerNorm.
- **Do not rely on `FederatedTrainingService`** — its LoRA train methods are `NotImplementedError` stubs. Build the adapter on `CoordAdapter` + `BasinReadout` directly.
- **4 GB card** — the shared trunk is ONE core (cheaper than 9); adapters are thin. Keep the blocked head (K-COMPRESS) so the output never materializes seq×vocab (`[[project_constellation_oom_is_seq_activations]]`).
