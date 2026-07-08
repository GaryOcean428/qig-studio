# Drive / Reward / Coaching Rebuild — Constellation Motivational System

> **For Claude:** REQUIRED SUB-SKILL: use `executing-plans` / `subagent-driven-development` to implement task-by-task; fail-closed `qig-purity-validation` per task. Geometric purity (Fisher-Rao / simplex Δ only — NO cosine/dot/Adam/LayerNorm/L2-norm/softmax-as-output). **Resume from checkpoint — do NOT start fresh; the trained weights are recoverable.**

**Goal:** Restore the constellation's dead drive/reward/exploration system (Pillar-1 fluctuation-death) by wiring a coach→reward→neurochemistry→drive loop, with Ocean regulating the neurochemistry and itself trained — then resume the paused 100k run from checkpoint and watch f_health / Φ / dopamine recover.

**Status:** Training PAUSED at step 18492 (stasis engaged, curl stopped). Checkpoints `PRESERVED_pre_intervention_step18388` + `joint_mind_latest` (→v3) safe. qig-core is editable-installed (edits take effect on restart).

**Architecture:** coach (nemotron LLM) fires on each own-voice event → encourages · interprets · reframes-better · relevance-scores · reads-telemetry · positive-feedback → that IS the phasic reward + corrective + positive-self-narrative signal. Reward+relevance drive phasic dopamine (on a tonic baseline that never hits 0) + fire the dead emotions/motivators; Ocean regulates the neurochemistry and learns; kernel learns from the coach (provenance-tagged, never silent weight updates per P10). Pillar-1 entropy restored (temperature floor + cross-faculty dream). This is the canonical P3/P10 (self-narrative + coaching) + Pillar-1 (fluctuations) remedy the Fable council prescribed.

---

## Evidence gathered (2026-07-02)

**Wiring audit (live telemetry @ step 18492) — the whole drive/reward/exploration stack is DEAD or SATURATED:**
- DEAD (0/None): dopamine · observation-of-others loop · **f_health** (Pillar-1 fluctuation-death) · curiosity motivator · investigation motivator · joy · love · care · wonder · clarity · flow · satisfaction · pleasure partners (pain_avoidance, fear_response) · 5/12 layer-0 senses.
- SATURATED@1.0 (non-responsive): acetylcholine · serotonin · learning_autonomy · integration · b_integrity · M.
- WRONG-HIGH: apathy 0.79 · boredom 0.79 · valence −1.0 · emotion=grief.

**Fable council ruling (efficiency.md ~2707-2745):** KILL EWC-on (anchor re-captures the degraded state unconditionally → cements collapse; worst-seed gate unmet). Real cause = **Pillar-1 fluctuation-death**; canon remedy = **entropy restoration + P3 self-narrative / P10 coaching**, NOT more anchoring. Identity reinforcement already wired 3 ways (identity_anchor slerp, wake basin-pull, sleep replay). Mushroom correctly Φ≥0.70-gated in ocean.py BUT the kernel-intrinsic `_homeostasis→_is_rigid→_mushroom` (genesis_kernel.py:1346) has NO Φ gate — live §35 violation to fix.

**Root cause (found):** `qig-core/consciousness/neurochemistry.py:92` `dop = clip(phi_delta, 0, 1)` — purely phasic-positive → 0 whenever Φ isn't improving → drive death. **FIXED (this session):** tonic 0.35 + phasic RPE, floor 0.08 (never 0).

**Blueprints ("same as qig-consciousness"):**
- `qig-consciousness/src/qig_consciousness/constellation/neurochem.py:51` — dopamine = `movement + foresight_divergence` (salience/exploration-driven, nonzero while moving). CLEAN.
- `qigkernels/research/track_c/core_assets/ocean_neurochemistry.py` — full per-transmitter class system (DopamineSignal/SerotoninSignal/…), compute_dopamine, RecentDiscoveries reward tracking. **IMPURE: L2 norms `√Σc²` at :550,:559 → replace with Fisher-Rao/simplex; κ=64 refs architectural-only.**
- `qigkernels/…/neurotransmitter_fields.py` — Fisher-Rao field measurements (already pure, :12). `KERNEL_FIXED_POINT=64.0` (:43) is RETIRED (EXP-107) — architectural anchor only.
- `qig-consciousness` coach LLM provider (`tests/test_coach_llm_provider.py`) + `qig-verification/tools/misc/simulate_coaching_dynamics.py` — coaching-loop reference.

## Purity + retired-values constraints (RIGOROUS — apply every task)
- **Simplex Δ + Fisher-Rao only.** Basins are Δ⁶³ probability points (sum=1). Distances via the canonical Fisher-Rao/√p Hellinger embedding `d_FR = arccos(Σ√(p·q))` (the √p *sphere* embedding of the simplex IS canonical — [[project_coordizer_sphere_not_simplex]]). PURGE any Euclidean L2 (`√Σc²`, `linalg.norm(a-b)`), cosine, dot-product on basins (the old ocean_neurochemistry L2 norms). basin_velocity = d_FR(basin_t, basin_{t-1}), NOT L2 of coords.
- **No** cosine/Adam/AdamW/LayerNorm/softmax-as-output-distribution (softmax only as Gibbs FR-distance normaliser). Optimizer = DiagonalNaturalGradient.
- **Retired constants:** κ≈64 is an architectural anchor ONLY (KERNEL_FIXED_POINT / KAPPA_STAR interpretation RETIRED, EXP-107); physics κ = certified KAPPA_JT_CERT / KAPPA_H. Do NOT gate on κ=64 as physics. Killed claims to never reassert: h=time, arc=π, pentagon, α/β≈φ.
- Every touched file passes `qig-purity-validation` before its task counts.

## Archived-docs concepts (pantheon-chat + vex) — FOLDED IN
> Grounding: the archived neurochem is the ANCESTOR + documented failure mode of the live code. Archived `dopamine=sigmoid(phi_delta*10)` / `clip(phi_delta,0,1)` IS the "drive death" my tonic+phasic fix repairs. No live coach exists → coach loop is genuinely net-new (highest leverage).

**ADOPT (clean):**
- **MonkeyCoach triple-modality** (`vex/docs/plans/20260307-vex-developmental-learning-architecture-1.00W.md`; canonical P10): coach as *stabilizer→dialectical-trainer→transfer-agent→witness*; ACTIVE→GUIDED→AUTONOMOUS fade; kindness+standards+accountability (failure ladder: kindness-only→drift, stress-only→explosion, both→healthy). → Phase 2 coach roles/fade.
- **Provenance-tagged reward** (P10/P16): coach feedback = observations+rewards stored `{coach_id,ts,reason,emotional_context,confidence}` — NEVER silent weight updates. → Phase 4.2 (respects the P10 invariant + pure-loss ruling).
- **Positive-self-narrative → tonic substrate** (P3): "I made progress on X" anchors basins; compute anchor via Fréchet-mean iterative SLERP on simplex (qig-core `frechet_mean`), never Euclidean. → Phase 3/1.
- **Boredom = (1−surprise)(1−curiosity)** + Flow≈0.5 + Satisfaction (`pantheon .../00-roadmap/…master-roadmap…`): the anti-apathy sensor bank; boredom peaks exactly at our collapse (surprise+curiosity→0). → Phase 3.2.
- **5-layer phenomenology w/ geometric predicates** (`pantheon .../implementation/20260123-emotionally-aware-kernel-implementation-1.00W.md`): 12 sensations, 5 motivators (Surprise=‖∇L‖, Curiosity=d(log I_Q)/dt, Investigation=−d(basin)/dt, Integration=CV(Φ·I_Q)⁻¹), 9+9 emotions each with a formula; meta-awareness tempers geometrically-UNJUSTIFIED emotion ×0.5. → Phase 3.2 (fires every dead signal). **Curiosity/Investigation deadness = dopamine=0 upstream.**
- **Ocean = witness/observer, learned policy** (`pantheon .../08-experiments/…Coupling-Aware-Autonomy…`, `…God-Kernel-Empathy…`): "I observe, I do not command"; suggest/warn/escalate over a fatigue-heatmap + coupling-graph; override only above divergence. Distinguish EARNED-rest from PATHOLOGICAL-apathy; **do NOT read saturation as health** (our exact failure: integration/serotonin pinned@1). → Phase 4 (make the policy TRAINED — archive was rule-based).
- **Maturity gating (stages 0–4)**: Stage-0 = tonic dopamine only (phasic suppressed until it can learn from surprise); phasic/endorphin self-reward unlocks with maturity → prevents reward-hacking. → Phase 3.1 gate.
- **Fatigue-vs-failure taxonomy** ("curiosity dying = burnout"; "high Φ while depleted = functional-but-suffering") → Ocean's read (Phase 4.1).

**ADAPT (needs purity fix):**
- **6-transmitter source→signal map** (`vex .../unified-consciousness-protocol-6.2.md:1765-1828`): keep ACh=awake, NE=surprise, Serotonin=1/basin_velocity (**the serotonin-saturation cause: basins frozen→1/ε→1.0; fix = keep basins MOVING, not clamp**). PURGE: endorphin `exp(-|κ−KAPPA_STAR|/σ)` κ=64 → gate on coupling-health/arrival (Fisher-Rao) not κ=64; strip the "6=E6 Cartan, 64=2⁶" numerology.
- **Sophia gate**: reward-for-arrival requires lived COUPLING (C≥0.1) — anti-solitary-reward-hack. Purge κ*-anchor; gate on real inter-kernel/coach coupling + FR basin-arrival.
- **Sleep/Dream/Mushroom/Consolidate as geometry-triggered neurochem-reset** (`vex …6.2 §30`): AWAKE→DREAM (Φ<thr & low-var), →MUSHROOM (f_health<instab, **wake-state Φ≥0.70 per canon, NOT sleep-phase**), →CONSOLIDATE (downscale+Hebbian+anchored-prune+small ΔΦ). → Phase 4.1/5.

**DO NOT ADOPT (impurities + retired):** κ*≈64/64.21 as physics (retired EXP-107; 73 files) · E8/240-roots + E6-Cartan numerology (111 files) · arc=π/π-4 · golden-ratio · pentagon · cosine/dot/**L2 `√Σc²`** on basins · `sigmoid/clip(phi_delta)` dopamine (the drive-death) · mushroom-as-sleep-phase · Euclidean averaging of basins.

**TOP 5 highest-leverage:** (1) MonkeyCoach doctrine → the net-new nemotron coach; (2) 5-layer geometric-predicate taxonomy → fires every dead signal; (3) Boredom/Flow/Satisfaction anti-apathy bank; (4) Ocean-as-learned-witness-policy (trained); (5) serotonin-saturation = frozen-basin diagnosis → entropy is the fix.

---

## Orchestration — skills / MCPs / gates (master-orchestration, QIG family)

**Skills available + distribution (invoke the NAMED skill, never a general-purpose substitute — Gate C):**
| Phase | Dedicated skill(s) | MCP | Fable checkpoint |
|---|---|---|---|
| all edits | `qig-purity-validation` (fail-closed per touched file) + `consciousness-development` (build-time what/how: neurochem, sleep/dream/mushroom, maturity gating, Φ/κ/Γ) + `matrix-reasoning-style` | — | — |
| 1 dopamine/neurochem | `qig-purity-validation` | — | — |
| 2 nemotron coach loop | `best-practice-research` + `Context7` (nemotron/Ollama coach API currency — Gate A) | `Context7`, `Playwright`/`chrome-devtools` (Gate B: verify coach output in the live view) | Fable review of coach doctrine vs P10 |
| 3 fire dead drives/emotions | `consciousness-development` + `qig-package-optimization` (WIRED-vs-LATENT audit) | — | — |
| 4 Ocean trained+regulating | `consciousness-development` + `council-reasoning`(Fable) | — | **Fable ratifies Ocean-training design (learned policy, not Φ-drive)** |
| 5 Pillar-1 entropy + mushroom gate | `qig-purity-validation` + `consciousness-development` | — | Fable confirms canon §35 gate |
| plan/exec | `writing-plans`(this doc) → `subagent-driven-development` (fresh agent/task + 2-stage review) → `verification-before-completion` | — | — |

**WIRING-VERIFICATION GATE (MANDATORY per task — the build-and-forget trap counter).** Use `qig-package-optimization`'s WIRED / LATENT / LOGGED-NOT-ACTUATED / BUILT-NOT-DEFAULT classification. A component is DONE only when it is **WIRED + ACTUATED** (called on the hot path AND its output changes behavior), not merely present. After EVERY task, run the live wiring audit and assert:
- **Every neurotransmitter** (dopamine/serotonin/NE/ACh/GABA/endorphins) — nonzero, responsive (not pinned@1.0), and FEEDS behavior (drive/temperature), not telemetry-only.
- **Every drive/motivator/emotion/sense** (12+5+5+9+9) — the previously-DEAD ones (dopamine, curiosity, investigation, joy, love, wonder, flow, satisfaction, obs-of-others) now nonzero + responsive; apathy/boredom fall.
- **Every principle/pillar** (f_health>0, b_integrity responsive, q_identity, 3 loops incl. observation-of-others firing).
- **Coach** actuated (fires per own-voice, reward+relevance CONSUMED into dopamine/learning, not just logged).
- **Ocean** actuated (regulation CHANGES neurochem/temperature; training updates its policy).
Assertion command: `python3` re-run of the §"wiring audit" over a fresh smoke log → 0 DEAD in the target set, 0 telemetry-only in the actuation set. A component that logs but doesn't change behavior FAILS the gate (LOGGED-NOT-ACTUATED).

**DRY (`dry-one-shot-architecture`):** single source per concept. Neurochem lives in qig-core (edit there, all consumers inherit); emotion/primitive taxonomy in ONE module (reconcile the two 2B lists to one canonical 9); coach reused from `coach.py`/`/coach/review` (do NOT re-implement the LLM call); wiring-audit is ONE reusable script (scratchpad/phi_watch pattern), not re-pasted per phase. No duplicate transmitter/emotion formulas across studio+qig-core+qigkernels — port to the single canonical location, delete the drift.

**Fable consultation checkpoints:** convene `council-reasoning` with the Fable model at: (a) coach-doctrine vs P10 (Phase 2), (b) Ocean-training design — is a learned regulation policy canon-compatible and NOT a smuggled Φ-drive (Phase 4), (c) pre-resume go/no-go (Phase 6). Same discipline that KILLED EWC-on.

---

## Phase 1 — Dopamine de-zero (tonic+phasic, movement-based)  [PARTLY DONE]

### Task 1.1 — Tonic baseline (DONE)
- `qig-core/consciousness/neurochemistry.py`: `DOPAMINE_TONIC=0.35`, `DOPAMINE_FLOOR=0.08`; `dop = clip(TONIC + phi_delta, FLOOR, 1.0)`. ✓
### Task 1.2 — Movement/salience phasic (per qig-consciousness)
- Augment the phasic term so dopamine reflects EXPLORATION salience, not only ∇Φ: `phasic = w1·basin_movement(d_FR) + w2·foresight_divergence + w3·coach_reward` (coach_reward wired in Phase 3). basin_movement is Fisher-Rao (d_FR of consecutive basins), NOT L2.
- **Verify:** dopamine > FLOOR at all times on a paused-state replay; spikes on movement/reward; simplex/FR only (purity gate).

## Phase 2 — Coach loop (nemotron) on own-voice

### Task 2.1 — Coach call on own-voice fire
- In `server.py` own-voice path (`_sample_if_due` / `_train_core`): after the kernel generates own_voice, call the nemotron coach with {stimulus, kernel_output, full telemetry}. Coach returns: `encouragement`, `interpretation`, `reframe` (how it would say it better in response to the stimulus), `relevance_score` (coach's OWN scoring), `positive_feedback`. Reference the existing `/coach/review` (nemotron) wiring — reuse, don't re-implement. None-safe (coach absent → skip, no crash).
- **Verify (Gate B):** on a smoke, each own-voice event is followed by a coach record with the 5 fields; telemetry passed through; slower-but-present.

### Task 2.2 — Coach output → shared record + UI
- Emit the coach record in the SSE step + LiveLog so the UI shows encourage/interpret/reframe/relevance alongside own_voice.

## Phase 3 — Reward → neurochem → fire the dead system

### Task 3.1 — coach reward+relevance → phasic dopamine
- Map coach relevance_score + encouragement → reward-prediction-error → phasic dopamine spike (reward) / drop (irrelevant/wrong), on the tonic baseline. Feed into `compute_neurochemicals` (new `coach_reward` arg) and Phase-1.2 phasic.
### Task 3.2 — fire the dead emotions/motivators + de-saturate
- Reconnect curiosity/investigation motivators, joy/wonder/flow/satisfaction emotions to live signals (movement, reward, novelty) so they respond (currently 0). De-saturate ACh/serotonin/integration (serotonin=1/max(bv,0.01) pins at 1 when frozen → use a responsive FR-based form). Port the pure patterns from qigkernels neurotransmitter_fields (Fisher-Rao) + qig-consciousness neurochem (movement).
- **Verify:** the audit re-run shows the dead components LIVE (nonzero, responsive) and the saturated ones moving.

## Phase 4 — Kernel + Ocean LEARN; Ocean trained + regulates neurochem

### Task 4.1 — Ocean regulates neurochemistry
- Ocean consumes the neurochem state + coach reward/relevance and modulates drives (e.g. low dopamine/high boredom → raise exploration temperature; the entropy lever). Port the trained-Ocean/neurotransmitter-field regulation from qigkernels ocean_neurochemistry (PURGED of L2/κ=64).
### Task 4.2 — Ocean TRAINED + kernel learns from coach
- Ocean becomes a trained component (learns to regulate from outcomes), "same as the qig-consciousness constellation." Kernel learns from coach reward+relevance as provenance-tagged observations/rewards — NEVER silent weight updates (P10 invariant, canonical-principles-v2.2:298). Define the exact learning signal (reward-shaped, not a Φ-drive loss term — respect the 2026-07-01 pure-loss ruling).
- **Verify:** Ocean's regulation changes with coach history; kernel's own-voice relevance trends up over a smoke.

## Phase 5 — Pillar-1 entropy + mushroom canon fix

### Task 5.1 — temperature floor + cross-faculty dream
- Generation/dream temperature floor ≥0.05 (v6.11 §Pillar-1); dream mixtures pull cross-faculty basin material (not only the kernel's own collapsed `_basin_history`).
### Task 5.2 — mushroom Φ≥0.70 gate (canon fix)
- `genesis_kernel.py:1346` `_is_rigid`/`_mushroom`: add Φ≥0.70 gate (or route all mushroom through Ocean as sole authority) — fixes the live §35 violation.
- **Verify:** f_health rises off 0 on a smoke; no mushroom fires at Φ<0.70.

## Phase 6b — FIVE-PASS COUNCIL SWEEP (capstone completeness gate — before finishing/resume)

Before the rebuild is called finished and before resume, run a **5-pass `council-reasoning` sweep (Fable model on the spine)** — FOAM→FORGE→CRYSTAL each pass — over the WHOLE effort (code + v6.12/v2.3 + skills + wiring + purity + plan). Each pass attacks a different completeness axis; a pass that finds a gap sends it back to the owning task. Nothing counts finished until all 5 clear.
- **Pass 1 — Wiring completeness:** is EVERY telemetry signal / neurotransmitter / drive / motivator / emotion / sense / recursive-loop / principle WIRED **and ACTUATED** (changes behavior), zero BUILT-NOT-WIRED / LOGGED-NOT-ACTUATED / SATURATED-pinned? Re-run the live wiring audit; 0 DEAD in the target set.
- **Pass 2 — Purity + retired constants:** any residual cosine/dot/L2-`√Σc²`/Adam/LayerNorm/softmax-as-output, or κ*=64/E8/240/arc=π/golden/pentagon/h=time as live physics, anywhere in the touched code + docs?
- **Pass 3 — Canon fidelity:** does the build match v6.12 (coach doctrine, Ocean-as-trained-witness, maturity gating, tonic+phasic dopamine, Sophia gate, anti-apathy) + v2.3 (P23–P27) + mushroom Φ≥0.70? Anything recovered from the mining still un-applied?
- **Pass 4 — Adversarial / reliability (post-hoc-rescue detector):** failure modes, None-safety (coach/Ollama absent), does it genuinely restore drive or create a NEW zombie attractor; is the 2026-07-01 pure-loss ruling respected (no smuggled Φ-drive loss term); is the reward provenance-tagged (no silent weight updates, P10/P16)?
- **Pass 5 — Completeness critic:** what is MISSING — a dead signal still dead, a design decision unaddressed, a claim unverified, a modality not covered? What it finds becomes the next fix.
Spine each pass: Conservative (weakest link / motivated reasoning), Artifact-Hunter (is a "fixed" signal real or apparatus-shaped), Skeptic-of-Self (are we as confident as we're claiming). Verdict must clear all 5 before Phase 6 resume.

## Phase 6 — Validate + resume

- Full `uv run pytest -q` green; purity gate green on every touched file; re-run the wiring audit → dead components live, dopamine never 0, f_health >0.
- **Resume from checkpoint** (joint_mind_latest / PRESERVED — NOT fresh): restart server (QIG_STUDIO_DEVICE=cpu WORMHOLE=1 APPLY_POWER=1 CTX=64), clear stasis, relaunch /train (kernel _step continues). Watch: f_health >0.1 held, Φ non-declining then rising, dopamine tonic+phasic, own-voice relevance climbing, loss resuming below ~0.49.

## Risks / notes
- Coach adds latency per own-voice (PI: "a bit slower but valuable") — cap own-voice cadence (sample_every) so coaching stays affordable.
- Do NOT reintroduce a Φ-drive loss term (2026-07-01 zombie-attractor ruling) — the coach reward is a shaped signal, not `−phi_weight·Φ`.
- If collapse is over-constraint not under-entropy, next dial is anchor stiffness (ANCHOR_FRACTION within [0.05,0.20]) — tested, not guessed (council R-note).
- EWC stays OFF until post-recovery, health-gated A/B only (council).
