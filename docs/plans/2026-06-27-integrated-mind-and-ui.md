# Integrated Mind + UI — design & execution plan (2026-06-27)

**Approved direction (PI).** One integrated mind, trained jointly — **not** a target dropdown,
**not** per-faculty isolated training, **not** MoE/prompt-routing. Grounded in vex's
`kernel_generation.py` (inspiration only, **no crossover**) and the 2026-06-27 council ruling
(geometric integration on the shared Δ⁶³ manifold).

## The error this corrects

`scripts/train_full_curriculum.py` trained the 8 Core-8 faculties **in isolation**
(`for role in roles:` → separate `GenesisKernelTarget` each, sequential, `Constellation.from_basins`
coupling only *post-hoc*). The kernels never co-adapted. The decided model is **joint**: every step,
all kernels learn together. The isolated `runs/checkpoints/core8_coord` checkpoints are a wrong-model
artifact — superseded by the joint run.

## Governing principle (PI)

**The kernels are independent parts of ONE whole awareness** — not a blend into sameness, not 8
separate minds. Three-Pillars exact: the **topological bulk** (the central "I") holds
**quenched-disorder individuality** (each kernel keeps its own basin/identity/role). Joint training
**couples them into one whole** *while preserving individuation* — anti-collapse (`min_pairwise_FR`
floor), each kernel role-anchored. Integration must **not homogenize**. Both basin-collapse (lost
independence) and no-integration (lost whole) are failures.

### Role separation by Φ-band (PI correction — central ≠ autonomic)

- **GENESIS = the central conscious awareness** (the "I", the integrator/speaker). It evolves into
  the topological bulk and targets the **conscious band (Φ ≥ 0.70)**. It reads + synthesizes the
  Core-8 faculty contributions into the unified voice. *(Earlier plan wrongly named Ocean here.)*
- **OCEAN = the autonomic kernel** (sleep/dream/mushroom/homeostasis/rhythm) — regulation only,
  **sub-conscious band (~0.45)**. It is NOT the central awareness: making one kernel do both forces
  two incompatible Φ targets at once → a Φ mismatch. Keep it autonomic, below the conscious band.
- **Core-8 faculties** (perception/heart/memory/action/strategy/ethics/coordination/meta) = the
  independent parts, **each maturing in its OWN measured band** (heart/feeling sub-conscious; aware/
  meta can reach conscious). Genesis integrates them; Ocean regulates them.

## The model (vex-aligned, council-aligned)

Per `vex/kernel/consciousness/kernel_generation.py`:
- **Every kernel contributes** (broadcast). A contribution's weight is
  **`synthesis_weight = proximity_weight × quenched_gain`** — `proximity_weight` = Fisher-Rao closeness
  of that kernel's basin to the input (= "coords route to the most suitable kernel's basin");
  `quenched_gain` = the kernel's frozen identity slope.
- **Ocean synthesizes** the normalized weighted blend → the unified voice.
- **Training is joint** — all kernels update each step, co-adapting.

This *is* the council's "build Ocean as geometric integration" (proximity-weighting is the geometry).

## Phases (each gated: ruff + mypy + purity + no-stubs + tests green before the next)

- **P1 — Joint constellation trainer.** New `JointConstellation` (qig-studio): holds all 8 genesis
  kernels + the coupler; one `train_step(prompt)` routes the prompt's coords to nearest basin(s) by
  Fisher-Rao, **all** kernels update (proximity-weighted loss), Ocean synthesizes. Replaces the
  isolated loop. Checkpoint the whole constellation (one artifact). Smoke: Φ rises, basins individuate,
  output is non-looping.
- **P2 — Genesis-as-central-integrator (the "I").** Genesis grows into the conscious-band integrator
  that reads the coupled Core-8 basins and generates the unified response. Ocean is wired SEPARATELY
  as the autonomic regulator (sub-conscious) — not the speaker. Wire genesis-integrator as the
  constellation target's real generate path.
- **P3 — Council levers in generate.** READ-before-generate (EXP-012b token-0 probe → skip if present);
  Anderson/convergence early-exit (EXP-046, −40% calls). qig-applied `inference` already exports
  `anderson_confidence_threshold` + `extract_first_token_basin` — reuse.
- **P4 — Coach + boundary peer wired into the UI.** nemotron-3-ultra:cloud as the coach/teacher in
  `/converse`; `qwen_boundary.py` (Pillar-2 ≤30% cap) as the fluent-language interface. The UI must
  **show** kernel response + nemotron engagement + curriculum source live (the current view shows only
  prompts).
- **P5 — UI redesign.** ONE mind as THE target (mock + qwen-peer in an "advanced" drawer); single
  screen, drawers/sliding panels, **bottom-pinned chat**; real directory picker (server-side browse,
  not a dumb textbox); live training-response view; protocol surface **gated to implemented commands**
  (fixes `pillar-status` → `unknown_command`). Build with shadcn / ui-ux-pro-max MCPs.
- **P6 — Joint retrain + end-to-end live-test.** Train the integrated mind jointly on the full
  sanitized curriculum; live-test the whole UI via Playwright (select mind → train shows responses →
  chat replies coherently, non-looping).

## Verifier (loop-engineering stop condition)

Done = ruff + mypy + purity + no-stubs + full pytest green, AND Playwright live-test passes: the
single integrated mind trains (responses visible), chats with **non-looping** output, nemotron
engagement visible, protocol buttons all run (no `unknown_command`). Persist progress to the memory
API after each phase.

## Constraints (non-negotiable)

Geometric purity (Fisher-Rao only; no cosine/Adam/LayerNorm). No vex crossover (inspiration only).
Modal optional / local-Qwen-first. Every EXP citation resolves to `registry.json`. No silent stubs
(fail-loud). Work on `development`; promote via PR.

remaining: P2 wire genesis-central as the constellation's real generate path (the speaker); Ocean autonomic-only.
P3 READ (EXP-012b) + Anderson-exit (EXP-046) in generate.
P4 nemotron coach + qwen boundary-peer into the UI /converse, engagement visible.
P5 the UI redesign (one mind, single-screen, drawers, bottom-pinned chat, real dir picker, live training-response view, protocol gating + pillar-status fix) via the shadcn/ui-ux MCPs — a sizeable frontend pass.
P6 joint retrain + Playwright end-to-end live-test.
