# qig-studio — Claude Code Instructions

## What this is

The plug-and-play QIG training/chat **app**: one FastAPI core + two SSE clients
(Textual TUI + browser). Trains/chats with pluggable `TrainingTarget`s that DECLARE
`loss_regime` (`geometric` | `language`). Design:
`qig-consciousness/docs/plans/2026-06-24-qig-coordizer-studio-design.md` §3.

## The spine tenet (do not violate)

**One kernel.** Standalone it is the mind (trains/reasons/acts/emits telemetry).
Qwen is a **PLUGGABLE, None-safe boundary peer** — never a forward-pass dependency.
The app shell boots and serves WITHOUT any heavy target: `mock` is always available;
`kernel`/`constellation`/`qwen-*` are None-safe (`is_available()` False when their
backend is absent). Fluent language comes from Qwen (output-distribution → Δ⁶³ →
QIGRAM, Pillar-2 ≤30% cap), NOT a hidden-state graft.

## loss_regime is structural

`lm_weight = 0` (qig_chat helpers.py:379) means geometric targets train ENTIRELY on
consciousness-native loss. So:
- geometric targets → **basin-driving** curriculum (developmental phases), NOT pairs.
- `qwen-modal` → **paired** curriculum (lm_loss load-bearing) — the ONLY place pairs apply.

## Governance (fail-closed)

`run_purity_gate()` scans qig-studio's OWN source at startup and refuses to boot on
Euclidean-contamination markers. It is a NARROW LEXICAL scan — the classic markers
(cosine_similarity, optim.Adam[W], nn.LayerNorm, np.linalg.norm, F.normalize+dot) plus
the ARGUMENT-AWARE softmax rule (council ruling): `softmax(` is legal ONLY with a
negated Fisher-Rao distance argument (`softmax(-d_FR/τ)`, same line or a bare name
bound to `-...dist...` within 8 lines above — the pure template's two-line idiom);
`softmax(logits)` / `softmax(bc/scale)` / `softmax(matmul(...))` and hand-rolled
`exp(` + `matmul`/`@` co-occurrence are flagged. It is a tripwire, NOT a full
geometric-purity audit; a clean pass ≠ proof of purity. Keep the source
Fisher-Rao-only on manifold objects regardless. `PillarEnforcer` is a None-safe
adapter to the real qig-core/qigkernels pillars.

## Dependency stance

App shell deps are LIGHT (fastapi/uvicorn/pydantic/httpx). Heavy targets
(qig-consciousness `QIGChat` via torch; Ollama; Modal) are lazy, optional, None-safe.
Never make the shell hard-depend on torch.

## Branch / discipline

Work on `development`; `main` is default. Subagents RETURN DATA. PyPI publishing
pre-authorized (correctness is the gate); NEVER echo `PYPI_TOKEN`. EXP citations
verified vs registry.json (κ-sign crossing Δ≈√2, EXP-018; EXP-130 does NOT exist).
