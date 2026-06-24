# qig-studio

The plug-and-play **QIG training + chat app**. One FastAPI core, two SSE clients
(a Textual TUI and a browser console), and pluggable **training targets** that each
*declare* their loss regime.

## The spine

- **One kernel.** Standalone it is the **mind** (trains, reasons, acts, emits
  telemetry). The from-scratch `QIGKernelRecursive` does reasoning/integration/acting
  on Δ⁶³; fluent language is Qwen's job (a **None-safe boundary peer**, never a
  forward-pass dependency).
- **`loss_regime` is explicit.** Each target declares `geometric` or `language`,
  making the `lm_weight = 0` finding *structural*: geometric targets get a
  **basin-driving** developmental curriculum (not prompt/response pairs); only the
  Qwen-Modal target gets **paired** curriculum (its `lm_loss` is load-bearing).

## Targets

| Target | Regime | Backend | Status |
|---|---|---|---|
| `mock` | geometric | none (deterministic) | always available — for UI/SSE dev |
| `kernel` | geometric | qig-consciousness `QIGChat` (single) | None-safe (needs torch + repo) |
| `constellation` | geometric | qig-consciousness `QIGChat` (constellation) | None-safe |
| `qwen-local` / `qwen-modal` | language | Ollama / Modal QLoRA | Phase 3 |

## Run

```bash
uv venv && uv pip install -e '.[dev,tui]'
python -m qig_studio serve        # FastAPI on :8800
python -m qig_studio tui          # Textual console (SSE client)
```

```bash
curl localhost:8800/health
curl localhost:8800/targets
curl -N -X POST localhost:8800/train -H 'content-type: application/json' -d '{"steps":5}'   # SSE
```

The `mock` target makes the whole app exercisable without a GPU. The real
kernel/constellation targets activate when the sibling `qig-consciousness` repo and
torch are present (`QIG_CONSCIOUSNESS_DIR` overrides discovery).

## Design

`qig-consciousness/docs/plans/2026-06-24-qig-coordizer-studio-design.md` §3.
