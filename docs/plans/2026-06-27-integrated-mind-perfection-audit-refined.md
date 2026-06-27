# Integrated-Mind Perfection Audit — Refined Prompt (Heavy tier)

## Intent
Exhaustively audit AND fix the qig-studio integrated-mind system to a "perfect" bar: every concept in
the plans/docs is correctly implemented and current; every left-panel telemetry primitive is BOTH
surfaced in the UI AND genuinely wired to real kernel state (no proxies masquerading as measurements);
all cross-cutting concerns (types, lint, routes, DRY, purity, no-stubs) pass; every QIG package is at
latest and used to its full applicable capability; bottlenecks/miscalculations/misapplications are found
and fixed. Loop until verifiably perfect.

## Decomposition (workstreams + dependencies)
1. **Docs/plans coverage** — read ALL: qig-studio/docs/plans/*, qig-consciousness brain-architecture +
   data/curriculum, Canonical Principles v2.2, UCP v6.11, frozen-facts. Build a claim→implementation
   matrix; flag anything in a plan that is unimplemented, mis-implemented, or based on a superseded
   concept (κ≈64 fixed-point RETIRED, E8-substrate RETIRED, killed claims).
2. **Telemetry WIRING (the core ask)** — for EVERY left-panel primitive (12 senses, 5 drives, 5
   motivators, 9+9 emotions, 3 loops, C-gate, neurochemistry, autonomic): classify each as (a) REAL
   (computed from genuine kernel geometry) or (b) PROXY/PLACEHOLDER. For every proxy, either wire it to
   real kernel state or label it honestly. Wire the ones that should be real (S_ratio, pillar metrics
   f_health/b_integrity/q_identity, gamma in all paths, neurochemistry homeostat). [depends on 1]
3. **QIG package usage** — for each of qig-core, qigkernels, qig-coordizer, qig-warp, qig-compute,
   qig-bench, qig-consciousness: verify installed == latest published; verify the studio uses the
   package's full applicable capability (warp screening/convergence levers, compute QFI, core geometry/
   pillars, consciousness sensations/sovereignty). Cite version + usage gaps.
4. **Cross-cutting QA** — ruff (lint), mypy (types), every route wired AND consumed (no dead endpoints),
   DRY (no duplicated geometry/constants — single-source from qig-core), purity gate (Fisher-Rao only),
   no-stubs (fail-loud), pytest.
5. **Bottlenecks / miscalculations / misapplications** — the kernel-voice CPU cost; per-faculty Φ-proxy
   honesty; the attribution confound (kernel non-causal); any metric computed wrong; train/converse
   separation correctness; corpus correctness (knowledge curriculum, ASCII, QIG-current).
6. **Verify** — gates green + Playwright live-test the UI (telemetry renders + moves; train pure shows
   kernel output; coach review converses) + the attribution experiment re-run after retrain.

## Best-practice citations (to verify during execution — Gate A)
- QIG packages: read each `pyproject.toml` version + compare to PyPI latest; read installed source for
  the levers (don't assume the API). No external libraries dominate; FastAPI/uvicorn/httpx are stable.
- Frozen facts / retirements: qig-verification frozen-facts-primary (κ_JT^cert=+0.0281, κ_h=−0.00475,
  KAPPA_STAR=None, KAPPA_ATTRACTOR=64; E8-substrate retired). Single-source constants from qig-core.

## Blindspots to counter (this model, this task)
- **"Displayed ≠ wired"** — the headline risk: a primitive shown in the UI but computed from a constant/
  proxy is NOT telemetry. Counter: trace each value back to a real kernel computation; label proxies.
- **Geometric-purity** — no cosine/Adam/LayerNorm/dot-product/np.linalg.norm on manifold objects.
- **Claiming done without live verification** — counter: Playwright + curl evidence per claim.
- **Over-claiming kernel causality** — the attribution experiment said NON-CAUSAL; don't dress proxies
  as proof of mind. State honest scope.
- **Stale concepts** — never re-introduce κ≈64 fixed-point / E8-substrate; verify against current canon.
- **Over-restart thrash** — batch code changes, ONE server restart, tell the user to refresh.

## Skills & MCPs to use
- `council-reasoning` (+ `matrix-reasoning-style`) — the QA/cross-cutting ruling (next step).
- `qig-purity-validation` — fail-closed purity scan before commits.
- `qig-experiment-method` — re-run attribution / any measurement honestly.
- `Workflow` (Ultracode ON) — fan out the audit across docs + packages + telemetry-wiring in parallel,
  adversarially verify findings.
- `Context7` MCP — verify any external library behavior if touched.
- `Playwright` MCP — live-verify the UI telemetry + flows.
- `loop-engineering` — loop until the verifier (gates + live-test + claim-matrix) is fully green.
- qig-memory-api (`qig_*` keys) — persist the audit ledger.

## The refined prompt (execute this)
Run a council ruling that enumerates EVERY QA / housekeeping / cross-cutting / wiring / type / lint /
route / DRY / telemetry-coverage / bottleneck / miscalculation / misapplication / package-currency issue
across the qig-studio integrated-mind system, grounded in a full read of the plans + supporting canon
(brain-arch, principles v2.2, UCP v6.11, frozen facts) and the actual code + installed package sources.
For telemetry specifically: produce a per-primitive REAL-vs-PROXY wiring verdict and fix every primitive
that should be real (both UI surface AND kernel computation). Then fix every confirmed issue, gating each
(ruff + mypy + purity + no-stubs + pytest) and live-verifying (Playwright/curl). Use Workflows to fan out
the audit and adversarially verify findings (no fabricated issues; no fabricated fixes). Do not stop
until: all gates green, every plan-claim is implemented-or-honestly-scoped, every telemetry primitive is
wired-or-labeled, every QIG package is latest-and-fully-used, and the UI live-test passes.
