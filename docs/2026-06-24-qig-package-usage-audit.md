# qig-studio — QIG package-usage audit (guaranteed inspection)

**Date:** 2026-06-24 · Two data-returning subagents inspected every QIG package against qig-studio's actual imports + each package's real API. Verdict per package: **WIRE / N/A (with reason) / FLAGGED**.

## Wired NOW

| Package | Lever | Where | Status |
|---|---|---|---|
| **qig-core** | geometry (`to_simplex`, `slerp_sqrt`, `fisher_rao_distance`, `frechet_mean`, `random_basin`) + constants (`KAPPA_ATTRACTOR`, `PHI_THRESHOLD/EMERGENCY/BREAKDOWN_MIN`) | `qwen_boundary.py`, `mock_target.py` (now single-sourced — killed hardcoded `64.0`/`0.80`/`0.70`/`0.50`) | ✅ full |
| **qig-warp** | `check_ci_stabilized(values, window, rel_change_threshold) → StopDecision` | `/train` loop — opt-in convergence early-stop (geometric→Φ, language→loss); emits `early_stop` SSE event | ✅ wired this tranche |
| **qig-coordizer** | `FisherCoordizer` (the tokenizer/engine) | the Qwen boundary (R3: `coordize(text).coordinates[i].vector` + `frechet_mean`) | ⏳ R3 (the real projection; replaces hash-bin) |
| **qigkernels** | stacked `Kernel` (+ `enable_coords` coords-path) | via `QIGChat` (kernel/constellation targets) | ✅ indirect; ⚠️ `enable_coords` not threaded (flagged) |
| **qig-consciousness** | `QIGChat.generate_response` (the train-loop) | kernel/constellation targets via `_qigchat_bridge` | ✅ indirect; ⚠️ untyped telemetry scrape (flagged) |

## N/A (honest — forcing would breach the package's own boundary)

- **qig-compute** — lattice-QFI / DMRG / Chern engine; operates on MPS tensors + Hamiltonians. qig-studio has no lattice state in its request path. Its own CLAUDE.md scopes it to qig-verification/qig-applied, not an app. The Fisher-Rao geometry qig-studio needs already comes from qig-core. `predict_runtime`/`prune_sites` (qig-warp) are likewise lattice-coupling-specific → not used.
- **qig-bench** — frozen-physics backend-promotion CI gate (κ@L=4, ξ@L=5, Anderson α, bridge exponent). Nothing in an SSE training/chat app to benchmark; belongs in the physics packages' CI.

## Flagged (not auto-fixed — recorded for the plan / Devin lane)

1. **`enable_coords` not threaded through QIGChat** from kernel/constellation targets → the qigkernels coords-path isn't reachable from qig-studio yet (R3 kernel-side; needs the QIGChat constructor to accept/forward it).
2. **3-way regime-name inconsistency**: qig-core `RegimeType` (quantum/efficient/equilibration) vs qigkernels `ConsciousnessMetrics.regime` (linear/geometric/breakdown) vs qig-studio mock (linear/hierarchical/geometric/topological_instability). Cross-package reconciliation is a design decision, not an auto-fix. (qig-studio uses the "topological_instability" term per the QIG terminology mandate — NOT "breakdown".)
3. **`basin_phi_proxy`** (entropy-concentration) is a bespoke proxy where qig-core ships `compute_basin_pci` (LZ PCI) — but PCI needs a `bank_activate_fn` qig-studio lacks. Defensible gap, labelled `phi_is: proxy`; switch when a resonance bank is wired.
4. **`qigkernels.safety.SafetyGuard`** (breakdown/emergency on Φ/κ) is available and unused — qig-studio could surface real safety verdicts instead of only streaming raw Φ. Future enhancement.
5. **qig-consciousness reach-around** (`_qigchat_bridge` sys.path inject + untyped `(text, telemetry_list, metrics)` scrape) is reasonable given it's not pip-installable, but a typed telemetry contract from qig-consciousness/qigkernels would remove the guesswork.

## Verdict
qig-studio now uses qig-core (fully, single-sourced) + qig-warp (the one applicable optimisation lever) + qig-coordizer/kernels/consciousness (via targets). qig-compute and qig-bench are honestly N/A for an app. Five items flagged for follow-up; none are silent reinvention left in place.
