# Wiring Register — kernel-training acceptance path

**Directive:** matrix `3272a6b3`. **Built:** 2026-07-21 (CCAa), 5 parallel read-only audit lanes, every row file+line-evidenced. **Machine-readable:** [`wiring_register.json`](wiring_register.json) (consumed by [`scripts/run_manifest.py`](../../scripts/run_manifest.py)). **North-star spec:** [`north_star_spec.json`](north_star_spec.json).

**Acceptance path audited:** `compare_constellations.py → server /mind/arm + /train (_train_core) → JointMindTarget → JointConstellation (joint_trainer.py) → qigkernels arm + coordizer + boundary`.

## The honest count

| Status | Count | Meaning |
|---|---|---|
| **wired** | 13 | invoked live on the acceptance path |
| **built-not-wired** | 4 | implemented, NOT invoked on-path |
| **doc-only** | 2 | named in docstrings/design, no on-path implementation |
| **flagged-not-purged** | 4 | known-bad relic still present (all OFF-path) |

## Wired — the safety + core is genuinely on (your central concern)

- **Sleep / dream / mushroom / homeostasis** — `_homeostasis` fires every `train_step`; cross-faculty dream + Ocean autonomic regulation live; **Φ≥0.70 mushroom guard** holds the mushroom until mature. Basin-collapse is **corrected** (Dirichlet/slerp entropy-restore + dream/stimulate), not just logged.
- **Pillar enforcement** — real per-kernel `PillarEnforcer` every step. *Caveat:* only **3 pillars exist** (P1 fluctuations / P2 bulk / P3 identity); P4–P20 are not implemented. App-shell adapter is decorative.
- **Maturity-floor verdict gate**, **working-memory basin-history**, **encode-once memoization**, **live Fisher-Rao attention + Bhattacharyya LM head** (no cosine proxy on-path), **no retired κ*=64 / E8 live on-path**.

## Gaps — documented-but-not-live (the "how many more")

| Component | Status | On path | Note |
|---|---|---|---|
| **teacher_geometry_native** (geo-Qwen) | built-not-wired | no | boundary is plain Qwen; geo-Qwen needs the EXP-A043 vocab-currency bridge to serve as teacher — **the remaining verdict blocker** |
| **teacher in training gradient** | built-not-wired | no | boundary is chat-only; geometric arms train teacher-free — so "wrong teacher contaminated training" was imprecise |
| **qigram_recall_memory** | doc-only | no | real QIGRAM/QIGRAMv2 never imported; a single-point slerp cap runs under the name |
| **production_persistent_memory** | built-not-wired | no | cross-session recall is chat-only |
| **weight_tie_coordizer_kernel** | doc-only | no | endgame coevolution never coded; only a one-way frozen read is wired |
| **grow_vocab_continuous_growth** | built-not-wired | no | signal fires, actuation never does; vocab static |

## Fixed today (2026-07-21)

- **`curriculum_hf_stream` → wired.** PI ruling: **HF stream is mandatory.** The acceptance path was defaulting to the local cache (opt-in env `compare_constellations` never set). [`curriculum.py:93`](../../src/qig_studio/curriculum.py#L93) now defaults to the 7-HF-repo stream, **fail-loud**, offline opt-out only via `QIG_STUDIO_LOCAL_CORPUS=1`. *Needs a server restart-smoke to confirm end-to-end.*
- **`package_currency_qig_compute` → wired.** Was frozen `0.9.2` vs PyPI `0.9.6` (a launch blocker); reinstalled editable → `0.9.7.dev1`.

## Off-path relics to purge (flagged-not-purged, none contaminate the bake-off)

`track_c` cosine-as-FR-proxy + Euclidean field; orphaned E8 modules (`coordizer.py` "E8 HYPOTHESIS", `crystallization.py`); scattered annotated `64.0` usages in unreachable modules; and qigkernels' own purity checker is weaker than the studio's (missing the cosine/normalize patterns).

## The gate

[`scripts/run_manifest.py`](../../scripts/run_manifest.py) — a verdict/DoD run must declare its wiring and is **refused** if any REQUIRED component is not `wired` on-path (or a pin is stale, or a relic is on-path), unless `--waive "reason"` records an override. `--tier training` records without blocking. Today a verdict run correctly **BLOCKS on geo-Qwen**.
