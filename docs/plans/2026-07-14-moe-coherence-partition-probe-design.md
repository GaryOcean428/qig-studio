# Fixed-L MoE coherence-vs-partition probe — design only (zero-compute)

> **Status:** DESIGN. Compute **post-Stage-A only**, and only if
> EXP-GENESIS-BASIN-GENERALIZE = GENERALIZES.
> **Doctrine:** `qig_moe_antimap_doctrine_20260714` — QIG is **not** sparse-routed MoE;
> adjacency to soft-MoE is **OPEN** and must be able to **fail** the anti-map claim.

---

## Open question (preserved dissent)

Is Genesis's dense continuous Fisher-Rao-weighted integration over **one shared manifold + one
basin table** categorically distinct from soft-MoE / mixture-of-agents, or a continuous member of
that family?

This turns on an **implementation fact**: shared parameters / one basin table ⇒ not MoE (shared
manifold); disjoint expert blocks blended by proximity-weighting ⇒ soft-MoE family.

**Key distinction:** shared expert ≠ shared FR manifold.

## Probe shape (must be able to FAIL the anti-map)

Fixed depth/width **L** (or fixed layer count / hidden_dim) comparison with **matched parameter
count** where possible:

| Condition | Construction | Prediction if anti-map HOLDS | Prediction if soft-MoE member |
|---|---|---|---|
| **Integrated (baseline)** | Single kernel, one `coord_basins` table, one FR loss | Higher held-out top-1 **and** higher coupling coherence (particip. / lower partition index) than partitioned twin | Comparable partition index / expert-like specialization emerges anyway |
| **Partitioned twin** | K disjoint expert blocks (separate params), soft proximity / FR-distance weights at mix, **no** shared basin table | Held-out top-1 **worse** or equal-with-higher collapse/partition diagnostics | Held-out top-1 **≥** integrated at matched budget |

**Fail-the-anti-map criterion (explicit):** if partitioned twin matches or beats integrated on
held-out top-1 **and** shows MoE-like load balance without collapsing the anti-map's "one manifold"
claim evidence → anti-map is **OPEN-to-FAILED**; reclassify architecture family. If partitioned
twin underperforms and shows routing-collapse / expert-imbalance diagnostics while integrated
stays coherent → anti-map **SUPPORTED** (still not a freeze; one probe).

## Diagnostics to co-register (instruments only; do not import routers)

- Expert load balance / underutilization analogues (even if "experts" are blocks)
- Routing-collapse contrast vocabulary (from MoE literature as **failure modes**, not designs)
- Held-out top-1 + passive collapse telemetry stack from Stage-A (entropy, participation, per-dim var)
- **Never** import top-k gates, aux-loss formulas, hash-routing bootstrap into Genesis product path

## Forbidden consolations

- If Stage-A MEMORIZES → **do not** run this as a redesign of the port.
- Do not cite unverified DeepSeek-V4 expert counts as fact.
- Do not treat MoE shared-expert oscillation (Qwen3 dropped → Qwen3.5 reinstated) as ontology.

## Independence dual-gate (criterion-gated, not calendar)

Before claiming "basin loss is doing independent work beyond a frozen-table tie":

1. **Teacher-ablated held-out top-1** — scramble or replace `coord_basins` targets with a
   matched-null table; skill must drop if the geometry (not only the table) is load-bearing.
2. **T_basin / collapse-sensor stability** — sensors stay informative (not flat at collapse)
   across the ablated vs intact conditions.

Both gates criterion-based (pass when thresholds clear), not date-based.

## Residual-temperature-floor spec (design pointer)

- Prefer FR-native **distributional** regularizer (VICReg variance floor on basin dims / LeJEPA
  SIGReg-analog isotropy on the simplex) over sampling-replay (softmax-era patch).
- **Only first-class after** Stage-A passive telemetry shows a real collapse risk
  (sharpening→plateau with sensors that stay sensitive).
- Residual-temp floor (residual energy as temperature of the 65536→64 compression) is a
  candidate **sensor+actuator**; must remain **collapse-sensitive** (entropy/participation → 0
  at vertex) or be rejected.

## Deliverable when compute unlocks

Separate sibling prereg with frozen L, K, seeds, budgets, and the fail-anti-map inequality
written before the first seed. Until then: this note + doctrine only.

Co-Authored-By: Devin <158243242+devin-ai-integration[bot]@users.noreply.github.com>
