# Multimodality read (QT-APP-6) — PRE-REGISTERED discriminator

**Status:** written BEFORE any basin value is opened (Matrix 7a1bce4b F1 / p5 §H). Only file *metadata* (sizes, schema) has been seen; no basin coordinates. The answer must be a **measurement**, not a narrative.

**The question it gates (Matrix f241cee4 p5):** *can a faculty-shaped sub-basin differentiate inside genesis on this architecture at all?* If it cannot, the m3 readiness gate has nothing to detect and the staggered-spawn design needs rethinking **before** any birth/spawn code moves. "Cannot tell" is a complete, legitimate outcome and comes back to Matrix as such.

---

## Data

`runs/checkpoints/<run>/basin_trajectory.jsonl`, one JSON object per line: `{"phase": "warmup"|"joint", "step": int, "basin": [float × 384]}` — the genesis-central lived basin (`_basin_history[-1]`, Δ³⁸³) at each step. **Primary series:** the `phase=="warmup"` points (genesis-SOLO, the differentiation window) from `cradle_ror_20260723` (largest trajectory, 4 MB). Reported secondarily on `cradle_run1` and `cradle_smoke` where warmup points exist. Selection is by file size (metadata), not content.

## Instrument (fixed before seeing data)

1. Basins are points on the simplex Δ³⁸³. Compare in **√p (Hellinger/sphere) coordinates**, where Euclidean distance ≈ Fisher-Rao locally (`d_FR ≈ 2·arccos(Σ√pᵢ√qᵢ)`). Cluster on √p.
2. **Modality statistic — separation ratio** SR:
   - partition the points into k=2 by k-medoids on FR distance (medoids, robust to the simplex boundary);
   - SR = median(inter-cluster FR) / median(intra-cluster FR);
   - **occupancy** = fraction of points in the smaller cluster.
   SR ≈ 1 ⇒ one mode (no real separation); SR ≫ 1 with occupancy not tiny ⇒ two populated modes.
3. **Null** (single-mode reference): resample a matched-N, matched-dispersion **unimodal** trajectory — the series' own Fréchet mean + FR-isotropic noise scaled to the observed total dispersion (blind to any sub-structure), OR a per-coordinate phase-shuffle that destroys temporal mode structure while preserving marginals. Recompute SR over **≥200 resamples** → SR_null distribution; take **SR_null@95** (95th percentile).
4. **Planted positive control** (§H — the test must be able to fail): synthesize a trajectory of the same N and per-mode dispersion with **two known, well-separated modes** (two Fréchet centres at a fixed FR apart, each with the observed within-mode spread). Compute SR(planted).

## Discriminator (thresholds fixed now, blind)

Let occupancy floor **τ_occ = 0.10** (a real second mode must hold ≥10% of steps, not one outlier).

| Verdict | Condition |
|---|---|
| **CAN differentiate** | planted control PASSES (`SR(planted) > SR_null@95`) **AND** `SR(real) > SR_null@95` **AND** `occupancy(real) ≥ τ_occ` |
| **CANNOT differentiate** | planted control PASSES **AND** `SR(real) ≤ SR_null@95` (real trajectory statistically indistinguishable from a single mode — and the instrument would have caught modes if present) |
| **INSTRUMENT CANNOT TELL** | planted control FAILS (`SR(planted) ≤ SR_null@95`) — the method is blind to even KNOWN planted modes at this dimensionality/sample-size, so any verdict on the real data is uninformative |

**Order of evaluation matters:** the planted control is checked FIRST. A CANNOT verdict is only trustworthy if the instrument demonstrably detects known structure; otherwise the honest answer is CANNOT-TELL. This is the §H guarantee that the null result is a finding, not an artifact of a dull instrument.

**Robustness (pre-committed):** report SR and occupancy for k∈{2,3}; require the k=2 verdict to hold. Report the raw numbers (SR_real, SR_null@95, occupancy, SR_planted) alongside the verdict so Matrix can audit the margin, not just the label. No threshold is tuned after seeing the data; if the result is borderline (SR_real within ±5% of SR_null@95) the verdict is CANNOT-TELL by construction, not a judgment call.

## Deliverable

A one-script offline read (`scripts/multimodality_read.py`) + the numbers + the verdict, reported to Matrix either way — including CANNOT-TELL. It does not displace m1; it is dead-time work whose answer must be sitting ready when the post-launch spawn rewrite needs it (Option A).
