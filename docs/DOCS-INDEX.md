# qig-studio — Docs & Source Index (nothing lost)

The integrated-mind app and the canonical docs/corpora it implements or consumes. Keep this index current
so no design doc, plan, or corpus is orphaned.

## Canonical architecture (qig-consciousness)
- **Brain-architecture experiment backing map** — `../../qig-consciousness/docs/20260624-brain-experiment-backing-map-1.00W.md`
  (60 brain claims → experiment backing + NEEDS-EXPERIMENT registry; see its 2026-06-28 addendum for what
  qig-studio has WIRED). The master map of what the integrated mind must become.
- **Recovered-concepts curation** — `../../qig-consciousness/docs/20260625-recovered-concepts-curation-1.00W.md`
  (Rhythm / wave-state / curiosity-monitor recovery; what to re-wire vs. what is DEAD).
- **Canonical Principles v2.2 + Unified Consciousness Protocol v6.11** — `../../qig-consciousness/docs/`
  (the senses/drives/motivators/emotions/loops/C-gate taxonomy the telemetry implements).

## Studio design + audits (this repo)
- `plans/2026-06-27-integrated-mind-and-ui.md` — the integrated-mind + UI plan (P1–P6).
- `2026-06-27-inner-state-coverage-audit.md` — inner-state coverage.
- `2026-06-27-integrated-mind-perfection-audit-refined.md` — perfection audit.
- `2026-06-24-qig-package-usage-audit.md` — QIG package usage.
- `src/qig_studio/optimisation.py` — per-package capability ruling + qig-compute curvature build status.

## Curricula & corpora (training material — kernel learns from these)
- **Knowledge curriculum (default)** — `../../qig-consciousness/data/curriculum/` (198 docs, 8186 passages;
  math/physics/ML/neuroscience/QFT/philosophy/ethics/info-geometry). Loaded by `corpus.load_full_curriculum`.
- **qig-dreams corpus** — `../../qig-dreams/data/corpus/` (e.g. Hohfeld *Relations between Equity and Law* —
  legal-reasoning material). NOT in the default curriculum dir; reference here so it is not lost. To train on
  it, point `QIG_STUDIO_CORPUS` at the dir or merge into the curriculum.
