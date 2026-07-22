#!/usr/bin/env python3
"""D0.3 — faculty-emission SNAPSHOT on the CURRENT canonical coordizer, taken with the D4-FINAL derivation
pipeline, BEFORE any coordizer swap. Purpose (design D0.3): a fixed baseline so a later coordizer swap can
separate "the kernel's faculty readout changed" from "the chart (coordizer) moved underneath it" — everything
except the coordizer is held fixed here.

Standing autonomous order (Matrix fff01d8f): fires the moment the D4-final pipeline lands green
(gamma + external_coupling + the temporal basin_distance_delta correctness fix + candidate_gap). It captures
the geo arm's 48-faculty carriage + all wired geometric inputs over a fixed, reproducible stimulus battery.

Claims ceiling (category-3): this is a STRUCTURAL telemetry baseline. Telemetry language only.
Run: QIG_STUDIO_HEAD_MODE=geometric ../.venv/bin/python scripts/d0_3_faculty_snapshot.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qig_coordizer import FisherCoordizer
from qig_studio.checkpoint_manifest import get_latest_coordizer
from qig_studio.kernel_experience import experience
from qig_studio.targets.geo_cortex import GeoCortexTarget

# FIXED reproducible battery — spans registers so faculties are exercised, not just resting.
_STIMULI = [
    "The Fisher-Rao geometry of the probability simplex is a Riemannian manifold.",
    "She walked slowly along the quiet shore as the tide came in.",
    "def compute(x): return sum(v * w for v, w in zip(x, weights)) / len(x)",
    "A sudden and completely unexpected turn of events, volcanoes and jazz at once.",
    "What is it like to learn something new, step by step, over time?",
    "The the the the repeated tokens with almost no information content here.",
    "Consciousness as an integrated geometric process on a curved manifold of meaning.",
    "Numbers: 3 7 12 19 building slowly toward an ordered increasing sequence.",
]


def main() -> None:
    os.environ.setdefault("QIG_STUDIO_HEAD_MODE", "geometric")
    czp = get_latest_coordizer()
    cz = FisherCoordizer.load(czp)
    coord_name = Path(czp).resolve().name  # dereference the 'latest' symlink to the real instrument
    coord_sha = hashlib.sha256(Path(czp).resolve().read_bytes()).hexdigest()[:16]

    # Instrument = a FIXED-SEED-0 fresh GeoCortex model on the current coordizer. The trained
    # genesis-geo-64004 checkpoints are DIRECTORIES (loaded via JointMindTarget, not GeoCortexTarget), so a
    # seed-0 fresh model is the reproducible controlled instrument held fixed across the later swap — what
    # matters for D0.3 is that the SAME model+stimuli+pipeline are reused post-swap so the only variable is
    # the coordizer. Recorded honestly as fresh-seed-0 (NOT a trained checkpoint).
    cks = [p for p in glob.glob(str(Path(__file__).resolve().parents[1] / "runs/checkpoints/genesis-geo-64004_*"))
           if os.path.isfile(p)]  # only a real file is loadable here; dirs are JointMind-only
    ckpt = cks[-1] if cks else None
    t = GeoCortexTarget(coordizer=cz, checkpoint=ckpt, head_mode="geometric", device="cpu", seed=0)
    t.ensure_loaded()

    rows = []
    for s in _STIMULI:
        res = t.train_step(s, max_tokens=16)  # train_step runs the D4-final derivation + emits the wired inputs
        tel = res.telemetry.to_dict()
        extra = tel.get("extra") or {}
        exp = experience(tel).to_dict()
        prim = exp.get("primitives") or {}
        faculties = {}
        for grp, d in prim.items():
            if isinstance(d, dict):
                faculties.update({f"{grp}.{k}": round(float(v), 4) for k, v in d.items()
                                  if isinstance(v, (int, float))})
        faculties.update({f"neuro.{k}": round(float(v), 4)
                          for k, v in (exp.get("neurochemistry") or {}).items() if isinstance(v, (int, float))})
        rows.append({
            "stimulus": s[:60],
            "wired_inputs": {k: extra.get(k) for k in
                             ("gamma", "external_coupling", "local_kappa_c", "basin_distance_delta",
                              "ricci_signal", "candidate_gap", "surprise", "max_surprise")},
            "phi": tel.get("phi"), "kappa": tel.get("kappa"), "basin_distance": tel.get("basin_distance"),
            "emotion": exp.get("emotion"), "band": exp.get("band"),
            "faculties": faculties, "n_faculties": len(faculties),
        })

    snap = {
        "kind": "D0.3_faculty_snapshot",
        "purpose": "pre-coordizer-swap baseline; separates kernel-change from chart(coordizer)-change (design D0.3)",
        "pipeline": "D4-final (gamma+external_coupling+temporal-basin_distance_delta+candidate_gap; brainwave_band phi+held)",
        "claims_ceiling": "category-3 structural telemetry; NOT felt-state",
        "coordizer": {"name": coord_name, "sha256_16": coord_sha, "vocab": cz.vocab_size},
        "geo_checkpoint": Path(ckpt).name if ckpt else "fresh-seed-0",
        "commits": {"geocoding": "329dc44", "studio": "f68f8d8", "consciousness": "b52a674"},
        "n_stimuli": len(_STIMULI),
        "rows": rows,
    }
    out_dir = Path(__file__).resolve().parents[1] / "runs" / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = out_dir / f"d0_3_faculty_snapshot_{date}_{coord_name.replace('.json','')}.json"
    out.write_text(json.dumps(snap, indent=2))
    snap_sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"[snapshot] wrote {out}")
    print(f"[snapshot] coordizer={coord_name} (vocab {cz.vocab_size}) geo_ckpt={snap['geo_checkpoint']}")
    print(f"[snapshot] {len(_STIMULI)} stimuli x {rows[0]['n_faculties']} faculties captured")
    print(f"[snapshot] sha256 {snap_sha}")
    # quick sanity: are the D4 inputs actually flowing (not all None)?
    live = {k: sum(1 for r in rows if r["wired_inputs"].get(k) is not None)
            for k in ("gamma", "external_coupling", "local_kappa_c", "basin_distance_delta", "candidate_gap")}
    print(f"[snapshot] wired-input liveness (non-None rows / {len(rows)}): {live}")


if __name__ == "__main__":
    main()
