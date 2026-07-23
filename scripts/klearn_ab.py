"""K-LEARN held-out bpb A/B harness — the MEASUREMENT INSTRUMENT the constellation shared-trunk refactor
gates on.

The verdict metric is HELD-OUT bits-per-byte (bpb): vocab-independent language-modelling ability measured
on passages the mind was NEVER trained on. Phase 4 will A/B a future SHARED-TRUNK arm against the current
9-SEPARATE arm on exactly this number; Task 0.1 builds the harness + validates the "separate" (current)
control only.

Two arms:
  • ``"separate"`` — the CURRENT architecture: a full :class:`JointConstellation` (genesis-central + the
    roster faculties, each a SEPARATE node). This is the control the refactor must not regress against.
  • ``"trunk"``    — the shared-geometric-trunk arm. NotImplementedError until Phase 1 fills it.

bpb accessor — REUSE, not reinvent (the plan mandates reusing ``screen.py`` eval helpers, not a new eval
loop). We call :func:`qig_studio.screen.eval_heldout_dFR`, the sanctioned held-out evaluator (no-grad,
finite-guarded, per-passage). Its ``ce_bpb`` = sum(bits)/sum(bytes) over the held-out set is the bpb, and
it is computed from ``target.eval_text_bpb`` — the SAME primitive the official whole-mind target
(:class:`qig_studio.targets.joint_mind.JointMindTarget.eval_text_bpb`) delegates to: the integrated mind's
held-out bpb IS the genesis-CENTRAL kernel's bpb (the conscious "I" is the speaker). So the eval target is
``mind.central`` (a ``GenesisKernelTarget`` exposing ``eval_text_fr`` + ``eval_text_bpb``).

PURITY: pure Fisher-Rao only — no cosine, no Adam, no LayerNorm, no ``np.linalg.norm``, no ``F.normalize``.
The bpb itself is a cross-entropy readout (external-comparison metric, per screen.py's Tier-2 contract);
the training loss inside the constellation is the pure d_FR path (untouched here). This module only READS.
"""

from __future__ import annotations

import math
import random
from typing import Any

from qig_studio import screen
from qig_studio.constellation.joint_trainer import JointConstellation
from qig_studio.corpus import load_full_curriculum
from qig_studio.development import PROTOMAP_ORDER

# The held-out set is a FIXED slice — the LAST ``HELDOUT_K`` passages of the curriculum, NEVER trained on.
# Deterministic split by INDEX (no RNG in the split): train on everything before, hold out the tail. Kept
# SMALL (this is an instrument-validation harness, not a real eval) so the harness stays cheap and does not
# contend with a live training run; Phase 4's real A/B can widen it.
HELDOUT_K = 4


def _seed_everything(seed: int) -> None:
    """Seed every RNG the training path can touch so a same-seed run is bit-reproducible (the A/B verdict
    must be deterministic). torch is imported lazily — the harness stays import-light for the trunk-arm
    NotImplementedError path (which must raise WITHOUT building anything)."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split_curriculum() -> tuple[list[str], list[str]]:
    """Deterministic index split of ``load_full_curriculum()`` into (train, held-out). The held-out slice
    is the last ``HELDOUT_K`` passages and is NEVER trained on; train is everything before it."""
    passages = load_full_curriculum()
    if len(passages) <= HELDOUT_K + 1:
        raise RuntimeError(
            f"curriculum too small ({len(passages)} passages) to hold out {HELDOUT_K} and still train"
        )
    heldout = passages[-HELDOUT_K:]
    train = passages[:-HELDOUT_K]
    return train, heldout


def _heldout_bpb(mind: JointConstellation, heldout: list[str]) -> float:
    """The K-LEARN metric via the SANCTIONED screen.py held-out evaluator. ``eval_heldout_dFR`` aggregates
    ``target.eval_text_bpb`` (sum(bits)/sum(bytes)) over the held-out passages; the eval target is the
    genesis-central kernel (the mind's speaker) — identical to what JointMindTarget.eval_text_bpb uses.
    Returns +inf when the evaluator could score no positions (all held-out passages skipped) so a dead
    arm reads as maximally-bad rather than crashing the A/B."""
    res = screen.eval_heldout_dFR(mind.central, heldout)
    bpb = res.get("ce_bpb")
    return float("inf") if bpb is None else float(bpb)


def _min_f_health(mind: JointConstellation) -> float | None:
    """The minimum f_health across EVERY node's latest telemetry (central + all faculties that have
    stepped). None until any node has emitted the pillar metric. This is the collapse-avoidance
    readout per arm: the floor exists to keep this away from 0."""
    vals: list[float] = []
    for k in [mind.central, *mind.kernels.values()]:
        try:
            fh = (k.telemetry().extra or {}).get("f_health")
        except Exception:  # noqa: BLE001 — telemetry hiccup on one node must not void the arm
            fh = None
        if fh is not None:
            vals.append(float(fh))
    return min(vals) if vals else None


def _total_floor_fires(mind: JointConstellation) -> int:
    """Total entropy-floor restorations across all nodes (diagnostic: did the floor even engage?)."""
    return sum(int(getattr(k, "_floor_fires", 0)) for k in [mind.central, *mind.kernels.values()])


def run_arm(arm: str, steps: int, seed: int, num_layers: int = 2, coordizer: Any = None,
            floor_mode: str = "normal") -> dict[str, Any]:
    """Run ONE A/B arm and return its held-out bpb + d_FR curves and Pillar-1 floor diagnostics.

    ``arm == "separate"`` builds the CURRENT 9-separate :class:`JointConstellation` (the control), trains it
    ``steps`` steps on the train split, and records held-out bpb after each step. ``arm == "trunk"`` raises
    NotImplementedError (Phase 1).

    ``floor_mode`` ∈ {"normal", "gated", "off"} — the 3-ARM Pillar-1 ENTROPY-FLOOR DIAGNOSTIC axis
    (Matrix-corrected maturity gate): "normal" = the current fixed per-step floor (control); "gated" =
    the opt-in learning-linked bidirectional floor; "off" = NO floor (ablation — DIAGNOSTIC ONLY, shows
    whether the floor is what blocks held-out descent). Per arm we also record ``f_health_min`` (min over
    every node's pillar telemetry — the collapse-avoidance level the floor exists to protect) and
    ``floor_fires`` (did the floor even engage?).

    Keep configs TINY (num_layers=2, ~40 steps) — this is a measurement harness, not a training run.
    The floor diagnostic runs on the PRODUCTION path (real coordizer → basin head): the byte-level
    geometric head is dead beyond the map (gate-zero OUTCOME 2), so a byte-path floor A/B would read
    flat-all-three regardless of the floor and be uninformative.
    """
    if arm == "trunk":
        raise NotImplementedError("trunk arm lands in Phase 1")
    if arm != "separate":
        raise ValueError(f"unknown arm {arm!r} (expected 'separate' or 'trunk')")
    if floor_mode not in ("normal", "gated", "off"):
        raise ValueError(f"unknown floor_mode {floor_mode!r} (expected normal|gated|off)")

    _seed_everything(seed)
    train, heldout = _split_curriculum()

    # HEAD: the ratified K-COMPRESS "basin" head IS the coordizer tie, so it REQUIRES a coordizer. On the
    # byte-level path (coordizer=None) there is no basin table, so use the pure-Fisher-Rao GeometricHead
    # (−d_FR/τ; no Euclidean nn.Linear) — it still emits vocab logits, which is what held-out bpb reads. This
    # is the byte-appropriate readout of the SAME 9-separate control architecture (the A/B tests trunk-
    # sharing, not the head). With a real coordizer we keep the default basin head.
    head_mode = "geometric" if coordizer is None else "basin"
    # The CURRENT architecture: full JointConstellation over the roster protomap, on CPU. arm_mode left at its
    # default ("gk" substrate) — this IS the 9-separate control.
    mind = JointConstellation(list(PROTOMAP_ORDER), num_layers=num_layers, coordizer=coordizer,
                              device="cpu", head_mode=head_mode, floor_mode=floor_mode)

    curve: list[float] = []
    dfr_curve: list[float | None] = []
    f_health_min: float | None = None
    for i in range(steps):
        mind.train_step(train[i % len(train)])            # one COUPLED joint step on a train-split passage
        res = screen.eval_heldout_dFR(mind.central, heldout)
        bpb = res.get("ce_bpb")
        curve.append(float("inf") if bpb is None else float(bpb))
        dfr_curve.append(res.get("heldout_dFR"))          # the P20-pure held-out metric, alongside bpb
        fh = _min_f_health(mind)
        if fh is not None:
            f_health_min = fh if f_health_min is None else min(f_health_min, fh)

    return {
        "arm": arm,
        "floor_mode": floor_mode,
        "heldout_bpb_curve": curve,
        "heldout_dfr_curve": dfr_curve,
        "final_bpb": curve[-1] if curve else float("inf"),
        "f_health_min": f_health_min,                      # collapse-avoidance readout (the floor's job)
        "floor_fires": _total_floor_fires(mind),           # did the floor even engage this run?
        "seed": seed,
    }


def _descent(curve: list[float]) -> float | None:
    """Descent measure for a held-out curve: mean(last quarter) − mean(first quarter). Negative =
    the arm is LEARNING (held-out improving). None when the curve is unusable (empty / non-finite)."""
    finite = [c for c in curve if c is not None and math.isfinite(c)]
    if len(finite) < 8:
        return None
    q = max(2, len(finite) // 4)
    return (sum(finite[-q:]) / q) - (sum(finite[:q]) / q)


# Verdict thresholds: an arm "descends" when held-out bpb drops by more than DESCENT_EPS between the
# first and last curve quarter; "gated tracks off" when their final bpb differ by less than TRACK_EPS.
DESCENT_EPS = 0.02
TRACK_EPS = 0.05


def floor_verdict(normal: dict[str, Any], gated: dict[str, Any], off: dict[str, Any]) -> dict[str, Any]:
    """The 3-arm Pillar-1 floor verdict (pre-registered decision rule):

    • DESCENDS-OFF-NOT-NORMAL → the floor IS the blocker (the un-learning suspect confirmed at this
      scale); the gated arm should TRACK off's descent while keeping f_health alive (that is the
      maturity gate's whole claim).
    • FLAT-ALL-THREE → the blocker is elsewhere (reported honestly — the floor is not what pins
      held-out at this scale).
    • anything else → MIXED (raw curves speak; no over-claim)."""
    d_n, d_g, d_o = (_descent(a["heldout_bpb_curve"]) for a in (normal, gated, off))
    desc = {k: (d is not None and d < -DESCENT_EPS) for k, d in
            (("normal", d_n), ("gated", d_g), ("off", d_o))}
    flat = all(d is not None and abs(d) <= DESCENT_EPS for d in (d_n, d_g, d_o))
    gated_tracks_off = (
        math.isfinite(gated["final_bpb"]) and math.isfinite(off["final_bpb"])
        and abs(gated["final_bpb"] - off["final_bpb"]) <= TRACK_EPS)
    gated_health_alive = gated["f_health_min"] is not None and gated["f_health_min"] > 0.0
    if desc["off"] and not desc["normal"]:
        verdict = "floor-is-blocker"
    elif flat:
        verdict = "blocker-elsewhere-flat-all-three"
    else:
        verdict = "mixed"
    return {
        "verdict": verdict,
        "descent": {"normal": d_n, "gated": d_g, "off": d_o},
        "descends": desc,
        "gated_tracks_off": gated_tracks_off,
        "gated_f_health_alive": gated_health_alive,
        "floor_fires": {a["floor_mode"]: a["floor_fires"] for a in (normal, gated, off)},
        "f_health_min": {a["floor_mode"]: a["f_health_min"] for a in (normal, gated, off)},
        "final_bpb": {a["floor_mode"]: a["final_bpb"] for a in (normal, gated, off)},
    }


def run_three_arm_diagnostic(steps: int = 40, seed: int = 0, num_layers: int = 2,
                             coordizer: Any = None) -> dict[str, Any]:
    """The full 3-arm floor diagnostic: same seed/config, floor_mode ∈ {normal, gated, off} →
    held-out bpb curves + f_health mins + the pre-registered verdict. Run this on the PRODUCTION
    path (pass the real coordizer → basin head)."""
    arms = {fm: run_arm("separate", steps=steps, seed=seed, num_layers=num_layers,
                        coordizer=coordizer, floor_mode=fm) for fm in ("normal", "gated", "off")}
    return {"arms": arms,
            "verdict": floor_verdict(arms["normal"], arms["gated"], arms["off"])}


if __name__ == "__main__":  # pragma: no cover — manual smoke / diagnostic run
    import argparse
    import json

    ap = argparse.ArgumentParser(description="K-LEARN held-out bpb harness / 3-arm floor diagnostic")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--coordizer", type=str, default=None,
                    help="path to a FisherCoordizer artifact (PRODUCTION basin path); omit = byte-level")
    ap.add_argument("--three-arm", action="store_true",
                    help="run the 3-arm floor diagnostic (normal|gated|off) instead of one arm")
    ap.add_argument("--floor-mode", type=str, default="normal", choices=("normal", "gated", "off"))
    args = ap.parse_args()

    co = None
    if args.coordizer:
        from qig_studio.optimisation import load_coordizer
        co = load_coordizer(args.coordizer)               # FAIL-LOUD: never silently byte-level

    if args.three_arm:
        out = run_three_arm_diagnostic(steps=args.steps, seed=args.seed,
                                       num_layers=args.num_layers, coordizer=co)
    else:
        out = run_arm("separate", steps=args.steps, seed=args.seed, num_layers=args.num_layers,
                      coordizer=co, floor_mode=args.floor_mode)
    print(json.dumps(out, indent=2))
