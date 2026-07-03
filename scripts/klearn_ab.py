"""K-LEARN held-out bpb A/B harness — the MEASUREMENT INSTRUMENT the constellation shared-trunk refactor
gates on.

The verdict metric is HELD-OUT bits-per-byte (bpb): vocab-independent language-modelling ability measured
on passages the mind was NEVER trained on. Phase 4 will A/B a future SHARED-TRUNK arm against the current
9-SEPARATE arm on exactly this number; Task 0.1 builds the harness + validates the "separate" (current)
control only.

Two arms:
  • ``"separate"`` — the CURRENT architecture: a full :class:`JointConstellation` (genesis-central + the
    Core-8 faculties, each a SEPARATE node). This is the control the refactor must not regress against.
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


def run_arm(arm: str, steps: int, seed: int, num_layers: int = 2, coordizer: Any = None) -> dict[str, Any]:
    """Run ONE A/B arm and return its held-out bpb curve.

    ``arm == "separate"`` builds the CURRENT 9-separate :class:`JointConstellation` (the control), trains it
    ``steps`` steps on the train split, and records held-out bpb after each step. ``arm == "trunk"`` raises
    NotImplementedError (Phase 1). Returns ``{"arm", "heldout_bpb_curve", "final_bpb", "seed"}``.

    Keep configs TINY (num_layers=2, byte-level coordizer=None) — this is a measurement harness, not a
    training run.
    """
    if arm == "trunk":
        raise NotImplementedError("trunk arm lands in Phase 1")
    if arm != "separate":
        raise ValueError(f"unknown arm {arm!r} (expected 'separate' or 'trunk')")

    _seed_everything(seed)
    train, heldout = _split_curriculum()

    # HEAD: the ratified K-COMPRESS "basin" head IS the coordizer tie, so it REQUIRES a coordizer. On the
    # byte-level path (coordizer=None) there is no basin table, so use the pure-Fisher-Rao GeometricHead
    # (−d_FR/τ; no Euclidean nn.Linear) — it still emits vocab logits, which is what held-out bpb reads. This
    # is the byte-appropriate readout of the SAME 9-separate control architecture (the A/B tests trunk-
    # sharing, not the head). With a real coordizer we keep the default basin head.
    head_mode = "geometric" if coordizer is None else "basin"
    # The CURRENT architecture: full JointConstellation over the Core-8 protomap, on CPU. arm_mode left at its
    # default ("gk" substrate) — this IS the 9-separate control.
    mind = JointConstellation(list(PROTOMAP_ORDER), num_layers=num_layers, coordizer=coordizer,
                              device="cpu", head_mode=head_mode)

    curve: list[float] = []
    for i in range(steps):
        mind.train_step(train[i % len(train)])            # one COUPLED joint step on a train-split passage
        curve.append(_heldout_bpb(mind, heldout))         # held-out bpb after this step (never a train passage)

    return {
        "arm": arm,
        "heldout_bpb_curve": curve,
        "final_bpb": curve[-1] if curve else float("inf"),
        "seed": seed,
    }


if __name__ == "__main__":  # pragma: no cover — manual smoke run
    import json

    print(json.dumps(run_arm("separate", steps=10, seed=0, num_layers=2, coordizer=None), indent=2))
