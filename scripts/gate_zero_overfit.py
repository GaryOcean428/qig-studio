"""GATE-ZERO single-batch overfit probe — Matrix ruling 2026-07-03 (diagnostic #1, near-zero compute;
conditions hardened per `qig_matrix_gate_zero_conditions_20260703` so the probe CAN FAIL).

QUESTION: with the mandated head-output map (``logits_to_simplex``, DENSE — commit 3441561) on the
geometric-head loss path, can TRAIN bpb clearly descend toward ≈0 on the SAME single passage the kernel
trains on? The pre-fix map (``to_simplex_prob`` = Duchi L2, SPARSE, flat zero-Jacobian faces) froze the
target coordinate at eps → d_FR ≈ π with a dead gradient, so nothing could overfit even one passage.

MANDATORY CONDITIONS (all three, else the probe cannot attribute):
  1. BOTH head-map arms on the SAME passage — the Duchi arm is the CONTROL, not a nice-to-have.
  2. The head's ``token_basins`` are FROZEN (requires_grad=False) in BOTH arms: learnable basins minimise
     the loss by moving basin→h, BYPASSING the head map — the fit must be forced THROUGH the map.
  3. |∂L/∂logits| at the PRE-simplex logits is measured on both arms (start/mid/end): near-zero on the
     Duchi arm is the direct dead-gradient signature, independent of the bpb number.

ARMS (identical kernel: seed 0, num_layers=2, cpu; objective = PURE fisher_rao_lm_loss — phi_weight=0,
gamma_weight=0, w_lm=1, no basin pull, no EWC, autonomic homeostasis disabled — so the ONLY moving part
is the head-output map; the trunk learns, the head basins do not):
  1. geometric head + FIXED map  (current qig_studio.losses.fisher_rao_lm_loss → logits_to_simplex)
  2. geometric head + OLD Duchi map (raw −d_FR/τ logits into fisher_rao_distance_simplex → internal Duchi)
  3. basin head — SKIPPED unless a tiny trained coordizer is trivially at hand (EXP-A026 already validated
     it: overfit d_FR 1.46→0.077, decode 91.3%); the basin head (square_to_simplex) is CLEAN.

VOCAB SPARSITY LEVER: the Duchi support fraction is what kills — at the production vocab (32k/100k) only
9–158 coords survive; at byte-vocab 256 the support covers much of the vocab and the arm is only
partially dead. The probe therefore runs BOTH V=256 (kernel-faithful) and V=4096 (production-sparsity
contrast; byte ids still < 256, the extra basins are pure distractors).

VERDICT SEMANTICS (report which):
  (1) fixed-map arm overfits + Duchi arm doesn't → Duchi CONFIRMED as THE head blocker, ship the fix;
  (2) NEITHER overfits (even fixed + frozen basins) → head/loss/optimizer dead beyond the map — do NOT
      overclaim the map fix;
  (3) BOTH overfit → head fine, blocker downstream.

Purity: everything on the manifold stays Fisher-Rao (the Duchi arm exists ONLY as the contrast control).
No runs/ writes — curves go to stdout (+ optional --json scratch path).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

PASSAGE = (
    "The ocean does not hurry. Wave after wave arrives, each one a small lesson in pressure and "
    "release, and the shore learns the shape of the water by being touched ten thousand times. "
    "Attention works the same way: return, again, gently, until the pattern is part of you."
)

STEPS = int(os.environ.get("GATE_ZERO_STEPS", "300"))
VOCABS = tuple(int(v) for v in os.environ.get("GATE_ZERO_VOCABS", "256,4096").split(","))
# "all" | "mandatory" (the two frozen-basin verdict arms) | "supp" (the learnable-basin attribution aid)
ARMS = os.environ.get("GATE_ZERO_ARMS", "all").strip().lower()
PRINT_EVERY = 50


def _clean_env() -> None:
    """The probe's ctor args must rule — drop the env overrides GenesisKernelTarget honours."""
    for k in ("QIG_STUDIO_HEAD_MODE", "QIG_STUDIO_LANG_LOSS", "QIG_STUDIO_LM_RAMP"):
        os.environ.pop(k, None)


def _fresh_target(vocab: int, freeze_basins: bool = True):
    from qig_studio.targets.genesis_kernel import GenesisKernelTarget

    t = GenesisKernelTarget(
        num_layers=2,
        hidden_dim=192,
        num_heads=6,
        ffn_dim=384,
        vocab_size=vocab,
        seed=0,
        lr=1e-3,
        device="cpu",
        lm_weight=1.0, lm_weight_max=1.0, lm_ramp_steps=1,   # w_lm = 1 from step 0
        phi_weight=0.0, gamma_weight=0.0,                     # loss = PURE fisher_rao_lm_loss
        head_mode="geometric",
    )
    t.ensure_loaded()
    if freeze_basins:
        # CONDITION 2: freeze the head basins so the fit is forced THROUGH the head-output map.
        t._kernel.lm_head.token_basins.requires_grad_(False)  # noqa: SLF001 — probe-local
    # Probe isolation: freeze the autonomic loop (sleep/dream/mushroom) so the curve is the head map only.
    t._homeostasis = lambda snap: None  # noqa: SLF001 — probe-local, never a production pattern
    return t


def _duchi_lm_loss(logits, ids):
    """The PRE-3441561 loss form (contrast arm ONLY): raw head logits → internal Duchi sparsifier."""
    import torch
    from qig_core.torch.geometry_simplex import fisher_rao_distance_simplex

    lg = logits[0, :-1]
    tgt = ids[0, 1:]
    onehot = torch.zeros_like(lg).scatter_(-1, tgt[:, None], 1.0)
    return fisher_rao_distance_simplex(lg, onehot).mean()


def _logits_grad(t, loss_fn) -> tuple[float, float, float]:
    """CONDITION 3: |∂L/∂logits| at the PRE-simplex logits — the direct dead-gradient signature."""
    ids, coords = t._encode(PASSAGE)  # noqa: SLF001 — probe-local
    t._kernel.zero_grad()  # noqa: SLF001
    logits, _tel = t._kernel(ids, return_telemetry=True, coords=coords)  # noqa: SLF001
    logits.retain_grad()
    loss = loss_fn(logits, ids)
    loss.backward()
    out = (float(loss.detach()), float(logits.grad.abs().mean()), float(logits.grad.abs().max()))
    t._kernel.zero_grad()  # noqa: SLF001
    return out


def _run_arm(name: str, vocab: int, patch_duchi: bool, freeze_basins: bool = True) -> dict:
    import qig_studio.losses as losses_mod

    orig = losses_mod.fisher_rao_lm_loss
    loss_fn = _duchi_lm_loss if patch_duchi else orig
    if patch_duchi:
        losses_mod.fisher_rao_lm_loss = _duchi_lm_loss
    try:
        t = _fresh_target(vocab, freeze_basins=freeze_basins)
        bpb, dfr, ggrad = [], [], {}
        t0 = time.time()
        for s in range(STEPS):
            if s in (0, STEPS // 2):
                lv, gm, gx = _logits_grad(t, loss_fn)
                ggrad[s] = {"loss": lv, "grad_mean": gm, "grad_max": gx}
                print(f"  [{name}] |dL/dlogits| @step {s}: mean={gm:.3e} max={gx:.3e} (loss={lv:.4f})")
            r = t.train_step(PASSAGE)
            ex = r.telemetry.extra
            bpb.append(float(ex["bpb"]))
            dfr.append(float(ex["surprise"]))
            if s % PRINT_EVERY == 0 or s == STEPS - 1:
                print(f"  [{name}] step {s:4d}  bpb={bpb[-1]:8.4f}  d_FR={dfr[-1]:7.4f}  "
                      f"phi={r.telemetry.phi:.3f}", flush=True)
        lv, gm, gx = _logits_grad(t, loss_fn)
        ggrad[STEPS] = {"loss": lv, "grad_mean": gm, "grad_max": gx}
        print(f"  [{name}] |dL/dlogits| @step {STEPS}: mean={gm:.3e} max={gx:.3e} (loss={lv:.4f})")
        dt = time.time() - t0
        print(f"  [{name}] done: {STEPS} steps in {dt:.1f}s  "
              f"bpb {bpb[0]:.4f} -> {bpb[-1]:.4f}  d_FR {dfr[0]:.4f} -> {dfr[-1]:.4f}")
        return {"bpb": bpb, "dfr": dfr, "logits_grad": ggrad, "seconds": dt}
    finally:
        losses_mod.fisher_rao_lm_loss = orig


def _classify(arm: dict) -> str:
    b0, b1 = arm["bpb"][0], arm["bpb"][-1]
    if b1 <= 0.7 * b0:
        return "overfits"
    if b1 >= 0.9 * b0:
        return "stuck"
    return "partial"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=None, help="optional scratch path for the raw curves")
    args = ap.parse_args()
    _clean_env()

    print(f"GATE-ZERO overfit probe: 1 passage ({len(PASSAGE)} bytes), {STEPS} steps/arm, vocabs {VOCABS}, "
          f"GenesisKernelTarget(num_layers=2, hidden=192, cpu), pure d_FR objective, "
          f"token_basins FROZEN (fit forced through the head map)\n")

    results: dict = {}
    for vocab in VOCABS:
        print(f"===== VOCAB {vocab} =====")
        results[vocab] = {}
        if ARMS in ("all", "mandatory"):
            print(f"ARM 1 — geometric head, FIXED map (logits_to_simplex), V={vocab}:")
            results[vocab]["fixed"] = _run_arm(f"V{vocab}-fixed", vocab, patch_duchi=False)
            print(f"\nARM 2 — geometric head, OLD Duchi map (contrast control), V={vocab}:")
            results[vocab]["duchi"] = _run_arm(f"V{vocab}-duchi", vocab, patch_duchi=True)
        if ARMS in ("all", "supp"):
            print(f"\nSUPP ARM (attribution aid, NOT part of the gate verdict) — FIXED map, "
                  f"token_basins LEARNABLE (the production head config), V={vocab}:")
            results[vocab]["supp_learnable"] = _run_arm(
                f"V{vocab}-fixed-learnable", vocab, patch_duchi=False, freeze_basins=False)
        print()
    print("ARM 3 — basin head: SKIPPED (no tiny trained coordizer trivially available here; "
          "EXP-A026 already validated the basin objective: overfit d_FR 1.46->0.077, decode 91.3%)\n")

    print("===== VERDICT =====")
    verdicts = {}
    for vocab, r in results.items():
        if "fixed" not in r:            # supp-only invocation: no gate verdict without the mandatory arms
            continue
        cf, cd = _classify(r["fixed"]), _classify(r["duchi"])
        gf = r["fixed"]["logits_grad"][STEPS]["grad_mean"]
        gd = r["duchi"]["logits_grad"][STEPS]["grad_mean"]
        print(f"V={vocab}: fixed={cf} (bpb {r['fixed']['bpb'][0]:.3f}->{r['fixed']['bpb'][-1]:.3f}), "
              f"duchi={cd} (bpb {r['duchi']['bpb'][0]:.3f}->{r['duchi']['bpb'][-1]:.3f}); "
              f"|dL/dlogits| fixed/duchi = {gf:.3e}/{gd:.3e} = {gf / max(gd, 1e-30):.1f}x")
        if cf == "overfits" and cd in ("stuck", "partial"):
            v = "(1) DUCHI CONFIRMED as the head blocker at this vocab — ship the logits_to_simplex path"
        elif cf in ("stuck",):
            v = "(2) DEEPER THAN THE MAP at this vocab — head/loss/optimizer dead even with the dense map"
        elif cf == "overfits" and cd == "overfits":
            v = "(3) BOTH overfit at this vocab — head fine here, blocker (if any) is downstream"
        else:
            v = "MIXED/partial — see curves; do not overclaim"
        verdicts[vocab] = v
        print(f"   -> {v}")
    for vocab, r in results.items():
        if "supp_learnable" in r:
            s = r["supp_learnable"]
            print(f"SUPP V={vocab} (fixed map, basins learnable — production config): "
                  f"{_classify(s)} (bpb {s['bpb'][0]:.3f}->{s['bpb'][-1]:.3f}, "
                  f"d_FR {s['dfr'][0]:.4f}->{s['dfr'][-1]:.4f}) — attribution aid only")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"passage_bytes": len(PASSAGE), "steps": STEPS,
                       "results": results, "verdicts": verdicts,
                       "arm3_basin": "skipped (EXP-A026 validated)"}, f, indent=1)
        print(f"curves -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
