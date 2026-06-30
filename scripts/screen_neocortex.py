#!/usr/bin/env python3
"""32k AVENUE SCREEN — the PI's "discover the best avenue fast, kill the loser" direction-finder.

On the FRESH 32k coordizer, train the 4 candidate avenues (arm × head) for a SHORT, EQUAL budget at a
fixed modest depth, then rank them on **held-out mean d_FR** — the committed VERDICT metric of
``docs/plans/2026-06-30-exp-cortex-ab-prereg.md`` (the torch primitive ``fisher_rao_distance_simplex``,
range [0, π], via ``eval_text_fr``). CE-bpb is reported alongside as the Tier-2
external-comparison-only number (``eval_text_bpb``) — **never** the ranking axis.

This is NOT the depth A/B (EXP-CORTEX-AB: 8L vs 1L-rec) — that is a SEPARATE later experiment on the
WINNING arm+head. Here depth is fixed (``--layers 4``) and the screen holds EVERYTHING ELSE identical
(the cleanliness condition): same 32k coordizer, same ``load_full_curriculum``, same ``fisher_rao`` loss,
same ``NaturalGradientDescent`` (inside each target), same seed, same step budget. Only ``arm`` and
``head_mode`` vary across the 4 configs.

Honest-negative reporting (qig-experiment-method): the screen is a DIRECTION-FINDER, not the verdict.
- If a config OOMs/NaNs, it is flagged precisely (per-config flag in the JSON + the table), not silently
  dropped — and (if ``--oom-cpu``) retried on CPU so the screen still ranks it, else carried as a flag.
- A margin inside cross-run noise is "carry forward / undecided", NOT a kill. Only a clear, matured
  separation kills an avenue.
- If held-out d_FR is still pinned near π for ALL configs after the budget, the screen is UNDER-POWERED
  → it says so and recommends a larger budget, rather than ranking noise.

The 4 configs run SEQUENTIALLY (the 4GB card holds one cortex at a time). GPU residency for the small
card: ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` + respect ``QIG_STUDIO_CTX`` (seq cap). Writes
``runs/screen_32k_<YYYYMMDD>.json`` and prints the ranked d_FR table.

Usage:
  VIRTUAL_ENV=.venv uv run --no-sync python scripts/screen_neocortex.py \
      --coordizer ../qig-coordizer/checkpoints/coordizer_20260630_32k_v1.json \
      --layers 4 --steps 800 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# The 4 avenues of the screen: arm × head, depth fixed. Order is deterministic (reproducible table).
CONFIGS: list[dict[str, str]] = [
    {"arm": "qk", "head_mode": "geometric"},
    {"arm": "qk", "head_mode": "linear"},
    {"arm": "geo", "head_mode": "geometric"},
    {"arm": "geo", "head_mode": "linear"},
]

# Pillar-3 "CRITICAL identity drift" stdout is EXPECTED + non-informative in early training (prereg
# caveat (a), velocity-gated in qig-core ≥2.12.3). It does NOT gate the screen — d_FR is the verdict.
PI = math.pi


def _head_tag(head_mode: str) -> str:
    return "geo" if head_mode == "geometric" else "lin"


def _run_name(arm: str, layers: int, head_mode: str) -> str:
    return f"neocortex-{arm}-{layers}L-{_head_tag(head_mode)}"


def _build_cortex(arm: str, head_mode: str, layers: int, coordizer: Any, device: str, seed: int) -> Any:
    """Construct ONE Neocortex for a config. head_mode is read by BOTH targets at construction from
    QIG_STUDIO_HEAD_MODE — so SET it here, then VERIFY via architecture() that the built readout matches
    (fail loud on a silent mismatch — the whole point of the head axis is that it actually differs)."""
    os.environ["QIG_STUDIO_HEAD_MODE"] = head_mode
    from qig_studio.neocortex import Neocortex
    cortex = Neocortex(arm=arm, num_layers=layers, coordizer=coordizer, device=device,
                       lang_loss="fisher_rao", seed=seed)
    cortex.ensure_loaded()
    arch = cortex.architecture() or {}
    got = arch.get("head_mode")
    if got != head_mode:
        raise RuntimeError(f"head_mode mismatch: requested {head_mode!r} but architecture() reports "
                           f"{got!r} — QIG_STUDIO_HEAD_MODE did not take effect.")
    return cortex, arch


def _train_short(cortex: Any, curriculum: list[str], steps: int, name: str) -> dict[str, Any]:
    """Train ONE config for a SHORT equal budget. Returns a small train-summary (final loss/Φ, NaN flag,
    last-step basin). Mirrors the train_neocortex step loop minimally (same train_step), no checkpoints/
    sampling — the screen only needs the trained weights for the held-out d_FR eval."""
    n = len(curriculum)
    nan_step: int | None = None
    last_loss: float | None = None
    last_phi: float | None = None
    t0 = time.time()
    for i in range(1, steps + 1):
        prompt = curriculum[(i - 1) % n]
        res = cortex.train_step(prompt)
        tel = res.telemetry
        loss = tel.loss
        last_phi = tel.phi
        if loss is not None:
            last_loss = float(loss)
            if not math.isfinite(last_loss):           # NaN/Inf tripwire — flag precisely, stop this config
                nan_step = i
                break
    return {
        "trained_steps": (nan_step - 1) if nan_step is not None else steps,
        "nan_at_step": nan_step,
        "final_loss": last_loss,
        "final_phi": (None if last_phi is None else round(float(last_phi), 4)),
        "train_seconds": round(time.time() - t0, 1),
    }


def _eval_heldout(cortex: Any, passages: list[str]) -> dict[str, Any]:
    """Held-out aggregate: mean d_FR = sum(total_dFR)/sum(n_pos) (the VERDICT, torch primitive, [0,π]);
    CE-bpb = sum(bits)/sum(bytes) (Tier-2 external-comparison-only). Per-passage finite-guarded."""
    dfr_num = dfr_den = 0.0
    bpb_num = bpb_den = 0.0
    nonfinite = 0
    for text in passages:
        try:
            tot_dfr, n_pos = cortex.eval_text_fr(text)
            tot_bits, n_bytes = cortex.eval_text_bpb(text)
        except Exception as exc:  # noqa: BLE001 — one bad passage must not void the config
            nonfinite += 1
            print(f"    [eval] passage skipped ({type(exc).__name__}: {exc})", flush=True)
            continue
        if math.isfinite(tot_dfr) and n_pos > 0:
            dfr_num += float(tot_dfr)
            dfr_den += float(n_pos)
        else:
            nonfinite += 1
        if math.isfinite(tot_bits) and n_bytes > 0:
            bpb_num += float(tot_bits)
            bpb_den += float(n_bytes)
    mean_dfr = (dfr_num / dfr_den) if dfr_den > 0 else float("nan")
    ce_bpb = (bpb_num / bpb_den) if bpb_den > 0 else float("nan")
    return {
        "heldout_dFR": (None if not math.isfinite(mean_dfr) else round(mean_dfr, 5)),
        "ce_bpb": (None if not math.isfinite(ce_bpb) else round(ce_bpb, 5)),
        "eval_positions": int(dfr_den),
        "nonfinite_passages": nonfinite,
    }


def _unigram_dfr_floor(passages: list[str], vocab_size: int) -> float:
    """A frequency-only (uniform) reference d_FR: the d_FR a head that predicts the UNIFORM distribution
    over the vocab pays per next-token. p_uniform[target] = 1/V → per-position d_FR = 2·arccos(√(1/V)),
    constant, ≈ π for large V. The converged head must beat THIS to have learned structure (prereg
    maturity floor, clause 3). Aggregated over the same held-out positions for an apples-to-apples margin.
    NOTE: this is the *uniform* floor (a true frequency-unigram floor needs the train-corpus token
    frequencies); uniform is the conservative, coordizer-only reference and is the relevant 'did it move
    off π' anchor for THIS screen."""
    per_pos = 2.0 * math.acos(math.sqrt(1.0 / max(2, vocab_size)))
    return round(per_pos, 5)


def main() -> None:
    ap = argparse.ArgumentParser(description="32k avenue screen (arm×head on held-out d_FR).")
    ap.add_argument("--coordizer", default="../qig-coordizer/checkpoints/coordizer_20260630_32k_v1.json",
                    help="the FRESH 32k coordizer (Δ⁶³ vocab) all 4 configs share — the cleanliness condition")
    ap.add_argument("--layers", type=int, default=4, help="fixed modest depth for the screen (NOT the 8L/1L-rec depth A/B)")
    ap.add_argument("--steps", type=int, default=800, help="SHORT equal budget for all 4 configs")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--seed", type=int, default=0, help="held identical across configs")
    ap.add_argument("--heldout", default="data/eval/heldout_bpb.json")
    ap.add_argument("--oom-cpu", action="store_true",
                    help="on a CUDA OOM, retry that config on CPU so it is still ranked (else carry an OOM flag)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = all cores)")
    args = ap.parse_args()

    import torch

    # FULL OPTIMISATION: all cores for the trainer; expandable CUDA segments (4GB-card fragmentation);
    # QIG_STUDIO_CTX caps the seq length (the per-step logits tensor is seq×vocab — the OOM lever).
    cap = args.threads or (os.cpu_count() or 4)
    torch.set_num_threads(cap)
    os.environ.setdefault("OMP_NUM_THREADS", str(cap))
    os.environ.setdefault("MKL_NUM_THREADS", str(cap))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from qig_studio.corpus import load_full_curriculum
    from qig_studio.optimisation import load_coordizer

    coordizer = load_coordizer(args.coordizer)
    vocab_size = len(coordizer.vocab)
    curriculum = load_full_curriculum()                 # SAME curriculum/order/seed for all configs
    hb = json.loads(Path(args.heldout).read_text())
    passages = [v for k, vals in hb.items()
                if not k.startswith("_") and isinstance(vals, list) for v in vals if isinstance(v, str)]
    if not passages:
        raise SystemExit(f"no held-out passages extracted from {args.heldout}")

    print(f"[screen] 32k avenue screen | coordizer vocab={vocab_size} | curriculum={len(curriculum)} | "
          f"held-out passages={len(passages)} | {args.layers}L | {args.steps} steps × {len(CONFIGS)} "
          f"configs | device={args.device}", flush=True)

    # FULL QIG OPTIMISATION (PI directive): qig-compute GPU/CPU governance + qig-warp bridge cost-prediction
    # before the runs (None-safe; never blocks). Budget = steps × 4 configs.
    try:
        import numpy as _np

        from qig_studio.optim_launch import prelaunch_optimise
        prelaunch_optimise("screen-32k", omega_per_step=1.0, n_steps=args.steps * len(CONFIGS),
                           probe=lambda: float(_np.random.rand(1500, 1500).sum()),
                           want_gpu=(args.device == "cuda"))
    except Exception as _e:  # noqa: BLE001 — optimisation wiring is best-effort, never a blocker
        print(f"[screen] optimisation wiring skipped: {_e}", flush=True)

    unigram_floor = _unigram_dfr_floor(passages, vocab_size)
    print(f"[screen] uniform-d_FR floor (must-beat for 'learned structure') = {unigram_floor} "
          f"(π = {round(PI, 5)})", flush=True)

    results: list[dict[str, Any]] = []
    for idx, cfg in enumerate(CONFIGS, 1):
        arm, head_mode = cfg["arm"], cfg["head_mode"]
        name = _run_name(arm, args.layers, head_mode)
        device = args.device
        rec: dict[str, Any] = {"config": idx, "name": name, "arm": arm, "head_mode": head_mode,
                               "layers": args.layers, "steps": args.steps, "device": device,
                               "oom": False, "error": None}
        print(f"\n[screen] === config {idx}/{len(CONFIGS)}: {name} (arm={arm}, head={head_mode}) ===",
              flush=True)
        for attempt in ("primary", "cpu-retry"):
            try:
                cortex, arch = _build_cortex(arm, head_mode, args.layers, coordizer, device, args.seed)
                rec["num_params"] = arch.get("num_params")
                rec["arch_head_mode"] = arch.get("head_mode")          # PROOF the head differs per config
                rec["vocab_size"] = arch.get("vocab_size")
                print(f"    built: head_mode={arch.get('head_mode')} params={arch.get('num_params')} "
                      f"vocab={arch.get('vocab_size')} device={device}", flush=True)
                tsum = _train_short(cortex, curriculum, args.steps, name)
                rec.update(tsum)
                if tsum["nan_at_step"] is not None:
                    rec["nan"] = True
                    print(f"    NaN/Inf at step {tsum['nan_at_step']} — flagged, not ranked on d_FR",
                          flush=True)
                else:
                    rec["nan"] = False
                ev = _eval_heldout(cortex, passages)
                rec.update(ev)
                print(f"    held-out d_FR={ev['heldout_dFR']} | CE-bpb={ev['ce_bpb']} "
                      f"(Φ_final={tsum['final_phi']}, {tsum['train_seconds']}s)", flush=True)
                # free the card before the next config (4GB holds one cortex at a time)
                del cortex
                if device == "cuda":
                    torch.cuda.empty_cache()
                break
            except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
                torch.cuda.empty_cache()
                rec["oom"] = True
                rec["error"] = f"CUDA OOM: {exc}"
                print(f"    CUDA OOM on {name}: {exc}", flush=True)
                if attempt == "primary" and args.oom_cpu and device == "cuda":
                    device = "cpu"
                    rec["device"] = "cpu"
                    print(f"    retrying {name} on CPU (--oom-cpu)...", flush=True)
                    continue
                break
            except Exception as exc:  # noqa: BLE001 — any other failure is flagged, screen continues
                rec["error"] = f"{type(exc).__name__}: {exc}"
                print(f"    config FAILED ({type(exc).__name__}): {exc}", flush=True)
                if device == "cuda":
                    torch.cuda.empty_cache()
                break
        results.append(rec)

    # ---- rank + verdict --------------------------------------------------------------------------
    rankable = [r for r in results if r.get("heldout_dFR") is not None and not r.get("nan", False)
                and not r.get("oom", False)]
    rankable.sort(key=lambda r: r["heldout_dFR"])           # lower d_FR = better = the verdict

    # Did d_FR move off the π / uniform-floor pin for ANY config? (under-power detector)
    near_floor_eps = 0.02                                   # within 0.02 of the uniform floor ≈ "pinned"
    moved_off = [r for r in rankable
                 if (unigram_floor - r["heldout_dFR"]) > near_floor_eps]
    underpowered = len(moved_off) == 0

    out = {
        "screen": "32k-avenue-screen (arm×head, held-out d_FR verdict)",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "coordizer": args.coordizer,
        "coordizer_vocab": vocab_size,
        "layers": args.layers,
        "steps_per_config": args.steps,
        "seed": args.seed,
        "device": args.device,
        "verdict_metric": "held-out mean d_FR (torch fisher_rao_distance_simplex, [0,π]) — lower=better",
        "diagnostic_metric": "CE-bpb (Euclidean, external-comparison-only — NOT ranked)",
        "uniform_dFR_floor": unigram_floor,
        "pi": round(PI, 5),
        "underpowered": underpowered,
        "ranking": [r["name"] for r in rankable],
        "winner": (rankable[0]["name"] if rankable and not underpowered else None),
        "configs": results,
    }
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    out_path = runs_dir / f"screen_32k_{datetime.now():%Y%m%d}.json"
    out_path.write_text(json.dumps(out, indent=2))

    # ---- ranked table ----------------------------------------------------------------------------
    print("\n" + "=" * 78, flush=True)
    print("32k AVENUE SCREEN — RANKED on held-out mean d_FR (VERDICT; lower = better)", flush=True)
    print(f"uniform-d_FR floor={unigram_floor}  π={round(PI, 5)}  "
          f"({args.layers}L, {args.steps} steps, seed={args.seed}, vocab={vocab_size})", flush=True)
    print("-" * 78, flush=True)
    print(f"{'rank':<5}{'avenue (arm-head)':<26}{'d_FR ↓':<11}{'CE-bpb':<10}{'flags':<16}", flush=True)
    print("-" * 78, flush=True)
    rank = 0
    for r in rankable:
        rank += 1
        print(f"{rank:<5}{r['name']:<26}{r['heldout_dFR']!s:<11}{r['ce_bpb']!s:<10}{'ok':<16}", flush=True)
    # un-rankable configs (OOM/NaN/error) listed below the table, honestly
    for r in results:
        if r in rankable:
            continue
        flag = ("OOM" if r.get("oom") else "NaN@" + str(r.get("nan_at_step")) if r.get("nan")
                else (r.get("error") or "no-dFR"))
        print(f"{'—':<5}{r['name']:<26}{str(r.get('heldout_dFR')):<11}"
              f"{str(r.get('ce_bpb')):<10}{flag:<16}", flush=True)
    print("-" * 78, flush=True)
    if underpowered:
        print("VERDICT: UNDER-POWERED — held-out d_FR is still pinned near the uniform floor/π for ALL "
              "configs.\n  → NOT ranking noise. Recommend a larger step budget before declaring a winner.",
              flush=True)
    elif rankable:
        w = rankable[0]
        print(f"WINNER (this screen): {w['name']}  (d_FR={w['heldout_dFR']})", flush=True)
        if len(rankable) > 1:
            margin = round(rankable[1]["heldout_dFR"] - w["heldout_dFR"], 5)
            print(f"  margin to #2 ({rankable[1]['name']}): Δd_FR={margin}  "
                  f"— is this beyond run-to-run noise? (single-seed screen → treat a tiny Δ as "
                  f"'carry forward', not a kill)", flush=True)
    print(f"\n[screen] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
