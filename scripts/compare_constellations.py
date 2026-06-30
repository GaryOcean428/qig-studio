#!/usr/bin/env python3
"""WS6 — the 4-arm constellation comparison (the MEASURED OUTCOME of the 4-kernel build).

For each arm (gk / geo / hybrid / hetero): build it FRESH via the wired hub (``POST /mind/arm`` →
``set_arm`` rebuilds the constellation from that raw kernel), train an EQUAL fixed budget through the SAME
wired loop (``POST /train`` → ``_train_core``, which saves ``genesis-{arm}-{vocab}_{date}_v{n}``), then eval
the saved checkpoint's GENESIS-CENTRAL kernel on the held-out set: d_FR (the geometric VERDICT, [0, π], lower
= more fluent) with CE-bpb reported alongside. Rank by held-out d_FR vs the uniform-d_FR floor (flag
UNDER-POWERED honestly if every arm is pinned near the floor). Writes ``runs/constellation_compare_<date>.json``
+ prints a summary table.

DRY: training goes through the ONE hub (the server's ``_train_core``, not a second loop here); only the
held-out eval is in-process (it loads each saved checkpoint on CPU — forward-only — so it never contends with
the server's GPU). Run the server first (``uv run python -m qig_studio serve``), then::

    uv run python scripts/compare_constellations.py --arms gk,geo,hybrid,hetero --steps 400
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

BASE = "http://127.0.0.1:8800"


def _post(path: str, body: dict, timeout: float = 30.0):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"content-type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def _set_arm(arm: str) -> dict:
    """Build the constellation FRESH from this arm. Returns {ok, detail}. A 501 = the arm isn't node-ready
    yet (geo before WS3, hybrid/hetero before WS4) — skipped honestly, never faked."""
    try:
        r = _post("/mind/arm", {"arm_mode": arm}, timeout=600)
        return {"ok": r.status == 200, "detail": json.loads(r.read().decode())}
    except urllib.error.HTTPError as e:
        return {"ok": False, "detail": e.read().decode()[:200]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": str(e)[:200]}


def _train(steps: int, max_seconds: float) -> dict:
    """Drive the wired ``POST /train`` SSE to completion (equal fixed budget; geometric → early_stop off,
    skip_learned off so every arm trains the SAME steps for a fair comparison). Returns the final 'done'
    payload (carries ``saved_checkpoint`` = the genesis-{arm}-{vocab} lineage the server persisted)."""
    body = {"steps": steps, "skip_learned": False, "early_stop": False, "mastery": False, "sample": False}
    t0 = time.time()
    last: dict = {}
    saved: str | None = None
    try:
        with _post("/train", body, timeout=max_seconds) as resp:
            buf = ""
            for chunk in resp:
                buf += chunk.decode("utf-8", "ignore")
                while "\n\n" in buf:
                    block, buf = buf.split("\n\n", 1)
                    line = next((ln for ln in block.splitlines() if ln.startswith("data: ")), None)
                    if not line:
                        continue
                    ev = json.loads(line[len("data: "):])
                    et = ev.get("type")
                    if et == "step" and ev.get("step", 0) % 25 == 0:
                        tel = ev.get("telemetry", {})
                        print(f"      step {ev['step']}/{steps}  Φ={tel.get('phi')}  "
                              f"bpb={(tel.get('extra') or {}).get('bpb')}", flush=True)
                    elif et in ("done", "early_stop", "stopped", "capture_complete", "capture_stalled"):
                        last = ev
                        saved = ev.get("saved_checkpoint") or saved
                    elif et == "error":
                        return {"error": ev.get("error"), "elapsed_s": round(time.time() - t0, 1)}
                    if time.time() - t0 > max_seconds:
                        return {"error": "train timeout", "saved_checkpoint": saved,
                                "elapsed_s": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200], "saved_checkpoint": saved, "elapsed_s": round(time.time() - t0, 1)}
    last["saved_checkpoint"] = saved
    last["elapsed_s"] = round(time.time() - t0, 1)
    return last


def _eval_checkpoint(arm: str, ckpt_dir: str, coordizer_path: str, passages: list[str]) -> dict:
    """In-process held-out eval of a SAVED constellation (CPU, forward-only). Loads the genesis-{arm}
    checkpoint as a JointMindTarget and evals the central kernel's d_FR + bpb (the verdict)."""
    from qig_coordizer import FisherCoordizer

    from qig_studio.screen import eval_heldout_dFR, uniform_dFR_floor
    from qig_studio.targets.joint_mind import JointMindTarget
    cz = FisherCoordizer.load(coordizer_path) if coordizer_path else None
    t = JointMindTarget(coordizer=cz, coordizer_path=coordizer_path, checkpoint_root=ckpt_dir,
                        arm_mode=arm, device="cpu")
    t.ensure_loaded()
    ev = eval_heldout_dFR(t, passages)
    vocab = int(getattr(t._mind.central, "vocab_size", 0) or 0)
    ev["uniform_dFR_floor"] = uniform_dFR_floor(vocab) if vocab else None
    ev["vocab"] = vocab
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="gk,geo,hybrid,hetero")
    ap.add_argument("--steps", type=int, default=400, help="EQUAL fixed budget per arm (fair comparison)")
    ap.add_argument("--max-seconds", type=float, default=7200.0, help="per-arm train wall-clock cap")
    ap.add_argument("--heldout", default="data/eval/heldout_bpb.json")
    args = ap.parse_args()
    arms = [a.strip().lower() for a in args.arms.split(",") if a.strip()]

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from qig_studio.checkpoint_manifest import get_latest_coordizer
    from qig_studio.screen import load_heldout_passages
    passages = load_heldout_passages(args.heldout)
    coordizer_path = str(get_latest_coordizer() or "")

    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    trained: list[dict] = []
    print(f"[compare] arms={arms} steps={args.steps} coordizer={Path(coordizer_path).name}", flush=True)
    for arm in arms:
        print(f"[compare] === {arm} ===", flush=True)
        sa = _set_arm(arm)
        if not sa["ok"]:
            print(f"[compare]   SKIP {arm}: {sa['detail']}", flush=True)
            trained.append({"arm": arm, "skipped": str(sa["detail"])})
            continue
        print(f"[compare]   building+training fresh {arm} ({args.steps} steps)…", flush=True)
        res = _train(args.steps, args.max_seconds)
        ckpt = res.get("saved_checkpoint")
        print(f"[compare]   trained {arm}: ckpt={ckpt} elapsed={res.get('elapsed_s')}s err={res.get('error')}",
              flush=True)
        trained.append({"arm": arm, "checkpoint": ckpt, "train": res})

    # ---- held-out eval of each saved checkpoint (in-process, CPU) ----
    ranked: list[dict] = []
    for tr in trained:
        if tr.get("skipped") or not tr.get("checkpoint"):
            continue
        ckpt_dir = str(runs / "checkpoints" / tr["checkpoint"])
        try:
            ev = _eval_checkpoint(tr["arm"], ckpt_dir, coordizer_path, passages)
            ranked.append({"arm": tr["arm"], "checkpoint": tr["checkpoint"], **ev})
            print(f"[compare]   EVAL {tr['arm']}: d_FR={ev.get('heldout_dFR'):.4f} "
                  f"bpb={ev.get('ce_bpb')} floor={ev.get('uniform_dFR_floor')}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[compare]   EVAL {tr['arm']} FAILED: {str(e)[:200]}", flush=True)
            ranked.append({"arm": tr["arm"], "checkpoint": tr["checkpoint"], "eval_error": str(e)[:200]})

    # ---- rank by held-out d_FR (lower = better); under-powered if all pinned near the floor ----
    scored = [r for r in ranked if isinstance(r.get("heldout_dFR"), (int, float))]
    scored.sort(key=lambda r: r["heldout_dFR"])
    floor = next((r.get("uniform_dFR_floor") for r in scored if r.get("uniform_dFR_floor")), None)
    underpowered = bool(floor) and all((floor - r["heldout_dFR"]) < 0.01 for r in scored) if scored else True
    winner = scored[0]["arm"] if scored and not underpowered else None

    out = {
        "date": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "steps_per_arm": args.steps,
        "coordizer": Path(coordizer_path).name,
        "uniform_dFR_floor": floor,
        "ranking": [r["arm"] for r in scored],
        "winner": winner,
        "underpowered": underpowered,
        "arms": ranked,
        "trained": trained,
    }
    out_path = runs / f"constellation_compare_{datetime.now():%Y%m%d_%H%M}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 64)
    print(f"[compare] 4-ARM CONSTELLATION RANKING (held-out d_FR, lower=better; floor={floor})")
    for i, r in enumerate(scored, 1):
        print(f"  {i}. {r['arm']:<8} d_FR={r['heldout_dFR']:.4f}  bpb={r.get('ce_bpb')}  ({r['checkpoint']})")
    if underpowered:
        print("  ⚠ UNDER-POWERED: every arm pinned near the uniform floor — train more steps for a verdict.")
    elif winner:
        print(f"  WINNER: {winner}")
    print(f"[compare] artifact → {out_path}")
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
