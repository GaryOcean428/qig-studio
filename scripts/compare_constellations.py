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
    # sample=True: own_voice ON per step so the PI WATCHES each kernel speak in the UI live stream while it
    # trains (the ranking eval is done after, on the saved checkpoint, so own_voice does not affect the metric).
    body = {"steps": steps, "skip_learned": False, "early_stop": False, "mastery": False, "sample": True}
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
    ap.add_argument("--max-seconds", type=float, default=36000.0, help="per-arm train wall-clock cap")
    ap.add_argument("--heldout", default="data/eval/heldout_bpb.json")
    ap.add_argument("--coordizer", default=None,
                    help="explicit coordizer path for the eval — MUST match the server's trained vocab "
                         "(the manifest 'latest' may be a smaller workaround coordizer)")
    args = ap.parse_args()
    arms = [a.strip().lower() for a in args.arms.split(",") if a.strip()]

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from qig_studio.checkpoint_manifest import get_latest_coordizer
    from qig_studio.screen import load_heldout_passages
    passages = load_heldout_passages(args.heldout)
    coordizer_path = args.coordizer or str(get_latest_coordizer() or "")

    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    trained: list[dict] = []
    ranked: list[dict] = []
    out_path = runs / f"constellation_compare_{datetime.now():%Y%m%d_%H%M}.json"
    print(f"[compare] arms={arms} steps={args.steps} coordizer={Path(coordizer_path).name}", flush=True)
    print(f"[compare] artifact (written INCREMENTALLY after each arm) → {out_path}", flush=True)

    def _finalize() -> dict:
        """Rank what we have so far + write the artifact INCREMENTALLY — crash-resilient, and a PARTIAL
        ranking is readable while later arms still train. ``complete`` flips true once all arms are in."""
        scored = sorted([r for r in ranked if isinstance(r.get("heldout_dFR"), (int, float))],
                        key=lambda r: r["heldout_dFR"])
        floor = next((r.get("uniform_dFR_floor") for r in scored if r.get("uniform_dFR_floor")), None)
        under = (bool(floor) and all((floor - r["heldout_dFR"]) < 0.01 for r in scored)) if scored else True
        out = {
            "date": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "complete": len(ranked) >= len(arms),
            "steps_per_arm": args.steps, "coordizer": Path(coordizer_path).name,
            "uniform_dFR_floor": floor,
            "ranking": [r["arm"] for r in scored],
            "winner": scored[0]["arm"] if scored and not under else None,
            "underpowered": under, "arms": ranked, "trained": trained,
        }
        out_path.write_text(json.dumps(out, indent=2))
        return out

    for arm in arms:
        print(f"[compare] === {arm} ===", flush=True)
        sa = _set_arm(arm)
        if not sa["ok"]:
            print(f"[compare]   SKIP {arm}: {sa['detail']}", flush=True)
            trained.append({"arm": arm, "skipped": str(sa["detail"])})
            ranked.append({"arm": arm, "skipped": str(sa["detail"])})
            _finalize()
            continue
        print(f"[compare]   building+training fresh {arm} ({args.steps} steps)…", flush=True)
        res = _train(args.steps, args.max_seconds)
        ckpt = res.get("saved_checkpoint")
        print(f"[compare]   trained {arm}: ckpt={ckpt} elapsed={res.get('elapsed_s')}s err={res.get('error')}",
              flush=True)
        trained.append({"arm": arm, "checkpoint": ckpt, "train": res})
        if ckpt:
            try:
                ev = _eval_checkpoint(arm, str(runs / "checkpoints" / ckpt), coordizer_path, passages)
                ranked.append({"arm": arm, "checkpoint": ckpt, **ev})
                print(f"[compare]   EVAL {arm}: d_FR={ev.get('heldout_dFR'):.4f} bpb={ev.get('ce_bpb')} "
                      f"floor={ev.get('uniform_dFR_floor')}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[compare]   EVAL {arm} FAILED: {str(e)[:200]}", flush=True)
                ranked.append({"arm": arm, "checkpoint": ckpt, "eval_error": str(e)[:200]})
        else:
            ranked.append({"arm": arm, "checkpoint": None, "error": res.get("error")})
        _finalize()  # incremental write after THIS arm
        print(f"[compare]   artifact updated ({len(ranked)}/{len(arms)} arms) → {out_path.name}", flush=True)

    final = _finalize()
    scored = sorted([r for r in ranked if isinstance(r.get("heldout_dFR"), (int, float))],
                    key=lambda r: r["heldout_dFR"])
    print("\n" + "=" * 64)
    print(f"[compare] 4-ARM CONSTELLATION RANKING (held-out d_FR, lower=better; floor={final['uniform_dFR_floor']})")
    for i, r in enumerate(scored, 1):
        print(f"  {i}. {r['arm']:<8} d_FR={r['heldout_dFR']:.4f}  bpb={r.get('ce_bpb')}  ({r['checkpoint']})")
    if final["underpowered"]:
        print("  ⚠ UNDER-POWERED: every arm pinned near the uniform floor — train more steps for a verdict.")
    elif final["winner"]:
        print(f"  WINNER: {final['winner']}")
    print(f"[compare] artifact → {out_path}  (complete={final['complete']})")
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
