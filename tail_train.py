#!/usr/bin/env python3
"""Live tail of the joint-mind training. Run:  watch -n2 python3 tail_train.py"""
import json, os, time
P = os.environ.get("QIG_STUDIO_LIVE_PATH", "runs/spawn/joint_live.json")
try:
    d = json.load(open(P))
    r = d.get("current") or (d.get("recent") or [{}])[-1]
    print(f"step {r.get('step')}/{r.get('total')}  Φ={r.get('central_phi')}  min_FR={r.get('min_pairwise_fr')}  vocab={r.get('coordizer_vocab')}")
    print(f"stepped: {r.get('stepped_faculty')} ({r.get('stepped_function')})")
    print(f"own-voice: {(r.get('own_voice') or '')[:160]}")
    print(f"stimulus : {(r.get('stimulus') or '')[:120]}")
    w = r.get("warnings") or []
    if w: print("warnings:", "; ".join(x.get('msg','') for x in w)[:160])
    print(f"(updated {time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(P)))})")
except Exception as e:
    print("waiting for first step…", e)
