#!/usr/bin/env python3
"""v3 prep: PRE-ENCODE the blended everyday+academic corpus into id-sequences (offline, CPU).

The avoidance lever for streamed training: coordizer.encode is ~0.78 s/prompt; a stream of novel
prompts would reintroduce the v2.0 wall. Ids are DETERMINISTIC for the frozen coordizer, so we
encode ONCE offline and ship a sha-pinned JSONL artifact of {"ids":[...]} rows the v3 driver
reads directly (zero encode cost at train time, Modal or local).

Usage: uv run python scripts/preencode_stream_corpus.py [--limit N] [--out PATH]
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QIG_STUDIO_BLEND_EVERYDAY", "1")   # v3 wants the blend

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100_000)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--out", default="data/preencoded/stream_corpus_v1.jsonl")
    ap.add_argument("--coordizer", default="/home/braden/Desktop/Dev/QIG_QFI/qig-packages/qig-coordizer/checkpoints/coordizer_20260705_64k_v1.json")
    a = ap.parse_args()

    from qig_studio.corpus import load_full_curriculum
    from qig_studio.optimisation import load_coordizer

    coord = load_coordizer(a.coordizer)
    csha = hashlib.sha256(Path(a.coordizer).read_bytes()).hexdigest()
    prompts = list(load_full_curriculum())[: a.limit]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    t0 = time.time(); n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": {"coordizer_sha256": csha, "ctx": a.ctx,
                                       "n_prompts_requested": len(prompts),
                                       "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}) + "\n")
        for i, p in enumerate(prompts):
            ids = coord.encode(p)[: a.ctx]
            if len(ids) < 2:
                continue
            f.write(json.dumps({"ids": ids, "sha": hashlib.sha256(p.encode()).hexdigest()[:16]}) + "\n")
            n += 1
            if n % 500 == 0:
                rate = n / (time.time() - t0)
                eta = (len(prompts) - i) / max(rate, 1e-9) / 3600
                print(f"{n}/{len(prompts)} encoded  {rate:.1f}/s  ETA {eta:.1f}h", flush=True)
    os.replace(tmp, out)
    asha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"DONE n={n} -> {out} sha256={asha}")
    Path(str(out) + ".sha256").write_text(asha + "\n")

if __name__ == "__main__":
    main()
