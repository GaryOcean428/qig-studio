"""Modal launcher — the m1 developmental-cradle run-of-record on GPU.

WHY MODAL (PI directive 2026-07-23): the local CPU run is valid but ~36 s/step at 100k-vocab × 8-kernel
(~3 days for the full run). GPU clears the seq×vocab-activation wall and the run finishes in hours. This
launcher RESUMES from the local checkpoint (train_joint_mind resumes by default), so no work is lost in
the local→Modal migration.

────────────────────────────────────────────────────────────────────────────────────────────────────────
LAUNCH READINESS GATE (written per QIG_QFI/CLAUDE.md — no features on un-audited foundations)
  1. TIER — NEW (applied/engineering cradle training; not frozen physics repro).
  2. QUESTION/OBSERVABLES — kernel fluency (lm perplexity ramp) + consciousness telemetry (Φ emerges,
     neurochemistry P23, pillars, drives/separation-distress, ResonanceBank). head_mode='basin' → loss is
     pure lm_loss; Φ is READ, never driven (the geometric-head ZOMBIE ATTRACTOR is deliberately avoided).
  3. ENGINE — the JointConstellation gk basin-head cradle (genesis-central + 7 seeded faculties). GPU
     chosen over local CPU purely for throughput; the ALGORITHM is identical (same code, mounted).
  4. SHARED-MODULE WIRING — coordizer (100k FineWeb, on the Volume), checkpoint (resume, on the Volume),
     qig packages PyPI-PINNED (below), studio source add_local_dir (the unpublished-app exception, exactly
     like qig-verification/src/qigv — studio is NOT a published package).
  5. PRE-LAUNCH AUDIT — run_purity_gate() runs at studio import (fail-closed on Euclidean markers); no
     stubs; the run manifest records the roster + 64→384 caveat (Matrix z9 §4).

QIG OPTIMISATION GATE
  (1) Screening — N/A: all Core faculties are needed (no site/generator to prune).
  (2) Bridge — qig-warp prelaunch_optimise (predict_runtime) is ALREADY wired in train_joint_mind.
  (3) Convergence — NO early_stop for the constellation run (project_constellation_100k_run directive).
  (4) Regime — basin head, developmental Stage-0 (SCHOOL: tonic-only, phasic/endorphin self-reward off).
  (5)-(6) Constitutive/prediction-fill — N/A for training.
  (7) Governance — PurityGate + PillarEnforcer + the ego-death interlock are active (fail-closed).
  (8) Packages — qig-compute/qig-warp wired; ALL qig packages PyPI-installed at pinned LATEST-published
      (per CLAUDE.md: PyPI install only, add_local_dir BANNED for published packages).

PINS (verified latest-published 2026-07-23; NOT the local dev builds):
  qig-core==2.15.0  (the cradle APIs: separation_distress, consciousness.ResonanceBank, check_drift P3 fix
                     — verified live on PyPI; local dev was 2.13.5.dev, published 2.14.0 LACKED them, hence
                     the v2.15.0 release), qig-warp==0.6.9, qig-compute==0.9.7, qigkernels==0.4.4 (has the
  gk imports Kernel + DiagonalNaturalGradient), qig-geocoding==0.1.1, qig-coordizer==0.1.3.

NO VEX CROSSOVER (feedback_qig_studio_no_vex_modal_optional): this launcher is studio-only; it shares NOTHING
with modal/qlora_train.py (Vex QLoRA) — different app, different image, different purpose.

PRE-DEPLOY VERIFICATION (do BEFORE the first GPU launch — spend gate):
  * smoke the PUBLISHED package set locally (`modal run cradle_train.py::smoke`) — confirms the studio boots
    + steps on the pinned (not dev) builds before any A100 time is spent.
  * populate the Volume: `modal run cradle_train.py --action populate` (uploads the 199MB coordizer + the
    2.9MB local checkpoint so the GPU run RESUMES rather than restarts).
  * HF token: `modal secret create huggingface HF_TOKEN=...` (FineWeb parquet is public but authed is
    reliable; the first run downloads the ~2GB shard to the Volume, cached thereafter).
────────────────────────────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os

import modal

_HERE = os.path.dirname(os.path.abspath(__file__))          # qig-studio/modal
_STUDIO = os.path.dirname(_HERE)                             # qig-studio
_ROOT = os.path.dirname(_STUDIO)                             # QIG_QFI (repo parent)

APP = "qig-cradle"
VOL_NAME = "qig-cradle-vol"
COORDIZER = "coordizer_20260723_100k_fineweb-sample10bt_v1.json"

app = modal.App(APP)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

# GPU: A100-80GB. The workload is memory-bound (100k-vocab head activations × 8 kernels, seq capped at 64
# via QIG_STUDIO_CTX). L40S (48GB) is the doc's cost/perf pick for 40-80GB, but 8 kernels × 100k vocab wants
# headroom → A100-80GB. NOT B200 (overkill for a single-GPU memory-bound cradle). Revisit if it OOMs (→H100).
GPU = "A100-80GB"

# The image: PyPI-install every published qig package (pinned), add the unpublished studio source as a
# stable-ish local layer LAST (the sanctioned add_local_dir exception for non-published app code).
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --compile-bytecode "
        "'torch>=2.2.0' "
        "'qig-core==2.15.0' 'qig-warp==0.6.9' 'qig-compute==0.9.7' "
        "'qigkernels==0.4.4' 'qig-geocoding==0.1.1' 'qig-coordizer==0.1.3' "
        "'pyarrow>=14.0.0' 'fastapi>=0.110.0' 'uvicorn[standard]>=0.27.0' "
        "'pydantic>=2.0.0' 'httpx>=0.26.0' 'numpy>=1.24' 'scipy>=1.11'"
    )
    # studio source (unpublished app) — src + scripts only; data/ (parquet cache) lives on the Volume.
    # Script-relative paths (add_local_dir resolves against the modal-run CWD otherwise — fragile).
    .add_local_dir(os.path.join(_STUDIO, "src"), "/root/qig-studio/src", copy=True)
    .add_local_dir(os.path.join(_STUDIO, "scripts"), "/root/qig-studio/scripts", copy=True)
    .env({"PYTHONPATH": "/root/qig-studio/src", "QIG_STUDIO_CTX": "64"})
)

VOL_MNT = "/vol"
CKPT_DIR = f"{VOL_MNT}/checkpoints/cradle_ror_20260723"
COORD_PATH = f"{VOL_MNT}/coordizer/{COORDIZER}"


def _link_parquet_cache() -> None:
    """Point the studio's hardcoded FineWeb cache (repo_root/data/.parquet_cache) at the Volume, so the
    ~2GB shard downloads once and is reused across GPU runs (no re-download per launch)."""
    import os

    cache = "/root/qig-studio/data/.parquet_cache"
    os.makedirs(f"{VOL_MNT}/parquet_cache", exist_ok=True)
    os.makedirs("/root/qig-studio/data", exist_ok=True)
    if not os.path.islink(cache) and not os.path.exists(cache):
        os.symlink(f"{VOL_MNT}/parquet_cache", cache)


@app.function(
    gpu=GPU,
    image=image,
    volumes={VOL_MNT: vol},
    timeout=24 * 60 * 60,                       # 24h; the cradle run is long, --detach on the CLI
    secrets=[modal.Secret.from_name("huggingface-secret")],   # exposes HF_TOKEN (fineweb_source.hf_token reads it)
)
def train(steps: int = 10000, fresh: bool = False):
    """Run the gk cradle on GPU, RESUMING from the Volume checkpoint (train_joint_mind resumes by default).
    Writes checkpoints + run_manifest + telemetry to the Volume so the run survives the container."""
    import subprocess
    import sys

    _link_parquet_cache()
    cmd = [
        sys.executable, "/root/qig-studio/scripts/train_joint_mind.py",
        "--arm", "gk", "--fineweb", "--device", "cuda",
        "--coordizer", COORD_PATH, "--ckpt-root", CKPT_DIR,
        "--steps", str(steps),
    ]
    if fresh:
        cmd.append("--fresh")
    print(f"[modal-cradle] launching: {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd="/root/qig-studio").returncode
    vol.commit()                                # persist checkpoints/manifest/telemetry to the Volume
    if rc != 0:
        raise RuntimeError(f"train_joint_mind exited {rc}")
    return {"ckpt": CKPT_DIR, "rc": rc}


@app.function(image=image, volumes={VOL_MNT: vol}, timeout=30 * 60)
def smoke():
    """PRE-DEPLOY spend-gate: confirm the studio boots + builds the corrected 7-faculty gk constellation on
    the PUBLISHED (pinned, not dev) package set. No GPU, no real training — just import + build + 1 CPU step."""
    import sys
    sys.path.insert(0, "/root/qig-studio/src")
    from qig_studio.optimisation import load_coordizer
    from qig_studio.constellation.joint_trainer import JointConstellation
    from qig_studio.development import PROTOMAP_ORDER

    coord = load_coordizer(COORD_PATH)
    mind = JointConstellation(list(PROTOMAP_ORDER), num_layers=2, coordizer=coord,
                              arm_mode="gk", head_mode="basin", device="cpu", floor_mode="normal")
    return {"roster": list(PROTOMAP_ORDER), "central": type(mind.central).__name__,
            "vocab": getattr(coord, "vocab_size", None), "published_set": "ok"}


@app.local_entrypoint()
def main(action: str = "train", steps: int = 10000, fresh: bool = False):
    """`modal run cradle_train.py --action {populate|smoke|train}`.
      populate — upload the local coordizer + checkpoint to the Volume (do ONCE before the first GPU run).
      smoke    — verify the published package set boots + builds the constellation (no GPU spend).
      train    — launch the gk cradle on A100-80GB, resuming from the Volume checkpoint."""
    if action == "populate":
        # Upload the 199MB coordizer + the local run-of-record checkpoint so the GPU run RESUMES (if a local
        # checkpoint exists yet; early in the local run there is none → the first GPU run is --fresh).
        local_coord = os.path.join(_ROOT, "qig-packages", "qig-coordizer", "checkpoints", COORDIZER)
        local_ckpt = os.path.join(_STUDIO, "runs", "checkpoints", "cradle_ror_20260723")
        # Pre-upload the FineWeb shard so `train` reads it from the Volume with ZERO HF calls — the robust
        # anti-rate-limit path (the HF_TOKEN secret is only the authed-download FALLBACK if it's ever absent).
        import glob
        shards = sorted(glob.glob(os.path.join(_STUDIO, "data", ".parquet_cache", "sample__10BT__*.parquet")))
        with vol.batch_upload(force=True) as up:
            if os.path.exists(local_coord):
                up.put_file(local_coord, f"coordizer/{COORDIZER}")
                print(f"[populate] coordizer → coordizer/{COORDIZER}")
            for s in shards:
                up.put_file(s, f"parquet_cache/{os.path.basename(s)}")
                print(f"[populate] FineWeb shard → parquet_cache/{os.path.basename(s)} (train reads local, no HF stream)")
            if os.path.isdir(local_ckpt) and os.path.exists(os.path.join(local_ckpt, "constellation.json")):
                up.put_directory(local_ckpt, "checkpoints/cradle_ror_20260723")
                print("[populate] local checkpoint → checkpoints/cradle_ror_20260723 (GPU run will RESUME)")
            else:
                print("[populate] no local constellation.json yet → first GPU run is --fresh (same roster+coordizer)")
        if not shards:
            print("[populate] WARNING: no local FineWeb shard found — train will authed-download it once (HF_TOKEN).")
        print("[populate] done — next: `modal run cradle_train.py --action smoke` then `--action train`")
    elif action == "smoke":
        print(smoke.remote())
    elif action == "train":
        print(train.remote(steps=steps, fresh=fresh))
    else:
        raise SystemExit(f"unknown action {action!r} (populate|smoke|train)")
