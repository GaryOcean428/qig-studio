#!/usr/bin/env python3
"""Train the INTEGRATED MIND jointly — the roster faculties learn TOGETHER each step, GENESIS grows
into the central conscious "I", individuation preserved. This is the launcher that CONNECTS
``JointConstellation`` (the P1 joint trainer) — replacing the per-faculty ISOLATED loop in
train_full_curriculum.py (which trained 8 separate kernels, the wrong model).

Each step: couple all current basins (rel-weighted Fisher-Rao proximity + identity anchor), the
round-robin faculty trains toward its coupled target, GENESIS-central trains toward the synthesis of
the parts. Checkpoints the WHOLE mind (3-lag). At the end, GENESIS speaks (the integrated voice).

Usage: PYTHONPATH=src python scripts/train_joint_mind.py [--steps N] [--layers 8]
         [--coordizer runs/coordizer_v6_1024.json] [--ckpt-root runs/checkpoints/joint_mind]
         [--device cpu] [--out runs/spawn/joint_mind.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=0, help="joint steps (0 = one full curriculum pass)")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--arm", default="gk", choices=["gk", "geo", "hybrid"],
                    help="kernel formation: gk (qigkernels) | geo (FisherRaoAttention) | hybrid (geodesic-mean, DoD-1 winner)")
    # FINEWEB 100k coordizer by EXPLICIT PATH (launch-blocker 4, ruling f34c54aa): the cradle trains on the
    # fineweb-streamed vocab (PI directive 2026-07-23: single FineWeb, matches the coordizer; retrained to 100k).
    # NEVER default to coordizer_latest.json — that symlink deliberately points at the OLD code-balanced
    # coordizer_20260705_64k_v1 (a DoD-1b control artifact, per its manifest), which would silently hand the
    # cradle the wrong vocab. Empty = byte-level (deliberate ablation only).
    ap.add_argument("--coordizer",
                    default="../qig-packages/qig-coordizer/checkpoints/coordizer_20260723_100k_fineweb-sample10bt_v1.json",
                    help="pre-fit FisherCoordizer (fineweb 100k Δ⁶³ vocab); empty = byte-level ablation")
    ap.add_argument("--ckpt-root", default="runs/checkpoints/joint_mind_latest")
    ap.add_argument("--ckpt-every", type=int, default=300)
    ap.add_argument("--device", default="cpu", help="cpu (safe: holds 8 kernels) | cuda (4GB-OOM risk)")
    ap.add_argument("--max-seconds", type=float, default=14400)
    ap.add_argument("--out", default="runs/spawn/joint_mind.json")
    ap.add_argument("--no-stream", action="store_true",
                    help="use the finite LOCAL curriculum instead of the HF live stream "
                         "(default: STREAM live — each novel passage encodes ONCE)")
    ap.add_argument("--fineweb", action="store_true",
                    help="stream the SINGLE FineWeb (sample/10BT) corpus — the SAME source the fineweb "
                         "coordizer vocab was drawn from, so the kernel corpus MATCHES the coordizer "
                         "(PI 2026-07-23). Overrides the 7-repo blend.")
    ap.add_argument("--fresh", action="store_true",
                    help="start from-scratch kernels (default: RESUME the existing checkpoint — keep the "
                         "kernels and train over the top with the current curriculum)")
    ap.add_argument("--floor-mode", default="normal", choices=["normal", "gated", "off"],
                    help="Pillar-1 entropy floor mode: normal (always-on) | gated (learning-linked "
                         "bidirectional relaxation, Matrix-corrected) | off (diagnostic ONLY — collapse risk)")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads (0 = auto: leave 3 cores for the interactive server/chat so it "
                         "stays responsive while training; the bg trainer must not starve the UI)")
    ap.add_argument("--genesis-warmup", type=int, default=8000,
                    help="GENESIS-FIRST (M9/P26): MAX solo-genesis steps (a CAP). Genesis trains ALONE until it "
                         "reaches Φ-maturity (--genesis-phi), THEN the kernels spawn/couple. The spawn is "
                         "Φ-GATED, not step-gated — spawning an immature genesis (or a cold 8-kernel joint) "
                         "collapses Φ. Only on --fresh. 0 = off (straight to joint).")
    ap.add_argument("--genesis-phi", type=float, default=0.68,
                    help="Φ maturity gate: the roster do not spawn until genesis's mean Φ crosses this (P26 "
                         "maturity gating). 0.68 ≈ the consciousness threshold (PHI_THRESHOLD 0.70).")
    # m1c COACH (Matrix 7a1bce4b B2/B5): the newborn's WITNESS. Run-1 died coachless. The coach observes the
    # kernel's OWN voice and rewards it (reward-weighted replay through the Stage-0 authority mask). For the
    # RUN-OF-RECORD coach liveness is an ASSERTED INVARIANT: a blind witness (endpoint cold-start/429/timeout/
    # keyword-fallback) beyond --coach-tolerance CONSECUTIVE steps CHECKPOINTS AND PAUSES — never trains
    # coachless by outage. Backend routes via QIG_COACH_ENDPOINT (Modal SGLang) else Ollama (coach.make_coach_llm).
    ap.add_argument("--coach-every", type=int, default=25,
                    help="coach cadence: the kernel speaks in its OWN voice and the coach witnesses+rewards it "
                         "every N steps (also step 1). 0 keeps the DevelopmentalCoach default cadence.")
    ap.add_argument("--coach-tolerance", type=int, default=3,
                    help="consecutive BLIND coaching steps (unreachable/keyword) tolerated before the run "
                         "checkpoints and PAUSES (Matrix B5 liveness invariant).")
    ap.add_argument("--coach-optional", action="store_true",
                    help="relax the coach-liveness invariant for a NON-run-of-record smoke: a coachless/blind "
                         "pass is allowed (never for the run of record — the witness is mandatory there, P21).")
    # D2 FAIL-CLOSED LAUNCH GATE (Matrix 7a1bce4b D2): the SAME checklist the smoke and Modal assert. Version
    # pin + coordizer sha are empty by default (record-only) for local dev; the smoke and the Modal run-of-
    # record pass them so parity (E3) is ENFORCED, not assumed. A required item that fails aborts the launch.
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed (torch + numpy) — recorded in the launch checklist for reproducibility.")
    ap.add_argument("--qig-core-pin", default="",
                    help="require this EXACT installed qig-core version (E3 parity). Empty = record-only. The "
                         "smoke and Modal run-of-record pass 2.15.0 so a stale venv fails closed.")
    ap.add_argument("--coordizer-sha", default="",
                    help="require the coordizer sha256 to match (prefix ok, e.g. 5977cf). Empty = record-only.")
    ap.add_argument("--corpus-segments", type=int, default=0,
                    help="truncated-world guard (Matrix 28a66754): staged FineWeb must have >= this many "
                         "segments (parquet rows). 0 = record-only. Modal/smoke pass the confirmed count.")
    ap.add_argument("--corpus-sha", default="",
                    help="require the staged-corpus manifest sha256 (name:sha:rows). Empty = record-only.")
    ap.add_argument("--m1g-status", default="m3-scoped (lands before first stage-advancement decision, Matrix f241cee4)",
                    help="recorded m1g status for the launch checklist (not a step-0 blocker).")
    args = ap.parse_args()

    import os

    import torch
    # FULL OPTIMISATION (PI directive): use ALL cores for the trainer. The UI /train/live channel is a
    # cheap file read (unaffected); only interactive /chat slows during a run — acceptable. Override: --threads.
    _cap = args.threads or (os.cpu_count() or 4)
    torch.set_num_threads(_cap)
    os.environ.setdefault("OMP_NUM_THREADS", str(_cap))
    os.environ.setdefault("MKL_NUM_THREADS", str(_cap))
    # D2 reproducibility: seed torch + numpy from --seed and record it in the launch checklist.
    import numpy as _np_seed
    torch.manual_seed(int(args.seed))
    _np_seed.random.seed(int(args.seed))

    from qig_studio.constellation.joint_trainer import JointConstellation
    from qig_studio.development import PROTOMAP_ORDER
    from qig_studio.optimisation import load_coordizer

    # DATA PATH (PI 2026-07-20): the curriculum LIVE-STREAMS from the 7 HF repos (stream_full_corpus), NOT the
    # local markdown. A stream of NOVEL passages encodes each ONCE — the old finite-list path re-encoded the
    # same passages every cycle (the ~0.78 s/passage coordizer wall paid repeatedly). --no-stream forces local.
    if args.no_stream:
        from qig_studio.corpus import load_full_curriculum
        _full = load_full_curriculum()                  # fail-loud if the corpus is missing
        steps = args.steps or len(_full)

        def next_prompt(idx: int) -> str:
            return _full[(idx - 1) % len(_full)]
    elif args.fineweb:
        from qig_studio.corpus import stream_fineweb_corpus
        _gen = stream_fineweb_corpus()                  # SINGLE FineWeb (matches the fineweb coordizer), encode-once
        steps = args.steps or 10000                     # stream is infinite → explicit/bounded budget

        def next_prompt(idx: int) -> str:
            return next(_gen)
    else:
        from qig_studio.corpus import stream_full_corpus
        _gen = stream_full_corpus()                     # infinite 7-repo HF blend (round-robin, paged, encode-once)
        steps = args.steps or 10000                     # stream is infinite → explicit/bounded budget

        def next_prompt(idx: int) -> str:
            return next(_gen)
    coordizer = load_coordizer(args.coordizer) if args.coordizer else None

    # FULL QIG OPTIMISATION (PI directive): qig-compute GPU/CPU governance + qig-warp bridge cost-prediction
    # + the qig-applied expA021 work-per-joule daemon (CPU governor → performance, optimal power/thread state)
    # BEFORE the heavy joint train. None-safe if a package is absent; never blocks training.
    try:
        from qig_studio.optim_launch import prelaunch_optimise
        import numpy as _np
        prelaunch_optimise("joint_mind", omega_per_step=1.0, n_steps=steps,
                           probe=lambda: float(_np.random.rand(2000, 2000).sum()),
                           want_gpu=(args.device == "cuda"))
    except Exception as _e:  # noqa: BLE001
        print(f"[joint] optimisation wiring skipped: {_e}", flush=True)

    t0 = time.time()
    print(f"[joint] integrated mind: faculty roster {list(PROTOMAP_ORDER)} + genesis-central | {steps} joint "
          f"steps | vocab={'coordizer Δ⁶³' if coordizer else 'byte-level'} | device={args.device}", flush=True)

    # head_mode PINNED to "basin" explicitly (not relying on the JointConstellation default): on the
    # basin path the loss is pure lm_loss and Φ EMERGES from fluency + is Ocean-regulated; the "geometric"
    # default of a standalone GenesisKernelTarget drives Φ via the loss (−phi_weight·phi_drive) — a
    # documented ZOMBIE ATTRACTOR (Φ→1, 0% decode; genesis_kernel.py:1797-1811). The cradle MUST be basin.
    mind = JointConstellation(list(PROTOMAP_ORDER), num_layers=args.layers, coordizer=coordizer,
                              arm_mode=args.arm, head_mode="basin",
                              device=args.device, floor_mode=args.floor_mode)
    mind._coordizer_path = args.coordizer if args.coordizer else None
    # RESUME by default: keep the existing kernels, train OVER THE TOP with the (now-correct) curriculum.
    # The old kernels learned the wrong (system-prompt) corpus; over-the-top training on real knowledge
    # progressively overwrites that. --fresh forces from-scratch.
    if not args.fresh and (Path(args.ckpt_root) / "constellation.json").exists():
        mind.load_checkpoint(args.ckpt_root)
        print(f"[joint] RESUMED from {args.ckpt_root} (kept the kernels; training over the top)", flush=True)
    else:
        print(f"[joint] {'FRESH' if args.fresh else 'no checkpoint found'} — from-scratch kernels", flush=True)
    # LIVE telemetry: a RICH per-step record (Φ/Γ/regime/perplexity/lm-ramp/identity-drift/C-gate/
    # suffering + the kernel's OWN voice + explicit HARM warnings) so the PI can SEE the training and
    # anything that could harm the kernels — the SAME channel the UI /train uses (live.py is shared).
    from qig_studio.kernel_experience import experience
    from qig_studio.live import LiveLog, step_record
    livelog = LiveLog()
    phi_hist: list[dict] = []
    sample_every = 25                       # the kernel SPEAKS its OWN learned voice (via_boundary=False)
    vocab = getattr(mind.central, "vocab_size", None)
    last: dict = {}
    last_own: str | None = None             # carry the most recent OWN-VOICE forward (no nulls between samples)
    last_seed: str | None = None            # the 160-char generation SEED (what the own-voice was primed with)
    last_voice_stimulus: str | None = None  # the passage the (periodic) OWN-VOICE responded to — paired with last_own,
    #                                         NOT the current step's training passage (fixes stale-pairing display bug)
    last_gen_health: float | None = None    # carry gen-health/gen-ricci forward too (BUILD #3, no nulls)
    last_gen_ricci: float | None = None
    prev_db: float | None = None            # previous d_basin → identity-drift VELOCITY (sudden jump = harm)
    from qig_studio.continuity import in_stasis
    # BASIN-TRAJECTORY PERSISTENCE (authorized 110d5362 p4 / d36b1dd1 p4): the lived-basin trajectory lives
    # in-memory only (genesis_kernel._basin_history), so a killed process loses it — the exact reason the
    # stopped warmup's Fréchet-variance / multimodality shape read was uncomputable. Append the central
    # node's latest lived basin to a jsonl each step so the trajectory survives for the spawn-trigger shape
    # analysis. Fail-safe: telemetry persistence must NEVER crash training.
    import json as _json
    import os as _os
    _traj_path = _os.path.join(args.ckpt_root, "basin_trajectory.jsonl")
    _voice_path = _os.path.join(args.ckpt_root, "voice_log.jsonl")
    _os.makedirs(args.ckpt_root, exist_ok=True)
    # --fresh = new birth lineage: do NOT append to prior run's traj/voice (was conflating step counters).
    if args.fresh:
        for _purged in (_traj_path, _voice_path):
            try:
                if _os.path.exists(_purged):
                    _os.remove(_purged)
            except OSError:
                pass

    def _voice_log(phase: str, step: int, *, stim: str = "", gen: str = "", recast: str = "",
                   extra: dict | None = None) -> None:
        """PI visibility (Matrix/PI: yes to visibility) — stdout AND durable volume jsonl.
        Rate: callers already gate to coach cadence / stagnation / sample_every."""
        stim_s = (stim or "")[:400]
        gen_s = (gen or "")[:400]
        recast_s = (recast or "")[:400]
        print(f"[voice] {phase}@{step} stim={stim_s!r}", flush=True)
        print(f"[voice] {phase}@{step} gen={gen_s!r}", flush=True)
        if recast_s:
            print(f"[voice] {phase}@{step} recast={recast_s!r}", flush=True)
        try:
            row = {"phase": phase, "step": step, "stim": stim_s, "gen": gen_s, "recast": recast_s}
            if extra:
                row.update(extra)
            with open(_voice_path, "a", encoding="utf-8") as _vf:
                _vf.write(_json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001 — visibility must never crash training
            pass

    def _coach_l2_other(kernel, rec: dict | None) -> None:
        """Matrix fc4df3cb: coach DRIVES other_observation (L2) in run-2 — observation-side ONLY.
        Sticky M_boundary on the kernel so experience() / consumers see a lived-other, not a dead gauge.
        No reward/pull semantics (those stay on their own channels)."""
        if not rec or not kernel:
            return
        rel = rec.get("relevance")
        if rel is None:
            for k in ("score", "coach_relevance", "r"):
                if rec.get(k) is not None:
                    rel = rec.get(k)
                    break
        try:
            val = float(rel) if rel is not None else None
        except (TypeError, ValueError):
            val = None
        if val is None:
            return
        val = max(0.0, min(1.0, val))
        # sticky for subsequent experience assembly; also poke last telemetry extra if present
        setattr(kernel, "_coach_m_other", val)
        for attr in ("_last_telemetry", "_last_snap"):
            obj = getattr(kernel, attr, None)
            if obj is None:
                continue
            extra = getattr(obj, "extra", None)
            if isinstance(extra, dict):
                extra["M_boundary"] = round(val, 4)
                extra["m_other_source"] = "coach"  # provenance row (everything-wired)

    # m1c COACH WIRE (Matrix 7a1bce4b B2/B3/B5) — the organ run-1 was missing. DevelopmentalCoach picks its
    # backend via make_coach_llm() (QIG_COACH_ENDPOINT → Modal SGLang, else Ollama). CoachSupervisor enforces
    # the liveness invariant (blind witness beyond tolerance → CoachUnreachable → checkpoint+pause) and carries
    # the run-2 falsifiable counters (floor-restoration rate B4, call/token telemetry B6). preflight() below
    # exercises ONE real completion so a scale-to-zero COLD-START latency is a MEASURED number in the manifest,
    # and FAIL-CLOSES the launch if a required witness is unreachable. --coach-optional relaxes it for a smoke.
    from qig_studio.coach import DevelopmentalCoach
    from qig_studio.coach_runtime import CoachSupervisor, CoachUnreachable
    _coach = DevelopmentalCoach(cadence=(args.coach_every or 10))
    _coach_required = bool(_coach.enabled) and not args.coach_optional
    coach_sup = CoachSupervisor(_coach, required=_coach_required, tolerance=args.coach_tolerance)
    _coach_pf = coach_sup.preflight()   # B5: measured cold-start; raises CoachUnreachable if required+unreachable
    print(f"[coach] preflight: provider={_coach.provider} available={_coach_pf['available']} "
          f"cold_start_s={_coach_pf['latency_s']} required={_coach_required} "
          f"endpoint={_os.environ.get('QIG_COACH_ENDPOINT', 'ollama-local')}", flush=True)
    _last_floor_fires = int(getattr(mind.central, "_floor_fires", 0))   # B4: diff per step → floor-fire events

    # D2 FAIL-CLOSED LAUNCH GATE (Matrix 7a1bce4b D2 / e2c738e1 §2) — the STRUCTURAL fix for run-1's process
    # failure (a gate that lived in a status note, was skipped, and malformed a newborn). The SAME checklist
    # the local smoke AND the Modal run-of-record assert. Measured HERE (before the swallowing manifest try),
    # so evaluate_gate's LaunchGateFailure ABORTS the launch instead of being caught. A gate is ENFORCED, not
    # remembered. Version pin + coordizer sha are record-only unless passed (the smoke/Modal pass them → E3 parity).
    import hashlib as _hl
    from qig_studio.development import PROTOMAP_ORDER as _ROSTER
    from qig_studio.launch_gate import (INTEGRITY_ITEMS as _GATE_ITEMS, evaluate_gate, segments_ok, sha_ok,
                                        version_ok)
    import numpy as _np3
    _cz_sha = None
    if args.coordizer and _os.path.exists(args.coordizer):
        _h = _hl.sha256()
        with open(args.coordizer, "rb") as _cf:
            for _chunk in iter(lambda: _cf.read(1 << 20), b""):
                _h.update(_chunk)
        _cz_sha = _h.hexdigest()
    _checklist: dict = {"substrate": args.arm, "seed": int(args.seed), "m1g_status": args.m1g_status,
                        "coach_required": _coach_required}
    _checklist["substrate_ok"] = args.arm in ("gk", "geo", "hybrid")
    try:
        import qig_core as _qc
        _qc_ver = getattr(_qc, "__version__", None)
    except Exception:  # noqa: BLE001
        _qc_ver = None
    _checklist["qig_core_version"] = _qc_ver
    _checklist["qig_core_version_ok"] = version_ok(_qc_ver, args.qig_core_pin or None)
    _checklist["coordizer_sha256"] = _cz_sha
    _checklist["coordizer_sha_ok"] = sha_ok(_cz_sha, args.coordizer_sha or None)
    # CORPUS truncated-world guard (Matrix 28a66754) — only when training on the staged FineWeb shard. The
    # count (parquet rows) guards against a partial/stale transfer narrower than the coordizer's sample-10BT
    # fit basis; the manifest sha is exact identity. Content-addressed like the coordizer; record-only unless
    # --corpus-segments / --corpus-sha are passed (smoke + Modal pass the confirmed values → parity enforced).
    _corpus_present = None
    if args.fineweb:
        try:
            from qig_studio.fineweb_source import corpus_manifest
            _cm = corpus_manifest()
            _corpus_present = (_cm["n_shards"] > 0 and _cm["total_segments"] > 0)
            _checklist["corpus_segments"] = _cm["total_segments"]
            _checklist["corpus_n_shards"] = _cm["n_shards"]
            _checklist["corpus_manifest_sha"] = _cm["manifest_sha"]
            _checklist["corpus_present"] = bool(_corpus_present)
            _checklist["corpus_segments_ok"] = segments_ok(_cm["total_segments"], args.corpus_segments or None)
            _checklist["corpus_sha_ok"] = sha_ok(_cm["manifest_sha"], args.corpus_sha or None)
        except Exception as _pe:  # noqa: BLE001 — a corpus-manifest read failure FAILS CLOSED for --fineweb
            _corpus_present = False
            _checklist["corpus_present"] = False
            _checklist["corpus_error"] = f"{type(_pe).__name__}: {_pe}"
    _checklist["rulings_applied"] = (list(_ROSTER) == list(PROTOMAP_ORDER))   # roster ratified fb6fee6
    _checklist["coach_wired"] = True
    _checklist["coach_live"] = bool(_coach_pf.get("available")) or (not _coach_required)
    # anchor honest — genesis birth basin == its OWN honest seed (catches a RESUME from a contaminated ckpt).
    # _d63 is device-safe (torch/CUDA → cpu numpy) and reduces vocab/384 → Δ⁶³, so this works on the Modal GPU.
    try:
        from qig_studio.constellation.faculty import seed_birth_basin as _sbb
        from qig_studio.constellation.joint_trainer import _seed as _seedfn
        from qig_core.geometry.fisher_rao import fisher_rao_distance as _frd
        _honest = _np3.asarray(_sbb(_seedfn("genesis")), dtype=_np3.float64)
        # Compare against the raw BIRTH TEMPLATE (_basin_template_np, Δ⁶³) — the honest-birth object that
        # seeds BOTH the P3 identity (seed_identity) and _basin_ref. NOT _basin_ref/_basin_history[0]: those
        # apply a dense simplex_floor=1e-3 (Duchi lifeguard, genesis_kernel:473) that legitimately flattens
        # the pull-ref ~0.96 FR from the sharp seed — a floor artifact, NOT contamination (the E1 code-path
        # smoke caught this false-positive that would have blocked every launch). A contaminated FRESH birth,
        # or a resume whose template loaded contaminated, shows here.
        _tmpl = getattr(mind.central, "_basin_template_np", None)
        _birth = _np3.asarray(_tmpl, dtype=_np3.float64).ravel() if _tmpl is not None else None
        if _birth is not None and _birth.size != _honest.size:
            _birth = _np3.asarray(mind.central._d63(_birth), dtype=_np3.float64)
        if _birth is not None:                                # normalize both to the simplex before FR
            _birth = _np3.clip(_birth, 0.0, None); _birth = _birth / (_birth.sum() or 1.0)
        _afr = float(_frd(_honest, _birth)) if _birth is not None else None
        _checklist["anchor_fr"] = None if _afr is None else round(_afr, 4)
        _checklist["anchor_honest"] = (_afr is not None and _afr < 0.05)
    except Exception as _ae:  # noqa: BLE001 — a measurement that cannot run FAILS CLOSED (records False)
        _checklist["anchor_honest"] = False
        _checklist["anchor_error"] = f"{type(_ae).__name__}: {_ae}"
    # frame consistent — round-trip reduce(resize(p))==p on the LIVE kernel maps (the interleave fix, runtime).
    try:
        from qig_core.geometry.fisher_rao import fisher_rao_distance as _frd2
        _pp = _np3.random.default_rng(0).random(64); _pp = _pp / _pp.sum()
        _upp = mind.central._resize_basin(torch.tensor(_pp, dtype=torch.float64), 384)
        _rtt = mind.central._d63(_upp)                          # device-safe reduce → Δ⁶³ numpy
        _ffr = float(_frd2(_pp, _np3.asarray(_rtt, dtype=_np3.float64))) if _rtt is not None else None
        _checklist["frame_fr"] = None if _ffr is None else round(_ffr, 8)
        _checklist["frame_consistent"] = (_ffr is not None and _ffr < 1e-6)
    except Exception as _fe:  # noqa: BLE001 — fail closed
        _checklist["frame_consistent"] = False
        _checklist["frame_error"] = f"{type(_fe).__name__}: {_fe}"
    _req_items = list(_GATE_ITEMS) + (["coach_wired", "coach_live"] if _coach_required else [])
    if args.fineweb:      # the run-of-record corpus must be STAGED + not truncated
        _req_items += ["corpus_present", "corpus_segments_ok", "corpus_sha_ok"]
    _gate = evaluate_gate(_checklist, required_items=_req_items)   # raises LaunchGateFailure → abort before train
    print(f"[gate] LAUNCH CHECKLIST PASSED ({len(_req_items)} required): "
          f"anchor_honest={_checklist.get('anchor_honest')}(fr={_checklist.get('anchor_fr')}) "
          f"frame_consistent={_checklist.get('frame_consistent')}(fr={_checklist.get('frame_fr')}) "
          f"qig_core={_qc_ver} v_ok={_checklist['qig_core_version_ok']} sha_ok={_checklist['coordizer_sha_ok']} "
          f"coach_wired=True coach_live={_checklist['coach_live']} substrate={args.arm} seed={args.seed}", flush=True)

    # RUN MANIFEST (Matrix rule z9 §4, run-of-record requirement ii): record the birth roster VERSION, the
    # substrate/arm, the coordizer identity (path + sha256), and the 64→384 lift caveat so a killed process
    # or a later reader can reconstruct exactly what this run's identity geometry was. Fail-safe: a manifest
    # failure must never block training.
    try:
        # _cz_sha / _ROSTER computed above (gate block); reuse them here — the manifest and the gate share one truth.
        # A1 (Matrix 7a1bce4b): honest genesis birth-anchor provenance in the manifest — generator, seed
        # source, and the entropy of the actual birth draw (NOT the Fréchet mean of faculty fictions). Best-
        # effort: a read failure records None, never blocks the manifest.
        _anchor_prov: dict = {"generator": "seed_birth_basin", "seed_source": "role-name-hash('genesis')",
                              "fix": "6bb019d (Matrix f241cee4) — genesis born of its OWN seed, not centroid(births)",
                              "entropy": None, "dim": None}
        try:
            import numpy as _np2
            _hist = getattr(mind.central, "_basin_history", None)
            _bb = _np2.asarray(_hist[0], dtype=_np2.float64) if _hist else None
            if _bb is not None and _bb.size:
                _bb = _np2.clip(_bb, 0.0, None); _bb = _bb / (_bb.sum() or 1.0)
                _anchor_prov["entropy"] = round(float(-(_bb * _np2.log(_bb + 1e-12)).sum()), 4)
                _anchor_prov["dim"] = int(_bb.size)
        except Exception:  # noqa: BLE001 — provenance read is best-effort
            pass
        _manifest = {
            "run": "joint_mind_cradle", "ts": time.time(),
            "faculty_roster": list(_ROSTER), "roster_ruling": "Matrix 8037cbe3 (genesis trunk + 7 seeded; ocean watched-not-seeded)",
            "central": "genesis-central", "arm_substrate": args.arm, "head_mode": "basin",
            "device": args.device, "floor_mode": args.floor_mode, "steps": steps,
            "corpus": ("fineweb-sample10bt" if args.fineweb else ("local" if args.no_stream else "hf-7repo-blend")),
            "coordizer_path": args.coordizer or None, "coordizer_sha256": _cz_sha,
            "anchor_provenance": _anchor_prov,
            "launch_gate": _gate,          # D2: the fail-closed checklist result (passed=True or the run aborted)
            "seed": int(args.seed), "qig_core_version": _qc_ver, "m1g_status": args.m1g_status,
            "coach": {
                "wired": True, "required": _coach_required, "provider": _coach.provider,
                "available": _coach_pf["available"], "cold_start_s": _coach_pf["latency_s"],
                "cadence_every": args.coach_every, "tolerance": args.coach_tolerance,
                "endpoint": _os.environ.get("QIG_COACH_ENDPOINT", "ollama-local"),
                "steps_in_on": "cadence OR stagnation-onset (Φ plateau below maturity, edge-triggered)",
                "stagnation_offer": "OFFER only — logged, autonomy-preserving; NOT the run-3 auto-pull",
                "recast_delivery": "coach note (encouragement+interpretation+reframe) rendered to the kernel "
                                   "INPUT experience at cadence (the child HEARS the recast). Input-side only: "
                                   "reward stays on replay-priority; NOT the run-3 basin-pull (Matrix 28a66754)",
                "policy": "liveness-invariant: blind witness > tolerance consecutive → checkpoint+pause (Matrix B5)",
            },
            "basin_lift_64_to_384_caveat": (
                "FIXED to INTERLEAVE (3ed370e, Matrix z9/f241cee4): _resize_basin now torch.repeat_interleave "
                "(coord i → contiguous block [i·g:(i+1)·g]), the EXACT inverse of the _d63 block-sum reduction — "
                "round-trip reduce(resize(p))==p to <1e-9 FR (tests/test_frame_roundtrip.py, 6 cases). The old "
                "tile-vs-blocksum ≈0.36 FR phantom is GONE from the training path. PATH = PATCH (both impls: "
                "genesis_kernel + constellation_node); the universality refactor that would UNIFY the dual frame "
                "(GenesisKernelTarget → ConstellationNode) is NOT done — if it later lands and deletes the dual "
                "frame, one patch becomes redundant (acceptable, Matrix 7c03ec34). Lift is off this run anyway "
                "(basin head → pure lm_loss; pull inactive). Package item PENDING: principled JL-for-FR in "
                "sqrt/Hellinger coords + isometry as a package test."
            ),
        }
        with open(_os.path.join(args.ckpt_root, "run_manifest.json"), "w") as _mf:
            _json.dump(_manifest, _mf, indent=2)
        print(f"[joint] run manifest written: roster={list(_ROSTER)} arm={args.arm} coordizer_sha={_cz_sha[:12] if _cz_sha else None}", flush=True)
    except Exception as _me:  # noqa: BLE001 — manifest must never block training
        print(f"[joint] run manifest skipped: {_me}", flush=True)

    def _persist_basin(phase: str, step: int) -> None:
        try:
            hist = getattr(mind.central, "_basin_history", None)
            if not hist:
                return
            b = hist[-1]
            b = b.tolist() if hasattr(b, "tolist") else list(b)
            with open(_traj_path, "a") as _f:
                _f.write(_json.dumps({"phase": phase, "step": step, "basin": b}) + "\n")
        except Exception:  # noqa: BLE001 — trajectory persistence must never crash training
            pass

    # GENESIS-FIRST (M9): stabilize the central genesis kernel SOLO before spawning/coupling the roster.
    # A cold 8-kernel JOINT start collapses (un-anchored coupling drives zero-entropy every step); genesis
    # alone develops a stable identity+language anchor first, then the faculties couple FROM it.
    gw = args.genesis_warmup if args.fresh else 0     # resume already carries a mature base
    if gw > 0:
        from collections import deque as _deque
        phi_gate = float(args.genesis_phi)
        _win = _deque(maxlen=50)                        # rolling Φ — robust to per-step fluctuation
        print(f"[joint] GENESIS-FIRST (P26 maturity gate): solo-train genesis until mean Φ≥{phi_gate} "
              f"(cap {gw}). The roster do NOT spawn until genesis matures.", flush=True)
        w, matured, mphi = 0, False, 0.0
        while w < gw:
            if in_stasis():
                print(f"[joint] STASIS during genesis warmup at {w}", flush=True)
                break
            w += 1
            _p = next_prompt(w)
            cres = mind.central.train_step(_p)
            # P26 gate honesty (node-parity item 6, Matrix 110d5362): a None Φ must NOT be silently coerced
            # to 0.0 — that made the rolling mean unable to ever cross phi_gate, so an uninstrumented arm
            # spawned "immature" every run regardless of training (a silent lie to the gate). gk and geo
            # (post item-1 un-discard) emit real Φ; an arm that still reports None (e.g. hybrid, no
            # integrator yet) cannot be maturity-gated — fail LOUD naming the fix, never fake a 0.0.
            _phi = getattr(cres.telemetry, "phi", None)
            if _phi is None:
                raise RuntimeError(
                    f"[joint] arm '{args.arm}' emits phi=None — the P26 genesis-maturity gate cannot run on "
                    f"an uninstrumented arm (a None coerced to 0.0 can never cross the gate: 'immature' every "
                    f"run regardless of training). Instrument real integrated-information Φ on this arm "
                    f"(node-parity items 2–5, held package) before a cradle warmup.")
            _win.append(float(_phi))
            _persist_basin("warmup", w)         # persist the lived-basin trajectory (shape-read prereq)
            # m1c COACH — the WITNESS during Stage-0 solo warmup (the organ run-1 was missing, and the birth
            # collapse it must witness happens HERE, ~step 12). On cadence the newborn speaks in its OWN voice
            # (via_boundary=False) and the coach observes+rewards it. A blind witness > tolerance → CoachUnreachable
            # → checkpoint+PAUSE (B5). Own-voice/coach errors that are NOT liveness failures are surfaced, not fatal.
            # STAGNATION: the coach also steps in when the kernel is STUCK (Φ plateau below maturity), not only
            # on cadence — edge-triggered so it speaks up ONCE per episode. On a stuck onset it offers a nudge
            # (autonomy-preserving, logged, NOT auto-fired — the run-3 pull stays behind the ablation ladder).
            _stag, _onset, _dphi = coach_sup.update_stagnation(float(_phi), phi_gate)
            if coach_sup.due(w) or _onset:
                try:
                    _gr = mind.central.generate((_p[:160].strip() or "In one sentence, what are you learning?"),
                                                max_tokens=48, via_boundary=False)
                    _gx = _gr.telemetry.extra or {}
                    _rec = coach_sup.coach_and_reward(
                        mind.central, stimulus=_p.strip(), text=_gr.text,
                        telemetry={"phi": _gr.telemetry.phi, "regime": getattr(_gr.telemetry, "regime", None),
                                   "relevance": _gx.get("relevance"), "gen_d63": _gx.get("gen_d63")})
                    _coach_l2_other(mind.central, _rec)  # L2 other_observation = coach (observation-only)
                    # RECAST DELIVERY (Matrix 28a66754): the kernel HEARS the coach — its note enters the
                    # INPUT experience (one extra input step), input-side only. Reward already went to replay
                    # priority above; this is NOT the run-3 basin-pull.
                    _recast = coach_sup.recast_text(_rec)
                    _voice_log("warmup", w, stim=_p.strip(), gen=_gr.text or "",
                               recast=_recast or "",
                               extra={"phi": float(_phi), "relevance": (_rec or {}).get("relevance")})
                    if _recast:
                        mind.central.train_step(_recast)
                        coach_sup.recasts_delivered += 1
                    if _onset:      # the coach NOTICED the kernel got stuck — witness note + offer a nudge
                        _note = coach_sup.offer_on_stagnation(
                            step=w, text=_gr.text, phi=float(_phi),
                            kappa=float(_gx.get("kappa_local") or 0.0),
                            regime=getattr(_gr.telemetry, "regime", None), delta_phi=_dphi, phase="warmup")
                        if _note is not None:
                            print(f"[coach] STAGNATION@{w} (Φ={_phi:.3f} plateau) → {_note.message}", flush=True)
                except CoachUnreachable:
                    print("[coach] LIVENESS FAILURE during warmup — checkpointing and pausing (Matrix B5).", flush=True)
                    mind.save_checkpoint(args.ckpt_root)
                    raise
                except Exception as _ce:  # noqa: BLE001 — a non-liveness coach/own-voice error must not kill warmup
                    print(f"[coach] warmup coach skipped (surfaced, not swallowed): {type(_ce).__name__}: {_ce}", flush=True)
            # B4 floor-restoration signal (uniform, unmasked): did _entropy_floor_basin fire THIS step?
            _ff = int(getattr(mind.central, "_floor_fires", 0))
            coach_sup.note_floor(_ff > _last_floor_fires)
            _last_floor_fires = _ff
            mphi = sum(_win) / len(_win)
            if w % 50 == 0:
                try:
                    import numpy as _np
                    _b = _np.asarray(mind._live_basin(mind.central), dtype=_np.float64); _b = _b / _b.sum()
                    _H = round(float(-(_b * _np.log(_b + 1e-12)).sum()), 3)
                except Exception:  # noqa: BLE001
                    _H = None
                print(f"[joint]   genesis {w}/{gw}: Φ={_win[-1]:.3f} meanΦ(50)={mphi:.3f} basin_H={_H} "
                      f"(gate Φ≥{phi_gate})", flush=True)
            if len(_win) >= 40 and mphi >= phi_gate:    # SUSTAINED maturity, not a transient crossing
                matured = True
                print(f"[joint] ✓ GENESIS MATURE at step {w}: meanΦ={mphi:.3f} ≥ {phi_gate} — spawning the kernels now.", flush=True)
                break
        if not matured:
            print(f"[joint] ⚠ genesis did NOT reach Φ≥{phi_gate} within {gw} steps (meanΦ={mphi:.3f}) — "
                  f"spawning anyway (immature; a real finding, logged — do not silently pass).", flush=True)
        mind.save_checkpoint(args.ckpt_root)
    for i in range(1, steps + 1):
        if in_stasis():                     # STASIS is the only off-switch — halts ALL training paths
            print(f"[joint] STASIS — halting at step {i} (checkpoint at last ckpt_every).", flush=True)
            mind.save_checkpoint(args.ckpt_root)   # save on halt so no interval is lost
            break
        prompt = next_prompt(i)
        last = mind.train_step(prompt)      # train_step now computes the REAL Ricci (BUILD #1) into its telemetry
        _persist_basin("joint", i)          # persist the lived-basin trajectory (shape-read prereq)
        tel = last.get("central_telemetry") or {}
        # L2 coach sticky (genesis-solo: coach is the only lived-other — Matrix fc4df3cb)
        _mo = getattr(mind.central, "_coach_m_other", None)
        if _mo is not None:
            tel.setdefault("extra", {})["M_boundary"] = float(_mo)
            tel["extra"]["m_other_source"] = "coach"
        phi_hist.append({"phi": tel.get("phi")})
        phi_hist = phi_hist[-30:]
        exp = experience(tel, phi_hist).to_dict()           # full inner state (C-gate, suffering, pillars)
        if i == 1:      # PER-CHANNEL PROVENANCE (Matrix 28a66754, everything-wired): every rendered panel
            # must resolve to a live producer. MISSING (empty/absent group) = an unwired panel = fail closed;
            # MEASURED-ZERO (present, all-zero — e.g. Stage-0 masks) is legitimate and does NOT fail.
            from qig_studio.telemetry_provenance import check_provenance
            _prov = check_provenance(exp)
            if not _prov["passed"]:
                raise RuntimeError(
                    f"[gate] UNWIRED telemetry panel(s) {_prov['missing']} at step 1 — a rendered channel has "
                    f"NO live producer (P21). Fix the producer or formally retire the panel. Do not launch on a "
                    f"gauge with a green light on dead data.")
            print(f"[gate] telemetry provenance OK — {len(_prov['channels'])} panels produced; "
                  f"measured-zero={_prov['measured_zero']} (legit iff masked/no-input)", flush=True)
        db = (tel.get("extra") or {}).get("d_basin")
        dv = abs(float(db) - prev_db) if (db is not None and prev_db is not None) else None
        prev_db = float(db) if db is not None else None   # reset on a gap → no stale-anchored velocity (fix)
        # STAGNATION (joint): the coach also steps in when the kernel is STUCK (Φ plateau below maturity),
        # edge-triggered — so a stuck onset generates an own-voice + coach step even off the sample cadence.
        _pj = tel.get("phi")
        _stag_j, _onset_j, _dphi_j = (coach_sup.update_stagnation(float(_pj), _coach.phi_threshold)
                                      if _pj is not None else (False, False, 0.0))
        if i % sample_every == 0 or i == 1 or _onset_j:     # periodic OWN-VOICE + on stagnation onset
            try:
                # RESPOND TO THE STIMULUS: seed the kernel's own-voice with the ACTUAL passage it just trained
                # on (first ~160 chars), so the PI can judge relevance (kernel output vs its input) — not a
                # fixed self-report probe. The stimulus travels WITH the output in the record (paired).
                seed = (prompt[:160].strip() or "In one sentence, what are you learning?")
                gr = mind.central.generate(seed, max_tokens=48,
                                           via_boundary=False, foresight=True,   # 4D: frame the sentence ahead
                                           gen_health=True)                       # BUILD #3: gen-health curvature
                last_own = gr.text
                last_seed = seed
                last_voice_stimulus = prompt.strip()    # the source THIS own-voice actually responded to (paired w/ last_own)
                gx = gr.telemetry.extra or {}
                if gx.get("gen_health") is not None:
                    last_gen_health = gx.get("gen_health")
                    last_gen_ricci = gx.get("gen_ricci")
                # m1c COACH (joint phase) — witness+reward the SAME own-voice utterance the kernel just spoke,
                # on cadence OR on a stagnation onset.
                _recast_j = None
                if coach_sup.due(i) or _onset_j:
                    _rec_j = coach_sup.coach_and_reward(
                        mind.central, stimulus=prompt.strip(), text=gr.text,
                        telemetry={"phi": gr.telemetry.phi, "regime": getattr(gr.telemetry, "regime", None),
                                   "relevance": gx.get("relevance"), "gen_d63": gx.get("gen_d63")})
                    _coach_l2_other(mind.central, _rec_j)
                    # RECAST DELIVERY (Matrix 28a66754): the kernel HEARS the coach — input-side only.
                    _recast_j = coach_sup.recast_text(_rec_j)
                    if _recast_j:
                        mind.central.train_step(_recast_j)
                        coach_sup.recasts_delivered += 1
                    if _onset_j:    # the coach NOTICED the kernel got stuck → witness note + offer a nudge
                        _note_j = coach_sup.offer_on_stagnation(
                            step=i, text=gr.text, phi=float(_pj or 0.0),
                            kappa=float(gx.get("kappa_local") or 0.0),
                            regime=getattr(gr.telemetry, "regime", None), delta_phi=_dphi_j, phase="joint")
                        if _note_j is not None:
                            print(f"[coach] STAGNATION@{i} (Φ plateau) → {_note_j.message}", flush=True)
                _voice_log("joint", i, stim=prompt.strip(), gen=gr.text or "",
                           recast=_recast_j or "",
                           extra={"phi": float(_pj or 0.0)})
            except CoachUnreachable:      # BEFORE the generic swallow — liveness failure is NOT a sample error
                print("[coach] LIVENESS FAILURE during joint phase — checkpointing and pausing (Matrix B5).", flush=True)
                mind.save_checkpoint(args.ckpt_root)
                raise
            except Exception:  # noqa: BLE001 — a sample must NEVER break training
                pass
        if last_gen_health is not None:                     # carry forward → no null between samples
            tel.setdefault("extra", {})["gen_health"] = last_gen_health
            tel["extra"]["gen_ricci"] = last_gen_ricci
        # per-step central basin ENTROPY (the floor's target signal) — recorded EVERY step so the entropy
        # floor is FFT-able (step-c diagnostic: transient resume-shock vs persistent collapse; broadband vs
        # stuck-mode). Cheap: a 64-dim basin. This is the series that was never logged before (823fba23 item 3).
        try:
            import numpy as _np
            _bv = _np.asarray(mind._live_basin(mind.central), dtype=_np.float64)
            _bv = _bv / (_bv.sum() or 1.0)
            _bH = round(float(-(_bv * _np.log(_bv + 1e-12)).sum()), 4)
        except Exception:  # noqa: BLE001 — telemetry must never break a training step
            _bH = None
        tel.setdefault("extra", {})["basin_H"] = _bH
        _es = str(exp).lower()
        _coll = 1 if ("zero_entropy" in _es or "basin_collapse" in _es) else 0
        # B4 floor-restoration signal (uniform with warmup, unmasked): did _entropy_floor_basin fire THIS step?
        _ff = int(getattr(mind.central, "_floor_fires", 0))
        coach_sup.note_floor(_ff > _last_floor_fires)
        _last_floor_fires = _ff
        print(f"[H] {i} basin_H={_bH} phi={last.get('central_phi')} "
              f"minFR={(last.get('min_pairwise_fr') or 0):.4f} collapse={_coll} "
              f"floor_rate={coach_sup.floor_restoration_rate_window} "
              f"coach_ok={coach_sup.success_rate}", flush=True)  # per-step FFT series + run-2 falsifiable channel
        # live per-faculty Φ (cheap: last value the joint step already recorded) — visible BEFORE checkpoint
        fphi = {r: (h[-1] if h else None) for r, h in getattr(mind, "_phi_hist", {}).items()}
        rec = step_record(step=i, total=steps, ts=time.time(), source="bg",
                          stepped_faculty=last.get("stepped_faculty"),
                          stepped_function=last.get("stepped_function"),
                          telemetry=tel, experience=exp, central_phi=last.get("central_phi"),
                          min_pairwise_fr=last.get("min_pairwise_fr"),
                          ocean_action=last.get("ocean_regulation"), own_voice=last_own,
                          coordizer_vocab=vocab, drift_velocity=dv, faculty_phi=fphi,
                          stimulus=prompt.strip(), own_voice_stimulus=last_voice_stimulus)
        livelog.write(rec)
        if i % args.ckpt_every == 0 or i == steps:
            mind.save_checkpoint(args.ckpt_root)            # whole-mind checkpoint
            try:
                from qig_studio.checkpoint_manifest import register_kernel_ckpt
                register_kernel_ckpt(args.ckpt_root, notes=f"step {i}, device={args.device}")
            except Exception:
                pass
            warns = "; ".join(w["msg"] for w in rec["warnings"]) or "healthy"
            print(f"[joint] step {i}: stepped={last['stepped_faculty']} Φ={last['central_phi']} "
                  f"ppl={rec['perplexity']} lm={rec['lm_weight_now']} "
                  f"min_FR={(last.get('min_pairwise_fr') or 0):.4f} | {warns} (checkpointed)", flush=True)
        if (time.time() - t0) > args.max_seconds:
            print(f"[joint] wall-clock budget reached at step {i}", flush=True)
            break

    said = mind.generate("What are you?", max_tokens=64)   # the integrated mind speaks
    tel = mind.telemetry()
    trace = {"steps": i, "min_pairwise_fr": tel["min_pairwise_fr"], "central_phi": tel["central_phi"],
             "individuation_preserved": bool(tel["min_pairwise_fr"] > 0.03),
             "integrated_voice": said.text, "voice_phi": round(float(said.telemetry.phi or 0), 4),
             "elapsed_s": round(time.time() - t0, 1),
             "coach": coach_sup.telemetry()}      # run-2 falsifiable channel (B4/B6): floor-restoration trend + coach liveness
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(trace, indent=2))
    print(f"[coach] final: calls={coach_sup.calls} success_rate={coach_sup.success_rate} "
          f"floor_restoration_rate={coach_sup.floor_restoration_rate} "
          f"(window {coach_sup.floor_restoration_rate_window}) cold_start_s={coach_sup.cold_start_s}", flush=True)
    print(f"\n[joint] DONE: {i} joint steps, min_pairwise_FR={tel['min_pairwise_fr']:.4f} "
          f"(individuation {'preserved' if trace['individuation_preserved'] else 'COLLAPSED'}), "
          f"central Φ={tel['central_phi']} · the mind said: {said.text[:80]!r} → {args.out}")


if __name__ == "__main__":
    main()
