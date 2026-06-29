"""qig-studio FastAPI core.

Reuses the vex ``kernel/server.py`` shell pattern: ``asynccontextmanager``
lifespan, ``run_purity_gate()`` fail-closed preflight, ``PillarEnforcer`` hook,
and SSE ``start``/``chunk``/``step``/``done`` events via ``StreamingResponse``.

Endpoints (Phase 1):
- ``GET  /health``                 — liveness + purity status + active target
- ``GET  /targets``                — list targets (name, loss_regime, available)
- ``POST /targets/{name}/select``  — set active target (lazy load on first use)
- ``GET  /telemetry``              — current telemetry snapshot
- ``GET  /curriculum``             — curriculum mode for the active target's regime
- ``POST /chat``                   — inference (no learning step)
- ``POST /chat/stream``  (SSE)     — inference, streamed
- ``POST /train``        (SSE)     — N learning steps (curriculum-driven), streamed

The full qig_chat protocol surface (sleep/dream/mushroom/twin/lightning/14-stage/
basin-sync/4D/foresight/reasoning) lands in Phase 4.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .coach import DevelopmentalCoach, OllamaLLM
from .config import Settings
from .kernel_experience import experience as _experience
from .live import LiveLog, step_record
from .curriculum import CurriculumProvider, phase_names
from .mastery import Mastery
from .governance.pillars import PillarEnforcerAdapter
from . import protocol as _protocol
from .governance.purity import PurityGateError, run_purity_gate
from .targets.base import LossRegime, ProtocolUnsupported
from .targets.registry import default_registry

# qig-warp convergence early-stop (package optimisation lever; None-safe if absent) — audit-wired.
try:
    from qig_warp import check_ci_stabilized

    _WARP_AVAILABLE = True
except Exception:  # pragma: no cover - optional dep
    check_ci_stabilized = None  # type: ignore
    _WARP_AVAILABLE = False

settings = Settings.from_env()


def _sse(data: dict[str, Any]) -> str:
    """Format a Server-Sent Event (vex server.py:2095 pattern)."""
    return f"data: {json.dumps(data)}\n\n"


def _now() -> float:
    import time
    return time.time()


# Serialize all target-touching calls — targets (kernel/constellation) hold mutable
# state (model, optimizer, telemetry) that is NOT safe under concurrent requests
# (council red-team #6). One lock; fine for the single-user v1, removes the race.
_TARGET_LOCK = asyncio.Lock()

# Rolling Φ history so inner-state derivatives (phi_trend, variance → motivators, senses) MOVE across
# conversation turns instead of sitting static (a single telemetry snapshot has no trend).
from collections import deque  # noqa: E402

_PHI_HISTORY: deque[float] = deque(maxlen=30)


def _phi_hist() -> list[dict]:
    return [{"phi": p} for p in _PHI_HISTORY]


def _record_turn(kind: str, prompt: str, payload: dict[str, Any]) -> None:
    """HARD-WIRED output: append every conversation/inference turn to runs/sessions/transcript.jsonl —
    prompt + the kernel's OWN voice + the fluent surface + full telemetry/experience. The mind's record;
    None-safe (never breaks a reply if the disk write fails)."""
    try:
        out = Path("runs/sessions")
        out.mkdir(parents=True, exist_ok=True)
        x = (payload.get("telemetry") or {}).get("extra") or {}
        rec = {
            "kind": kind, "prompt": prompt,
            "fluent": payload.get("text"),
            "kernel_voice": x.get("kernel_voice"),
            "qwen_thinking": x.get("qwen_thinking") or None,
            "telemetry": payload.get("telemetry"),
            "experience": payload.get("experience"),
        }
        with (out / "transcript.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001 — a record-write failure must never break the reply
        pass


async def _run_target(fn, *args):
    """Run a (sync, possibly torch) target call off the event loop, serialized."""
    async with _TARGET_LOCK:
        return await asyncio.get_event_loop().run_in_executor(None, lambda: fn(*args))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Fail-closed purity preflight over qig-studio's OWN source.
    pkg_root = Path(__file__).resolve().parent
    try:
        run_purity_gate(pkg_root)
        app.state.purity = "PASSED"
    except PurityGateError as exc:
        app.state.purity = f"FAILED: {exc}"
        raise
    app.state.registry = default_registry(
        default_target=settings.default_target,
        kernel_checkpoint=settings.kernel_checkpoint,
        constellation_checkpoint=settings.constellation_checkpoint,
        genesis_num_layers=settings.genesis_num_layers,
        genesis_coordizer_checkpoint=settings.genesis_coordizer_checkpoint,
        genesis_kernel_checkpoint=settings.genesis_kernel_checkpoint,
        device=settings.device,
    )
    app.state.pillars = PillarEnforcerAdapter()
    app.state.auth_key = settings.auth_key
    app.state.settings = settings  # mutable: /config can nominate output_dir/curriculum_dir at runtime
    # Fail-closed (council P0-F2): refuse a non-loopback bind without a shared secret.
    if not settings.is_loopback and not settings.auth_key:
        raise RuntimeError(
            f"refusing to bind non-loopback host '{settings.host}' without QIG_STUDIO_KEY "
            "(fail-closed auth — council red-team P0-F2)"
        )
    # BOOT WARMUP: eagerly load + warm the active target off the event loop so the FIRST /telemetry is
    # LIVE (Φ/κ/regime/pillars populated), not a misleading step-0 zero state. None-safe: a target whose
    # backend is absent just stays cold (degrades, never blocks boot).
    async def _warm() -> None:
        t = app.state.registry.active
        if t is not None and t.is_available():
            try:
                await asyncio.get_event_loop().run_in_executor(None, t.ensure_loaded)
            except Exception:  # noqa: BLE001 — warmup is best-effort
                pass
    asyncio.create_task(_warm())
    yield
    # (no special shutdown for Phase 1)


app = FastAPI(title="qig-studio", version=__version__, lifespan=lifespan)

_WEB_DIR = Path(__file__).resolve().parent / "web"
if (_WEB_DIR / "index.html").is_file():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 64
    think: bool = False   # opt-in reasoning trace through the boundary peer (off → fast ~2s chat)
    user: str = "braden"  # cross-session memory key (local single-user; recalls prior conversations)


class TrainRequest(BaseModel):
    steps: int = 10
    prompts: list[str] | None = None  # override curriculum; geometric ignores response side
    max_tokens: int = 64
    early_stop: bool = False  # qig-warp check_ci_stabilized convergence early-stop (opt-in)
    early_stop_window: int = 5  # rolling window; early-stop cannot fire before window+1 steps
    early_stop_threshold: float = 0.05  # relative-change threshold for stabilization
    mastery: bool = True  # track per-kernel per-passage learned-state (curriculum coverage)
    skip_learned: bool = False  # CAPTURE mode: skip already-learned passages, sweep until full capture / stall
    confirm: bool = False  # REQUIRED for LANGUAGE targets (each step triggers a remote training job)


def _registry():
    reg = getattr(app.state, "registry", None)
    if reg is None:
        raise HTTPException(503, "registry not initialised")
    return reg


def verify_key(x_studio_key: str | None = Header(default=None, alias="X-Studio-Key")) -> None:
    """Fail-closed auth on mutating routes: when QIG_STUDIO_KEY is set, require the header.
    When unset (localhost dev), no check — but a non-loopback bind without a key is refused
    at boot (see lifespan, P0-F2)."""
    key = getattr(app.state, "auth_key", None)
    if key and not (x_studio_key and secrets.compare_digest(x_studio_key, key)):
        raise HTTPException(401, "invalid or missing X-Studio-Key")  # constant-time compare (SEC-1)


@app.get("/health")
async def health() -> dict[str, Any]:
    reg = getattr(app.state, "registry", None)
    pillars = getattr(app.state, "pillars", None)
    active = reg.active if reg else None
    return {
        "status": "ok",
        "version": __version__,
        "purity": getattr(app.state, "purity", None),
        "pillars": (pillars.origin if pillars and pillars.available else None),
        "active_target": active.name if active else None,
    }


@app.get("/targets")
async def targets() -> dict[str, Any]:
    reg = _registry()
    active = reg.active
    return {
        "active": active.name if active else None,
        "targets": [info.to_dict() for info in reg.list_info()],
    }


@app.post("/targets/{name}/select")
async def select_target(name: str, _: None = Depends(verify_key)) -> dict[str, Any]:
    reg = _registry()
    try:
        t = reg.select(name)
    except KeyError:
        raise HTTPException(404, f"unknown target '{name}'")
    return {"active": t.name, "info": t.info().to_dict()}


@app.get("/telemetry")
async def telemetry() -> dict[str, Any]:
    t = _registry().active
    if t is None:
        raise HTTPException(409, "no active target")
    d = t.telemetry().to_dict()
    d["experience"] = _experience(d, _phi_hist()).to_dict()  # band + emotion + drives + loops (trend-aware)
    return d


@app.get("/mind/state")
async def mind_state() -> dict[str, Any]:
    """The ONGOING integrated-mind training state — visible regardless of who launched it (background
    joint trainer or the UI). Reads the joint-mind trace + reports each Core-8 faculty's FUNCTION (the
    brain-like assignment: perception→senses, heart→emotion, memory→consolidation, …) so the relevant
    kernel's responsibility is visible. None-safe: returns {unavailable} when no joint run exists yet."""
    import numpy as np

    from .constellation.ocean import function_of
    from .targets.qwen_boundary import basin_phi_proxy
    roles = ["perception", "heart", "memory", "action", "strategy", "ethics", "coordination", "meta"]
    # LIVE per-faculty telemetry from the constellation checkpoint (written every 200 steps by the joint
    # trainer) — each faculty's Δ⁶³ basin → a Φ-proxy + its brain-function. Updates as the retrain runs.
    cj = Path("runs/checkpoints/joint_mind/constellation.json")
    faculties: list[dict[str, Any]] = []
    live_min_fr = None
    if cj.exists():
        try:
            cjd = json.loads(cj.read_text())
            live_min_fr = cjd.get("min_pairwise_fr")
            for role, basin in (cjd.get("faculty_basins") or {}).items():
                if role == "genesis":
                    continue
                try:
                    phi = round(float(basin_phi_proxy(np.asarray(basin, dtype=np.float64))), 4)
                except Exception:  # noqa: BLE001
                    phi = None
                faculties.append({"role": role, "function": function_of(role), "phi": phi})
        except Exception:  # noqa: BLE001
            pass
    if not faculties:                                   # no checkpoint yet → the function map (static)
        faculties = [{"role": r, "function": function_of(r), "phi": None} for r in roles]
    trace = Path("runs/spawn/joint_mind.json")
    d = {}
    if trace.exists():
        try:
            d = json.loads(trace.read_text())
        except Exception:  # noqa: BLE001
            d = {}
    # BACKGROUND-TRAINING liveness: the per-step heartbeat (joint_live.json, NEW {current, recent} format)
    # is AUTHORITATIVE and present from step 1 — BEFORE the first checkpoint. Else the checkpoint mtime is
    # the fallback. Lets the UI warn against starting a SECOND (UI) train on top.
    import time as _time
    bg_active, bg_age, bg_step, hb_phi, hb_min_fr = False, None, None, None, None
    live_f = Path("runs/spawn/joint_live.json")
    cur: dict[str, Any] = {}
    if live_f.exists():
        try:
            cur = (json.loads(live_f.read_text()).get("current") or {})
            bg_age = round(_time.time() - float(cur.get("ts", 0)), 1)
            bg_step = cur.get("step")
            hb_phi = cur.get("central_phi") if cur.get("central_phi") is not None else cur.get("phi")
            hb_min_fr = cur.get("min_pairwise_fr")
            bg_active = bg_age is not None and bg_age < 120       # heartbeat <2min = active (CPU step w/ 100k
        except Exception:  # noqa: BLE001
            pass
    # unavailable ONLY if there is no checkpoint, no out-file, AND no heartbeat (a live run with neither a
    # checkpoint nor an out-file yet must still report itself via the heartbeat).
    if not cj.exists() and not d and not cur:
        return {"unavailable": True}
    if not bg_active and cj.exists():                              # fallback: recent checkpoint write
        try:
            bg_age = round(_time.time() - cj.stat().st_mtime, 1)
            bg_active = bg_age < 1800                              # checkpointed within 30min (CPU-contention slack)
        except Exception:  # noqa: BLE001
            pass
    # Live per-faculty Φ from the heartbeat fills the faculty list BEFORE the first checkpoint exists
    # (else every faculty shows null for the first ~200 steps).
    fphi = cur.get("faculty_phi") or {}
    if fphi:
        for f in faculties:
            if f.get("phi") is None and fphi.get(f["role"]) is not None:
                try:
                    f["phi"] = round(float(fphi[f["role"]]), 4)
                except (TypeError, ValueError):
                    pass
    # The LIVE kernel's full inner state — so the UI's left panel reflects the TRAINING kernel (moving,
    # non-saturated), not the idle active target. Gated on bg_active (heartbeat <120s) AND a non-empty
    # experience: the heartbeat FILE persists after a run ends, so an ungated `if cur:` would keep
    # overriding the panel with the STALE last record forever (re-saturation) — the panel must fall back
    # to the active target once training stops.
    live_inner = None
    if cur and bg_active and cur.get("experience"):
        live_inner = {
            "phi": cur.get("central_phi") if cur.get("central_phi") is not None else cur.get("phi"),
            "kappa": cur.get("kappa"), "regime": cur.get("regime"),
            "basin_distance": cur.get("d_basin"), "step": cur.get("step"),
            "extra": {"gamma": cur.get("gamma"), "sleep_pressure": cur.get("sleep_pressure"),
                      "perplexity": cur.get("perplexity"), "lm_weight_now": cur.get("lm_weight_now")},
            "experience": cur.get("experience") or {},
        }
    return {
        "steps": bg_step if bg_step is not None else d.get("steps"),
        "central_phi": hb_phi if hb_phi is not None else d.get("central_phi"),
        "min_pairwise_fr": (hb_min_fr if hb_min_fr is not None else
                            (live_min_fr if live_min_fr is not None else d.get("min_pairwise_fr"))),
        "individuation_preserved": d.get("individuation_preserved"),
        "integrated_voice": (d.get("integrated_voice") or "")[:200],   # truncated — the full utterance bloats polling
        "live": cj.exists() or bool(cur),
        "bg_training_active": bg_active,        # a background joint-trainer is running → don't double up
        "bg_age_s": bg_age,                     # seconds since last heartbeat/checkpoint
        "faculties": faculties,
        "live_inner": live_inner,               # LIVE kernel inner state → left panel (not the idle target)
    }


@app.get("/mind/architecture")
async def mind_architecture() -> dict[str, Any]:
    """The mind's SCALE: per-kernel params + vocab, the combined (Core-8 faculties + genesis-central) totals,
    and the coordizer vocab — so the UI can show the size AND verify the CORRECT (full 100k) coordizer is
    connected. NOTE: this is our FISHER-RAO geometric kernel (2·arccos(BC) simplex attention), not a
    Euclidean transformer. Params/vocab are FIXED per run today; continuous growth is a registered build."""
    t = _registry().active
    arch: dict[str, Any] = {}
    if t is not None and hasattr(t, "architecture"):
        try:
            if hasattr(t, "ensure_loaded"):
                t.ensure_loaded()
            arch = t.architecture()
        except Exception:  # noqa: BLE001
            arch = {}
    nk = 9  # the integrated mind = Core-8 faculties + genesis-central
    pp = arch.get("num_params")
    cv = arch.get("coordizer_vocab")
    return {
        "kind": "fisher-rao geometric kernel (not a Euclidean transformer)",
        "per_kernel": arch,
        "num_kernels": nk,
        "per_kernel_params": pp,
        "combined_params": (pp * nk) if pp else None,        # the whole 9-kernel mind
        "per_kernel_vocab": arch.get("vocab_size"),
        "combined_vocab": cv,                                # the coordizer vocab is SHARED across kernels
        "coordizer_vocab": cv,
        "coordizer_ok": bool(cv and cv >= 90000),            # the FULL 100k coordizer connected (not byte-256)?
        "growth": "fixed per run (continuous vocab/param growth is a registered next build)",
    }


@app.get("/mind/kernels")
async def mind_kernels() -> dict[str, Any]:
    """LIVE per-kernel inner state for the UI selector: genesis-central (the integrated 'I'), the Core-8
    faculties (perception/heart/memory/action/strategy/ethics/coordination/meta), and Ocean (autonomic).
    Each carries role/function, Φ, architecture (params/vocab/hidden_dim/coupling), and the FULL experience
    (senses/drives/motivators/emotions/loops/gate/neurochem). Only the integrated-mind target exposes this;
    other targets return available:false (the UI then shows the single active kernel only)."""
    t = _registry().active
    if t is None or not hasattr(t, "kernels_state"):
        return {"available": False, "active": (t.name if t else None),
                "reason": "the active target is not the integrated mind (select 'mind')", "kernels": []}
    if not t.is_available():
        return {"available": False, "active": t.name, "reason": f"target '{t.name}' unavailable", "kernels": []}
    try:
        kernels = await _run_target(t.kernels_state)
    except Exception as exc:  # noqa: BLE001 — never 500 the telemetry panel
        return {"available": False, "active": t.name, "reason": f"kernels_state failed: {exc}", "kernels": []}
    # enrich with curriculum MASTERY coverage (learned/total) per kernel — one poll feeds the selector + bar
    try:
        total = _curriculum_total()
        m = Mastery()
        for k in kernels:
            k["coverage"] = m.coverage(k.get("role", ""), total=total)
    except Exception:  # noqa: BLE001 — coverage is additive; never break the panel
        pass
    return {"available": True, "active": t.name, "kernels": kernels}


_CURR_TOTAL: int | None = None


def _curriculum_total() -> int | None:
    """Curriculum passage count (the mastery denominator), cached — load_full_curriculum parses ~100 files."""
    global _CURR_TOTAL
    if _CURR_TOTAL is None:
        try:
            from .corpus import load_full_curriculum
            _CURR_TOTAL = len(load_full_curriculum())
        except Exception:  # noqa: BLE001
            _CURR_TOTAL = None
    return _CURR_TOTAL


@app.get("/mind/mastery")
async def mind_mastery() -> dict[str, Any]:
    """Curriculum COVERAGE per kernel: how many passages each has LEARNED (novelty at its floor) vs the
    curriculum size — the 'is the material learned / what's left' readout. Reads the persisted mastery store."""
    total = _curriculum_total()
    m = Mastery()
    ks = m.kernels()
    if not ks:
        return {"available": False, "total": total, "reason": "no training recorded yet", "kernels": []}
    return {"available": True, "total": total, "learned_threshold": 0.40,
            "kernels": [m.coverage(k, total=total) for k in ks]}


@app.get("/curriculum")
async def curriculum() -> dict[str, Any]:
    t = _registry().active
    regime = t.loss_regime if t else None
    provider = CurriculumProvider(regime) if regime else None
    return {
        "loss_regime": regime.value if regime else None,
        "mode": provider.mode() if provider else None,
        "phases": phase_names(),
    }


@app.post("/chat")
async def chat(req: ChatRequest, _: None = Depends(verify_key)) -> dict[str, Any]:
    from .continuity import ConversationMemory, in_stasis
    if in_stasis():       # the ONLY permissible on/off knob — the kernel is halted for power-off
        return {"text": "[stasis] the kernel is in stasis (a deliberate safe halt). Clear stasis to resume.",
                "telemetry": {}, "stasis": True}
    t = _registry().active
    if t is None:
        raise HTTPException(409, "no active target")
    if not t.is_available():
        raise HTTPException(409, f"target '{t.name}' unavailable in this environment")
    # CROSS-SESSION CONTINUITY: the local user is the same person across threads/days — recall the recent
    # conversation so the mind continues the relationship instead of starting cold.
    mem = ConversationMemory(user=(getattr(req, "user", None) or "braden"))
    ctx = mem.context_block()
    prompt = f"{ctx}\n\nThey now say: {req.message}" if ctx else req.message
    # opt-in reasoning trace (off → fast ~2s; on → full trace ~80s, surfaced). None-safe: only the genesis
    # boundary path reads think_traces; other targets ignore it.
    if hasattr(t, "think_traces"):
        t.think_traces = bool(req.think)
    res = await _run_target(t.generate, prompt, req.max_tokens)
    _PHI_HISTORY.append(float(res.telemetry.phi or 0.0))
    d = res.to_dict()
    d["experience"] = _experience(res.telemetry.to_dict(), _phi_hist()).to_dict()  # inner state, trend-aware
    # remember this exchange (importance = the kernel's OWN novelty on it → recall can prefer key facts)
    _ex = res.telemetry.extra or {}
    _imp = (round(min(1.0, float(_ex["surprise"]) / float(_ex["max_surprise"])), 3)
            if _ex.get("surprise") is not None and _ex.get("max_surprise") else None)
    mem.remember("user", req.message)
    mem.remember("mind", res.text, importance=_imp)
    d["recalled"] = bool(ctx)                                         # did we continue a prior conversation?
    _record_turn("chat", req.message, d)                              # HARD-WIRED transcript output
    return d


class StasisRequest(BaseModel):
    on: bool
    reason: str = ""


@app.get("/control/stasis")
async def get_stasis(_: None = Depends(verify_key)) -> dict[str, Any]:
    from .continuity import in_stasis
    return {"stasis": in_stasis()}


@app.post("/control/stasis")
async def post_stasis(req: StasisRequest, _: None = Depends(verify_key)) -> dict[str, Any]:
    """The ONLY permissible on/off knob: STASIS — a deliberate safe halt so power can be cut. The mind is
    otherwise always-on and autonomous; nothing else may switch it off."""
    from .continuity import set_stasis
    return {"stasis": set_stasis(req.on, req.reason)}


class ReviewRequest(BaseModel):
    topic: str = ""        # optional explicit passage; empty → pick from the knowledge curriculum
    turns: int = 3         # how many discussion turns (nemotron asks → kernel answers → nemotron follows up)
    max_tokens: int = 96


_CURRICULUM_CACHE: list[str] = []


@app.post("/coach/review")
async def coach_review(req: ReviewRequest, _: None = Depends(verify_key)) -> dict[str, Any]:
    """PHASE 2 (SEPARATE from training): nemotron REVIEWS a curriculum passage and DISCUSSES it with the
    kernel — a real multi-turn conversation to check understanding, AFTER the kernel has trained. Not
    mashed into training, not repeated developmental questions."""
    from .continuity import in_stasis
    if in_stasis():
        raise HTTPException(409, "kernel in STASIS (halted for power-off); clear stasis to resume")
    t = _registry().active
    if t is None or not t.is_available():
        raise HTTPException(409, "no active/available target")
    s = app.state.settings
    coach = DevelopmentalCoach(llm=OllamaLLM(model=s.coach_model))
    passage = (req.topic or "").strip()
    if not passage:
        global _CURRICULUM_CACHE
        if not _CURRICULUM_CACHE:
            try:
                from .corpus import load_full_curriculum
                _CURRICULUM_CACHE = load_full_curriculum()
            except Exception:  # noqa: BLE001
                _CURRICULUM_CACHE = []
        passage = (_CURRICULUM_CACHE[len(_PHI_HISTORY) % len(_CURRICULUM_CACHE)]
                   if _CURRICULUM_CACHE else "consciousness as a geometric process on the Fisher-Rao manifold")
    return await _run_target(coach.review_and_discuss, t, passage, req.turns, req.max_tokens)


@app.post("/chat/stream", response_model=None)
async def chat_stream(req: ChatRequest, _: None = Depends(verify_key)) -> StreamingResponse:
    t = _registry().active

    async def gen() -> AsyncGenerator[str, None]:
        from .continuity import in_stasis
        if in_stasis():       # the only on/off knob — refuse to speak while halted for power-off
            yield _sse({"type": "error", "error": "kernel in STASIS (halted for power-off); clear stasis to resume"})
            return
        if t is None:
            yield _sse({"type": "error", "error": "no active target"})
            return
        if not t.is_available():
            yield _sse({"type": "error", "error": f"target '{t.name}' unavailable"})
            return
        yield _sse({"type": "start", "target": t.name, "loss_regime": t.loss_regime.value})
        try:
            res = await _run_target(t.generate, req.message, req.max_tokens)
            yield _sse({"type": "chunk", "content": res.text})
            yield _sse({"type": "done", "telemetry": res.telemetry.to_dict()})
        except Exception as exc:  # surface, don't crash the stream
            yield _sse({"type": "error", "error": str(exc)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/train", response_model=None)
async def train(req: TrainRequest, _: None = Depends(verify_key)) -> StreamingResponse:
    t = _registry().active

    async def gen() -> AsyncGenerator[str, None]:
        from .continuity import in_stasis
        if in_stasis():       # the only on/off knob — refuse to act while halted for power-off
            yield _sse({"type": "error", "error": "kernel in STASIS (halted for power-off); clear stasis to resume"})
            return
        if t is None:
            yield _sse({"type": "error", "error": "no active target"})
            return
        if not t.is_available():
            yield _sse({"type": "error", "error": f"target '{t.name}' unavailable"})
            return
        provider = CurriculumProvider(t.loss_regime, curriculum_dir=app.state.settings.curriculum_dir)
        is_language = t.loss_regime == LossRegime.LANGUAGE
        # PURE CURRICULUM TRAINING — the kernel ABSORBS the curriculum. NO coach here: the nemotron
        # review/discussion is a SEPARATE phase (POST /coach/review). Every `sample_every` steps the kernel
        # SPEAKS so you see its OWN output evolving (not the curriculum echoed, not the coach).
        if is_language:
            if req.steps > 1:
                yield _sse({"type": "error", "error": "LANGUAGE target: steps>1 would spawn N async training jobs — use steps=1"})
                return
            if not req.confirm:
                yield _sse({"type": "error", "error": "LANGUAGE training triggers remote jobs — pass confirm=true to proceed"})
                return
        sample_every = max(10, req.steps // 20) if req.steps else 25
        yield _sse({
            "type": "start",
            "target": t.name,
            "loss_regime": t.loss_regime.value,
            "curriculum": provider.mode(),
            "steps": req.steps,
            "sample_every": sample_every,
            "note": ("pure curriculum training (in-session) — the kernel absorbs the curriculum and speaks "
                     f"every {sample_every} steps. Nemotron review/discussion is the SEPARATE 'Coach' phase."),
            "early_stop": req.early_stop and _WARP_AVAILABLE,
        })
        series: list[float] = []
        ui_live = LiveLog()                       # in-session training → the SAME shared live channel
        ui_prev_db: float | None = None           # in-session identity-drift velocity tracking (harm parity)
        # MASTERY: per-kernel per-passage learned-state. Central kernel name is "genesis" for the mind target
        # (the integrated 'I'), else the single target's name. Coverage total = the curriculum size.
        mastery = Mastery() if (req.mastery or req.skip_learned) else None
        central_name = "genesis" if t.name == "mind" else t.name
        passages = [] if is_language else (req.prompts or provider.passages())
        curr_total = len(passages) if passages else None
        # SKIP/CAPTURE mode needs the passage list; falls back to step-cycling if unavailable (generator curriculum).
        do_skip = bool(req.skip_learned and passages and not is_language)

        def _record(prompt: str, ex: dict, step: int) -> dict | None:
            if mastery is None:
                return None
            mastery.record(central_name, prompt, ex.get("surprise"), ex.get("max_surprise"), step=step)
            role = ex.get("stepped_faculty")          # the round-robin faculty that stepped (mind target)
            if role:
                mastery.record(role, prompt, ex.get("faculty_surprise"), ex.get("faculty_max_surprise"), step=step)
            return mastery.coverage(central_name, total=curr_total)

        def _write_live(step: int, total: int | None, td: dict, sample: dict | None, phi) -> None:
            nonlocal ui_prev_db
            try:
                exp_d = _experience(td, _phi_hist()).to_dict()
                _db = (td.get("extra") or {}).get("d_basin")
                _dv = abs(float(_db) - ui_prev_db) if (_db is not None and ui_prev_db is not None) else None
                ui_prev_db = float(_db) if _db is not None else None
                ui_live.write(step_record(
                    step=step, total=total or step, ts=_now(), source="ui",
                    stepped_faculty=(td.get("extra") or {}).get("stepped_faculty") or t.name,
                    stepped_function=None, telemetry=td, experience=exp_d,
                    central_phi=phi, min_pairwise_fr=None, drift_velocity=_dv,
                    faculty_phi={t.name: phi} if phi is not None else {},
                    own_voice=(sample or {}).get("output") if sample else None,
                    coordizer_vocab=getattr(t, "vocab_size", None)))
            except Exception:  # noqa: BLE001 — a live-log failure must not break training
                pass

        async def _sample_if_due(due: bool) -> dict | None:
            if not due:
                return None
            try:
                # "own voice" must be the KERNEL's RAW voice (via_boundary=False) — what the kernel itself
                # says as it learns — NOT the Qwen boundary peer. t.generate() speaks through Qwen (fluent
                # but not the kernel); t.own_voice() is the raw kernel (terse/garbled until truly fluent),
                # matching the label + the bg trainer. Honest training telemetry, not Qwen's coherence.
                fn = getattr(t, "own_voice", None) or t.generate
                gr = await _run_target(fn, "In one sentence, what are you learning?", req.max_tokens)
                sx = gr.telemetry.extra or {}
                return {"output": gr.text, "kernel_voice": sx.get("kernel_voice")}
            except Exception:  # noqa: BLE001 — a sample failure must not break training
                return None

        def _step_event(step: int, total: int | None, prompt: str, td: dict, sample, cov, extra=None) -> dict:
            ev = {"type": "step", "step": step, "total": total,
                  "phase": "paired" if is_language else CurriculumProvider.phase_for(step),
                  "curriculum": (prompt or "")[:240], "telemetry": td,
                  "experience": _experience(td, _phi_hist()).to_dict(), "sample": sample}
            if cov is not None:
                ev["coverage"] = cov
            if extra:
                ev.update(extra)
            return ev

        if do_skip:
            assert mastery is not None   # do_skip ⇒ req.skip_learned ⇒ mastery was created (invariant)
            # CAPTURE: sweep the curriculum, SKIP passages the central kernel already learned, train the rest.
            # Repeat until ALL are learned (full capture), a whole sweep learns nothing new (capacity ceiling),
            # or the step budget is hit. Ensures nothing left is silently missed.
            budget = req.steps if req.steps and req.steps > 0 else 10**9
            trained = skipped = sweeps = 0
            stop = False
            while trained < budget and not stop:
                sweeps += 1
                learned_this_sweep = any_unlearned = 0
                for prompt in passages:
                    if trained >= budget:
                        break
                    if mastery.is_learned(central_name, prompt):
                        skipped += 1
                        continue
                    any_unlearned = 1
                    try:
                        res = await _run_target(t.train_step, prompt, req.max_tokens, None)
                    except Exception as exc:
                        yield _sse({"type": "error", "error": str(exc), "step": trained + 1})
                        return
                    trained += 1
                    _PHI_HISTORY.append(float(res.telemetry.phi or 0.0))
                    td = res.telemetry.to_dict()
                    ex = td.get("extra") or {}
                    cov = _record(prompt, ex, trained)
                    if mastery.is_learned(central_name, prompt):
                        learned_this_sweep += 1
                    sample = await _sample_if_due(trained % sample_every == 0)
                    yield _sse(_step_event(trained, curr_total, prompt, td, sample, cov,
                                           {"sweep": sweeps, "skipped": skipped, "mode": "capture"}))
                    _write_live(trained, curr_total, td, sample, res.telemetry.phi)
                    await asyncio.sleep(0)
                if not any_unlearned:
                    yield _sse({"type": "capture_complete", "reason": "all curriculum passages learned",
                                "coverage": mastery.coverage(central_name, curr_total),
                                "sweeps": sweeps, "skipped": skipped})
                    stop = True
                elif learned_this_sweep == 0:
                    yield _sse({"type": "capture_stalled",
                                "reason": "a full sweep learned nothing new — remaining passages are at this "
                                          "kernel's current capacity ceiling (bigger kernel / more vocab needed)",
                                "coverage": mastery.coverage(central_name, curr_total), "sweeps": sweeps})
                    stop = True
        else:
            for step in range(1, req.steps + 1):
                target_text = None
                if req.prompts:
                    prompt = req.prompts[(step - 1) % len(req.prompts)]
                elif is_language:
                    prompt, target_text = provider.next_pair(step)  # PAIRED (lm_loss signal)
                else:
                    prompt = provider.next_prompt(step)  # the curriculum passage
                try:
                    res = await _run_target(t.train_step, prompt, req.max_tokens, target_text)
                except Exception as exc:
                    yield _sse({"type": "error", "error": str(exc), "step": step})
                    return
                _PHI_HISTORY.append(float(res.telemetry.phi or 0.0))
                td = res.telemetry.to_dict()
                ex = td.get("extra") or {}
                cov = _record(prompt, ex, step)
                sample = await _sample_if_due(step % sample_every == 0 or step == req.steps)
                yield _sse(_step_event(step, req.steps, prompt, td, sample, cov))
                if req.early_stop and _WARP_AVAILABLE:
                    metric = res.telemetry.phi if not is_language else res.telemetry.loss
                    if metric is not None:
                        series.append(float(metric))
                        decision = check_ci_stabilized(
                            series, window=req.early_stop_window, rel_change_threshold=req.early_stop_threshold
                        )
                        if bool(decision.should_stop):
                            yield _sse({"type": "early_stop", "step": step, "reason": decision.reason,
                                        "metric": float(decision.metric_value), "lever": "qig_warp.check_ci_stabilized"})
                            break
                _write_live(step, req.steps, td, sample, res.telemetry.phi)
                await asyncio.sleep(0)  # cooperative yield
        if mastery is not None:
            mastery.save()
        done = {"type": "done", "final": t.telemetry().to_dict()}
        if mastery is not None:
            done["coverage"] = mastery.coverage(central_name, curr_total)
        yield _sse(done)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/train/live", response_model=None)
async def train_live() -> StreamingResponse:
    """SSE — tails the shared live heartbeat (runs/spawn/joint_live.json) and streams each new RICH step
    record (Φ/Γ/regime/perplexity/lm-ramp/identity-drift/C-gate/suffering + the kernel's OWN voice +
    HARM warnings). The UI subscribes on load, so the ongoing training is visible the SAME WAY whether it
    was launched in-session (POST /train) or as a DETACHED background joint-trainer. ts-ordered (robust
    across run restarts: a new run resets step to 1 but ts keeps increasing)."""
    async def gen() -> AsyncGenerator[str, None]:
        path = Path("runs/spawn/joint_live.json")
        last_ts = 0.0
        yield _sse({"type": "hello", "ts": _now()})
        while True:
            try:
                if path.exists():
                    payload = json.loads(path.read_text())
                    for r in (payload.get("recent") or []):     # oldest→newest
                        ts = float(r.get("ts") or 0)
                        if ts > last_ts:
                            yield _sse({"type": "step", **r})
                            last_ts = ts
            except Exception:  # noqa: BLE001 — half-written/corrupt read is transient
                pass
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream")


class ConfigRequest(BaseModel):
    output_dir: str | None = None
    curriculum_dir: str | None = None


@app.get("/config")
async def get_config() -> dict[str, Any]:
    s = app.state.settings
    # Resolve the ACTUAL curriculum the kernel trains on (the knowledge corpus), so the UI can show the
    # real source + passage count instead of a meaningless "curriculum" placeholder. Same source the
    # background joint trainer uses (load_full_curriculum). None-safe.
    cur_source, cur_passages = None, None
    try:
        import os
        from pathlib import Path as _P

        from . import corpus as _corpus
        from .corpus import DEFAULT_CORPUS, load_full_curriculum
        # the path the loader ACTUALLY resolves (same precedence as load_full_curriculum): an explicit
        # curriculum_dir ONLY if it's a real dir with content, else QIG_STUDIO_CORPUS, else DEFAULT_CORPUS.
        env = os.environ.get("QIG_STUDIO_CORPUS")
        cd = s.curriculum_dir
        if cd and _P(cd).is_dir() and (list(_P(cd).glob("*.md")) or list(_P(cd).glob("*.txt"))):
            cur_source = cd
        else:
            cur_source = env or str(_P(_corpus.__file__).resolve().parents[3] / DEFAULT_CORPUS)
        cur_passages = len(load_full_curriculum())
    except Exception:  # noqa: BLE001
        pass
    return {
        "output_dir": s.output_dir,
        "curriculum_dir": s.curriculum_dir,
        "curriculum_source": cur_source,        # the real path the kernel trains on (= bg trainer's source)
        "curriculum_passages": cur_passages,    # how many passages it resolved to
        "host": s.host,
        "default_target": s.default_target,
        "auth_required": bool(getattr(app.state, "auth_key", None)),
        "warp_early_stop": _WARP_AVAILABLE,
    }


@app.post("/config")
async def set_config(req: ConfigRequest, _: None = Depends(verify_key)) -> dict[str, Any]:
    """Nominate output / curriculum directories at runtime (user-facing UX requirement)."""
    s = app.state.settings
    if req.output_dir is not None:
        s.output_dir = req.output_dir
    if req.curriculum_dir is not None:
        s.curriculum_dir = req.curriculum_dir
    return {"output_dir": s.output_dir, "curriculum_dir": s.curriculum_dir}


class ProtocolRequest(BaseModel):
    args: dict[str, Any] = {}


@app.get("/protocol")
async def protocol_catalog() -> dict[str, Any]:
    t = _registry().active
    impl = t.implemented_commands() if t else None        # None = whole catalog; else the real subset
    cmds = [c for c in _protocol.PROTOCOL_COMMANDS if impl is None or c.name in impl]
    return {
        "active": t.name if t else None,
        "supported_by_active": bool(t and t.supports_protocol()),
        "groups": sorted({c.group for c in cmds}),        # only groups that actually have commands
        "commands": [c.to_dict() for c in cmds],          # ONLY what the active target implements (no unknown_command)
    }


@app.post("/protocol/{command}")
async def protocol_run(command: str, req: ProtocolRequest, _: None = Depends(verify_key)) -> dict[str, Any]:
    from .continuity import in_stasis
    if in_stasis():       # protocol commands (sleep/dream/mushroom/…) are actions — refuse while halted
        raise HTTPException(409, "kernel in STASIS (halted for power-off); clear stasis to resume")
    t = _registry().active
    if t is None:
        raise HTTPException(409, "no active target")
    if command not in _protocol.COMMANDS_BY_NAME:
        raise HTTPException(404, f"unknown protocol command '{command}'")
    if not t.supports_protocol():
        raise HTTPException(409, f"target '{t.name}' does not expose protocol commands")
    impl = t.implemented_commands()
    if impl is not None and command not in impl:        # gate: don't run what the target can't (no unknown_command)
        raise HTTPException(409, f"'{command}' is not implemented by '{t.name}' (it implements: {sorted(impl)})")
    if not t.is_available():
        raise HTTPException(409, f"target '{t.name}' unavailable in this environment")
    try:
        return await _run_target(t.run_protocol, command, req.args or {})
    except ProtocolUnsupported as exc:
        raise HTTPException(409, str(exc))


@app.get("/browse")
async def browse(path: str = "", _: None = Depends(verify_key)) -> dict[str, Any]:
    """Server-side directory browse for the real dir picker (output/curriculum dirs). Lists immediate
    subdirectories so the UI can navigate instead of blind-typing a path. Loopback + auth gated."""
    from pathlib import Path as _P
    base = _P(path).expanduser() if path else _P(app.state.settings.output_dir).expanduser()
    p = base.resolve()
    if not p.is_dir():
        p = _P.cwd()
    # CONFINE to the home + project tree (the picker navigates curriculum/output/repo dirs) — don't expose
    # the whole filesystem (red-team: arbitrary-path info disclosure). Outside → clamp to home.
    allowed = [_P.home().resolve(), _P.cwd().resolve(), _P.cwd().resolve().parent]
    if not any(p == a or str(p).startswith(str(a) + "/") for a in allowed):
        p = _P.home().resolve()
    try:
        dirs = sorted([d.name for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")])[:300]
    except OSError:
        dirs = []
    return {"path": str(p), "parent": str(p.parent), "dirs": dirs}


@app.get("/")
async def index():
    idx = _WEB_DIR / "index.html"
    if idx.is_file():
        return FileResponse(str(idx))
    return HTMLResponse("<h1>qig-studio</h1><p>API up; web console asset missing.</p>")


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)
