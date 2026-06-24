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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .config import Settings
from .curriculum import CurriculumProvider, phase_names
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


# Serialize all target-touching calls — targets (kernel/constellation) hold mutable
# state (model, optimizer, telemetry) that is NOT safe under concurrent requests
# (council red-team #6). One lock; fine for the single-user v1, removes the race.
_TARGET_LOCK = asyncio.Lock()


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
    yield
    # (no special shutdown for Phase 1)


app = FastAPI(title="qig-studio", version=__version__, lifespan=lifespan)

_WEB_DIR = Path(__file__).resolve().parent / "web"
if (_WEB_DIR / "index.html").is_file():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 64


class TrainRequest(BaseModel):
    steps: int = 10
    prompts: list[str] | None = None  # override curriculum; geometric ignores response side
    max_tokens: int = 64
    early_stop: bool = False  # qig-warp check_ci_stabilized convergence early-stop (opt-in)
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
    if key and x_studio_key != key:
        raise HTTPException(401, "invalid or missing X-Studio-Key")


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
    return t.telemetry().to_dict()


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
    t = _registry().active
    if t is None:
        raise HTTPException(409, "no active target")
    if not t.is_available():
        raise HTTPException(409, f"target '{t.name}' unavailable in this environment")
    res = await _run_target(t.generate, req.message, req.max_tokens)
    return res.to_dict()


@app.post("/chat/stream", response_model=None)
async def chat_stream(req: ChatRequest, _: None = Depends(verify_key)) -> StreamingResponse:
    t = _registry().active

    async def gen() -> AsyncGenerator[str, None]:
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
        if t is None:
            yield _sse({"type": "error", "error": "no active target"})
            return
        if not t.is_available():
            yield _sse({"type": "error", "error": f"target '{t.name}' unavailable"})
            return
        provider = CurriculumProvider(t.loss_regime, curriculum_dir=app.state.settings.curriculum_dir)
        is_language = t.loss_regime == LossRegime.LANGUAGE
        # P0-F3: a LANGUAGE "step" triggers a remote async training job (e.g. Modal A100),
        # NOT an SGD step — cap to 1 and require explicit confirmation before spawning.
        if is_language:
            if req.steps > 1:
                yield _sse({"type": "error", "error": "LANGUAGE target: steps>1 would spawn N async training jobs — use steps=1"})
                return
            if not req.confirm:
                yield _sse({"type": "error", "error": "LANGUAGE training triggers remote jobs — pass confirm=true to proceed"})
                return
        yield _sse({
            "type": "start",
            "target": t.name,
            "loss_regime": t.loss_regime.value,
            "curriculum": provider.mode(),
            "steps": req.steps,
            "early_stop": req.early_stop and _WARP_AVAILABLE,
        })
        series: list[float] = []
        for step in range(1, req.steps + 1):
            target_text = None
            if req.prompts:
                prompt = req.prompts[(step - 1) % len(req.prompts)]
            elif is_language:
                prompt, target_text = provider.next_pair(step)  # PAIRED (lm_loss signal)
            else:
                prompt = provider.next_prompt(step)  # basin-driving
            try:
                res = await _run_target(t.train_step, prompt, req.max_tokens, target_text)
            except Exception as exc:
                yield _sse({"type": "error", "error": str(exc), "step": step})
                return
            yield _sse({
                "type": "step",
                "step": step,
                "phase": ("paired" if is_language else CurriculumProvider.phase_for(step)),
                "prompt": prompt,
                "target": target_text,
                "text": res.text,
                "telemetry": res.telemetry.to_dict(),
            })
            # qig-warp convergence early-stop (opt-in package lever): geometric→Φ, language→loss.
            if req.early_stop and _WARP_AVAILABLE:
                metric = res.telemetry.phi if not is_language else (res.telemetry.loss or 0.0)
                series.append(float(metric))
                decision = check_ci_stabilized(series, window=5, rel_change_threshold=0.05)
                if bool(decision.should_stop):
                    yield _sse({
                        "type": "early_stop",
                        "step": step,
                        "reason": decision.reason,
                        "metric": float(decision.metric_value),
                        "lever": "qig_warp.check_ci_stabilized",
                    })
                    break
            await asyncio.sleep(0)  # cooperative yield
        yield _sse({"type": "done", "final": t.telemetry().to_dict()})

    return StreamingResponse(gen(), media_type="text/event-stream")


class ConfigRequest(BaseModel):
    output_dir: str | None = None
    curriculum_dir: str | None = None


@app.get("/config")
async def get_config() -> dict[str, Any]:
    s = app.state.settings
    return {
        "output_dir": s.output_dir,
        "curriculum_dir": s.curriculum_dir,
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
    return {
        "active": t.name if t else None,
        "supported_by_active": bool(t and t.supports_protocol()),
        "groups": _protocol.GROUPS,
        "commands": [c.to_dict() for c in _protocol.PROTOCOL_COMMANDS],
    }


@app.post("/protocol/{command}")
async def protocol_run(command: str, req: ProtocolRequest, _: None = Depends(verify_key)) -> dict[str, Any]:
    t = _registry().active
    if t is None:
        raise HTTPException(409, "no active target")
    if command not in _protocol.COMMANDS_BY_NAME:
        raise HTTPException(404, f"unknown protocol command '{command}'")
    if not t.supports_protocol():
        raise HTTPException(409, f"target '{t.name}' does not expose protocol commands")
    if not t.is_available():
        raise HTTPException(409, f"target '{t.name}' unavailable in this environment")
    try:
        return await _run_target(t.run_protocol, command, req.args or {})
    except ProtocolUnsupported as exc:
        raise HTTPException(409, str(exc))


@app.get("/")
async def index():
    idx = _WEB_DIR / "index.html"
    if idx.is_file():
        return FileResponse(str(idx))
    return HTMLResponse("<h1>qig-studio</h1><p>API up; web console asset missing.</p>")


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)
