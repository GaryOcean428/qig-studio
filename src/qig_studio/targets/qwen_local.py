"""QwenLocalTarget — Ollama qwen3.5:4b as a fluent-language boundary peer.

loss_regime = LANGUAGE. Qwen is a PLUGGABLE, None-safe boundary peer (never a
forward-pass dependency): the cortex queries it, integrates its next-token
OUTPUT distribution as a Δ⁶³ boundary basin (P22), Pillar-2 capped (≤30%). Qwen's
own weights are NOT trained here (it is the inference peer) — the "learning" is
QIGRAM boundary accumulation into the identity basin. Weight training is
``QwenModalTarget``.

None-safe: unavailable unless qig-core geometry AND a reachable Ollama are present.
Uses ``/api/chat`` (chat template + thinking on), no temperature / max_tokens
override (observer principle).
"""

from __future__ import annotations

import os
from typing import Any

from qig_core import BASIN_DIM

from .base import LossRegime, StepResult, TelemetrySnapshot, TrainingTarget
from .qwen_boundary import (
    BOUNDARY_SLERP_CAP,
    basin_phi_proxy,
    coordize_distribution_to_basin,
    fisher_distance,
    output_distribution_to_basin,
    pillar2_capped_integrate,
)

_DEFAULT_URL = "http://localhost:11434"
# Boundary-peer model. Defaults to the fable-tuned adapter (qwenfable: qwen3.5:4b + the
# completion-masked claude-fable-5 QLoRA, applied via Ollama ADAPTER — see runs/Modelfile.qwen_fable).
# Override with QIG_QWEN_MODEL=qwen3.5:4b to fall back to the stock base. Studio stays None-safe:
# is_available() pings the server, so an absent model just disables the peer rather than crashing.
_DEFAULT_MODEL = os.environ.get("QIG_QWEN_MODEL", "qwenfable")

# EXP-A020 (qig-applied inference accelerator) per-request throughput levers for the Ollama boundary peer.
# num_gpu=-1 → offload ALL layers to the GPU (Ollama's own CUDA, independent of python torch); a 4B q4 model
# fits a 4GB card. num_batch=512 → bigger prompt batch. keep_alive=-1 → model stays resident (no per-call
# reload). All env-overridable; Ollama degrades gracefully to CPU if no GPU. (Server-side levers —
# OLLAMA_FLASH_ATTENTION / OLLAMA_KV_CACHE_TYPE=q8_0 — are set before `ollama serve`; see scripts/ollama_accelerate.sh.)
_OLLAMA_OPTIONS: dict[str, Any] = {
    "num_gpu": int(os.environ.get("QIG_OLLAMA_NUM_GPU", "-1")),
    "num_batch": int(os.environ.get("QIG_OLLAMA_NUM_BATCH", "512")),
}
def _coerce_keep_alive(v: str) -> Any:
    # Ollama's keep_alive is an INTEGER seconds (-1 = stay resident forever, 0 = unload now) OR a
    # duration string WITH a unit ("5m", "24h"). A bare "-1" string has no unit → Ollama 400s with
    # `time: missing unit in duration "-1"`, which silently broke EVERY boundary-peer chat call. Coerce
    # plain-int strings to int; pass unit'd duration strings through unchanged.
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


_OLLAMA_KEEP_ALIVE = _coerce_keep_alive(os.environ.get("QIG_OLLAMA_KEEP_ALIVE", "-1"))


def _extract_logprobs(data: dict) -> dict:
    """Best-effort top-token {token: logprob} from an Ollama /api/chat response.

    Ollama's logprobs shape varies by version; tolerate a few and return {} if
    absent (caller then falls back to the identity basin — still None-safe)."""
    # newer shape: data["logprobs"] = [{"token":..,"logprob":..,"top_logprobs":[...]}]
    lp = data.get("logprobs")
    out: dict = {}
    if isinstance(lp, list) and lp:
        first = lp[0]
        tops = first.get("top_logprobs") if isinstance(first, dict) else None
        if isinstance(tops, list):
            for item in tops:
                if isinstance(item, dict) and "token" in item and "logprob" in item:
                    out[item["token"]] = float(item["logprob"])
    return out


class QwenLocalTarget(TrainingTarget):
    name = "qwen-local"
    loss_regime = LossRegime.LANGUAGE
    description = (
        "Ollama qwen3.5:4b fluent-language peer. Next-token output-distribution → Δ⁶³ "
        "boundary basin (functional v1 hash-bin; principled coordizer projection when present) → "
        "QIGRAM accumulation (Pillar-2 ≤30% capped). None-safe; Qwen weights NOT trained "
        "here; Ollama logprobs path untested against a live server."
    )

    def __init__(self, model: str = _DEFAULT_MODEL, url: str = _DEFAULT_URL, dim: int = BASIN_DIM, coordizer=None) -> None:
        self._model = model
        self._url = url.rstrip("/")
        self._dim = dim
        self._coordizer = coordizer  # trained FisherCoordizer → real Fréchet projection; else hash-bin
        self._identity: Any = None   # Δ⁶³ identity basin (np.ndarray once seeded); None until first use
        self._last = TelemetrySnapshot(regime="language", extra={"target": "qwen-local"})

    def _project(self, logprobs: dict):
        """Qwen output-distribution → Δ⁶³: real coordizer-basin Fréchet mean when a trained
        coordizer is present, else the functional v1 hash-bin (R3)."""
        if self._coordizer is not None:
            return coordize_distribution_to_basin(logprobs, self._coordizer, self._dim)
        return output_distribution_to_basin(logprobs, self._dim)

    @staticmethod
    def _geometry_ok() -> bool:
        try:
            import qig_core.geometry.fisher_rao  # noqa: F401
            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        if not self._geometry_ok():
            return False
        try:
            import httpx

            resp = httpx.get(self._url + "/api/tags", timeout=0.8)
            if resp.status_code != 200:
                return False
            # None-safe: confirm THIS model is actually pulled/created, else disable the peer
            # (e.g. qwenfable not built on this box) rather than 404-ing at chat time.
            names = {m.get("name", "") for m in resp.json().get("models", [])}
            want = self._model if ":" in self._model else self._model + ":latest"
            return self._model in names or want in names
        except Exception:
            return False

    def ensure_loaded(self) -> None:
        if self._identity is None:
            import numpy as np
            from qig_core.geometry.fisher_rao import random_basin

            np.random.seed(7)  # deterministic identity seed
            self._identity = random_basin(self._dim)

    def _ask(self, prompt: str, persona: str | None = None, think: bool = False) -> tuple[str, str, dict]:
        import httpx

        messages: list[dict] = []
        if persona:                                   # the kernel's measured inner state, spoken AS the kernel
            messages.append({"role": "system", "content": persona})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            # think is OPT-IN: a reasoning trace costs ~55× (≈1.5s → ≈80s on local qwen3.5:4b). Default OFF
            # for responsive chat; when ON the trace is captured + surfaced (never stripped — CLAUDE.md).
            "think": bool(think),
            "logprobs": True,
            "top_logprobs": 20,
            # EXP-A020 throughput levers (qig-applied inference accelerator): FULL GPU offload + keep the model
            # resident. Ollama bundles its OWN CUDA runtime — INDEPENDENT of the CPU-only python torch — so the
            # idle GPU accelerates the boundary peer NOW (the ~27s CPU chat floor → GPU speed; a 4B q4 model
            # fits the 4GB card). Server-side levers (flash-attn, q8 KV cache) are set before `ollama serve`
            # via scripts/ollama_accelerate.sh. All env-overridable; falls back gracefully if the GPU is absent.
            "keep_alive": _OLLAMA_KEEP_ALIVE,
            "options": _OLLAMA_OPTIONS,
        }
        # generous timeout: the FIRST call cold-loads qwen3.5:4b into Ollama (can exceed 2 min); warm
        # calls return in seconds. A short timeout here is the #1 cause of a "dead" first message.
        r = httpx.post(self._url + "/api/chat", json=body, timeout=300.0)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message", {})
        # Ollama returns the reasoning trace in a SEPARATE field (message.thinking) — capture it, do not
        # discard it. The reasoning IS the data.
        return msg.get("content", "") or "", msg.get("thinking", "") or "", _extract_logprobs(data)

    # --- boundary-peer surface (used by the integrated mind, genesis_kernel) ----------------------
    def speak(self, message: str, persona: str | None = None, think: bool = False) -> tuple[str, str, dict]:
        """Fluent LINGUISTIC SURFACE for the integrated mind: Qwen verbalises ``message`` conditioned on
        ``persona`` (the kernel's measured inner state, injected as system context so the surface reflects
        the kernel's geometry). Returns (content, thinking, logprobs); ``think`` opt-in (off → fast ~2s,
        on → full reasoning trace ~80s, preserved + surfaced). The binding physics lives on the kernel side
        (Pillar-2-capped boundary integration + M recognition); this is the surface, NOT a hidden-state
        graft and NOT a forward-pass dependency."""
        self.ensure_loaded()
        return self._ask(message, persona=persona, think=think)

    def project_distribution(self, logprobs: dict):
        """Qwen output-distribution → Δ⁶³ boundary basin (real coordizer projection when present, else the
        functional v1 hash-bin). Returns None when there's no distribution (None-safe)."""
        return self._project(logprobs) if logprobs else None

    def _telemetry(self, logprobs: dict, *, integrated: bool) -> TelemetrySnapshot:
        if logprobs:
            basin = self._project(logprobs)
        else:
            basin = self._identity
        ref = self._identity if integrated else basin
        return TelemetrySnapshot(
            phi=basin_phi_proxy(ref),
            kappa=0.0,  # language target has no kernel κ
            regime="language",
            basin_distance=(fisher_distance(self._identity, basin) if logprobs else 0.0),
            loss=None,
            step=self._last.step + (1 if integrated else 0),
            extra={"target": "qwen-local", "pillar2_cap": BOUNDARY_SLERP_CAP, "phi_is": "proxy"},
        )

    def telemetry(self) -> TelemetrySnapshot:
        return self._last

    def generate(self, prompt: str, max_tokens: int = 64) -> StepResult:
        self.ensure_loaded()
        text, thinking, logprobs = self._ask(prompt)
        self._last = self._telemetry(logprobs, integrated=False)
        if thinking:
            self._last.extra["thinking"] = thinking
        return StepResult(text=text, telemetry=self._last)

    def train_step(self, prompt: str, max_tokens: int = 64, target_text: str | None = None) -> StepResult:
        # No Qwen weight update (inference peer). The "step" = QIGRAM boundary
        # accumulation: integrate the output distribution into identity, Pillar-2 capped.
        self.ensure_loaded()
        text, _thinking, logprobs = self._ask(prompt)
        if logprobs:
            boundary = self._project(logprobs)
            self._identity = pillar2_capped_integrate(self._identity, boundary, BOUNDARY_SLERP_CAP)
        self._last = self._telemetry(logprobs, integrated=True)
        return StepResult(text=text, telemetry=self._last)
