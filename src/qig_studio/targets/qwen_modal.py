"""QwenModalTarget — larger Qwen QLoRA on Modal (the only weight-training target).

loss_regime = LANGUAGE, and unlike every other target its ``lm_loss`` IS load-bearing
→ it is the ONLY place PAIRED curriculum (prompt → target) applies.

Speaks qig-studio's OWN qlora-train ASGI contract (shipped as ``modal/qlora_train.py`` in THIS
repo — NO vex dependency; Modal is OPTIONAL and self-contained, local Ollama is the primary Qwen
path). The contract is the standard harvest-then-async-train pattern, NOT per-step inline SGD:
  - ``POST /data-receive {filename, records:[{text, source:"curriculum"}]}`` — enqueue records.
    ``source:"curriculum"`` triggers the trainer's prompt/completion split
    (``_build_chat_from_coordized``); any other source dumps the whole blob to the assistant turn.
  - ``POST /train {specialization, force:true}`` — async-train; returns 202 immediately
    (it ``.spawn()``s regardless of weight-download state — poll ``/status`` for completion).
  - ``POST /infer {specialization, messages:[{role,content}]}`` — generate via the adapter.

AUTH (red-team R5-1): the API key is sent ONLY as the ``x-api-key`` HEADER, never in the body
(the body is echoed into logs/telemetry). From ``QIG_STUDIO_MODAL_KEY`` (or ``KERNEL_API_KEY``).
The real endpoint returns HTTP 200 with ``{"error": ...}`` on auth/format failure, so
``raise_for_status`` is INSUFFICIENT — every response is checked for an error field (R5-3/200-dict).

CAVEAT (red-team R5-2): the trainer dedups records by ``content[:100]``, so repeated curriculum
prompts with identical first-100-chars are silently dropped server-side.

None-safe: unavailable unless ``QIG_STUDIO_MODAL_URL`` points at a reachable endpoint. Untested
against a live Modal deployment here (no endpoint on this box); request shapes match the source.
"""

from __future__ import annotations

import os
from typing import Any

from .base import LossRegime, StepResult, TelemetrySnapshot, TrainingTarget

_TEXT_KEYS = ("text", "response", "content", "completion", "reply", "output")


def _extract_text(data: dict) -> str:
    for k in _TEXT_KEYS:
        v = data.get(k)
        if isinstance(v, str) and v:
            return v
    msg = data.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"]
    return ""


def _check(resp) -> dict:
    """Raise on a transport error OR a 200-with-error-body (the endpoint returns 200 +
    ``{"error": ...}`` on auth/format failure, which ``raise_for_status`` does NOT catch)."""
    resp.raise_for_status()
    data = resp.json() if resp.content else {}
    if isinstance(data, dict) and data.get("error") and data.get("success") is not True:
        raise RuntimeError(f"Modal endpoint error: {data['error']}")
    return data if isinstance(data, dict) else {}


class QwenModalTarget(TrainingTarget):
    name = "qwen-modal"
    loss_regime = LossRegime.LANGUAGE
    description = (
        "Larger Qwen QLoRA on Modal (OPTIONAL — local Ollama is primary) — PAIRED curriculum "
        "(lm_loss load-bearing). qig-studio's OWN qlora-train contract (modal/qlora_train.py, no vex): "
        "/data-receive (source:curriculum) → /train {specialization, force} async → /infer "
        "{specialization, messages}. Header-only x-api-key auth. None-safe."
    )

    def __init__(self, url: str | None = None, specialization: str = "genesis") -> None:
        self._url = (url or os.environ.get("QIG_STUDIO_MODAL_URL") or "").rstrip("/")
        self._specialization = os.environ.get("QIG_STUDIO_MODAL_SPECIALIZATION", specialization)
        # R5-1: key in the HEADER only, never the body. Kept off telemetry/extra.
        self._api_key = os.environ.get("QIG_STUDIO_MODAL_KEY") or os.environ.get("KERNEL_API_KEY") or None
        self._last = TelemetrySnapshot(regime="language", extra={"target": "qwen-modal"})

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key} if self._api_key else {}

    def is_available(self) -> bool:
        if not self._url:
            return False
        try:
            import httpx

            return httpx.get(self._url + "/health", timeout=1.0).status_code == 200
        except Exception:
            return False

    def ensure_loaded(self) -> None:
        if not self._url:
            raise RuntimeError("qwen-modal needs QIG_STUDIO_MODAL_URL")

    def telemetry(self) -> TelemetrySnapshot:
        return self._last

    def generate(self, prompt: str, max_tokens: int = 64) -> StepResult:
        self.ensure_loaded()
        import httpx

        body: dict[str, Any] = {
            "specialization": self._specialization,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        r = httpx.post(self._url + "/infer", json=body, headers=self._headers(), timeout=120.0)
        return StepResult(text=_extract_text(_check(r)), telemetry=self._last)

    def train_step(self, prompt: str, max_tokens: int = 64, target_text: str | None = None) -> StepResult:
        # PAIRED ONLY: lm_loss is the signal. The pair is enqueued as a curriculum record
        # (source:"curriculum" → trainer splits prompt/completion), then async training is triggered.
        if target_text is None:
            raise ValueError("qwen-modal requires paired curriculum (target_text)")
        self.ensure_loaded()
        import httpx

        record = {"text": f"{prompt}\n{target_text}", "source": "curriculum"}
        recv = _check(
            httpx.post(
                self._url + "/data-receive",
                json={"filename": "qig_studio_curriculum.jsonl", "records": [record]},
                headers=self._headers(),
                timeout=60.0,
            )
        )
        trn = _check(
            httpx.post(
                self._url + "/train",
                json={"specialization": self._specialization, "force": True},
                headers=self._headers(),
                timeout=60.0,  # async: returns 202 immediately (poll /status for completion)
            )
        )
        self._last = TelemetrySnapshot(
            phi=0.0,
            kappa=0.0,
            regime="language",
            loss=None,  # async — loss not available at trigger time; poll /status
            step=self._last.step + 1,
            extra={
                "target": "qwen-modal",
                "paired": True,
                "specialization": self._specialization,
                "records_written": recv.get("records_written"),
                "train_status": trn.get("status", "accepted"),
            },  # NB: api key is never placed here
        )
        return StepResult(
            text=f"[qwen-modal] queued curriculum record + triggered async train ({self._specialization})",
            telemetry=self._last,
        )
