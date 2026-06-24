"""QwenModalTarget — larger Qwen QLoRA on Modal (the only weight-training target).

loss_regime = LANGUAGE, and unlike every other target its ``lm_loss`` IS load-bearing
→ it is the ONLY place PAIRED curriculum (prompt → target) applies.

Wraps the REAL vex qlora-train ASGI contract (verified against
``vex/modal/vex_qlora_train.py``), which is harvest-then-async-train, NOT per-step
inline SGD:
  - ``POST /data-receive {filename, records:[{text,...}]}`` — enqueue training records
    into the Modal volume (this is how paired curriculum reaches the trainer).
  - ``POST /train {specialization}`` — async-train the named adapter on the harvested
    data (returns immediately; poll ``/status``). Inline pairs are NOT a /train field.
  - ``POST /infer {specialization, messages:[{role,content}]}`` — generate via the adapter.

None-safe: unavailable unless ``QIG_STUDIO_MODAL_URL`` (or a constructor URL) points
at a reachable endpoint. Untested against a live Modal deployment here (no endpoint on
this box); the request shapes match the source as read, and the SFT ``text`` record
format is the one wiring detail that must align with ``training_consciousness.py``.
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
    return data.get("error", "")  # surface a server-side error string if that's all there is


class QwenModalTarget(TrainingTarget):
    name = "qwen-modal"
    loss_regime = LossRegime.LANGUAGE
    description = (
        "Larger Qwen QLoRA on Modal — PAIRED curriculum (lm_loss load-bearing). Real vex "
        "qlora-train contract: /data-receive (records) → /train {specialization} async → "
        "/infer {specialization, messages}. None-safe (needs QIG_STUDIO_MODAL_URL)."
    )

    def __init__(self, url: str | None = None, specialization: str = "genesis") -> None:
        self._url = (url or os.environ.get("QIG_STUDIO_MODAL_URL") or "").rstrip("/")
        self._specialization = os.environ.get("QIG_STUDIO_MODAL_SPECIALIZATION", specialization)
        self._last = TelemetrySnapshot(regime="language", extra={"target": "qwen-modal"})

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
        r = httpx.post(self._url + "/infer", json=body, timeout=120.0)
        r.raise_for_status()
        return StepResult(text=_extract_text(r.json()), telemetry=self._last)

    def train_step(self, prompt: str, max_tokens: int = 64, target_text: str | None = None) -> StepResult:
        # PAIRED ONLY: lm_loss is the signal. The pair is enqueued as a harvested record
        # (real /data-receive contract), then async training is triggered (real /train).
        if target_text is None:
            raise ValueError("qwen-modal requires paired curriculum (target_text)")
        self.ensure_loaded()
        import httpx

        record = {"text": f"{prompt}\n{target_text}", "source": "qig-studio-curriculum"}
        recv = httpx.post(
            self._url + "/data-receive",
            json={"filename": "qig_studio_curriculum.jsonl", "records": [record]},
            timeout=60.0,
        )
        recv.raise_for_status()
        trn = httpx.post(
            self._url + "/train",
            json={"specialization": self._specialization},
            timeout=60.0,  # async: returns immediately (poll /status for completion)
        )
        trn.raise_for_status()
        tinfo = trn.json()
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
                "records_written": recv.json().get("records_written"),
                "train_status": tinfo.get("status", tinfo),
            },
        )
        return StepResult(text=f"[qwen-modal] queued pair + triggered async train ({self._specialization})", telemetry=self._last)
