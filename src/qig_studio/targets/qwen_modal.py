"""QwenModalTarget — larger Qwen QLoRA on Modal (the only weight-training target).

loss_regime = LANGUAGE, and unlike every other target its ``lm_loss`` IS load-bearing
→ it is the ONLY place PAIRED curriculum (prompt → target) applies. Wraps a Modal
QLoRA ASGI endpoint (PyPI-pinned image, atomic adapter swap on the Modal side).

None-safe: unavailable unless ``QIG_STUDIO_MODAL_URL`` (or a constructor URL) points
at a reachable endpoint. Live training is exercised against the deployed Modal app;
here it stays dormant (no URL → not available).
"""

from __future__ import annotations

import os

from .base import LossRegime, StepResult, TelemetrySnapshot, TrainingTarget


class QwenModalTarget(TrainingTarget):
    name = "qwen-modal"
    loss_regime = LossRegime.LANGUAGE
    description = (
        "Larger Qwen QLoRA on Modal — PAIRED curriculum (lm_loss load-bearing), "
        "PyPI-pinned image, atomic adapter swap. None-safe (needs QIG_STUDIO_MODAL_URL)."
    )

    def __init__(self, url: str | None = None, model: str | None = None) -> None:
        self._url = (url or os.environ.get("QIG_STUDIO_MODAL_URL") or "").rstrip("/")
        self._model = model or os.environ.get("QIG_STUDIO_MODAL_MODEL")
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

        r = httpx.post(
            self._url + "/infer",
            json={"prompt": prompt, "max_tokens": max_tokens},
            timeout=120.0,
        )
        r.raise_for_status()
        data = r.json()
        return StepResult(text=data.get("text", data.get("completion", "")), telemetry=self._last)

    def train_step(self, prompt: str, max_tokens: int = 64, target_text: str | None = None) -> StepResult:
        # PAIRED ONLY: lm_loss is the signal, so a target is required.
        if target_text is None:
            raise ValueError("qwen-modal requires paired curriculum (target_text)")
        self.ensure_loaded()
        import httpx

        r = httpx.post(
            self._url + "/train",
            json={"pairs": [{"prompt": prompt, "completion": target_text}]},
            timeout=None,
        )
        r.raise_for_status()
        data = r.json()
        loss = data.get("loss")
        self._last = TelemetrySnapshot(
            phi=0.0,
            kappa=0.0,
            regime="language",
            loss=float(loss) if loss is not None else None,
            step=self._last.step + 1,
            extra={"target": "qwen-modal", "paired": True, "adapter": data.get("adapter")},
        )
        return StepResult(text=data.get("sample", ""), telemetry=self._last)
