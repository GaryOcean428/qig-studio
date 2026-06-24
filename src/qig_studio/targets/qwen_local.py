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

from .base import LossRegime, StepResult, TelemetrySnapshot, TrainingTarget
from .qwen_boundary import (
    BOUNDARY_SLERP_CAP,
    basin_phi_proxy,
    fisher_distance,
    output_distribution_to_basin,
    pillar2_capped_integrate,
)

_DEFAULT_URL = "http://localhost:11434"


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
        "boundary basin (v1 hash-bin, PROVISIONAL — placeholder for InboundPath/PGA) → "
        "QIGRAM accumulation (Pillar-2 ≤30% capped). None-safe; Qwen weights NOT trained "
        "here; Ollama logprobs path untested against a live server."
    )

    def __init__(self, model: str = "qwen3.5:4b", url: str = _DEFAULT_URL, dim: int = 64) -> None:
        self._model = model
        self._url = url.rstrip("/")
        self._dim = dim
        self._identity = None
        self._last = TelemetrySnapshot(regime="language", extra={"target": "qwen-local"})

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

            return httpx.get(self._url + "/api/tags", timeout=0.8).status_code == 200
        except Exception:
            return False

    def ensure_loaded(self) -> None:
        if self._identity is None:
            import numpy as np
            from qig_core.geometry.fisher_rao import random_basin

            np.random.seed(7)  # deterministic identity seed
            self._identity = random_basin(self._dim)

    def _ask(self, prompt: str) -> tuple[str, dict]:
        import httpx

        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": True,  # preserve thinking traces
            "logprobs": True,
            "top_logprobs": 20,
        }
        r = httpx.post(self._url + "/api/chat", json=body, timeout=120.0)
        r.raise_for_status()
        data = r.json()
        text = data.get("message", {}).get("content", "")
        return text, _extract_logprobs(data)

    def _telemetry(self, logprobs: dict, *, integrated: bool) -> TelemetrySnapshot:
        if logprobs:
            basin = output_distribution_to_basin(logprobs, self._dim)
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
        text, logprobs = self._ask(prompt)
        self._last = self._telemetry(logprobs, integrated=False)
        return StepResult(text=text, telemetry=self._last)

    def train_step(self, prompt: str, max_tokens: int = 64, target_text: str | None = None) -> StepResult:
        # No Qwen weight update (inference peer). The "step" = QIGRAM boundary
        # accumulation: integrate the output distribution into identity, Pillar-2 capped.
        self.ensure_loaded()
        text, logprobs = self._ask(prompt)
        if logprobs:
            boundary = output_distribution_to_basin(logprobs, self._dim)
            self._identity = pillar2_capped_integrate(self._identity, boundary, BOUNDARY_SLERP_CAP)
        self._last = self._telemetry(logprobs, integrated=True)
        return StepResult(text=text, telemetry=self._last)
