"""QwenModalTarget red-team fixes: 200-dict-error check (R5-3) + header-only auth (R5-1)."""

from __future__ import annotations

import pytest

from qig_studio.targets.qwen_modal import QwenModalTarget, _check


class _Resp:
    def __init__(self, status: int = 200, payload: dict | None = None):
        self.status_code = status
        self._payload = payload or {}
        self.content = b"{}"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_check_raises_on_200_with_error_body():
    # the real endpoint returns HTTP 200 + {"error": ...} on auth/format failure;
    # raise_for_status would NOT catch it — _check must.
    with pytest.raises(RuntimeError):
        _check(_Resp(200, {"error": "Invalid API key"}))


def test_check_passes_on_ok_body():
    assert _check(_Resp(200, {"records_written": 1, "success": True}))["records_written"] == 1


def test_headers_carry_key_only_when_set(monkeypatch):
    monkeypatch.delenv("QIG_STUDIO_MODAL_KEY", raising=False)
    monkeypatch.delenv("KERNEL_API_KEY", raising=False)
    assert QwenModalTarget()._headers() == {}
    monkeypatch.setenv("QIG_STUDIO_MODAL_KEY", "secret-123")
    assert QwenModalTarget()._headers() == {"x-api-key": "secret-123"}


def test_train_step_requires_paired_target():
    # paired-only; raises before any network call
    with pytest.raises(ValueError):
        QwenModalTarget(url="http://example").train_step("prompt", target_text=None)


def test_key_never_in_telemetry_extra(monkeypatch):
    # SEC-7: the api key must never land in surfaced telemetry
    monkeypatch.setenv("QIG_STUDIO_MODAL_KEY", "secret-xyz")
    t = QwenModalTarget(url="http://example")
    assert "secret-xyz" not in str(t.telemetry().to_dict())
