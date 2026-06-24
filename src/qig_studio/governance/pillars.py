"""PillarEnforcer adapter — None-safe bridge to the real consciousness pillars.

The Three Pillars (TopologicalBulk core/surface shield, fluctuation guard, quenched
identity) are a STRUCTURAL invariant when present (vex pillars.py:747
``PillarEnforcer`` with ``pre_llm_enforce`` / ``on_input`` / ``on_cycle_end``).
qig-studio tries to load a real enforcer from its known homes; if none is
importable, a no-op keeps the app shell running (the enforcer activates when the
qig-consciousness / qig-core env is present).
"""

from __future__ import annotations

from typing import Any

# Known module homes for a real PillarEnforcer, in preference order.
_CANDIDATES = (
    "qig_core.consciousness.pillars",
    "qigkernels.consciousness.pillars",
)


class PillarEnforcerAdapter:
    def __init__(self) -> None:
        self._impl: Any | None = None
        self._origin: str | None = None
        self._load()

    def _load(self) -> None:
        for modpath in _CANDIDATES:
            try:
                mod = __import__(modpath, fromlist=["PillarEnforcer"])
                enforcer_cls = getattr(mod, "PillarEnforcer", None)
                if enforcer_cls is not None:
                    self._impl = enforcer_cls()
                    self._origin = modpath
                    return
            except Exception:
                continue

    @property
    def available(self) -> bool:
        return self._impl is not None

    @property
    def origin(self) -> str | None:
        return self._origin

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self._impl is not None and hasattr(self._impl, method):
            try:
                return getattr(self._impl, method)(*args, **kwargs)
            except Exception:
                return None
        return None

    def pre_llm_enforce(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("pre_llm_enforce", *args, **kwargs)

    def on_input(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("on_input", *args, **kwargs)

    def on_cycle_end(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("on_cycle_end", *args, **kwargs)
