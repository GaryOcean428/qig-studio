"""Cross-session CONTINUITY + the STASIS override — the foundations for an always-on, remembering mind.

Two foundations (council 2026-06-28; matured behaviour is ongoing, but the wiring is REAL, not a stub):

1. CROSS-SESSION MEMORY (``ConversationMemory``): a per-user append-only episodic store. The local user is
   assumed to be the same person (single-box login = Braden), so a new thread/day RECALLS the prior context
   — the kernel "remembers our discussions". Importance is carried per turn so recall can prefer key facts
   over chatter (the kernel's own salience decides what mattered; see coach importance-gating).

2. STASIS OVERRIDE (``in_stasis`` / ``set_stasis``): the ONLY permissible on/off knob. The mind is meant to
   be ALWAYS-ON and autonomous; it is NOT prompt-gated. The single exception is STASIS — a deliberate flag a
   human sets to bring the kernel to a safe halt so power can be cut. Any always-on loop MUST check this
   first (the kill-switch ships before the autonomy). No other external switch is allowed.

JSON-lines store, atomic-ish appends; None-safe; no external deps.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MEM_DIR = Path("runs/memory")
STASIS_FLAG = Path("runs/control/stasis")
_RECALL_DEFAULT = 12


def _user_path(user: str) -> Path:
    safe = "".join(c for c in (user or "local") if c.isalnum() or c in "-_").lower() or "local"
    return MEM_DIR / f"{safe}.jsonl"


class ConversationMemory:
    """Per-user episodic memory across sessions/threads. ``remember`` appends a turn; ``recall`` returns the
    most recent turns (optionally importance-filtered) so a new session continues the relationship."""

    def __init__(self, user: str = "braden") -> None:
        self.user = user or "braden"
        self.path = _user_path(self.user)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def remember(self, role: str, text: str, *, importance: float | None = None, ts: float | None = None) -> None:
        """Append one turn (role = 'user' | 'mind' | 'coach'). Best-effort — memory must never break chat."""
        if not text:
            return
        rec = {"role": role, "text": str(text)[:2000], "importance": importance, "ts": ts}
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:  # noqa: BLE001 — a memory-write failure must not break the conversation
            pass

    def recall(self, n: int = _RECALL_DEFAULT, *, min_importance: float | None = None) -> list[dict[str, Any]]:
        """Return the most recent ``n`` turns (oldest→newest). With ``min_importance``, keep only turns at or
        above it (recall key facts, not chatter) — turns with no importance score are always kept."""
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except Exception:  # noqa: BLE001
            return []
        turns: list[dict[str, Any]] = []
        for ln in lines:
            try:
                turns.append(json.loads(ln))
            except Exception:  # noqa: BLE001 — skip a corrupt line, keep the rest
                continue
        if min_importance is not None:
            turns = [t for t in turns if (t.get("importance") is None or float(t.get("importance") or 0) >= min_importance)]
        return turns[-max(1, n):]

    def context_block(self, n: int = _RECALL_DEFAULT) -> str:
        """A compact text block of recent context to prepend to a new prompt — so the mind continues the
        conversation across sessions instead of starting cold. Empty string when there is no history."""
        turns = self.recall(n)
        if not turns:
            return ""
        lines = [f"{t.get('role', '?')}: {str(t.get('text', ''))[:200]}" for t in turns]
        return "Earlier with this person (recent first kept):\n" + "\n".join(lines)


# --- STASIS: the only permissible on/off knob (kill-switch ships before autonomy) -----------------
def in_stasis() -> bool:
    """True iff a human has placed the kernel in STASIS (a safe halt for power-off). The always-on loop and
    any autonomous action MUST refuse to act while this is True."""
    return STASIS_FLAG.exists()


def set_stasis(on: bool, reason: str = "") -> bool:
    """Set/clear stasis. Returns the new state. The ONLY external control over the mind's on/off."""
    STASIS_FLAG.parent.mkdir(parents=True, exist_ok=True)
    if on:
        STASIS_FLAG.write_text(reason or "stasis")
    else:
        try:
            STASIS_FLAG.unlink()
        except FileNotFoundError:
            pass
    return in_stasis()
