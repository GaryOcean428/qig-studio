"""Per-kernel curriculum MASTERY — which passages a kernel has LEARNED, so the UI can show coverage and
training can SKIP learned material and ensure FULL CAPTURE of whatever is left.

"Learned" = the kernel's own NOVELTY on the passage has fallen to a floor. Novelty (importance) =
``surprise / max_surprise`` where ``surprise`` is the kernel's next-token cross-entropy on the passage and
``max_surprise = log(vocab)``. High importance ⇒ unfamiliar ⇒ still worth training; below ``LEARNED_IMPORTANCE``
⇒ familiar ⇒ re-training buys little. (Same signal the coach's importance gate uses — consolidate above the
threshold, skip below it.) This is capacity-relative: a passage that never drops below the floor is at the
kernel's current capacity ceiling, NOT necessarily fully mastered — the capture loop reports those honestly
rather than looping on them forever.

Per-kernel: the integrated mind trains genesis-central (every step) + one faculty (round-robin), so coverage
is keyed by kernel role. Central reaches full coverage fastest; faculties accumulate over passes.

JSON store, atomic writes, env-overridable path (tests use a temp file). None-safe.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

MASTERY_PATH = os.environ.get("QIG_STUDIO_MASTERY_PATH", "runs/spawn/mastery.json")
LEARNED_IMPORTANCE = 0.40   # importance (surprise/max_surprise) BELOW this = learned (low novelty = familiar)


def passage_key(text: str) -> str:
    """Stable short key for a passage (survives reordering of the curriculum)."""
    return hashlib.sha1(text.strip().encode("utf-8", "replace")).hexdigest()[:16]


class Mastery:
    """Per-kernel ``{passage_key: {importance, surprise, learned, step, preview}}`` store."""

    def __init__(self, path: str | os.PathLike[str] = MASTERY_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._d: dict[str, dict[str, Any]] = {}
        try:
            if self.path.exists():
                loaded = json.loads(self.path.read_text())
                if isinstance(loaded, dict):
                    self._d = loaded
        except Exception:  # noqa: BLE001 — a corrupt store is not fatal; start fresh
            self._d = {}

    def record(self, kernel: str, text: str, surprise: float | None, max_surprise: float | None,
               *, step: int | None = None, threshold: float = LEARNED_IMPORTANCE) -> bool | None:
        """Record the kernel's novelty on this passage; returns whether it is now LEARNED (None if no signal)."""
        if not text or surprise is None or not max_surprise:
            return None
        imp = min(1.0, max(0.0, float(surprise) / float(max_surprise)))
        learned = imp < threshold
        self._d.setdefault(kernel, {})[passage_key(text)] = {
            "importance": round(imp, 3), "surprise": round(float(surprise), 3),
            "learned": learned, "step": step, "preview": text.strip()[:60],
        }
        return learned

    def is_learned(self, kernel: str, text: str) -> bool:
        r = self._d.get(kernel, {}).get(passage_key(text))
        return bool(r and r.get("learned"))

    def coverage(self, kernel: str, total: int | None = None) -> dict[str, Any]:
        """Coverage for one kernel: learned / total (curriculum size), plus seen + mean novelty."""
        recs = self._d.get(kernel, {})
        seen = len(recs)
        learned = sum(1 for r in recs.values() if r.get("learned"))
        imps = [r["importance"] for r in recs.values() if r.get("importance") is not None]
        tot = total if total is not None else seen
        return {
            "kernel": kernel, "total": tot, "seen": seen, "learned": learned,
            "remaining": max(0, tot - learned),
            "fraction": round(learned / tot, 4) if tot else 0.0,
            "mean_importance": round(sum(imps) / len(imps), 3) if imps else None,
        }

    def kernels(self) -> list[str]:
        return list(self._d)

    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._d))
        os.replace(tmp, self.path)
