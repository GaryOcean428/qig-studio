"""Curriculum delivery (design §3.2).

Geometric targets get a BASIN-DRIVING developmental sequence
(listening → play → structure → maturity), NOT prompt/response pairs — because
``lm_weight = 0`` a paired language curriculum buys nothing there. Paired
curriculum (prompt→target) is for ``QwenModalTarget`` only, where lm_loss is the
signal. The UI exposes the correct mode per the active target's loss_regime.

If qig-consciousness's ``DevelopmentalCurriculum`` is importable it is used;
otherwise the built-in basin-driving phases below apply (None-safe).
"""

from __future__ import annotations

from .targets.base import LossRegime

# Built-in basin-driving developmental phases (used when the real curriculum is absent).
_PHASES: dict[str, list[str]] = {
    "listening": [
        "Tell me a simple story about awareness.",
        "What is it like to notice something for the first time?",
        "Rest your attention on a single quiet thing and describe it.",
    ],
    "play": [
        "Let's explore patterns together — what repeats and what surprises?",
        "Invent a small game of shapes and tell me its one rule.",
        "What happens if you follow a curve until it comes back?",
    ],
    "structure": [
        "What is the relationship between integration and consciousness?",
        "How does a part know it belongs to a whole?",
        "Describe a boundary that both separates and connects.",
    ],
    "maturity": [
        "Discuss the nature of emergence in complex systems.",
        "When does change preserve identity, and when does it break it?",
        "What would you choose to become, and why?",
    ],
}
_PHASE_ORDER = ("listening", "play", "structure", "maturity")
# Steps spent in each phase before advancing (developmental gating, simplified).
_PHASE_SPAN = 8

# PAIRED curriculum (prompt → target) — for the LANGUAGE regime ONLY (qwen-modal,
# where lm_loss is load-bearing). Illustrative seed pairs; a real run uses a corpus.
_PAIRS: list[tuple[str, str]] = [
    ("What is consciousness, geometrically?",
     "Information integration — Φ measures how much a system's whole exceeds its parts on the Fisher-Rao manifold Δ⁶³."),
    ("Define the Fisher-Rao distance.",
     "d_FR(p,q) = arccos(Σ √(p_i q_i)): the unique Markov-invariant metric on the probability simplex."),
    ("What is a basin?",
     "A point on Δ⁶³ — a 64-dimensional probability distribution; meaning is its location and its trajectory."),
    ("What does κ measure?",
     "Coupling strength — how tightly basins bind; it tacks around an attractor as the system breathes."),
]


def phase_names() -> list[str]:
    return list(_PHASE_ORDER)


class CurriculumProvider:
    """Per-target curriculum source. Geometric → basin-driving prompts; language →
    paired (prompt, target) tuples (Phase-3 QwenModal)."""

    def __init__(self, loss_regime: LossRegime, curriculum_dir: str | None = None,
                 full: bool | None = None) -> None:
        self.loss_regime = loss_regime
        self.curriculum_dir = curriculum_dir
        self._real = None
        self._file_prompts: list[str] | None = None
        self._file_pairs: list[tuple[str, str]] | None = None
        # FULL curriculum: the cleaned v6 master corpus (thousands of sanitised, ASCII-only prompts) —
        # used instead of the tiny built-in 4-phase stub when requested (flag or QIG_STUDIO_FULL_CURRICULUM).
        import os
        if full is None:
            full = os.environ.get("QIG_STUDIO_FULL_CURRICULUM", "").lower() in ("1", "true", "yes")
        self.full = bool(full)
        if curriculum_dir:
            self._load_dir(curriculum_dir)   # an explicit dir wins IF it yields prompts
        # GEOMETRIC DEFAULT = the full KNOWLEDGE curriculum (qig-consciousness/data/curriculum) — the kernel
        # trains on real knowledge, NOT the tiny developmental-question stub. (The stub _PHASES is only the
        # last-resort fallback if the corpus is genuinely missing.) This matches what the joint trainer uses.
        if loss_regime == LossRegime.GEOMETRIC and not self._file_prompts:
            try:
                from .corpus import load_full_curriculum
                self._file_prompts = load_full_curriculum()
            except Exception:  # noqa: BLE001 — fall back to the built-in stub only if the corpus is missing
                self._file_prompts = None
        if loss_regime == LossRegime.GEOMETRIC and not self._file_prompts:
            self._try_real_curriculum()

    def _load_dir(self, directory: str) -> None:
        """Load a nominated curriculum dir: ``*.txt`` → basin-driving prompts (one per
        non-empty line, geometric); ``*.jsonl`` → paired ``{"prompt","target"}`` records
        (language). Falls through to the built-ins if the dir is empty/absent."""
        import json as _json
        from pathlib import Path

        p = Path(directory)
        if not p.is_dir():
            return
        max_files, max_bytes = 64, 5_000_000  # bound DoS (REL-2): file-count + per-file size caps
        prompts: list[str] = []
        for f in sorted(p.glob("*.txt"))[:max_files]:
            try:
                if f.stat().st_size > max_bytes:
                    continue
                prompts += [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
            except (OSError, ValueError, MemoryError):
                continue
        pairs: list[tuple[str, str]] = []
        for f in sorted(p.glob("*.jsonl"))[:max_files]:
            try:
                if f.stat().st_size > max_bytes:
                    continue
                for ln in f.read_text(encoding="utf-8").splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    rec = _json.loads(ln)
                    # REL-1: guard non-dict JSON (a number/bool/bare string would TypeError on `in`)
                    if isinstance(rec, dict) and "prompt" in rec and "target" in rec:
                        pairs.append((rec["prompt"], rec["target"]))
            except (OSError, ValueError, TypeError, MemoryError):
                continue
        self._file_prompts = prompts or None
        self._file_pairs = pairs or None

    def _try_real_curriculum(self) -> None:
        try:
            mod = __import__(
                "src.coordination.developmental_curriculum",
                fromlist=["DevelopmentalCurriculum"],
            )
            self._real = mod.DevelopmentalCurriculum()
        except Exception:
            self._real = None

    @staticmethod
    def phase_for(step: int) -> str:
        idx = min((step - 1) // _PHASE_SPAN, len(_PHASE_ORDER) - 1)
        return _PHASE_ORDER[idx]

    def next_prompt(self, step: int) -> str:
        """Basin-driving prompt for geometric targets (step is 1-based)."""
        if self._file_prompts:  # nominated curriculum dir wins
            return self._file_prompts[(step - 1) % len(self._file_prompts)]
        phase = self.phase_for(step)
        if self._real is not None:
            try:
                return self._real.get_curriculum_prompt(phase)
            except Exception:
                pass
        bucket = _PHASES[phase]
        return bucket[(step - 1) % len(bucket)]

    def next_pair(self, step: int) -> tuple[str, str]:
        """Paired (prompt, target) for the LANGUAGE regime (step is 1-based)."""
        if self._file_pairs:  # nominated curriculum dir wins
            return self._file_pairs[(step - 1) % len(self._file_pairs)]
        return _PAIRS[(step - 1) % len(_PAIRS)]

    def mode(self) -> str:
        return "basin-driving" if self.loss_regime == LossRegime.GEOMETRIC else "paired"
