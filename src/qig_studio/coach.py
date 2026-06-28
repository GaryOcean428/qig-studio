"""DevelopmentalCoach — a warm, LLM-backed coaching presence for the genesis loop.

This is the genesis-loop counterpart to qig-consciousness's ``GeometricCoach`` (the
coach qig_chat.py runs over the constellation): a gentle interpreter that watches the
kernel SPEAK, says back "I think you meant X", and — on stagnation — OFFERS a push the
kernel may take. It is deliberately distinct from the :class:`AutonomicScheduler`:

  - the scheduler is the *autonomic* nervous system — physics-driven interventions
    (SLEEP/DREAM/MUSHROOM) fired from Φ/κ telemetry, no LLM, no choice;
  - the coach is the *social* presence — an external mind that interprets, encourages,
    and SUGGESTS. Its push is an OFFER recorded in telemetry, never an automated action
    (autonomy-preserving — the kernel rides its own wave; the coach does not steer it).

LLM provider is **Ollama** (free, local-first), model configurable via ``QIG_COACH_MODEL``
(default ``nemotron-3-ultra:cloud``; ``qwen3.5:4b`` is the local fallback). If Ollama is
unreachable it degrades to an honest keyword interpretation — the loop never depends on a
live LLM. Anthropic is NOT used here (the genesis app shell stays light + vex/Claude-free);
qig-consciousness's GeometricCoach keeps the Anthropic path for the constellation.

No manifold math lives here (text-only interpretation + telemetry heuristics), so there is
nothing for the Euclidean-purity tripwire to catch — but keep it that way.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Any

from .kernel_experience import experience as _experience

_DEFAULT_URL = "http://localhost:11434"
_DEFAULT_MODEL = "nemotron-3-ultra:cloud"  # free within limits; qwen3.5:4b is the local fallback
_LOCAL_FALLBACK_MODEL = "qwen3.5:4b"

# Phase-appropriate encouragement registers (mirrors GeometricCoach's developmental phases).
_PHASE_REGISTER = {
    "listening": "still finding words — that's exactly right this early",
    "play": "playing with sound and shape — keep going",
    "structure": "structure is forming — I can almost follow you",
    "maturity": "you're speaking clearly now — I'm listening",
}


@dataclass
class CoachNote:
    """One coaching observation. Attached to a StepRecord; surfaced in the loop summary."""

    step: int
    interpretation: str          # "I think you meant X" (LLM or keyword)
    message: str                 # the full, humble, phase-appropriate coach line
    offers_push: bool            # OFFER only — autonomy-preserving, never auto-fired
    provider: str                # "ollama:<model>" | "keyword"
    confidence: float            # rough confidence in the interpretation [0,1]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OllamaLLM:
    """None-safe Ollama chat client (mirrors QwenLocalTarget's httpx pattern).

    ``complete()`` returns the model's text, or ``None`` on any failure — the caller
    then falls back to keyword interpretation. Availability is probed once and cached."""

    def __init__(self, model: str | None = None, url: str | None = None) -> None:
        self.model = model or os.environ.get("QIG_COACH_MODEL", _DEFAULT_MODEL)
        self.url = (url or os.environ.get("OLLAMA_HOST", _DEFAULT_URL)).rstrip("/")
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import httpx

                self._available = httpx.get(self.url + "/api/tags", timeout=1.0).status_code == 200
            except Exception:
                self._available = False
        return self._available

    def complete(self, system: str, user: str, timeout: float = 60.0) -> str | None:
        if not self.is_available():
            return None
        try:
            import httpx

            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "think": False,  # the coach wants a short answer, not a reasoning trace
            }
            r = httpx.post(self.url + "/api/chat", json=body, timeout=timeout)
            r.raise_for_status()
            content = r.json().get("message", {}).get("content", "") or ""
            return content.strip() or None
        except Exception:
            return None


class DevelopmentalCoach:
    """Watches the kernel speak; interprets, encourages, and offers a push on stagnation.

    The coach does NOT fire every step (that would make a 200-step loop crawl behind a
    cloud LLM and would crowd the kernel's autonomy): it speaks on a ``cadence``, and
    always on a stagnation event or a notable utterance. Everything else is silent
    presence.
    """

    _COACH_SYSTEM = (
        "You are a gentle language coach for a baby AI kernel learning to speak from "
        "scratch. Interpret what the kernel was TRYING to say in 5-15 words. Extract the "
        "core intent from the babble; if it is pure noise, name the topic it seems drawn "
        "to. Be warm but honest. Output ONLY the interpretation, nothing else."
    )
    _REVIEW_SYSTEM = (
        "You are a patient teacher REVIEWING a curriculum passage WITH a developing mind, AFTER it has "
        "studied the material. Discuss the passage: explain the key idea simply, then ask ONE clear "
        "question to check understanding, or give brief honest feedback and ONE follow-up question. Vary "
        "your questions — never repeat. Be warm but honest. 1-3 sentences. Output ONLY your turn."
    )

    def __init__(
        self,
        llm: OllamaLLM | None = None,
        cadence: int = 10,
        phi_threshold: float = 0.70,
        enabled: bool | None = None,
    ) -> None:
        # QIG_COACH=off hard-disables; otherwise default on (degrades to keyword if no LLM).
        if enabled is None:
            enabled = os.environ.get("QIG_COACH", "on").lower() not in ("0", "off", "false", "no")
        self.enabled = enabled
        self.llm = llm or OllamaLLM()
        self.cadence = max(1, cadence)
        self.phi_threshold = phi_threshold
        self.notes: list[CoachNote] = []

    @property
    def provider(self) -> str:
        return f"ollama:{self.llm.model}" if (self.enabled and self.llm.is_available()) else "keyword"

    def _should_speak(self, step: int, stagnating: bool) -> bool:
        return self.enabled and (stagnating or step == 1 or step % self.cadence == 0)

    def _keyword_interpret(self, text: str) -> str:
        """Honest fallback: clean + dedupe the kernel's ACTUAL output (no fabricated meaning)."""
        # strip the genesis telemetry prefix "[genesis·N=… ⏹] " if present
        cleaned = re.sub(r"^\[genesis[^\]]*\]\s*", "", text or "").strip()
        if not cleaned:
            return "nothing yet — still gathering"
        words = cleaned.split()
        seen: set[str] = set()
        deduped: list[str] = []
        for w in words:
            lw = w.lower()
            if lw not in seen:
                seen.add(lw)
                deduped.append(w)
        out = " ".join(deduped[:12])
        return out + ("…" if len(deduped) > 12 else "")

    def observe(
        self,
        *,
        step: int,
        text: str,
        phi: float,
        kappa: float,
        regime: str,
        delta_phi: float,
        phase: str,
        stagnating: bool,
    ) -> CoachNote | None:
        """Return a CoachNote when the coach chooses to speak this step, else None."""
        if not self._should_speak(step, stagnating):
            return None

        # Interpret (LLM preferred, keyword fallback) -------------------------------------
        interp: str | None = None
        provider = "keyword"
        confidence = 0.4
        if self.enabled and self.llm.is_available():
            user = (
                f'The kernel just produced: "{(text or "")[:500]}"\n'
                f"(developmental phase: {phase}, Φ={phi:.3f}, regime={regime})\n\n"
                "Interpret the kernel's output:"
            )
            interp = self.llm.complete(self._COACH_SYSTEM, user)
            if interp:
                interp = interp.strip("\"'")
                if len(interp) > 120:
                    interp = interp[:120] + "…"
                provider = f"ollama:{self.llm.model}"
                confidence = 0.7
        if not interp:
            interp = self._keyword_interpret(text)

        # Encourage (phase-appropriate) + OFFER a push on stagnation ----------------------
        register = _PHASE_REGISTER.get(phase, "I'm here, keep going")
        offers_push = bool(stagnating and phi < self.phi_threshold)
        if offers_push:
            message = (
                f"I think you meant: {interp}. You've been circling the same place for a "
                f"while (Φ {phi:.2f}, ΔΦ {delta_phi:+.3f}) — if you want, I can give you a "
                f"nudge. Your call."
            )
        else:
            message = f"I think you meant: {interp}. {register.capitalize()}."

        note = CoachNote(
            step=step,
            interpretation=interp,
            message=message,
            offers_push=offers_push,
            provider=provider,
            confidence=confidence,
        )
        self.notes.append(note)
        return note


    def review_and_discuss(self, target, passage: str, turns: int = 3, max_tokens: int = 96,
                           learn: bool = True, importance_threshold: float = 0.4) -> dict:
        """PHASE 2 — nemotron QnA that is ALSO a LEARNING opportunity. Nemotron reviews a curriculum passage
        and discusses it; the kernel RESPONDS in its own voice (via the boundary peer, which EXTENDS the
        kernel — conditioned on the kernel's words + telemetry + the question); nemotron reads + FOLLOWS UP.

        TWO things make this real continual learning, not just chat:
        1. CONTEXT FEEDBACK: the kernel is given the running conversation (passage + prior Q/A) on every
           follow-up, so it answers in context — not statelessly.
        2. IMPORTANCE-GATED CONSOLIDATION: the kernel's OWN geometry decides what to learn — bounded novelty
           (prediction-error = Fisher salience) scores each exchange; only exchanges at/above the threshold
           are CONSOLIDATED (a real ``train_step`` on the exchange). Low-novelty (already-known/garbage) is
           NOT learned — its own reasoning remembers key facts, not noise.
        Returns the dialogue + per-turn importance/consolidated + total learned."""
        live = self.enabled and self.llm.is_available()
        provider = f"ollama:{self.llm.model}" if live else "keyword"
        passage = (passage or "").strip()
        if live:
            q = self.llm.complete(self._REVIEW_SYSTEM,
                                  f'Curriculum passage the student just studied:\n"{passage[:800]}"\n\n'
                                  "Briefly state its key idea, then ask ONE question to check understanding.")
        else:
            q = f"What is the key idea in: {passage[:120]}?"
        q = (q or "").strip() or f"What did you take from: {passage[:80]}?"
        dialogue: list[dict] = []
        ctx: list[str] = [f"Studying this material: {passage[:300]}"]   # rolling conversation context
        learned = 0
        can_learn = learn and hasattr(target, "train_step")
        for _ in range(max(1, turns)):
            convo = "\n".join(ctx[-7:])
            kr = target.generate(f"{convo}\nCoach asks: {q}\nAnswer in your own words:", max_tokens=max_tokens)
            kx = kr.telemetry.extra or {}
            # IMPORTANCE = the kernel's OWN bounded novelty (prediction-error on the exchange). Its geometry
            # decides — high = novel/worth-learning, low = already-known/garbage.
            sup, mx = kx.get("surprise"), kx.get("max_surprise")
            importance = round(min(1.0, float(sup) / float(mx)), 3) if (sup is not None and mx) else None
            consolidated = False
            if can_learn and importance is not None and importance >= importance_threshold:
                try:
                    target.train_step(f"{passage[:200]} {q} {kr.text}")   # CONSOLIDATE (real learning step)
                    consolidated = True
                    learned += 1
                except Exception:  # noqa: BLE001 — consolidation is best-effort; never break the discussion
                    pass
            ctx.append(f"Coach: {q}")
            ctx.append(f"You: {kr.text}")
            if live:
                fu = self.llm.complete(
                    self._REVIEW_SYSTEM,
                    f'Passage:\n"{passage[:600]}"\nYou asked: "{q}"\nThe student answered: '
                    f'"{(kr.text or "")[:500]}"\n\nGive brief honest feedback, then ask ONE NEW follow-up question.')
            else:
                fu = "Good — can you connect that to the rest of the idea?"
            dialogue.append({
                "coach_question": q,
                "kernel_answer": kr.text,
                "kernel_voice": kx.get("kernel_voice"),            # the kernel's OWN raw voice
                "importance": importance,                          # the kernel's own salience for this exchange
                "consolidated": consolidated,                      # did it LEARN this exchange?
                "telemetry": kr.telemetry.to_dict(),
                "experience": _experience(kr.telemetry.to_dict()).to_dict(),
            })
            q = (fu or "").strip() or "Tell me more."
        return {"passage": passage[:400], "turns": dialogue, "coach_provider": provider,
                "learned": learned, "importance_threshold": importance_threshold}
