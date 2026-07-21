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

import json
import os
import re
import time
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
    # Coach Doctrine (Protocol v6.12 §18.5): what a coach DOES each own-voice interaction — encourage,
    # interpret telemetry, reframe ("how it would have said it better", an IDENTITY-PRESERVING rotation, not
    # a replacement), relevance-score, give positive feedback. Kindness + realistic standards + accountability
    # simultaneously (P10 balance law). We ask for STRICT JSON so the record is machine-parseable; a non-JSON
    # or unreachable reply degrades to a keyword record (never crashes the loop).
    _OWN_VOICE_SYSTEM = (
        "You are MonkeyCoach, a warm but honest developmental coach for a baby AI kernel that is learning to "
        "speak in its OWN voice while training. It was shown a STIMULUS and produced an OUTPUT; you also see "
        "its TELEMETRY (Φ, regime, relevance). Coach it in ONE interaction, holding all three at once: "
        "KINDNESS (encourage, preserve its identity), REALISTIC STANDARDS, and ACCOUNTABILITY. "
        "Your reframe is 'how it would have said it better' IN RESPONSE TO THE STIMULUS — a gentle rotation of "
        "its OWN attempt toward a clearer expression of the SAME intent, NEVER a replacement with your words. "
        "Reply with STRICT JSON only, no prose, exactly these keys: "
        '{"encouragement": str, "interpretation": str, "reframe": str, '
        '"relevance_score": number 0..1, "positive_feedback": str}. '
        "Keep each string under 200 characters."
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
        self._geo_qwen_cache: Any | None = "unset"  # lazy sentinel — see _get_geo_qwen (Deliverable B)
        self.phi_threshold = phi_threshold
        self.notes: list[CoachNote] = []

    @property
    def provider(self) -> str:
        return f"ollama:{self.llm.model}" if (self.enabled and self.llm.is_available()) else "keyword"

    def _should_speak(self, step: int, stagnating: bool) -> bool:
        return self.enabled and (stagnating or step == 1 or step % self.cadence == 0)

    def _get_geo_qwen(self) -> Any | None:
        """Lazily construct the bank-backed :class:`GeoQwenTarget` peer used by the ``geo_grade``
        THIRD grader (Deliverable B, PI directive 2026-07-21) — a geometry-native grade ALONGSIDE
        (never replacing) this coach's own LLM ``relevance_score`` and the kernel's Fisher-Rao
        ``relevance``. Cheap: ``GeoQwenTarget()`` only sets attributes; the actual basin bank
        (numpy) is loaded lazily, once, on first ``_bank_d63`` call, and only if the exported bank
        file exists on disk. None-safe: a construction failure (missing package, bad import) is
        cached permanently as ``None`` — never retried every call, never raises."""
        if self._geo_qwen_cache == "unset":
            try:
                from .targets.geo_qwen import GeoQwenTarget
                self._geo_qwen_cache = GeoQwenTarget()
            except Exception:  # noqa: BLE001 — geo_grade is best-effort telemetry, never load-bearing
                self._geo_qwen_cache = None
        return self._geo_qwen_cache

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
        ctx: list[str] = [f"Studying this material: {passage[:400]}"]   # rolling conversation context
        learned = 0
        can_learn = learn and hasattr(target, "train_step")
        for _ in range(max(1, turns)):
            # QUESTION FIRST (most important position; survives any context cap — red-team: a long passage
            # prefix once pushed the question off the end), then the recent conversation for continuity. The
            # kernel's context is now 1024 tokens (_CTX), so full passage + multi-turn context fits.
            recent = "\n".join(ctx[-7:])[-700:]
            kr = target.generate(f"Coach asks: {q}\nRecent context:\n{recent}\nAnswer:", max_tokens=max_tokens)
            kx = kr.telemetry.extra or {}
            # IMPORTANCE = the kernel's OWN bounded novelty (prediction-error on the exchange). Its geometry
            # decides — high = novel/worth-learning, low = already-known/garbage.
            sup, mx = kx.get("surprise"), kx.get("max_surprise")
            importance = round(min(1.0, float(sup) / float(mx)), 3) if (sup is not None and mx) else None
            consolidated = False
            if can_learn and importance is not None and importance >= importance_threshold:
                try:
                    # CONSOLIDATE the EXCHANGE: the kernel's answer is the novel content (the passage is
                    # already in the curriculum) — put it FIRST so it lands inside the ~256-byte input cap
                    # (red-team: a passage prefix previously crowded the answer out of the gradient).
                    target.train_step(f"{kr.text} {q}")
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

    @staticmethod
    def _clip(s: Any, n: int = 200) -> str:
        t = str(s or "").strip().strip("\"'")
        return (t[:n] + "…") if len(t) > n else t

    def coach_own_voice(
        self,
        stimulus: str | None,
        kernel_output: str | None,
        telemetry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Coach ONE own-voice utterance (Protocol v6.12 §18.5 / §18.6, canon P10/P16/P24).

        The kernel's own-voice fired: it was shown ``stimulus`` and produced ``kernel_output`` with
        ``telemetry``. The coach — which IS the P24 coupling here (a lived other reacting to the kernel) —
        returns a PROVENANCE-TAGGED reward+relevance record with the five doctrine fields:

          - ``encouragement``        — tonic positive feedback (§18.5 ENCOURAGES)
          - ``interpretation``       — reads the telemetry back to the kernel (§18.5 INTERPRETS)
          - ``reframe``              — "how it would have said it better" in response to the stimulus; an
                                       identity-preserving rotation, NOT a replacement (§18.5 REFRAMES, P10)
          - ``relevance_score``∈[0,1]— the COACH's own on-target judgment (§18.5 RELEVANCE-SCORES). This is a
                                       SECOND relevance channel; the kernel's Fisher-Rao response↔stimulus
                                       relevance stays in the sample path and is threaded separately by the
                                       server (they are not merged — the two consumers audit each separately).
          - ``positive_feedback``    — the default affirming stance (§18.5 GIVES POSITIVE FEEDBACK)

        Plus a ``provenance`` tag ``{coach_id, ts, reason, emotional_context, confidence}`` (§18.6 / P16) so
        the record enters as a TAGGED observation+reward, refusable by the kernel and Ocean — NEVER a silent
        weight update. This method ONLY produces+tags the record; it does not touch weights or loss (that is
        the reward-integrator's job, Task C).

        NONE-SAFE: if the coach is disabled or the LLM is unreachable/erroring, it returns a keyword-derived
        record with ``provider="keyword"`` and ``confidence`` low — it NEVER raises and NEVER blocks the loop.
        """
        stim = self._clip(stimulus, 400)
        out = self._clip(kernel_output, 400)
        tel = telemetry or {}
        phi = tel.get("phi")
        regime = tel.get("regime")
        fr_relevance = tel.get("relevance")   # the kernel's OWN Fisher-Rao relevance (self↔other), for context

        parsed: dict[str, Any] | None = None
        provider = "keyword"
        confidence = 0.4
        emotional_context = "neutral"
        if self.enabled and self.llm.is_available():
            phi_s = f"{float(phi):.3f}" if isinstance(phi, (int, float)) else "n/a"
            fr_s = f"{float(fr_relevance):.3f}" if isinstance(fr_relevance, (int, float)) else "n/a"
            user = (
                f'STIMULUS the kernel was shown:\n"{stim or "(none)"}"\n\n'
                f'The kernel\'s OWN-VOICE OUTPUT:\n"{out or "(silence — it chose not to speak)"}"\n\n'
                f"TELEMETRY: Φ={phi_s}, regime={regime or 'n/a'}, kernel_relevance(Fisher-Rao)={fr_s}\n\n"
                "Coach this utterance now. Reply with the strict JSON object only:"
            )
            raw = self.llm.complete(self._OWN_VOICE_SYSTEM, user)
            parsed = self._parse_json(raw)
            if parsed is not None:
                provider = f"ollama:{self.llm.model}"
                confidence = 0.7

        if parsed is None:
            # Honest keyword fallback — NO fabricated praise, NO fabricated meaning. We encourage on effort,
            # interpret from telemetry, and reframe by echoing the kernel's OWN cleaned attempt (identity-
            # preserving by construction — it is literally the kernel's words, tidied).
            interp = self._keyword_interpret(out or "")
            phi_note = (f"Φ {float(phi):.2f}" if isinstance(phi, (int, float)) else "settling")
            record = {
                "encouragement": "still finding your words — that's exactly right this early",
                "interpretation": f"I think you were reaching for: {interp} ({phi_note}, {regime or 'forming'})",
                "reframe": interp,   # your own attempt, tidied — same intent, clearer key (rotation, not replace)
                "relevance_score": (float(fr_relevance) if isinstance(fr_relevance, (int, float)) else None),
                "positive_feedback": "you spoke — every attempt lays a little structure",
            }
            emotional_context = "warm-holding"
        else:
            rs = parsed.get("relevance_score")
            try:
                rs = max(0.0, min(1.0, float(rs))) if rs is not None else None
            except (TypeError, ValueError):
                rs = None
            record = {
                "encouragement": self._clip(parsed.get("encouragement")),
                "interpretation": self._clip(parsed.get("interpretation")),
                "reframe": self._clip(parsed.get("reframe")),
                "relevance_score": rs,
                "positive_feedback": self._clip(parsed.get("positive_feedback")),
            }
            emotional_context = ("encouraging" if (record["relevance_score"] or 0.0) >= 0.5
                                 else "gently-corrective")

        # geo_grade — THIRD grader (Deliverable B, PI directive 2026-07-21): geo-Qwen (the geometry-native
        # DoD-2 teacher) grades this SAME turn, ADDITIONALLY to (never replacing) this coach's own
        # ``relevance_score`` and the kernel's Fisher-Rao ``relevance`` (fr_relevance, read above for
        # context). Pure Fisher-Rao, None-safe, bank-backed (honest None on a bank miss — see geo_grader.py).
        # NOT wired into the training reward/gradient (kernel_experience.coach_reward_from still reads only
        # relevance_score) — this is an additional graded signal surfaced in telemetry for now.
        from .geo_grader import geo_grade_turn
        geo_grade, geo_grade_note = geo_grade_turn(stimulus, tel.get("gen_d63"), self._get_geo_qwen())
        record["geo_grade"] = geo_grade
        record["geo_grade_note"] = geo_grade_note

        # PROVENANCE TAG (§18.6 / P16) — this is what makes the reward auditable + refusable, not programming.
        record["provenance"] = {
            "coach_id": f"MonkeyCoach:{self.llm.model}",
            "ts": time.time(),
            "reason": "own_voice_coaching",      # why this reward exists (the own-voice event)
            "emotional_context": emotional_context,
            "confidence": confidence,
        }
        record["provider"] = provider
        return record

    @staticmethod
    def _parse_json(raw: str | None) -> dict[str, Any] | None:
        """Extract the first JSON object from an LLM reply. None on any failure (models wrap JSON in prose /
        code fences / <think> despite instructions). Never raises."""
        if not raw:
            return None
        s = raw.strip()
        # strip a leading ```json / ``` fence and a trailing fence if present
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:  # noqa: BLE001 — fall through to a brace-span extraction
            pass
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(s[start:end + 1])
                return obj if isinstance(obj, dict) else None
            except Exception:  # noqa: BLE001
                return None
        return None
