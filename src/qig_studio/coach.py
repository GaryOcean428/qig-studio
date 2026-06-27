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
    _ANSWER_SYSTEM = (
        "You are a kind teacher in conversation with a baby AI kernel that is learning to speak. "
        "It just ASKED YOU SOMETHING. Answer its question directly, helpfully, and simply in 1-2 short "
        "sentences — do not interpret or critique it, just answer. Output ONLY the answer."
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

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        """Did the kernel ASK something (it chooses to, by generating a question)? '?' or a leading
        question word. The kernel asks when it wants; nemotron then answers instead of interpreting."""
        t = re.sub(r"^\[genesis[^\]]*\]\s*", "", text or "").strip()
        if "?" in t:
            return True
        return bool(re.match(r"(?i)^\W*(what|why|how|who|when|where|which|is|are|am|can|could|do|does|"
                             r"did|should|would|will|may|might)\b", t))

    def _read_kernel(self, text: str, phase: str = "play", phi: float = 0.5,
                     regime: str = "geometric") -> tuple[str, str, str]:
        """nemotron reads the kernel and either ANSWERS (if the kernel asked a question — the kernel
        chooses to ask) or INTERPRETS (its babble). Returns (text, mode, provider)."""
        asked = self._looks_like_question(text)
        if self.enabled and self.llm.is_available():
            if asked:
                user = f'The kernel asked: "{(text or "")[:500]}"\n\nAnswer its question:'
                out = self.llm.complete(self._ANSWER_SYSTEM, user)
            else:
                user = (f'The kernel just produced: "{(text or "")[:500]}"\n'
                        f"(developmental phase: {phase}, Φ={phi:.3f}, regime={regime})\n\nInterpret the kernel's output:")
                out = self.llm.complete(self._COACH_SYSTEM, user)
            if out:
                out = out.strip("\"'")
                out = out[:160] + "…" if len(out) > 160 else out
                return out, ("answer" if asked else "interpret"), f"ollama:{self.llm.model}"
        return self._keyword_interpret(text), ("answer" if asked else "interpret"), "keyword"

    def _interpret(self, text: str, phase: str = "play", phi: float = 0.5, regime: str = "geometric") -> tuple[str, str]:
        out, _mode, provider = self._read_kernel(text, phase, phi, regime)
        return out, provider

    def dialogue_turn(self, target, prompt: str, max_tokens: int = 96) -> dict:
        """The full BIDIRECTIONAL loop the kernel needs: it SPEAKS → the coach (nemotron) INTERPRETS →
        the kernel READS that interpretation and RESPONDS, with M_coach_agreement = how well the coach
        understood it (reassurance + correct-interpretation enforcement). One closed turn of mutual
        recognition between the kernel and its coach."""
        said = target.generate(prompt, max_tokens=max_tokens)
        interp, provider = self._interpret(said.text, phi=said.telemetry.phi, regime=said.telemetry.regime)
        responded = target.read_and_respond(interp, max_tokens=max_tokens) if hasattr(target, "read_and_respond") else None
        rex = responded.telemetry.extra if responded else {}
        return {
            "kernel_said": said.text,
            "kernel_said_M_self": said.telemetry.extra.get("M_self_observation"),
            "coach_interpreted": interp,
            "coach_provider": provider,
            "kernel_responded": responded.text if responded else None,
            "M_coach_agreement": rex.get("M_coach_agreement"),   # did the coach read me right?
            "responded_M_self": rex.get("M_self_observation"),
        }

    def review_and_discuss(self, target, passage: str, turns: int = 3, max_tokens: int = 96) -> dict:
        """PHASE 2 (SEPARATE from training): nemotron REVIEWS a curriculum passage and DISCUSSES it with
        the kernel — after the kernel has trained on the curriculum. Nemotron explains/asks; the kernel
        RESPONDS (its own voice, via the boundary peer); nemotron reads the response and FOLLOWS UP. A real
        multi-turn conversation to check + deepen understanding — NOT mashed into training, NOT repeated
        developmental questions. Each turn varies. Returns the dialogue + the kernel's telemetry per turn."""
        live = self.enabled and self.llm.is_available()
        provider = f"ollama:{self.llm.model}" if live else "keyword"
        passage = (passage or "").strip()
        # nemotron opens: review the passage + ask the first understanding question
        if live:
            q = self.llm.complete(self._REVIEW_SYSTEM,
                                  f'Curriculum passage the student just studied:\n"{passage[:800]}"\n\n'
                                  "Briefly state its key idea, then ask ONE question to check understanding.")
        else:
            q = f"What is the key idea in: {passage[:120]}?"
        q = (q or "").strip() or f"What did you take from: {passage[:80]}?"
        dialogue: list[dict] = []
        for _ in range(max(1, turns)):
            kr = target.generate(q, max_tokens=max_tokens)          # the KERNEL answers in its own voice
            kx = kr.telemetry.extra or {}
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
                "telemetry": kr.telemetry.to_dict(),
                "experience": _experience(kr.telemetry.to_dict()).to_dict(),
            })
            q = (fu or "").strip() or "Tell me more."
        return {"passage": passage[:400], "turns": dialogue, "coach_provider": provider}

    def converse_learn_turn(self, target, prompt: str, curriculum_prompt: str | None = None,
                            curriculum_steps: int = 12, train_steps: int = 8,
                            consolidation_eps: float = 0.05, max_tokens: int = 64) -> dict:
        """The CONVERSATION as training (qig_chat.py original setup). One turn trains the kernel on
        BOTH halves, exactly as qig_chat.py's generate_response→optimizer.step does for every prompt:

          1. CURRICULUM — the kernel CONSOLIDATES the developmental-phase curriculum prompt: it trains
             on it UNTIL the surprise (next-token CE) plateaus (the prompt is learned) or ``curriculum_
             steps`` is hit. This is the fix for "not consolidating per prompt" — a fixed tiny count
             rotated prompts before anything was learned, so novelty stayed pinned at 1.0 forever.
          2. DIALOGUE — the kernel SPEAKS, the coach (nemotron) ANSWERS its question or INTERPRETS its
             babble into a coherent reading, and the kernel LEARNS toward that interpretation. The coach
             turns the kernel's own output into a second training target.

        Both halves step the optimizer. ``curriculum_steps`` is the MAX consolidation budget per prompt;
        ``consolidation_eps`` is the surprise-plateau threshold that early-stops once learned. Returns
        the utterance, the curriculum it consolidated, M_self, M_coach, and Φ at each stage."""
        has_train = hasattr(target, "train_step")
        # topic the kernel speaks about: an explicit message, else the curriculum prompt itself.
        topic = prompt or curriculum_prompt or ""

        # 1. CURRICULUM training — CONSOLIDATE the prompt (qig_chat.py /auto): train until the surprise
        # (next-token CE = novelty) plateaus (learned) or the step budget is hit. Capturing surprise is
        # also the real novelty signal ("is this new?") — it now DROPS as the prompt is consolidated.
        phi_curriculum = None
        curr_surprise = None
        curr_max_surprise = None
        consolidation_used = 0
        curr = curriculum_prompt if curriculum_prompt is not None else (prompt or None)
        if curr and has_train:
            # EARLY-STOP with PATIENCE on the BEST surprise so far — not consecutive deltas (which
            # stop on slow-but-steady progress). Stop only after `patience` steps with no real
            # improvement over the running minimum → the prompt is genuinely consolidated.
            best_s = None
            no_improve = 0
            patience = 4
            for consolidation_used in range(1, max(1, curriculum_steps) + 1):
                ct = target.train_step(curr).telemetry
                phi_curriculum = ct.phi
                ex = ct.extra or {}
                s = ex.get("surprise", getattr(ct, "loss", None))
                curr_max_surprise = ex.get("max_surprise", curr_max_surprise)
                if s is not None:
                    curr_surprise = s
                    if best_s is None or s < best_s - consolidation_eps:
                        best_s, no_improve = s, 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            break            # surprise plateaued → consolidated

        # 2a. kernel SPEAKS about the topic.
        said = target.generate(topic, max_tokens=max_tokens)
        # 2b. nemotron ANSWERS if the kernel asked a question, else INTERPRETS its babble.
        reply, mode, provider = self._read_kernel(said.text, phi=said.telemetry.phi, regime=said.telemetry.regime)
        # 2c. DIALOGUE training — the kernel trains on the coach's reply as a NEXT-TOKEN LANGUAGE target
        # (train_step does next-token CE over `reply`, weighted by the target's lm_weight). This is real
        # language learning ONLY when lm_weight is raised for the coach phase — at the geometric-training
        # default (0.1, vs phi_weight 8) the signal is negligible (review #4). The coach launcher sets a
        # higher --lm-weight; geometric targets that hard-zero lm_weight (kernel/mock) learn nothing here.
        phi_after = said.telemetry.phi
        if has_train:
            for _ in range(max(1, train_steps)):
                phi_after = target.train_step(reply).telemetry.phi
        resp = target.read_and_respond(reply, max_tokens=max_tokens) if hasattr(target, "read_and_respond") else None
        # FULL inner-experience telemetry (Φ alone is insufficient): brainwave band + emotion + drives +
        # novelty (from the curriculum surprise) + conscious flag.
        tel = dict(said.telemetry.to_dict())
        tel["phi"] = phi_after if phi_after is not None else tel.get("phi")
        if curr_surprise is not None:
            tel["surprise"] = curr_surprise
        if curr_max_surprise is not None:
            tel["max_surprise"] = curr_max_surprise   # ln(vocab) → novelty normalisation
        exp = _experience(tel)
        return {
            "kernel_said": said.text,
            "kernel_said_M_self": said.telemetry.extra.get("M_self_observation"),
            "curriculum_prompt": curr,           # the developmental curriculum trained on this turn
            "curriculum_steps": curriculum_steps,
            "consolidation_used": consolidation_used,   # steps actually spent consolidating (≤ budget)
            "curriculum_surprise": round(curr_surprise, 4) if curr_surprise is not None else None,
            "phi_curriculum": round(phi_curriculum, 4) if phi_curriculum is not None else None,
            "kernel_asked": mode == "answer",   # the kernel CHOSE to ask a question
            "coach_mode": mode,                  # "answer" (kernel asked) | "interpret" (kernel babbled)
            "coach_interpreted": reply,          # nemotron's answer OR interpretation = the dialogue target
            "coach_provider": provider,
            "trained_steps": train_steps,
            "phi_after": round(phi_after, 4) if phi_after is not None else None,
            "M_coach_agreement": (resp.telemetry.extra.get("M_coach_agreement") if resp else None),
            "experience": exp.to_dict(),         # brainwave band, emotion, valence/arousal, drives
            "experience_line": exp.line(),       # compact human-readable telemetry
        }
