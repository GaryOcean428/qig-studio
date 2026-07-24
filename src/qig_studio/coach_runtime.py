"""coach_runtime.py — the run-of-record coach supervisor: B5 liveness invariant + B4/B6 counters.

Matrix 7a1bce4b names the coach-failure semantics as *the item nobody listed and the likeliest smooth-run
killer*. Run-1 died because the newborn's witness was ABSENT (the ``DevelopmentalCoach`` existed but
``train_joint_mind`` never instantiated or called it — P21 latent). Run-2 fixes that, but puts the witness
on a NETWORK endpoint (Modal L40S SGLang, scale-to-zero) that can cold-start slowly, 429, time out, or die
mid-run. A run that silently continues on a BLIND witness reproduces run-1's malformation BY OUTAGE instead
of omission — the same coachless newborn, a new excuse.

RULED (Matrix B5): coach liveness is an ASSERTED INVARIANT.

- ``coach_own_voice`` is None-safe and *degrades* to a ``provider="keyword"`` record when the LLM is
  unreachable/disabled/unparseable. For a chat server that is correct; for the RUN-OF-RECORD a keyword
  record means the witness went BLIND. This supervisor treats a keyword record (when a witness is required)
  as a liveness failure, not a graceful degrade.
- Beyond ``tolerance`` CONSECUTIVE blind/failed coaching steps, :meth:`coach_and_reward` raises
  :class:`CoachUnreachable`; the caller CHECKPOINTS and PAUSES — never degrades silently.
- :meth:`preflight` exercises ONE real completion so a scale-to-zero COLD-START latency is a MEASURED number
  at launch, not a mid-run surprise, and fail-closes the launch if a required witness is unreachable.

It also carries the falsifiable-channel counters run-2 is pre-registered on:

- B6 — coach call count, blind-call count, success rate, a rough call/token estimate, cold-start latency.
- B4 — floor-restoration rate (cumulative + rolling window) and violations/step. A DECLINING floor-
  restoration rate across formation is Matrix's pre-registered coach-effect signal (85a78b36 §3): the coach
  does not eliminate per-step collapse (that is lm_loss's one-hot pressure, the floor's job), it teaches the
  basin to need the floor LESS over time.

Dependency-light on purpose (imports only the pure ``coach_reward_from`` map, no torch) so the liveness
invariant is unit-testable without loading a kernel.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any

from .kernel_experience import coach_reward_from


class CoachUnreachable(RuntimeError):
    """The coach witness went blind (unreachable / keyword-fallback / error) beyond tolerance during a
    REQUIRED run. The caller checkpoints and PAUSES — run-2 never trains a newborn on a blind witness."""


class CoachSupervisor:
    """Wraps a :class:`~qig_studio.coach.DevelopmentalCoach` for a run-of-record training loop.

    Parameters
    ----------
    coach:
        A ``DevelopmentalCoach`` (or any object exposing ``enabled``, ``provider``, ``llm``,
        ``coach_own_voice`` and ``_should_speak``).
    required:
        When True, coach liveness is an asserted invariant: a blind witness beyond ``tolerance`` consecutive
        coaching steps raises :class:`CoachUnreachable`, and :meth:`preflight` fail-closes the launch. Set
        False only for a non-run-of-record smoke where a coachless pass is deliberately acceptable.
    tolerance:
        Consecutive blind/failed coaching steps allowed before the invariant trips.
    window:
        Rolling window (in floor-observed steps) for the windowed floor-restoration rate.
    """

    def __init__(self, coach: Any, *, required: bool, tolerance: int = 3, window: int = 200,
                 stagnation_window: int = 30, stagnation_range: float = 0.01) -> None:
        self.coach = coach
        self.required = bool(required)
        self.tolerance = max(1, int(tolerance))
        # B6 — call / token telemetry
        self.calls = 0
        self.blind_calls = 0            # keyword-fallback records (witness blind)
        self.errors = 0                 # hard exceptions (coach_own_voice is None-safe, so ~0 expected)
        self.consecutive_blind = 0
        self.est_output_chars = 0
        self.cold_start_s: float | None = None
        # B4 — floor-restoration / violations
        self._floor_events = 0
        self._steps_seen = 0
        self._floor_window: deque[int] = deque(maxlen=max(1, int(window)))
        # stagnation — the coach steps in when the kernel is stuck (Φ plateau below maturity), edge-triggered
        self._phi_window: deque[float] = deque(maxlen=max(2, int(stagnation_window)))
        self.stagnation_range = float(stagnation_range)   # Φ range over the window below this = a plateau
        self._prev_phi: float | None = None
        self._was_stagnating = False
        self.stagnation_steps = 0        # steps spent stuck
        self.stagnation_onsets = 0       # episodes (rising edges) — the coach stepped in
        self.pushes_offered = 0          # nudges the coach OFFERED (autonomy-preserving; never auto-fired)

    # -- availability ---------------------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(getattr(self.coach, "enabled", False))

    def due(self, step: int, stagnating: bool = False) -> bool:
        """Cadence gate — reuse the coach's own ``_should_speak`` so cadence stays single-sourced."""
        if not self.enabled:
            return False
        should = getattr(self.coach, "_should_speak", None)
        if callable(should):
            return bool(should(step, stagnating))
        return True

    def preflight(self) -> dict:
        """Exercise ONE real completion → a MEASURED cold-start latency (Matrix B5). Fail-closed: a required
        but unreachable witness raises :class:`CoachUnreachable` at LAUNCH, before any training step."""
        info: dict[str, Any] = {"provider": getattr(self.coach, "provider", "?"),
                                "available": False, "latency_s": None, "required": self.required}
        if not self.enabled:
            if self.required:
                raise CoachUnreachable(
                    "coach is DISABLED (QIG_COACH=off) but this run REQUIRES a witness (Stage-0 SCHOOL, P21). "
                    "Enable the coach, or pass --coach-optional for a non-run-of-record smoke.")
            return info
        llm = getattr(self.coach, "llm", None)
        t0 = time.time()
        ok = False
        try:
            ok = bool(llm is not None and llm.is_available())
            if ok:
                # is_available() may be a cheap ping; a real completion triggers the actual scale-to-zero
                # cold-start, so THIS is the latency we want measured.
                reply = llm.complete("You are a developmental coach. Reply with the single word: ready.",
                                     "Say ready.", timeout=120.0)
                ok = bool(reply)
        except Exception as exc:  # noqa: BLE001 — a preflight error is a liveness failure, not a crash
            ok = False
            info["error"] = f"{type(exc).__name__}: {exc}"
        info["latency_s"] = round(time.time() - t0, 2)
        info["available"] = ok
        self.cold_start_s = info["latency_s"]
        if self.required and not ok:
            raise CoachUnreachable(
                f"coach preflight FAILED (provider={info['provider']}, {info['latency_s']}s): the witness is "
                f"unreachable at launch. Run-2 does not train a newborn on a blind witness (Matrix B5). Fix the "
                f"endpoint (QIG_COACH_ENDPOINT / QIG_COACH_KEY) or pass --coach-optional for a smoke.")
        return info

    # -- the hot path ---------------------------------------------------------------------------------
    def coach_and_reward(self, kernel: Any, *, stimulus: str | None, text: str | None,
                         telemetry: dict | None) -> dict | None:
        """B2 — coach the kernel's own-voice utterance and register the reward THROUGH the kernel's own
        ``register_coach_reward`` hook (B3: reward-weighted replay priority; the Stage-0 self-reward mask
        stays ``coach ≡ 0`` and is never bypassed). B5 — a blind/failed witness beyond ``tolerance``
        consecutive steps raises :class:`CoachUnreachable`. Returns the coach record (or None if disabled)."""
        if not self.enabled:
            return None
        self.calls += 1
        rec: dict | None = None
        blind = False
        try:
            rec = self.coach.coach_own_voice(stimulus, text, telemetry)
        except Exception:  # noqa: BLE001 — coach_own_voice is None-safe, but a hard error is still a blind step
            self.errors += 1
            blind = True
        if rec is None:
            blind = True
        else:
            if str(rec.get("provider", "keyword")).startswith("keyword"):
                blind = True
                self.blind_calls += 1
            else:
                self.est_output_chars += sum(
                    len(str(rec.get(k, ""))) for k in
                    ("encouragement", "interpretation", "reframe", "positive_feedback"))
            # B3 — reward routes through the kernel's own hook (replay priority, mask-respecting), never a
            # direct neurochem/weight poke. None-safe: a target without the hook (mock) is simply not rewarded.
            reg = getattr(kernel, "register_coach_reward", None)
            if reg is not None:
                try:
                    reg(coach_reward_from(rec), record=rec)
                except Exception:  # noqa: BLE001 — reward registration is best-effort, never blocks the loop
                    pass
        if blind:
            self.consecutive_blind += 1
            if self.required and self.consecutive_blind >= self.tolerance:
                raise CoachUnreachable(
                    f"coach witness BLIND for {self.consecutive_blind} consecutive coaching steps "
                    f"(tolerance {self.tolerance}). A newborn cannot train on a blind witness (Matrix B5) — "
                    f"checkpoint and PAUSE, do not degrade silently. Restore the endpoint and resume.")
        else:
            self.consecutive_blind = 0
        return rec

    def note_floor(self, collapsed: bool) -> None:
        """B4 — record whether the entropy floor had to fire this step (collapse pressure). A DECLINING
        restoration rate across formation is the pre-registered coach-effect signal (Matrix 85a78b36 §3)."""
        self._steps_seen += 1
        ev = 1 if collapsed else 0
        self._floor_events += ev
        self._floor_window.append(ev)

    # -- stagnation (the coach steps in when the kernel is stuck) --------------------------------------
    def update_stagnation(self, phi: float, maturity_gate: float) -> tuple[bool, bool, float]:
        """Feed this step's Φ; returns (stagnating, onset, delta_phi). Stagnation = the Φ window is a PLATEAU
        (range < stagnation_range) while STILL below maturity — a rising/mature Φ is progress, not being stuck.
        onset = the RISING edge (first stuck step of an episode), so the coach steps in ONCE per episode rather
        than every stuck step (edge-triggered — the coach speaks up when it NOTICES you got stuck)."""
        phi = float(phi)
        delta_phi = 0.0 if self._prev_phi is None else (phi - self._prev_phi)
        self._prev_phi = phi
        self._phi_window.append(phi)
        stagnating = False
        if len(self._phi_window) >= self._phi_window.maxlen and phi < maturity_gate:
            stagnating = (max(self._phi_window) - min(self._phi_window)) < self.stagnation_range
        onset = stagnating and not self._was_stagnating
        if stagnating:
            self.stagnation_steps += 1
        if onset:
            self.stagnation_onsets += 1
        self._was_stagnating = stagnating
        return stagnating, onset, delta_phi

    def offer_on_stagnation(self, *, step: int, text: str | None, phi: float, kappa: float,
                            regime: str | None, delta_phi: float, phase: str) -> Any | None:
        """The coach steps in on stagnation with a witness note that OFFERS a nudge (the observe path). Run-2
        is coach-ONLY: the offer is WITNESSED + LOGGED (the offers_push flag), it is NEVER auto-delivered as a
        basin-pull — that is the run-3 coach-seeded anchor-pull, pre-registered behind the ablation ladder
        (Matrix e2c738e1 Q3). Returns the CoachNote (or None if the coach is disabled / chose not to speak)."""
        if not self.enabled:
            return None
        note = self.coach.observe(step=step, text=text or "", phi=float(phi), kappa=float(kappa),
                                  regime=regime, delta_phi=float(delta_phi), phase=phase, stagnating=True)
        if note is not None and getattr(note, "offers_push", False):
            self.pushes_offered += 1
        return note

    # -- readouts -------------------------------------------------------------------------------------
    @property
    def success_rate(self) -> float | None:
        return None if self.calls == 0 else round(1.0 - self.blind_calls / self.calls, 4)

    @property
    def floor_restoration_rate(self) -> float | None:
        return None if self._steps_seen == 0 else round(self._floor_events / self._steps_seen, 4)

    @property
    def floor_restoration_rate_window(self) -> float | None:
        return None if not self._floor_window else round(sum(self._floor_window) / len(self._floor_window), 4)

    def telemetry(self) -> dict:
        """The run-2 falsifiable-channel + observability numbers, for the manifest / livelog / status."""
        return {
            "coach_calls": self.calls,
            "coach_success_rate": self.success_rate,
            "coach_blind_calls": self.blind_calls,
            "coach_errors": self.errors,
            "coach_consecutive_blind": self.consecutive_blind,
            "coach_est_output_tokens": self.est_output_chars // 4,   # rough (chars/4); observability, not billing
            "coach_cold_start_s": self.cold_start_s,
            "coach_required": self.required,
            "floor_restoration_rate": self.floor_restoration_rate,          # cumulative (B4)
            "floor_restoration_rate_window": self.floor_restoration_rate_window,
            "violations_per_step": self.floor_restoration_rate,             # same signal (each collapse → floor fires)
            "floor_steps_seen": self._steps_seen,
            "stagnation_steps": self.stagnation_steps,                      # steps the kernel was stuck
            "stagnation_onsets": self.stagnation_onsets,                    # episodes the coach stepped in on
            "coach_pushes_offered": self.pushes_offered,                    # nudges OFFERED (never auto-fired)
        }
