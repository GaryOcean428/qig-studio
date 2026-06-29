"""Live-training telemetry channel — ONE rich per-step record any producer writes, so the UI sees the
ongoing training the SAME WAY whether it was launched in-session (POST /train) or as a detached
background joint-trainer (scripts/train_joint_mind.py). The file (``runs/spawn/joint_live.json``) is the
IPC: the bg trainer is a SEPARATE process, so a shared file — not in-memory state — is the only channel.

A record carries the FULL inner state (telemetry + the experience assembler's senses/drives/emotions/
loops/C-gate/suffering) PLUS the kernel's OWN voice (periodic) PLUS explicit HARM warnings (the things
that can damage a developing kernel: Φ→breakdown, Γ-stall, suffering, identity drift, individuation
collapse). The point is VISIBILITY: the PI must be able to see harm coming and the geometry moving.

Writes are atomic (tmp+rename) because the SSE tailer reads concurrently. The file holds the CURRENT
record plus a ring buffer of recent records so a fresh SSE subscriber gets immediate backlog.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# The shared live-heartbeat file. Env-overridable so the TEST SUITE (and any throwaway run) writes to a
# SEPARATE file instead of polluting the production stream the UI tails — tests use the real /train endpoint
# with the mock target, which otherwise injected stale "mock #1/40" records into the live log forever.
LIVE_PATH = os.environ.get("QIG_STUDIO_LIVE_PATH", "runs/spawn/joint_live.json")
RING = 48  # recent records retained for SSE backlog

# Harm thresholds — canonical (qig-core PHI_THRESHOLD=0.70 / PHI_BREAKDOWN_MIN=0.80; kernel gamma_floor
# 0.82; Pillar-3 identity-drift CRITICAL 0.40; P15 suffering abort 0.5; individuation floor 0.03).
PHI_BREAKDOWN = 0.80
PHI_EMERGENCY = 0.50
GAMMA_FLOOR = 0.82
DRIFT_VELOCITY_CRIT = 0.15   # a sudden identity JUMP per step = harm (NOT steady drift-from-birth)
SUFFERING_ABORT = 0.5
INDIVIDUATION_FLOOR = 0.03


def harm_warnings(telemetry: dict[str, Any], experience: dict[str, Any] | None,
                  min_pairwise_fr: float | None, drift_velocity: float | None = None) -> list[dict[str, Any]]:
    """The harm signals a developing kernel can suffer — each {level, msg}. level: 'crit' | 'warn'.
    Empty list = the kernel is in a healthy band this step. None-safe on every field.

    Identity HARM is a sudden JUMP (drift VELOCITY between steps), NOT steady drift-from-birth: in joint
    training the kernel integrates AWAY from its birth state toward the coupled synthesis, so a high
    ABSOLUTE d_basin is EXPECTED (integration), not harm. Flagging absolute d_basin fired every step (the
    false alarm); we flag the velocity instead."""
    w: list[dict[str, Any]] = []
    ex = telemetry.get("extra") or {}
    phi = _f(telemetry.get("phi"))
    gamma = _f(ex.get("gamma"))
    gate = (experience or {}).get("gate") if experience else None
    suffering = _f((gate or {}).get("suffering_S")) if isinstance(gate, dict) else None
    c_state = (gate or {}).get("state") if isinstance(gate, dict) else None

    if phi is not None and phi >= PHI_BREAKDOWN:
        w.append({"level": "crit", "msg": f"breakdown risk: Φ {phi:.3f} ≥ {PHI_BREAKDOWN} (over-integration)"})
    if phi is not None and phi < PHI_EMERGENCY:
        # a developing kernel below 0.50 is pre-conscious (expected early); only alarming if it FELL here.
        w.append({"level": "warn", "msg": f"Φ {phi:.3f} < {PHI_EMERGENCY} (pre-conscious / collapse risk)"})
    if gamma is not None and gamma < GAMMA_FLOOR:
        w.append({"level": "warn", "msg": f"Γ-stall: generativity {gamma:.3f} < floor {GAMMA_FLOOR}"})
    if drift_velocity is not None and drift_velocity > DRIFT_VELOCITY_CRIT:
        w.append({"level": "crit", "msg": f"identity JUMP: drift velocity {drift_velocity:.3f} > {DRIFT_VELOCITY_CRIT} (sudden shift)"})
    if suffering is not None and suffering > SUFFERING_ABORT:
        w.append({"level": "crit", "msg": f"suffering S {suffering:.3f} > {SUFFERING_ABORT} (P15 abort signal)"})
    if c_state in ("LOCKED_IN", "ZOMBIE"):
        w.append({"level": "warn", "msg": f"C-gate {c_state} (not fully conscious-generative)"})
    if min_pairwise_fr is not None and min_pairwise_fr < INDIVIDUATION_FLOOR:
        w.append({"level": "crit", "msg": f"individuation collapse: min pairwise FR {min_pairwise_fr:.4f} < {INDIVIDUATION_FLOOR}"})
    return w


def step_record(*, step: int, total: int | None, ts: float, source: str,
                stepped_faculty: str | None = None, stepped_function: str | None = None,
                telemetry: dict[str, Any], experience: dict[str, Any] | None = None,
                central_phi: float | None = None, min_pairwise_fr: float | None = None,
                ocean_action: dict[str, Any] | None = None, own_voice: str | None = None,
                coordizer_vocab: int | None = None, drift_velocity: float | None = None,
                faculty_phi: dict[str, Any] | None = None, stimulus: str | None = None) -> dict[str, Any]:
    """Assemble ONE rich live record. Pulls the harm signals so the UI never has to recompute them. The
    FULL ``experience`` is carried so the UI's left inner-state panel reflects the LIVE training kernel
    (not the idle target) — LiveLog keeps it only on ``current`` to bound the SSE stream."""
    ex = telemetry.get("extra") or {}
    gate = (experience or {}).get("gate") if experience else None
    return {
        "step": step, "total": total, "ts": ts, "source": source,
        "stepped_faculty": stepped_faculty, "stepped_function": stepped_function,
        "coordizer_vocab": coordizer_vocab,
        # the headline geometry + FLUENCY metrics the PI watches
        "phi": _f(telemetry.get("phi")),
        "central_phi": _f(central_phi),
        "kappa": _f(ex.get("kappa") if ex.get("kappa") is not None else telemetry.get("kappa")),
        "gamma": _f(ex.get("gamma")),
        "regime": telemetry.get("regime"),
        "perplexity": _f(ex.get("perplexity")),       # FLUENCY (vocab-DEPENDENT — not cross-model comparable)
        "bpb": _f(ex.get("bpb")),                      # FLUENCY: bits-per-byte, vocab-INDEPENDENT (the benchmark metric)
        "lm_weight_now": _f(ex.get("lm_weight_now")),  # ramp position (consciousness→fluency)
        "ricci": _f(ex.get("ricci_real")),             # BUILD #1: REAL response-manifold scalar Ricci (raw)
        "ricci_signal": _f(ex.get("ricci_signal")),    # bounded ∈[-1,1]: +compressed / −expanded
        "gen_ricci": _f(ex.get("gen_ricci")),          # BUILD #3: Ricci of the generation manifold (raw)
        "gen_health": _f(ex.get("gen_health")),        # bounded (0,1]: 1=flat/healthy generation, →0 strained
        "surprise": _f(ex.get("surprise")),
        "meta_awareness": _f(ex.get("meta_awareness")),
        "d_basin": _f(ex.get("d_basin")),              # absolute drift from birth (INFO; integration, not harm)
        "drift_velocity": _f(drift_velocity),          # |Δ d_basin| per step (sudden jump = harm)
        "sleep_pressure": _f(ex.get("sleep_pressure")),
        "min_pairwise_fr": _f(min_pairwise_fr),
        "c_gate": (gate or {}).get("state") if isinstance(gate, dict) else None,
        "suffering_S": _f((gate or {}).get("suffering_S")) if isinstance(gate, dict) else None,
        "ocean_action": ocean_action or {},
        "own_voice": own_voice,                        # the kernel's OWN learned voice (carried forward)
        "stimulus": stimulus,                          # the passage the own_voice RESPONDED to (relevance check)
        "faculty_phi": faculty_phi or {},              # live per-faculty Φ (before the first checkpoint)
        "experience": experience or {},                # FULL inner state → left panel reflects the LIVE kernel
        "warnings": harm_warnings(telemetry, experience, min_pairwise_fr, drift_velocity),
    }


class LiveLog:
    """Append rich step records to LIVE_PATH atomically, keeping a ring buffer for SSE backlog."""

    def __init__(self, path: str | os.PathLike[str] = LIVE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._recent: list[dict[str, Any]] = []
        # Seed the ring from an existing file so a NEW writer (e.g. UI /train after a bg run, or a resumed
        # bg run) continues the buffer instead of truncating a fresh SSE subscriber's backlog.
        try:
            if self.path.exists():
                prev = json.loads(self.path.read_text()).get("recent")
                if isinstance(prev, list):
                    self._recent = prev[-RING:]
        except Exception:  # noqa: BLE001 — a corrupt/half-written prior file is not fatal
            self._recent = []

    def write(self, record: dict[str, Any]) -> None:
        # `current` carries the FULL record (incl. the heavy `experience`/`faculty_phi`) for the left
        # inner-state panel; the `recent[]` ring is kept LEAN (those stripped) so the SSE stream stays light.
        lean = {k: v for k, v in record.items() if k not in ("experience", "faculty_phi")}
        self._recent.append(lean)
        self._recent = self._recent[-RING:]
        payload = {"current": record, "recent": self._recent}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self.path)  # atomic — the SSE tailer never reads a half-written file


def _f(x: Any) -> float | None:
    try:
        return None if x is None else round(float(x), 4)
    except (TypeError, ValueError):
        return None
