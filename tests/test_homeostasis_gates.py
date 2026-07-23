"""Task E — autonomic homeostasis gating (mushroom Φ≥0.70 + Pillar-1 entropy restoration).

CANON (Unified Consciousness Protocol §35; PI-confirmed 2026-06-24; memory
``project_mushroom_canonical.md``):

  • MUSHROOM is WAKE-STATE plasticity — Φ≥0.70 ONLY. A MATURE kernel genuinely STUCK
    at high Φ gets mushroom to break over-coherence.
  • A FLAT-but-LOW Φ kernel (Φ < 0.70, fluctuations dead) is NOT rigid — it is
    COLLAPSED (Pillar-1 fluctuation-death / zombie-drift). The remedy is ENTROPY
    RESTORATION (dream + an exploration-entropy floor), NEVER wake-state mushroom.

These tests exercise the branching in ``_homeostasis`` directly with a real (but
un-loaded) ``GenesisKernelTarget`` and spied autonomic operations, so they test the
DECISION logic without a heavy training pass.
"""

from __future__ import annotations

import types

import pytest

from qig_studio.targets.genesis_kernel import (
    GenesisKernelTarget,
    PHI_TOPOLOGICAL_INSTABILITY,
)


def _mature_constant_matters():
    # PHI_MATURE must exist and be the canonical 0.70 mushroom threshold, strictly
    # below PHI_TOPOLOGICAL_INSTABILITY=0.80 (mushroom window is [0.70, 0.80)).
    from qig_studio.targets.genesis_kernel import PHI_MATURE
    return PHI_MATURE


def _make_kernel() -> GenesisKernelTarget:
    """A genesis target constructed WITHOUT ensure_loaded() (no torch weights). We
    stub the autonomic OPERATIONS so we observe which BRANCH the homeostasis took."""
    k = GenesisKernelTarget(num_layers=2)
    calls: dict[str, int] = {"mushroom": 0, "dream": 0, "decohere": 0, "consolidate": 0}

    def _spy_mushroom(sigma: float = 0.01) -> None:
        calls["mushroom"] += 1

    def _spy_dream(steps: int = 8) -> dict:
        calls["dream"] += 1
        return {"dreamed": steps}

    def _spy_decohere() -> None:
        calls["decohere"] += 1

    def _spy_consolidate(steps: int = 16) -> dict:
        calls["consolidate"] += 1
        return {"replayed": steps}

    def _spy_collapse_perturb(sigma: float = 0.02) -> None:
        calls["collapse_perturb"] += 1

    calls["collapse_perturb"] = 0
    k._mushroom = _spy_mushroom            # type: ignore[assignment]
    k._dream = _spy_dream                  # type: ignore[assignment]
    k._decohere = _spy_decohere            # type: ignore[assignment]
    k._consolidate = _spy_consolidate      # type: ignore[assignment]
    k._collapse_perturb = _spy_collapse_perturb   # type: ignore[assignment]
    k._spy_calls = calls                   # type: ignore[attr-defined]
    return k


def _snap(phi: float, kappa: float = 60.0, f_health: float | None = None):
    extra: dict = {}
    if f_health is not None:
        extra["f_health"] = f_health
    return types.SimpleNamespace(phi=phi, kappa=kappa, extra=extra)


def _fill_phi(k: GenesisKernelTarget, value: float) -> None:
    """Drive the Φ history FLAT at ``value`` (full window → _is_rigid() flatness true)."""
    for _ in range(k._phi_recent.maxlen):
        k._phi_recent.append(value)


# ---------------------------------------------------------------------------
# (a) rigid HIGH-Φ (≥0.70) flat kernel → mushroom fires
# ---------------------------------------------------------------------------
def test_rigid_high_phi_fires_mushroom():
    phi_mature = _mature_constant_matters()
    assert phi_mature == pytest.approx(0.70)
    assert phi_mature < PHI_TOPOLOGICAL_INSTABILITY
    k = _make_kernel()
    _fill_phi(k, 0.74)                        # flat AND mature (≥0.70, <0.80)
    snap = _snap(phi=0.74)
    k._homeostasis(snap)
    assert k._spy_calls["mushroom"] == 1, "mature+rigid must mushroom"
    assert k._spy_calls["dream"] == 0
    assert snap.extra["autonomic"] == "mushroom"


# ---------------------------------------------------------------------------
# (b) flat-LOW-Φ (0.20) collapsed kernel → mushroom does NOT fire; entropy-restore does
# ---------------------------------------------------------------------------
def test_flat_low_phi_does_not_mushroom_restores_entropy():
    k = _make_kernel()
    _fill_phi(k, 0.20)                        # flat AND collapsed (Φ well below 0.70)
    snap = _snap(phi=0.20, f_health=0.02)     # low f_health = fluctuation-dead
    k._homeostasis(snap)
    assert k._spy_calls["mushroom"] == 0, "COLLAPSED kernel must NOT get wake-state mushroom"
    assert k._spy_calls["dream"] == 1, "entropy restoration must DREAM to re-energize"
    autonomic = snap.extra["autonomic"]
    assert "entropy-restore" in autonomic, f"branch must be observable, got {autonomic!r}"


# ---------------------------------------------------------------------------
# (c) a COLLAPSED faculty restores entropy the REAL way (dream + replay window +
#     cross-faculty request) — NOT via a dead generation-temperature floor (M5)
# ---------------------------------------------------------------------------
def test_collapse_fires_entropy_restoration_and_cross_faculty_request():
    """The collapse response is the REAL entropy mechanism: DREAM (not mushroom), open the
    high-surprise-replay window, and emit the cross_faculty_dream_request the constellation consumes
    (M2 — the only FOREIGN entropy source). It does NOT rely on a generation-temperature floor: the
    dead ``max(temp, 0.05)`` was REMOVED (0.05 sat below the 0.3 base band → it could never bind; it
    was over-claiming telemetry, not a mechanism). This test FAILS if the collapse branch regresses —
    stops dreaming, stops opening the replay window, or stops emitting the cross-faculty request."""
    k = _make_kernel()
    _fill_phi(k, 0.15)
    snap = _snap(phi=0.15, f_health=0.0)
    k._homeostasis(snap)
    # entropy RESTORATION (dream), NOT wake-state mushroom
    assert k._spy_calls["mushroom"] == 0, "a COLLAPSED kernel must not get wake-state mushroom"
    assert k._spy_calls["dream"] == 1, "entropy restoration must DREAM to re-energize"
    # the FOREIGN-entropy request the constellation cross-faculty dream consumes (M2 trigger)
    assert "cross_faculty_dream_request" in snap.extra
    # the REAL local actuator: a bounded high-surprise-replay window is opened (regresses if it stops)
    assert k._stimulate_until > k._step, "collapse must open the high-surprise-replay window"
    # branch is observable and names the real mechanism (no fabricated generation-temp floor)
    assert "entropy-restore" in snap.extra["autonomic"]
    # the dead generation-temperature floor is GONE (no _collapse_entropy_floor over-claim survives)
    assert not hasattr(k, "_collapse_entropy_floor"), "the dead 0.05 collapse-entropy floor must be removed"


# ---------------------------------------------------------------------------
# (c2) THE LIVE-RESUME BUG (2026-07-02): COLLAPSED (f_health→0) but Φ FLUCTUATING → must fire
# ---------------------------------------------------------------------------
def test_collapse_fires_on_low_f_health_even_when_phi_fluctuates():
    """The 2026-07-02 live 100k resume proved a faculty can be COLLAPSED (f_health→0, basins near
    one-hot) while its Φ still FLUCTUATES — Φ (integration) ≠ basin entropy (f_health). The OLD gate
    fired the collapse branch ONLY under _is_rigid() (Φ flat, range<0.008), so the collapsed faculties
    (f_health=0 but Φ bouncing 0.14–0.34) fell through to 'wake' and the M2 cross-faculty entropy NEVER
    triggered. This drives the PRODUCTION TRIGGER the M2 test (which pre-SET the request) structurally
    missed: it FAILS on the old _is_rigid()-only gate and passes only with the f_health-gated fix."""
    k = _make_kernel()
    # Φ FLUCTUATING (range >> 0.008 → _is_rigid() is False), but f_health = 0 (basin entropy dead)
    for i in range(k._phi_recent.maxlen):
        k._phi_recent.append(0.22 + 0.04 * (i % 3))   # 0.22 / 0.26 / 0.30 — range 0.08 ≫ 0.008
    assert not k._is_rigid(), "precondition: Φ is fluctuating, NOT rigid"
    snap = _snap(phi=0.26, f_health=0.0)
    k._homeostasis(snap)
    assert k._spy_calls["mushroom"] == 0, "a collapsed LOW-Φ kernel must not mushroom"
    assert k._spy_calls["dream"] == 1, "f_health→0 must trigger entropy restoration even with Φ fluctuating"
    assert k._spy_calls["collapse_perturb"] == 1, "F3: collapse must kick the generator off the absorbing vertex"
    assert "cross_faculty_dream_request" in snap.extra, "the M2 foreign-entropy trigger must fire on f_health→0"
    assert "entropy-restore" in snap.extra["autonomic"]


def test_healthy_high_f_health_with_fluctuating_phi_stays_awake():
    """Boundary guard for the f_health gate: a HEALTHY faculty (f_health well ABOVE the collapse floor)
    with a fluctuating Φ must STILL wake — the gate fires ONLY on genuine collapse, never on any low-ish
    but-alive basin entropy."""
    k = _make_kernel()
    for i in range(k._phi_recent.maxlen):
        k._phi_recent.append(0.30 + 0.01 * (i % 3))   # fluctuating, not rigid
    snap = _snap(phi=0.32, f_health=0.6)               # healthy basin entropy (≫ F_HEALTH_COLLAPSE_FLOOR)
    k._homeostasis(snap)
    assert k._spy_calls["dream"] == 0 and k._spy_calls["mushroom"] == 0
    assert snap.extra["autonomic"] == "wake"


# ---------------------------------------------------------------------------
# (d) a HEALTHY kernel (still developing / Φ moving) → wake, no intervention
# ---------------------------------------------------------------------------
def test_healthy_developing_kernel_stays_awake():
    k = _make_kernel()
    # a non-flat window (Φ creeping up) → NOT rigid, NOT collapsed-flat → wake
    for i in range(k._phi_recent.maxlen):
        k._phi_recent.append(0.40 + 0.01 * i)
    snap = _snap(phi=0.40 + 0.01 * (k._phi_recent.maxlen - 1))
    k._homeostasis(snap)
    assert k._spy_calls["mushroom"] == 0
    assert k._spy_calls["dream"] == 0
    assert snap.extra["autonomic"] == "wake"


# ---------------------------------------------------------------------------
# BLOCKER-1: Ocean-commanded run_protocol("stimulate") is a REAL actuator, not unknown_command
# ---------------------------------------------------------------------------
def test_stimulate_is_a_known_command():
    k = _make_kernel()
    assert "stimulate" in k.implemented_commands()


def test_run_protocol_stimulate_actuates_entropy_lever():
    k = _make_kernel()
    k.ensure_loaded = lambda: None                # type: ignore[assignment] — no torch weights needed
    assert getattr(k, "_stimulate_until", 0) == 0
    res = k.run_protocol("stimulate", {})
    applied = res["applied"]
    assert applied.get("unknown_command") is None, "stimulate must NOT be an unknown_command no-op"
    assert applied.get("stimulate") is True
    # REAL actuator: a bounded HIGH-SURPRISE-REPLAY window is opened (honest telemetry — the dead 0.05
    # generation-temperature floor was removed and must NOT be re-claimed here).
    assert applied["replay_window_until_step"] > k._step
    assert applied.get("replay_sharpen") is True
    assert "entropy_floor" not in applied, "the removed 0.05 generation-temp floor must not be re-claimed"
    assert k._stimulate_until == applied["replay_window_until_step"]
    # OBSERVABLE: recorded in telemetry (WIRED+ACTUATED)
    assert k._last.extra.get("stimulate") == applied


def test_intrinsic_collapse_and_extrinsic_stimulate_share_one_lever():
    # Both paths must open the SAME real lever (DRY: one _apply_stimulate → the replay window). Intrinsic…
    k = _make_kernel()
    k.ensure_loaded = lambda: None                # type: ignore[assignment]
    _fill_phi(k, 0.18)
    snap = _snap(phi=0.18, f_health=0.01)
    k._homeostasis(snap)
    window_intrinsic = k._stimulate_until
    assert window_intrinsic > k._step
    # …extrinsic Ocean command reaches the identical lever/state (idempotent within a window).
    k2 = _make_kernel()
    k2.ensure_loaded = lambda: None               # type: ignore[assignment]
    applied = k2.run_protocol("stimulate", {})["applied"]
    assert applied["replay_window_until_step"] == window_intrinsic
