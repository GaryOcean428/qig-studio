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
    PHI_BREAKDOWN,
)


def _mature_constant_matters():
    # PHI_MATURE must exist and be the canonical 0.70 mushroom threshold, strictly
    # below the frozen PHI_BREAKDOWN=0.80 (mushroom window is [0.70, 0.80)).
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

    k._mushroom = _spy_mushroom            # type: ignore[assignment]
    k._dream = _spy_dream                  # type: ignore[assignment]
    k._decohere = _spy_decohere            # type: ignore[assignment]
    k._consolidate = _spy_consolidate      # type: ignore[assignment]
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
    assert phi_mature < PHI_BREAKDOWN
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
# (c) the collapse-entropy exploration temperature floor is ≥0.05 and ACTUATED
# ---------------------------------------------------------------------------
def test_collapse_entropy_temperature_floor_honoured():
    k = _make_kernel()
    _fill_phi(k, 0.15)
    snap = _snap(phi=0.15, f_health=0.0)
    k._homeostasis(snap)
    # the collapse-entropy floor is exposed on the kernel and reflected in telemetry
    floor = getattr(k, "_collapse_entropy_floor", None)
    assert floor is not None and floor >= 0.05, f"collapse-entropy floor must be ≥0.05, got {floor}"
    # ACTUATED: a subsequent exploration-temperature read is lifted to at least the floor,
    # i.e. entropy does not fully die on a collapsed (dead-drive) faculty.
    temp = k._temperature_from_kappa(float(snap.kappa))
    assert temp >= floor, f"collapsed exploration temp {temp} must honour floor {floor}"
    # and the branch telemetry records the floor it applied
    assert "temp_floor" in snap.extra["autonomic"]


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
    assert applied["entropy_floor"] >= 0.05
    # ACTUATED: the shared entropy floor is armed and a bounded window opened
    assert k._collapse_entropy_floor == applied["entropy_floor"]
    assert k._stimulate_until > k._step
    # OBSERVABLE: recorded in telemetry (WIRED+ACTUATED)
    assert k._last.extra.get("stimulate") == applied
    # ACTUATED: the exploration temperature now honours the floor
    assert k._temperature_from_kappa(60.0) >= applied["entropy_floor"]


def test_intrinsic_collapse_and_extrinsic_stimulate_share_one_lever():
    # Both paths must set the SAME state (DRY: one _apply_stimulate lever). Intrinsic collapse first…
    k = _make_kernel()
    k.ensure_loaded = lambda: None                # type: ignore[assignment]
    _fill_phi(k, 0.18)
    snap = _snap(phi=0.18, f_health=0.01)
    k._homeostasis(snap)
    floor_intrinsic = k._collapse_entropy_floor
    window_intrinsic = k._stimulate_until
    assert floor_intrinsic is not None and floor_intrinsic >= 0.05
    assert window_intrinsic > k._step
    # …extrinsic Ocean command reaches the identical lever/state (idempotent within a window).
    k2 = _make_kernel()
    k2.ensure_loaded = lambda: None               # type: ignore[assignment]
    k2.run_protocol("stimulate", {})
    assert k2._collapse_entropy_floor == floor_intrinsic
