"""MATURITY-GATED entropy floor (Matrix-corrected) — LEARNING-linked, bidirectional, never-zero.

The proactive Pillar-1 entropy floor (``GenesisKernelTarget._entropy_floor_basin``) cured the
fresh-birth f_health→0 collapse but is the prime suspect for RESETTING first-learning (efficiency.md
§2316 "learns then un-learns"): every time a sharpening (entropy-concentrating) basin dips below the
FIXED floor it gets Dirichlet-mixed back up. The Matrix-corrected spec:

  1. LEARNING-linked relaxation (NOT age/step-count): the effective floor relaxes only on
     DEMONSTRATED self-sustaining sharpening — the per-step train bpb readout held below its
     early-window mean for K consecutive steps. No clock decay: a kernel that is not sharpening
     keeps its full floor forever.
  2. BIDIRECTIONAL + hysteresis: the floor RISES again when f_health re-approaches collapse;
     asymmetric rates (relax slow, tighten fast); a dead zone between the tighten and relax
     f_health bands where the gate holds.
  3. DYNAMIC never-zero minimum: the floor can never relax below the measured collapse-avoidance
     level (a hard never-zero seed, raised by the deepest collapse onset actually observed).
  4. DEFAULT BIT-IDENTICAL: ``floor_mode="normal"`` (the default) reproduces the current behavior
     bit-for-bit; ``"gated"`` is opt-in; ``"off"`` is the diagnostic arm only.

Contract exercised here (no heavy training pass — un-loaded kernel + injected PillarEnforcer,
mirroring test_entropy_floor_stabilizer.py):
  • default mode == bit-identical reference (same np seed → same bytes);
  • gated-at-birth (fully tight) == normal, bit-identical — fresh-collapsed still fully restored;
  • gated relaxes under sustained sharpening: a mildly-collapsed basin passes UNTOUCHED where
    normal mode would reset it;
  • gate re-tightens FAST on f_health re-approach (tighten rate >> relax rate, hysteresis dead zone);
  • never-zero minimum at max relaxation: deep collapse is STILL restored;
  • ``off`` mode is a pure pass-through (diagnostic arm).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qig_core import BASIN_DIM
from qig_core.consciousness.pillars import (
    ENTROPY_FLOOR,
    TEMPERATURE_FLOOR,
    FluctuationGuard,
    PillarEnforcer,
)

from qig_studio.targets.genesis_kernel import EntropyFloorGate, GenesisKernelTarget

HIDDEN = 384  # the K-COMPRESS identity-basin width (Δ³⁸³) = 64 × 6


def _kernel(floor_mode: str = "normal") -> GenesisKernelTarget:
    """Un-loaded genesis target with a real PillarEnforcer injected (floor LIVE, no heavy load)."""
    k = GenesisKernelTarget(num_layers=2, floor_mode=floor_mode)
    k._pillars = PillarEnforcer()
    return k


def _collapsed(width: int = HIDDEN) -> "torch.Tensor":
    """Near-one-hot (deep Pillar-1 fluctuation-death) basin — 64-dim entropy ≈ 0."""
    v = np.full(width, 1e-9, dtype=np.float64)
    v[width // 3] = 1.0
    v = v / v.sum()
    return torch.as_tensor(v, dtype=torch.float32)


def _mildly_collapsed() -> "torch.Tensor":
    """A basin whose 64-dim entropy sits BETWEEN the relaxed floor and ENTROPY_FLOOR (≈0.097 nats):
    the sharpening-kernel case the fixed floor keeps resetting. Built on the exact _d63 block
    binning (repeat by 6) so the reduction recovers the 64-dim distribution exactly."""
    v = np.full(BASIN_DIM, 0.01 / (BASIN_DIM - 1), dtype=np.float64)
    v[0] = 0.99
    v = v / v.sum()
    return torch.as_tensor(np.repeat(v / 6.0, 6), dtype=torch.float32)


def _entropy64(basin, k: GenesisKernelTarget) -> float:
    return float(FluctuationGuard().basin_entropy(k._d63(basin)))


def _reference_floor(k: GenesisKernelTarget, cur: "torch.Tensor", seed: int) -> "torch.Tensor":
    """The ORIGINAL (pre-gate) _entropy_floor_basin restoration, replicated as the bit-identity
    reference: qig-core check_and_enforce on the Δ⁶³ reduction + the divisible-width max-entropy
    block lift. Deterministic under np.random.seed."""
    guard = k._pillars.fluctuation
    d63 = k._d63(cur)
    if guard.basin_entropy(d63) >= ENTROPY_FLOOR:
        return cur
    np.random.seed(seed)
    corrected64, _t, _s = guard.check_and_enforce(np.asarray(d63, dtype=np.float64),
                                                  float(TEMPERATURE_FLOOR))
    corrected64 = np.asarray(corrected64, dtype=np.float64).ravel()
    width = cur.numel()
    g = width // BASIN_DIM
    lifted = np.repeat(corrected64 / g, g)
    lifted = lifted / float(lifted.sum())
    return torch.as_tensor(lifted, dtype=cur.dtype, device=cur.device).reshape(cur.shape)


def _relax(gate: EntropyFloorGate, n_relax_steps: int) -> None:
    """Drive a gate into demonstrated-sharpening relaxation via its public observe API only:
    healthy f_health + early window at bpb=1.0, then sustained sharpening at bpb=0.5."""
    for _ in range(gate.EARLY_WINDOW):
        gate.observe_health(0.9)
        gate.observe_signal(1.0)
    for _ in range(gate.SUSTAIN_K + n_relax_steps):
        gate.observe_health(0.9)
        gate.observe_signal(0.5)


# ---------------------------------------------------------------- default bit-identical


def test_default_floor_mode_is_normal_and_gate_absent():
    k = GenesisKernelTarget(num_layers=2)
    assert k.floor_mode == "normal"
    assert k._floor_gate is None


def test_default_mode_bit_identical_to_current_behavior():
    """floor_mode='normal' (the default) must reproduce the pre-gate restoration BIT-FOR-BIT
    (same np seed → torch.equal), and leave a healthy basin as the SAME object."""
    k = _kernel("normal")
    cur = _collapsed()
    ref = _reference_floor(k, cur, seed=1234)
    np.random.seed(1234)
    out = k._entropy_floor_basin(cur)
    assert torch.equal(out, ref), "default mode drifted from the current (pre-gate) behavior"
    # healthy basin: same object, untouched (the existing floor contract)
    healthy = torch.as_tensor(np.random.default_rng(7).dirichlet(np.ones(HIDDEN) * 3.0),
                              dtype=torch.float32)
    assert k._entropy_floor_basin(healthy) is healthy


def test_gated_fresh_kernel_fully_restores_collapsed_bit_identical_to_normal():
    """At birth the gate is fully tight (tightness=1) → gated == normal bit-for-bit: a
    fresh-collapsed kernel is STILL fully restored (the cure the floor exists for is kept)."""
    k_gated = _kernel("gated")
    cur = _collapsed()
    ref = _reference_floor(k_gated, cur, seed=99)
    np.random.seed(99)
    out = k_gated._entropy_floor_basin(cur)
    assert out is not cur, "fresh-collapsed basin must be restored in gated mode"
    assert torch.equal(out, ref), "gated-at-birth must be bit-identical to normal mode"
    assert _entropy64(out, k_gated) >= ENTROPY_FLOOR


# ---------------------------------------------------------------- learning-linked relaxation


def test_gated_relaxes_under_sustained_sharpening_where_normal_resets():
    """LEARNING-linked (Matrix): after sustained sharpening (bpb held below its early-window mean
    for K+ steps with healthy f_health) the effective floor drops, so a mildly-collapsed
    (sharpening) basin passes UNTOUCHED — where normal mode resets it with Dirichlet noise."""
    mild = _mildly_collapsed()
    k_norm = _kernel("normal")
    e = _entropy64(mild, k_norm)
    assert 0.05 < e < ENTROPY_FLOOR, f"fixture entropy {e:.4f} not in the mild-collapse band"

    # normal mode RESETS it (the suspected un-learning)
    np.random.seed(5)
    out_norm = k_norm._entropy_floor_basin(mild)
    assert out_norm is not mild, "normal mode should floor a below-ENTROPY_FLOOR basin"

    # gated mode after demonstrated sharpening: effective floor below the basin's entropy → untouched
    k_gated = _kernel("gated")
    _relax(k_gated._floor_gate, n_relax_steps=30)  # tightness 1.0 → ~0.4
    assert k_gated._floor_gate.tightness < 0.7
    assert k_gated._floor_gate.effective_floor(float(ENTROPY_FLOOR)) < e
    out_gated = k_gated._entropy_floor_basin(mild)
    assert out_gated is mild, "a sharpening basin above the relaxed floor must pass untouched"


def test_relaxation_requires_sustained_signal_not_a_single_dip():
    """One good step is NOT demonstrated learning: the sustain counter resets on any
    non-sharpening step, so alternating bpb never relaxes the floor. NO step-count decay."""
    gate = EntropyFloorGate()
    for _ in range(gate.EARLY_WINDOW):
        gate.observe_health(0.9)
        gate.observe_signal(1.0)
    for _ in range(100):  # alternate: never K consecutive sharpening steps
        gate.observe_health(0.9)
        gate.observe_signal(0.5)
        gate.observe_health(0.9)
        gate.observe_signal(1.0)
    assert gate.tightness == 1.0, "floor relaxed without SUSTAINED sharpening (age/step leak)"


# ---------------------------------------------------------------- bidirectional + hysteresis


def test_gate_retightens_fast_on_f_health_reapproach():
    """BIDIRECTIONAL: after relaxation, f_health re-approaching collapse RAISES the floor again —
    and asymmetrically (tighten per-step rate >> relax per-step rate)."""
    gate = EntropyFloorGate()
    _relax(gate, n_relax_steps=30)
    relaxed = gate.tightness
    assert relaxed < 1.0
    floor_relaxed = gate.effective_floor(float(ENTROPY_FLOOR))

    gate.observe_health(0.20)  # re-approaching collapse (F_HEALTH_COLLAPSE_FLOOR=0.15)
    tighten_delta = gate.tightness - relaxed
    assert tighten_delta > 0, "floor did not rise on f_health re-approach"
    assert tighten_delta > gate.RELAX_RATE * 5, "tightening must be much faster than relaxation"
    assert gate.effective_floor(float(ENTROPY_FLOOR)) > floor_relaxed

    # a few more collapse-approach steps → fully tight again
    for _ in range(10):
        gate.observe_health(0.20)
    assert gate.tightness == 1.0


def test_hysteresis_dead_zone_holds_the_gate():
    """Between the tighten band (f_health < F_TIGHTEN) and the relax-permission band
    (f_health ≥ F_RELAX_OK) the gate HOLDS: no tightening, and sharpening does not relax."""
    gate = EntropyFloorGate()
    _relax(gate, n_relax_steps=10)
    held = gate.tightness
    mid = (gate.F_TIGHTEN + gate.F_RELAX_OK) / 2.0
    for _ in range(50):
        gate.observe_health(mid)     # inside the dead zone
        gate.observe_signal(0.5)     # sharpening signal, but health not demonstrably safe
    assert gate.tightness == pytest.approx(held), "gate moved inside the hysteresis dead zone"


# ---------------------------------------------------------------- dynamic never-zero minimum


def test_never_zero_minimum_at_max_relaxation():
    """The effective floor NEVER reaches 0: at maximal relaxation it rests at the measured
    collapse-avoidance level — a hard never-zero seed, raised by the deepest observed onset."""
    gate = EntropyFloorGate()
    gate.tightness = 0.0  # force max relaxation
    fmin = gate.effective_floor(float(ENTROPY_FLOOR))
    assert fmin > 0.0, "effective floor collapsed to zero"
    assert fmin == pytest.approx(gate.MIN_FRAC * float(ENTROPY_FLOOR))

    # a measured collapse onset RAISES the minimum (measured collapse-avoidance level)
    gate.record_fire(0.06)
    fmin2 = gate.effective_floor(float(ENTROPY_FLOOR))
    assert fmin2 >= gate.ONSET_MARGIN * 0.06 - 1e-12
    assert fmin2 <= float(ENTROPY_FLOOR)


def test_deep_collapse_still_restored_at_max_relaxation():
    """Even a fully-relaxed gate still restores DEEP collapse (entropy below the never-zero
    minimum) — relaxation can never disable collapse-avoidance."""
    k = _kernel("gated")
    k._floor_gate.tightness = 0.0
    cur = _collapsed()
    assert _entropy64(cur, k) < k._floor_gate.effective_floor(float(ENTROPY_FLOOR))
    out = k._entropy_floor_basin(cur)
    assert out is not cur, "deep collapse passed through a fully-relaxed gate"
    assert _entropy64(out, k) >= ENTROPY_FLOOR, "restoration must still fully restore"
    assert k._floor_gate.fires == 1, "the onset must be recorded (measured collapse level)"


# ---------------------------------------------------------------- off mode + validation


def test_floor_off_mode_is_pure_passthrough():
    """The diagnostic arm: floor_mode='off' returns even a deep-collapsed basin unchanged."""
    k = _kernel("off")
    cur = _collapsed()
    assert k._entropy_floor_basin(cur) is cur


def test_unknown_floor_mode_raises():
    with pytest.raises(ValueError):
        GenesisKernelTarget(num_layers=2, floor_mode="sometimes")
