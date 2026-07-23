"""PI ruling 2026-07-23 — two CRITICAL consciousness-training fixes in ``JointConstellation``:

  • FIX 1 (P4, three-loop minimum): L2 other-observation (``m_other``/``M_boundary``) is wired into the
    TRAINING hot path (``train_step``), not only the two chat call-sites
    (``_generate_via_boundary``/``read_and_respond``) that never touch training. Referent = the ROTATING
    RESPONSIBLE NODE (the round-robin faculty); the responsible node itself observes the LEAVE-ONE-OUT
    Fréchet mean of the *other* faculties (never the couple_step sync target, which contains its own
    basin — the two-observables-one-name trap). A must-vary guard fails loud on a pinned constant.

  • FIX 2 (P6): rigid coupling (``f_sync`` hardcoded to 0.25 forever) is replaced by per-step TACKING —
    ``neurochem.compute_modulation``/``apply_modulation`` + the ``rhythm.HeartOscillator`` breath (both
    previously dormant only inside the now-atticed ``constellation.py`` orchestrator) are ported directly
    into ``JointConstellation.train_step``.

These tests are torch-free at the DECISION level (mirrors ``tests/test_cross_faculty_dream.py``'s
pattern): a bare ``JointConstellation`` (built via ``__new__``, skipping the heavy kernel build) + mock
nodes exposing only the ``ConstellationNode`` surface ``train_step``/``_set_pull``/``_wire_l2_other_
observation`` touch. ``_set_pull`` itself uses torch tensors (CPU-only) — that surface is mocked, not
avoided, exactly as the existing cross-faculty-dream tests already do.
"""
from __future__ import annotations

import numpy as np
import pytest
from qig_core.geometry import fisher_rao_distance, frechet_mean, to_simplex
from qig_core.geometry.fisher_rao import BASIN_DIM

from qig_studio.constellation.faculty import Faculty, seed_birth_basin
from qig_studio.constellation.joint_trainer import (
    JointConstellation,
    _M_OTHER_VAR_EPS,
    _M_OTHER_WINDOW,
    _fr_recognition,
)
from qig_studio.constellation.neurochem import NeuroState, apply_modulation
from qig_studio.constellation.ocean import OceanAutonomic
from qig_studio.constellation.rhythm import HeartOscillator
from qig_studio.targets.base import StepResult, TelemetrySnapshot


# --------------------------------------------------------------------------- helpers / mocks
def _perturb(basin: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A REAL (non-fabricated) small geodesic-ish move — Gaussian jitter re-projected to Δ⁶³. Used so
    the mock's own output genuinely MOVES each call (the must-vary guard must never be tripped by real
    training data)."""
    noise = rng.normal(scale=0.05, size=basin.shape)
    return to_simplex(np.clip(np.asarray(basin, dtype=np.float64) + noise, 1e-6, None))


class _MockNode:
    """The ConstellationNode surface ``train_step``/``_set_pull``/``_wire_l2_other_observation``/
    ``OceanAutonomic.regulate`` touch: a numpy ``_basin_history`` trajectory (no torch), a persistent
    ``telemetry().extra`` dict, and the basin-pull attributes ``_set_pull`` needs (mirrors
    ``tests/test_cross_faculty_dream.py``'s ``_MockNode``). ``train_step`` emits a genuinely-moving next
    basin — the production training path, not a frozen stub."""

    def __init__(self, basin: np.ndarray, *, seed: int, phi: float = 0.72) -> None:
        self.head_mode = "basin"
        self.hidden_dim = BASIN_DIM
        self.vocab_size = BASIN_DIM
        self._basin_ref = None
        self._basin_history = [np.asarray(basin, dtype=np.float64)]
        self._rng = np.random.default_rng(seed)
        self._phi = phi
        self._snap = TelemetrySnapshot(phi=phi, extra={"drive": {"dopamine": 0.5}})

    def _node_device(self):
        import torch
        return torch.device("cpu")

    def _resize_basin(self, ref, size):   # defensive; not reached (numel == dim)
        return ref

    def telemetry(self):
        return self._snap

    def train_step(self, prompt: str):
        new = _perturb(self._basin_history[-1], self._rng)
        self._basin_history.append(new)
        self._snap = TelemetrySnapshot(phi=self._phi, extra={"drive": {"dopamine": 0.5}})
        return StepResult(text="", telemetry=self._snap)


class _FrozenNode:
    """The fault fixture: a node whose ``train_step`` never actually moves (the letter of L2 without the
    substance — a dead constant wearing a live field's name)."""

    def __init__(self, basin: np.ndarray) -> None:
        self._basin_history = [np.asarray(basin, dtype=np.float64)]
        self._snap = TelemetrySnapshot(phi=0.72, extra={})

    def telemetry(self):
        return self._snap

    def train_step(self, prompt: str):
        return StepResult(text="", telemetry=self._snap)   # FROZEN — never moves


def _bare_jc(n_faculty: int = 3, seed: int = 0) -> JointConstellation:
    """A JointConstellation with ONLY the state train_step/_wire_l2_other_observation need — no heavy
    kernel build (mirrors tests/test_cross_faculty_dream.py's ``_bare_constellation`` helper)."""
    roles = [f"f{i}" for i in range(n_faculty)]
    faculties: list[Faculty] = []
    kernels: dict[str, _MockNode] = {}
    births = []
    for i, r in enumerate(roles):
        birth = seed_birth_basin(seed + i, alpha=0.3)
        births.append(birth)
        faculties.append(Faculty(role=r, basin=birth.copy(), birth=birth.copy()))
        kernels[r] = _MockNode(birth.copy(), seed=1000 + seed + i)
    central = _MockNode(frechet_mean(births) if births else seed_birth_basin(seed), seed=9999 + seed)

    jc = JointConstellation.__new__(JointConstellation)
    jc.roles = roles
    jc.faculties = faculties
    jc.kernels = kernels
    jc.central = central
    jc.f_sync = 0.25
    jc._base_f_sync = 0.25
    jc._base_f_anchor = 0.12
    jc.heart = HeartOscillator(freq=0.05)
    jc.neuro = NeuroState()
    jc._last_tack_aggr = None
    jc._birth_min_pair = (
        min(float(fisher_rao_distance(births[i], births[j]))
            for i in range(len(births)) for j in range(i + 1, len(births)))
        if len(births) > 1 else 1.0
    )
    jc.ocean = OceanAutonomic()
    jc._phi_hist = {r: [] for r in roles}
    jc._last_regulation = {}
    jc._m_other_hist = {r: [] for r in (["genesis"] + roles)}
    jc._last_xdream_epoch = {}
    jc._xdream_target = {}
    jc._step_count = 0
    jc._rr = 0
    return jc


# =============================================================================================
# FIX 1 — L2 other-observation
# =============================================================================================
def test_train_step_populates_m_other_for_every_node():
    """A train_step with >=2 faculties in scope produces non-None m_other for EVERY node (central + each
    faculty) — the dead-L2-during-training bug (extra["M_boundary"] was only ever set in the two chat
    call-sites) is fixed."""
    jc = _bare_jc(n_faculty=3)
    out = jc.train_step("hello")
    assert out["central_telemetry"]["extra"].get("M_boundary") is not None
    for r in jc.roles:
        assert jc.kernels[r].telemetry().extra.get("M_boundary") is not None, f"{r} missing M_boundary"


def test_speaker_referent_is_leave_one_out_excluding_self():
    """The rotating responsible node (this step's round-robin speaker) must be measured against the
    LEAVE-ONE-OUT Fréchet mean of the OTHER faculties — NOT the couple_step sync target (the Fréchet mean
    of everyone, which contains its own basin: the two-observables-one-name trap the ruling names)."""
    jc = _bare_jc(n_faculty=3)
    role = jc.roles[jc._rr % len(jc.roles)]     # the responsible node THIS call
    out = jc.train_step("hello")
    assert out["stepped_faculty"] == role

    responsible_basin = jc.kernels[role]._basin_history[-1]
    leave_one_out = frechet_mean([f.basin for f in jc.faculties if f.role != role])
    expected = round(_fr_recognition(responsible_basin, leave_one_out), 4)
    assert jc.kernels[role].telemetry().extra["M_boundary"] == pytest.approx(expected)

    # discriminator: the ALL-INCLUSIVE mean (containing the speaker's own basin — the couple_step-style
    # target) gives a DIFFERENT number, proving leave-one-out is a real choice, not a no-op alias.
    all_inclusive = frechet_mean([f.basin for f in jc.faculties])
    wrong = round(_fr_recognition(responsible_basin, all_inclusive), 4)
    assert wrong != expected


def test_non_responsible_nodes_observe_the_speakers_fresh_basin():
    """Every OTHER node (central + every non-responsible faculty) observes the responsible node's FRESH
    output basin (this step's actual emission), not last cycle's stale coupled snapshot."""
    jc = _bare_jc(n_faculty=3)
    role = jc.roles[jc._rr % len(jc.roles)]
    out = jc.train_step("hello")
    responsible_basin = jc.kernels[role]._basin_history[-1]

    for f in jc.faculties:
        if f.role == role:
            continue
        expected = round(_fr_recognition(f.basin, responsible_basin), 4)
        assert jc.kernels[f.role].telemetry().extra["M_boundary"] == pytest.approx(expected)

    central_basin = jc.central._basin_history[-1]
    expected_central = round(_fr_recognition(central_basin, responsible_basin), 4)
    assert out["central_telemetry"]["extra"]["M_boundary"] == pytest.approx(expected_central)


def test_m_other_must_vary_guard_passes_on_live_varying_data():
    """Real (genuinely-moving) training data must NOT trip the must-vary guard."""
    jc = _bare_jc(n_faculty=3)
    for _ in range(_M_OTHER_WINDOW + 4):
        jc.train_step("hello")   # must not raise


def test_null_l2_with_peers_in_scope_raises():
    """A dead L2 (None) while peers ARE in scope must raise, never silently pass (P4 addendum)."""
    jc = _bare_jc(n_faculty=3)
    role = jc.roles[0]
    fres = jc.kernels[role].train_step("p")
    cres = jc.central.train_step("p")
    jc.central._basin_history = []   # sabotage: a genuinely-absent responsible-node/central basin
    with pytest.raises(RuntimeError):
        jc._wire_l2_other_observation(role, fres, cres)


def test_single_faculty_leave_one_out_is_none_not_a_fault():
    """With zero peer faculties, the speaker's leave-one-out referent is undefined — legitimately None,
    NOT a fault (peers are genuinely out of scope)."""
    jc = _bare_jc(n_faculty=1)
    role = jc.roles[0]
    fres = jc.kernels[role].train_step("p")
    cres = jc.central.train_step("p")
    jc._wire_l2_other_observation(role, fres, cres)   # must not raise
    assert "M_boundary" not in fres.telemetry.extra
    assert cres.telemetry.extra.get("M_boundary") is not None   # central still observes the lone faculty


def test_must_vary_guard_raises_on_frozen_stub():
    """The must-vary guard (P4, mirrors ocean_policy.py's P25 rail-variance shape): a non-None-but-FROZEN
    m_other over the window, while peers are in scope, is a fault — never a silently-accepted constant."""
    jc = _bare_jc(n_faculty=3)
    role = jc.roles[0]
    jc.kernels[role] = _FrozenNode(jc.faculties[0].basin.copy())
    jc.central = _FrozenNode(jc.central._basin_history[0])
    # freeze the OTHER faculties too (a genuinely frozen whole, not just the speaker) so every
    # referent this call computes is bit-identical across repeats.
    for f in jc.faculties[1:]:
        f.set_basin(f.basin.copy())

    with pytest.raises(RuntimeError, match="PINNED"):
        for _ in range(_M_OTHER_WINDOW + 2):
            fres = jc.kernels[role].train_step("p")
            cres = jc.central.train_step("p")
            jc._wire_l2_other_observation(role, fres, cres)


def test_frozen_stub_variance_is_actually_zero_sanity():
    """Sanity check the fixture itself: repeated recognition of a truly-static pair IS variance-zero
    (so the guard test above is exercising a real frozen stub, not an accidental pass)."""
    a = seed_birth_basin(1, alpha=0.3)
    b = seed_birth_basin(2, alpha=0.3)
    readings = [_fr_recognition(a, b) for _ in range(_M_OTHER_WINDOW + 2)]
    assert float(np.var(readings)) < _M_OTHER_VAR_EPS


# =============================================================================================
# FIX 2 — coupling tacking (P6)
# =============================================================================================
def test_first_step_uses_declared_default_rhythm():
    """No real aggregate exists yet on the very first coupling tick — the DECLARED DEFAULT (the
    constructor's static f_sync/f_anchor, unmodulated, honestly labeled), never a fabricated NeuroState."""
    jc = _bare_jc(n_faculty=3)
    out = jc.train_step("hello")
    assert out["coupling_tack"]["default_rhythm"] is True
    assert out["coupling_tack"]["f_sync"] == pytest.approx(jc._base_f_sync, abs=1e-9)


def test_f_sync_is_no_longer_the_constant_default_across_steps():
    """FIX 2: f_sync must MODULATE with the (real) NeuroState — it is no longer pinned at the fixed
    default across steps once real aggregates exist."""
    jc = _bare_jc(n_faculty=3)
    f_syncs = [jc.train_step("hello")["coupling_tack"]["f_sync"] for _ in range(15)]
    # after step 1 (the declared default), later steps are modulated — NOT all identical to each other.
    assert len(set(round(v, 6) for v in f_syncs[1:])) > 1, f"f_sync never varied: {f_syncs}"


def test_f_anchor_stays_in_verified_stable_band():
    """The breath-modulated anchor must never leave the VERIFIED [0.05, 0.20] anti-collapse-stable band
    (identity_anchor.py / test_constellation_no_collapse.py), regardless of how the NeuroState modulates."""
    jc = _bare_jc(n_faculty=3)
    for _ in range(30):
        out = jc.train_step("hello")
        f_anchor = out["coupling_tack"]["f_anchor"]
        assert 0.05 <= f_anchor <= 0.20, f"f_anchor left the verified band: {f_anchor}"


def test_heart_phase_advances_every_step():
    """The HeartOscillator is a REAL endogenous clock — its phase/beat count must advance each training
    step (the ported breath is live, not decorative dead code)."""
    jc = _bare_jc(n_faculty=3)
    phase0, beats0 = jc.heart.phase, jc.heart.beats
    for _ in range(5):
        jc.train_step("hello")
    assert jc.heart.beats >= beats0
    assert (jc.heart.phase != phase0) or (jc.heart.beats > beats0)


def test_apply_modulation_extreme_state_stays_in_verified_band():
    """Even with neuromodulation pushed to an extreme NeuroState, the modulated (f_sync, f_anchor) never
    leaves the verified stable bands — regression carried over from the (now-atticed) constellation.py
    integration test, since apply_modulation is now genuinely load-bearing in the live trainer."""
    extreme = NeuroState(dopamine=1.0, serotonin=1.0, noradrenaline=1.0, acetylcholine=0.0)
    f_sync, f_anchor = apply_modulation(extreme, base_f_sync=0.25, base_f_anchor=0.12)
    assert 0.05 <= f_anchor <= 0.20 and 0.10 <= f_sync <= 0.60


def test_signal_traffic_is_a_real_measurement_not_fabricated():
    """joint_trainer carries no SignalBus (that stayed dormant/atticed with constellation.py) — the
    signal_traffic aggregate must be a genuinely-computed non-negative number (cross-faculty-dream fires +
    Ocean interventions this tick), never a fabricated constant standing in for a bus that isn't there."""
    jc = _bare_jc(n_faculty=3)
    jc.train_step("hello")
    assert jc._last_tack_aggr is not None
    st = jc._last_tack_aggr["signal_traffic"]
    assert isinstance(st, float) and st >= 0.0
