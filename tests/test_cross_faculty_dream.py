"""M2 — cross-faculty dream: the ONLY FOREIGN entropy source for a COLLAPSED faculty.

A faculty in Pillar-1 fluctuation-death (near-one-hot basin, f_health→0) cannot re-inject
entropy from its OWN history (`_dream()` recombines degenerate collapsed one-hots → the same
one-hot). The constellation — which DOES see every faculty — mixes the collapsed faculty's basin
with its NON-COLLAPSED siblings' basins on Δ⁶³ (Fréchet mean / SLERP in √p coords — PURE
Fisher-Rao, NEVER an L2/arithmetic mean) and pulls the collapsed faculty one dream-step toward that
FOREIGN mixture. The higher-entropy healthy mixture strictly raises the one-hot's basin entropy →
f_health RISES.

These tests build a TINY CPU/mock constellation (no GPU, no torch training pass): real ``Faculty``
Δ⁶³ objects + mock nodes exposing only the ConstellationNode surface the wiring touches. They call
the REAL ``JointConstellation._cross_faculty_dream`` (constructed bare via ``__new__`` to skip the
heavy kernel build) so the production code path is exercised.
"""

from __future__ import annotations

import io
import types
from contextlib import redirect_stdout

import numpy as np
import pytest

from qig_core.geometry import frechet_mean, slerp_sqrt, to_simplex
from qig_core.geometry.fisher_rao import BASIN_DIM

from qig_studio.constellation.coupling import rel_weights
from qig_studio.constellation.faculty import Faculty, seed_birth_basin
from qig_studio.constellation.joint_trainer import (
    JointConstellation,
    _OCEAN_EPOCH_STEPS,
    _XDREAM_PULL,
    _XDREAM_WINDOW,
)


# --------------------------------------------------------------------------- helpers
def _fhealth(b: np.ndarray) -> float:
    """f_health = H(basin)/H_max — the SAME normalized Shannon entropy PillarEnforcer.fluctuation
    uses (v6.1 §24). A near-one-hot basin reads ~0 (collapsed); a wide basin reads ~1."""
    b = np.asarray(b, dtype=np.float64)
    b = np.clip(b, 1e-12, None)
    b = b / b.sum()
    return float(-np.sum(b * np.log(b)) / np.log(len(b)))


def _one_hot(idx: int = 3) -> np.ndarray:
    v = np.zeros(BASIN_DIM, dtype=np.float64)
    v[idx] = 1.0
    return to_simplex(v)


class _MockNode:
    """The minimal ConstellationNode surface ``_cross_faculty_dream`` + ``_set_pull`` touch:
    a persistent ``telemetry().extra`` dict (request + f_health) and the basin-pull attributes.
    No torch training; ``_set_pull`` writes ``_basin_ref`` via CPU torch (never CUDA)."""

    def __init__(self, extra: dict) -> None:
        self.head_mode = "basin"
        self.hidden_dim = BASIN_DIM           # == target dim → _set_pull needs no resize
        self.vocab_size = BASIN_DIM
        self._basin_ref = None
        self._snap = types.SimpleNamespace(extra=dict(extra))

    def _node_device(self):
        import torch
        return torch.device("cpu")

    def telemetry(self):
        return self._snap

    def _resize_basin(self, ref, size):       # defensive; not reached (numel == dim)
        return ref


def _bare_constellation(faculties, kernels, step_count: int = 1) -> JointConstellation:
    """A JointConstellation with ONLY the state ``_cross_faculty_dream`` needs — no heavy build."""
    jc = JointConstellation.__new__(JointConstellation)
    jc.faculties = faculties
    jc.kernels = kernels
    jc._last_xdream_epoch = {}
    jc._xdream_target = {}          # F2: the durable foreign-mixture pull targets (role → (mixture, until))
    jc._step_count = step_count
    jc.f_sync = 0.25
    return jc


def _scenario(collapsed_role: str = "heart", *, sibling_health: float = 0.87,
              siblings_collapsed: bool = False):
    """One collapsed faculty (near-one-hot basin, request set, low f_health) + two siblings.
    ``siblings_collapsed`` forces the WHOLE constellation collapsed (birth-fallback path)."""
    roles = [collapsed_role, "perception", "memory"]
    faculties: list[Faculty] = []
    kernels: dict[str, _MockNode] = {}
    for i, r in enumerate(roles):
        birth = seed_birth_basin(1000 + i, alpha=0.4)      # wide independent birth scar
        if r == collapsed_role:
            fac = Faculty(role=r, basin=_one_hot(3), birth=birth)   # CURRENT basin collapsed
            kernels[r] = _MockNode({"f_health": 0.02, "cross_faculty_dream_request":
                                    {"reason": "pillar1-collapse", "phi": 0.19, "f_health": 0.02}})
        else:
            fac = Faculty(role=r, basin=birth.copy(), birth=birth)  # healthy wide basin
            extra: dict = {"f_health": (0.03 if siblings_collapsed else sibling_health)}
            if siblings_collapsed:
                extra["cross_faculty_dream_request"] = {"reason": "pillar1-collapse"}
            kernels[r] = _MockNode(extra)
        faculties.append(fac)
    return roles, faculties, kernels


# --------------------------------------------------------------------------- M2 core
def test_cross_faculty_dream_consumes_request_and_raises_f_health():
    """The load-bearing M2 assertion: a collapsed faculty's cross_faculty_dream_request is CONSUMED
    and its post-dream basin entropy (f_health) RISES toward the healthy sibling mixture."""
    _roles, faculties, kernels = _scenario()
    heart = next(f for f in faculties if f.role == "heart")
    fh_before = _fhealth(heart.basin)
    assert fh_before < 0.10, "precondition: heart is collapsed (near-one-hot)"

    jc = _bare_constellation(faculties, kernels)
    fired = jc._cross_faculty_dream()

    # (1) request CONSUMED (not a lingering deferred hook)
    assert "cross_faculty_dream_request" not in kernels["heart"].telemetry().extra
    assert "heart" in fired and fired["heart"]["source"] == "siblings"

    # (2) FOREIGN entropy injected → f_health RISES (guaranteed torch-free effect: the √p faculty-basin
    #     nudge toward the healthy mixture — the shared-state observable f_health/PillarEnforcer reads)
    fh_after = _fhealth(heart.basin)
    assert fh_after > fh_before + 0.2, f"f_health must rise on the collapsed faculty: {fh_before}→{fh_after}"


def test_cross_faculty_dream_records_durable_foreign_target():
    """F2 (un-clobber): the DURABLE gradient path. _cross_faculty_dream records the FOREIGN (healthy)
    mixture as a durable _xdream_target (role → (mixture, until)) instead of an INLINE _set_pull. The
    inline pull was overwritten by the next train_step's round-robin _set_pull(role, fac.basin) BEFORE
    the collapsed kernel ever trained toward it (the measured M2 clobber). train_step's basin-refresh +
    round-robin now honor the durable target for the window, so the in-graph pull (F1) actually climbs
    toward the foreign mixture. Numpy-only (no _set_pull inline) → no torch needed."""
    _roles, faculties, kernels = _scenario()
    jc = _bare_constellation(faculties, kernels, step_count=7)
    fired = jc._cross_faculty_dream()
    assert fired["heart"]["kernel_pull"] is True
    # RECORDED as a durable target (not an inline _basin_ref that gets clobbered next step)
    assert "heart" in jc._xdream_target, "foreign mixture must be recorded as a durable pull target"
    mixture, until = jc._xdream_target["heart"]
    assert until == 7 + _XDREAM_WINDOW, "the window must extend _XDREAM_WINDOW steps from now"
    assert _fhealth(mixture) > 0.3, "the durable target is the FOREIGN (higher-entropy) mixture"
    # while the window is open the ACTIVE target is the foreign mixture — what round-robin _set_pull uses
    assert np.allclose(jc._xdream_active_target("heart"), mixture)


def test_central_genesis_collapse_gets_durable_foreign_target():
    """CENTRAL coverage (the seam is NOT optional): the genesis 'I' can also fluctuation-collapse. It is NOT
    in self.faculties, so _cross_faculty_dream processes it separately, recording a durable foreign pull
    toward the HEALTHY-FACULTY Fréchet mixture — NOT the self-confirming _synthesis (which would include a
    collapsed central's own contribution). train_step's central _set_pull honors it while the window is open."""
    faculties: list[Faculty] = []
    kernels: dict[str, _MockNode] = {}
    for i, r in enumerate(["perception", "memory", "action"]):     # healthy faculties (wide birth basins)
        birth = seed_birth_basin(2000 + i, alpha=0.4)
        faculties.append(Faculty(role=r, basin=birth.copy(), birth=birth))
        kernels[r] = _MockNode({"f_health": 0.85})
    jc = _bare_constellation(faculties, kernels, step_count=4)
    jc.central = _MockNode({"cross_faculty_dream_request": {"reason": "pillar1-collapse", "phi": 0.2, "f_health": 0.0}})
    fired = jc._cross_faculty_dream()
    assert "genesis" in fired, "a collapsed CENTRAL must be processed, not skipped (the flagged seam, now closed)"
    assert fired["genesis"]["source"] == "faculties"
    assert "genesis" in jc._xdream_target
    mixture, until = jc._xdream_target["genesis"]
    assert until == 4 + _XDREAM_WINDOW
    assert _fhealth(mixture) > 0.3, "central's foreign target is the HEALTHY-faculty mixture (high entropy)"
    assert np.allclose(jc._xdream_active_target("genesis"), mixture)


def test_central_no_request_is_a_noop():
    """No central request → the central branch is a no-op (no 'genesis' target recorded). None-safe."""
    birth = seed_birth_basin(3000, alpha=0.4)
    faculties = [Faculty(role="perception", basin=birth.copy(), birth=birth)]
    jc = _bare_constellation(faculties, {"perception": _MockNode({"f_health": 0.85})}, step_count=1)
    jc.central = _MockNode({})
    jc._cross_faculty_dream()
    assert "genesis" not in jc._xdream_target


def test_cross_faculty_mixture_is_frechet_slerp_not_l2():
    """PURITY: the mixture is the Fréchet mean on Δ⁶³ and the move is a √p SLERP — NOT an L2/
    arithmetic mean. Exact reconstruction pins the ops; the Fréchet≠L2 gap proves it discriminates."""
    _roles, faculties, kernels = _scenario()
    heart = next(f for f in faculties if f.role == "heart")
    siblings = [f.basin.copy() for f in faculties if f.role != "heart"]
    before = heart.basin.copy()

    # reconstruct the EXACT mixture the impl must use (Fréchet mean, proximity-weighted)
    centroid = frechet_mean(siblings)
    w = rel_weights(centroid, siblings)
    wn = (w / float(w.sum())).tolist()
    expected_mix = frechet_mean(siblings, weights=wn)
    expected_moved = slerp_sqrt(before, expected_mix, _XDREAM_PULL)

    jc = _bare_constellation(faculties, kernels)
    jc._cross_faculty_dream()

    # exact match to the Fréchet-mean + √p-SLERP path
    assert np.allclose(heart.basin, expected_moved, atol=1e-9), "must be frechet_mean + slerp_sqrt on Δ⁶³"

    # the arithmetic (L2) mean is a DIFFERENT point → the exact match above is a real discriminator
    l2 = np.mean(np.stack(siblings), axis=0)
    l2 = l2 / l2.sum()
    assert not np.allclose(expected_mix, l2, atol=1e-6), "Fréchet≠L2 here (else the purity check is vacuous)"


def test_cross_faculty_dream_skips_collapsed_siblings_birth_fallback_and_logs():
    """Guard: if EVERY sibling is itself collapse-requesting / low-f_health, there is no FOREIGN
    entropy among the faculties → fall back to the birth-basin anchor mixture (still non-degenerate)
    and LOG the fallback (never a silent no-op)."""
    _roles, faculties, kernels = _scenario(siblings_collapsed=True)
    heart = next(f for f in faculties if f.role == "heart")
    fh_before = _fhealth(heart.basin)

    jc = _bare_constellation(faculties, kernels)
    buf = io.StringIO()
    with redirect_stdout(buf):
        fired = jc._cross_faculty_dream()

    assert fired["heart"]["source"] == "birth-fallback"
    assert fired["heart"]["n_siblings"] == 0
    assert "cross_faculty_dream_request" not in kernels["heart"].telemetry().extra
    assert _fhealth(heart.basin) > fh_before + 0.2, "birth mixture is wide → still lifts entropy"
    assert "FALLBACK" in buf.getvalue(), "the fallback must be logged, not silent"


def test_cross_faculty_dream_cooldown_one_fire_per_epoch_window():
    """A10 dream-storm guard: fire at most once per faculty per OCEAN epoch window. A request arriving
    inside the cooldown is CONSUMED (cleared) but does NOT re-fire the pull."""
    _roles, faculties, kernels = _scenario()
    heart = next(f for f in faculties if f.role == "heart")

    jc = _bare_constellation(faculties, kernels, step_count=1)
    jc._cross_faculty_dream()                       # first fire (this epoch window)
    moved_basin = heart.basin.copy()

    # re-arm the request WITHIN the same epoch window and collapse the basin again
    kernels["heart"].telemetry().extra["cross_faculty_dream_request"] = {"reason": "pillar1-collapse"}
    heart.set_basin(_one_hot(3))
    jc._step_count += 1                              # still the same epoch (window = _OCEAN_EPOCH_STEPS)
    assert jc._step_count // _OCEAN_EPOCH_STEPS == 1 // _OCEAN_EPOCH_STEPS
    fired2 = jc._cross_faculty_dream()

    assert "heart" not in fired2, "cooldown must prevent a second fire in the same epoch window"
    assert "cross_faculty_dream_request" not in kernels["heart"].telemetry().extra, "request consumed"
    assert _fhealth(heart.basin) < 0.10, "basin NOT re-pulled during cooldown (still collapsed)"

    # a NEW epoch window clears the cooldown → it fires again
    kernels["heart"].telemetry().extra["cross_faculty_dream_request"] = {"reason": "pillar1-collapse"}
    jc._step_count += _OCEAN_EPOCH_STEPS            # advance a whole epoch window
    fired3 = jc._cross_faculty_dream()
    assert "heart" in fired3, "a new epoch window must allow the cross-faculty dream to fire again"


def test_no_request_no_fire():
    """No collapsed faculty → the cross-faculty dream is a clean no-op (never perturbs healthy basins)."""
    _roles, faculties, kernels = _scenario()
    # clear the collapse: heart healthy, no request
    kernels["heart"] = _MockNode({"f_health": 0.9})
    heart = next(f for f in faculties if f.role == "heart")
    heart.set_basin(faculties[0].birth.copy())
    before = {f.role: f.basin.copy() for f in faculties}

    jc = _bare_constellation(faculties, kernels)
    fired = jc._cross_faculty_dream()

    assert fired == {}
    for f in faculties:
        assert np.allclose(f.basin, before[f.role]), "healthy basins must be untouched"
