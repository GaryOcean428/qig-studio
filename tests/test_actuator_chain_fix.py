"""Actuator-chain fix (Fable-council 2026-07-02) — F1 in-graph basin-pull + F3 collapse perturbation.

The 100k constellation collapsed by step ≤200 because the entropy actuators never reached the thing
f_health measures. These tests pin the two genesis_kernel legs of the fix:

  • F1 — the basin-pull term ``basin_weight·d_FR(cur, role_attractor)`` must reach ``self._kernel``
    params. It was DEAD because ``cur`` was ``.detach()``-ed (line 1305, for history/telemetry) and the
    pull reused that detached value, so the pull contributed ZERO gradient. F1 recomputes ``cur``
    IN-GRAPH at the pull site (``simplex_floor=1e-3``, no no_grad/detach).
  • F3 — the collapse branch of ``_homeostasis`` must apply a bounded ISOTROPIC weight perturbation
    (``_collapse_perturb``) so the forward leaves the vertex where the pull gradient is dead. f_health is
    recomputed each step from a FRESH forward, so ONLY a weight-space change moves it.

Both build a REAL loaded (tiny) ``GenesisKernelTarget`` so the production path is exercised.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("qigkernels")
pytest.importorskip("qig_coordizer")

from qig_studio.targets.genesis_kernel import GenesisKernelTarget  # noqa: E402


def _coordizer():
    from qig_coordizer import FisherCoordizer

    cz = FisherCoordizer(target_vocab_size=300)
    cz.train(b"the heart learns to feel; patterns flow through basins integrating space. " * 200,
             context_window=3, min_pair_count=2, verbose=False)
    return cz


def _load_basin_kernel(basin_weight: float, *, seed: int = 7, coordizer=None) -> GenesisKernelTarget:
    """A tiny REAL basin-head kernel with weights loaded. Basin mode requires a coordizer (it IS the basin
    tie; F1's dead-gradient bug is basin-mode-specific — geometric mode already computes ``cur`` in-graph
    via ``logits``). ``basin_ramp_steps=1`` so the pull is at full weight on step 1 (w_t = basin_weight).
    Homeostasis is stubbed OFF so the measured param delta is the optimizer step ALONE (isolates the
    pull's contribution; F3 exercises homeostasis separately)."""
    from qig_core.geometry.fisher_rao import BASIN_DIM

    tmpl = np.random.default_rng(3).dirichlet(np.ones(BASIN_DIM))   # a fixed role attractor (≠ cur)
    k = GenesisKernelTarget(
        num_layers=1, hidden_dim=64, num_heads=2, ffn_dim=64, seed=seed, role="heart",
        basin_template=tmpl, head_mode="basin", basin_weight=basin_weight, basin_ramp_steps=1,
        coordizer=coordizer if coordizer is not None else _coordizer(),
    )
    k.ensure_loaded()
    k._homeostasis = lambda snap: None            # type: ignore[assignment] — isolate the optimizer step
    return k


def _param_delta_after_one_step(basin_weight: float, coordizer) -> float:
    k = _load_basin_kernel(basin_weight, coordizer=coordizer)
    torch.manual_seed(0)                          # identical RNG across the two runs → only basin_weight differs
    before = [p.detach().clone() for p in k._kernel.parameters()]
    k.train_step("the heart learns to feel")
    return sum((p.detach() - b).abs().sum().item() for p, b in zip(k._kernel.parameters(), before))


def test_f1_basin_pull_gradient_reaches_kernel_params():
    """F1: with everything else identical (same seed, same RNG, same prompt, same coordizer), turning the
    basin pull ON must CHANGE the optimizer step — proving the pull's gradient reaches ``self._kernel``.
    Was DEAD before (detached ``cur`` → the pull added no gradient → the param update was identical
    regardless of weight)."""
    cz = _coordizer()                             # ONE shared coordizer → the two runs differ ONLY in basin_weight
    d_no_pull = _param_delta_after_one_step(0.0, cz)
    d_pull = _param_delta_after_one_step(5.0, cz)
    assert d_no_pull > 0.0, "the bare basin d_FR loss must already move params (sanity)"
    # if cur were detached the pull would add ZERO gradient → the two deltas would be bit-identical
    assert abs(d_pull - d_no_pull) > 1e-6, (
        f"basin pull did not reach _kernel params (dead gradient): {d_no_pull} vs {d_pull}"
    )


def test_f3_collapse_branch_fires_perturb_and_changes_kernel_weights():
    """F3: on the collapse branch (_homeostasis, f_health→0 + low Φ) ``_collapse_perturb`` must FIRE and
    ACTUALLY change ``self._kernel`` weights (non-self-confirming — a weight-space move, not a flag)."""
    import types

    k = _load_basin_kernel(basin_weight=0.5)
    # restore the REAL homeostasis (the isolation stub above turned it off)
    k._homeostasis = types.MethodType(GenesisKernelTarget._homeostasis, k)

    # spy that the collapse branch reaches _collapse_perturb
    fired = {"n": 0}
    real_perturb = k._collapse_perturb

    def _spy_perturb(sigma: float = 0.02):
        fired["n"] += 1
        return real_perturb(sigma)

    k._collapse_perturb = _spy_perturb            # type: ignore[assignment]

    before = [p.detach().clone() for p in k._kernel.parameters()]
    # a genuine COLLAPSE snapshot: f_health well below the floor, Φ low (not mature) → collapse branch
    snap = types.SimpleNamespace(phi=0.15, kappa=60.0, step=1, extra={"f_health": 0.0})
    k._homeostasis(snap)

    assert fired["n"] == 1, "collapse branch must fire _collapse_perturb"
    moved = sum((p.detach() - b).abs().sum().item() for p, b in zip(k._kernel.parameters(), before))
    assert moved > 0.0, "the collapse perturbation must actually change _kernel weights (weight-space)"
    assert "entropy-restore" in snap.extra.get("autonomic", ""), "collapse branch must be observable"


def test_f3_collapse_perturb_is_isotropic_and_target_free():
    """F4: the perturbation is ISOTROPIC (target-free) bounded weight noise — non-self-confirming. It
    mutates every parameter tensor by a bounded amount and does NOT reference any basin/target."""
    k = _load_basin_kernel(basin_weight=0.5)
    before = [p.detach().clone() for p in k._kernel.parameters()]
    k._collapse_perturb(sigma=0.02)
    per_tensor_moves = [(p.detach() - b).abs().sum().item() for p, b in zip(k._kernel.parameters(), before)]
    assert all(m > 0.0 for m in per_tensor_moves), "every parameter tensor is perturbed (isotropic)"


# ---------------------------------------------------------------------------
# F2 — un-clobber the M2 cross-faculty pull (joint_trainer). The per-step round-robin
# _set_pull(...fac.basin) used to OVERWRITE the foreign mixture _cross_faculty_dream set, so a
# collapsed faculty never actually trained toward its healthy siblings. The durable _xdream_target
# takes precedence for a window. These tests are torch-free (bare constellation + mock nodes).
# ---------------------------------------------------------------------------
import types  # noqa: E402

import numpy as _np  # noqa: E402

from qig_core.geometry import frechet_mean, slerp_sqrt, to_simplex  # noqa: E402
from qig_core.geometry.fisher_rao import BASIN_DIM  # noqa: E402

from qig_studio.constellation.coupling import rel_weights  # noqa: E402
from qig_studio.constellation.faculty import Faculty, seed_birth_basin  # noqa: E402
from qig_studio.constellation.joint_trainer import (  # noqa: E402
    JointConstellation,
    _XDREAM_PULL,
    _XDREAM_WINDOW,
)


def _fhealth(b) -> float:
    b = _np.clip(_np.asarray(b, dtype=_np.float64), 1e-12, None)
    b = b / b.sum()
    return float(-_np.sum(b * _np.log(b)) / _np.log(len(b)))


def _one_hot(idx: int = 3):
    v = _np.zeros(BASIN_DIM, dtype=_np.float64)
    v[idx] = 1.0
    return to_simplex(v)


class _MockNode:
    """Minimal ConstellationNode surface _cross_faculty_dream + _set_pull touch."""

    def __init__(self, extra: dict) -> None:
        self.head_mode = "basin"
        self.hidden_dim = BASIN_DIM
        self.vocab_size = BASIN_DIM
        self._basin_ref = None
        self._snap = types.SimpleNamespace(extra=dict(extra))

    def _node_device(self):
        import torch
        return torch.device("cpu")

    def telemetry(self):
        return self._snap

    def _resize_basin(self, ref, size):
        return ref


def _bare_constellation(faculties, kernels, step_count: int = 1) -> JointConstellation:
    jc = JointConstellation.__new__(JointConstellation)
    jc.faculties = faculties
    jc.kernels = kernels
    jc._last_xdream_epoch = {}
    jc._xdream_target = {}
    jc._step_count = step_count
    jc.f_sync = 0.25
    return jc


def _collapsed_scenario():
    roles = ["heart", "perception", "memory"]
    faculties, kernels = [], {}
    for i, r in enumerate(roles):
        birth = seed_birth_basin(1000 + i, alpha=0.4)
        if r == "heart":
            faculties.append(Faculty(role=r, basin=_one_hot(3), birth=birth))   # collapsed
            kernels[r] = _MockNode({"f_health": 0.02, "cross_faculty_dream_request":
                                    {"reason": "pillar1-collapse", "phi": 0.19, "f_health": 0.02}})
        else:
            faculties.append(Faculty(role=r, basin=birth.copy(), birth=birth))  # healthy
            kernels[r] = _MockNode({"f_health": 0.87})
    return roles, faculties, kernels


def test_f2_cross_faculty_dream_records_durable_foreign_target():
    """F2: _cross_faculty_dream must RECORD the foreign mixture as a durable _xdream_target (role →
    (mixture, until)) — not set _basin_ref inline (which the next round-robin step clobbered). The
    recorded mixture is the FOREIGN (higher-entropy) sibling consensus and the window is _XDREAM_WINDOW."""
    _roles, faculties, kernels = _collapsed_scenario()
    jc = _bare_constellation(faculties, kernels, step_count=5)
    jc._cross_faculty_dream()

    assert "heart" in jc._xdream_target, "the foreign mixture must be recorded as a durable target"
    mixture, until = jc._xdream_target["heart"]
    assert until == 5 + _XDREAM_WINDOW, "the window must extend _XDREAM_WINDOW steps from the current step"
    assert _fhealth(mixture) > 0.3, "the recorded target is the FOREIGN (higher-entropy) sibling mixture"

    # it must match the EXACT Fréchet-mean + proximity-weight path (PURE Δ⁶³, never L2)
    siblings = [f.basin for f in faculties if f.role != "heart"]
    centroid = frechet_mean(siblings)
    w = rel_weights(centroid, siblings)
    wn = (w / float(w.sum())).tolist()
    assert _np.allclose(mixture, frechet_mean(siblings, weights=wn), atol=1e-9)


def test_f2_active_target_takes_precedence_then_reverts_after_window():
    """F2: while the window is live, _xdream_active_target(role) returns the FOREIGN mixture (this is
    what the round-robin _set_pull uses INSTEAD of fac.basin). Once _step_count passes `until`, it pops
    the entry and returns None → the pull reverts to the faculty's own coupled basin."""
    _roles, faculties, kernels = _collapsed_scenario()
    jc = _bare_constellation(faculties, kernels, step_count=10)
    foreign = seed_birth_basin(42, alpha=0.5)                 # a wide foreign mixture
    jc._xdream_target["heart"] = (foreign, 10 + _XDREAM_WINDOW)

    # inside the window → precedence: the active target IS the foreign mixture (not fac.basin)
    jc._step_count = 10 + _XDREAM_WINDOW
    tgt = jc._xdream_active_target("heart")
    assert tgt is not None and _np.allclose(tgt, foreign), "foreign pull must take precedence in-window"

    # past the window → reverts (None) and the stale entry is popped (no leak)
    jc._step_count = 10 + _XDREAM_WINDOW + 1
    assert jc._xdream_active_target("heart") is None, "expired window must revert to the coupled basin"
    assert "heart" not in jc._xdream_target, "expired target must be popped (no leak)"

    # a role with no recorded target is a clean None (round-robin falls back to fac.basin)
    assert jc._xdream_active_target("perception") is None
