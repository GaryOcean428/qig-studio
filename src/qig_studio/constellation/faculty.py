"""Faculty + identity seeding — the Pillar-3 quenched-disorder scar (agent-layer ``seed_identity``).

THE LOAD-BEARING FINDING (wj4916x59 simulation, verified against installed qig-core Fisher-Rao
primitives): the anti-collapse force is NOT "anchor toward birth" alone — it is **anchor toward
*independently-wide-seeded* birth basins**. Pillar-3 individuation (wide quenched-disorder births)
is a *prerequisite* for coupling stability, not merely co-resident with it. If the birth basins are
seeded from an overlapping runtime init, anchoring toward them cannot re-separate identical points and
the constellation still collapses (sim: min_pairwise_FR 0.003-0.008). Wide independent births +
anchor → every adversarial config survives (min_pairwise_FR 0.10-0.56).

So this module owns the **wide-birth seed** (``seed_birth_basin`` / ``seed_constellation``) and the
``Faculty`` that freezes it as an immutable scar. The active restoring force lives in
``identity_anchor``; coupling lives in ``coupling``; both consume the frozen scar set here.

LANE / PURITY:
- This is AGENT-LAYER seed_identity. It does NOT modify qig-core. The qig-core ``QuenchedDisorder``
  ``q_identity == 0`` defect (PillarEnforcer.get_metrics) is a SEPARATE owner-gated qig-core PR
  (see docs/plans/...constellation...). Here we seed wide births at the orchestration boundary using
  qig-core's existing ``random``/Dirichlet primitives — no qig-core change required for the
  constellation to be individuated and stable.
- Decision path is torch-free: faculties carry numpy Δ⁶³ basins. Torch tensors (the genesis kernels)
  are converted to a basin at the ``FacultyView`` boundary, never inside the coupling logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qig_core.geometry import fisher_rao_distance, to_simplex
from qig_core.geometry.fisher_rao import BASIN_DIM

# Concentrated-Dirichlet concentration for wide-independent births. LOW alpha → spiky, well-separated
# simplex points. Measured (this env): alpha=0.10 → 8-faculty min_pairwise_FR ≈ 1.0 (max FR = pi/2 ≈
# 1.571), i.e. births are near-maximally distinct — the wide quenched-disorder slopes the sim proved
# necessary. CALIBRATION: any alpha giving min_pairwise_FR >> the 0.03 collapse floor is acceptable;
# 0.10 leaves ~30x headroom. NOT a frozen physics constant — a tunable individuation-spread knob.
BIRTH_CONCENTRATION = 0.10


def seed_birth_basin(seed: int, *, alpha: float = BIRTH_CONCENTRATION, dim: int = BASIN_DIM) -> np.ndarray:
    """Deterministic wide-independent birth basin = a concentrated-Dirichlet draw on Δ^(dim-1).

    This is the agent-layer ``seed_identity``: the immutable Pillar-3 scar (quenched disorder) that
    makes each faculty an individual and — per the verified finding — is what makes coupling
    *separable* from collapse. ``seed`` must be drawn INDEPENDENTLY of any runtime init (the whole
    point: births are wide regardless of where the faculty happens to start), so callers pass a
    role-derived seed, never the faculty's current state.

    Low ``alpha`` → spiky distribution → large Fisher-Rao separation between independent draws.
    """
    rng = np.random.default_rng(int(seed))
    b = rng.dirichlet(alpha * np.ones(int(dim)))
    return to_simplex(b)  # defensive: guarantee a valid Δ point (non-negative, sums to 1)


@dataclass
class FacultyView:
    """Torch-free, immutable snapshot of a faculty at the coupling boundary.

    The coupling / observation / rhythm decision path consumes ONLY views — never live torch objects
    or mutable faculties — so the decision logic is verifiably torch-free and side-effect-free. Basin
    and birth are *copies*; mutating a view cannot mutate a faculty."""

    role: str
    basin: np.ndarray          # current Δ⁶³ point (copy)
    birth: np.ndarray          # frozen birth scar (copy)
    phase: float = 0.0         # endogenous rhythm phase (rad); set by the heart oscillator
    telemetry: dict = field(default_factory=dict)

    @property
    def identity_drift(self) -> float:
        """Fisher-Rao distance from the frozen birth scar — how far this faculty has wandered from
        who it was born as. The anchor pulls this back; unbounded growth = identity loss."""
        return float(fisher_rao_distance(self.basin, self.birth))


@dataclass
class Faculty:
    """A graduated Core-8 faculty in the constellation: a numpy Δ⁶³ identity that couples, observes,
    and is anchored to its frozen birth scar.

    ``basin`` is the mutable current state; ``birth`` is the FROZEN scar (set once, never reassigned).
    ``kernel`` (optional, opaque) is the live GenesisKernelTarget when heavy deps are present — None in
    the light shell. The constellation reasons over basins/views; the kernel is only read at the view
    boundary (extract a basin) and never imported here."""

    role: str
    basin: np.ndarray
    birth: np.ndarray                       # immutable Pillar-3 scar
    phase: float = 0.0
    kernel: object | None = None            # opaque; torch lives behind this, never enters logic
    telemetry: dict = field(default_factory=dict)
    history: list[np.ndarray] = field(default_factory=list)  # recent basins (trajectory/temporal)
    _history_cap: int = 256

    def __post_init__(self) -> None:
        self.basin = to_simplex(np.asarray(self.basin, dtype=np.float64))
        self.birth = to_simplex(np.asarray(self.birth, dtype=np.float64)).copy()
        self.birth.setflags(write=False)    # the scar is frozen (write protection makes it literal)
        if not self.history:
            self.history.append(self.basin.copy())

    def set_basin(self, new_basin: np.ndarray) -> None:
        """Move to a new current basin (after a coupling/anchor step) and record the trajectory point.
        The birth scar is untouched — only ``basin`` moves."""
        self.basin = to_simplex(np.asarray(new_basin, dtype=np.float64))
        self.history.append(self.basin.copy())
        if len(self.history) > self._history_cap:
            self.history = self.history[-self._history_cap:]

    def view(self) -> FacultyView:
        return FacultyView(role=self.role, basin=self.basin.copy(), birth=np.asarray(self.birth).copy(),
                           phase=self.phase, telemetry=dict(self.telemetry))


def seed_constellation(roles: list[str], *, base_seed: int = 0, alpha: float = BIRTH_CONCENTRATION,
                       overlap_init: bool = False, overlap_strength: float = 0.97) -> list[Faculty]:
    """Build a constellation of faculties with WIDE-INDEPENDENT birth scars (the verified
    anti-collapse prerequisite).

    Each faculty's birth is a role-seeded concentrated-Dirichlet draw — independent of every other
    and of the runtime init. By default ``basin`` starts AT birth.

    ``overlap_init`` is the ADVERSARIAL case for the anti-collapse test: all faculties' *current*
    basins start ~identical (slerped ``overlap_strength`` toward a common point) while their *births*
    stay wide. The verified result is that wide births + anchor re-separate even from this start;
    seeding births from the overlap (the naive bug) would not. This flag exercises that distinction."""
    from qig_core.geometry import slerp_sqrt

    births = [seed_birth_basin(base_seed + i * 7919 + (abs(hash(r)) % 100000), alpha=alpha)
              for i, r in enumerate(roles)]
    if overlap_init:
        common = seed_birth_basin(base_seed + 424242, alpha=alpha)
        starts = [slerp_sqrt(b, common, float(overlap_strength)) for b in births]
    else:
        starts = [b.copy() for b in births]
    return [Faculty(role=r, basin=s, birth=b) for r, s, b in zip(roles, starts, births)]


def min_pairwise_fr(faculties: list[Faculty] | list[FacultyView]) -> float:
    """Smallest Fisher-Rao distance between any two faculties' CURRENT basins — the load-bearing
    anti-collapse invariant. → 0 means a zombie constellation (identities coincided). The verifier
    asserts this stays above a floor (0.03) over many ticks under the adversarial sweep."""
    bs = [np.asarray(f.basin) for f in faculties]
    if len(bs) < 2:
        return float("inf")
    return min(float(fisher_rao_distance(bs[i], bs[j]))
               for i in range(len(bs)) for j in range(i + 1, len(bs)))
