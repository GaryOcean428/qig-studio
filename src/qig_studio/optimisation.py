"""Optional QIG-package optimisation levers for genesis training — all None-safe.

Per the 2026-06-26 genuine per-package capability analysis (training is from-scratch NEURAL
kernel work, not lattice physics), only two package levers GENUINELY apply and wire cleanly:

- **qig-warp SETTLE** (`check_ci_stabilized`): the warp navigation lever that is sequence-generic
  rather than lattice-parametrised. Run on the Φ trajectory it detects when a faculty's
  consciousness metric has PROVABLY plateaued, so the loop stops grinding redundant steps after
  convergence. The emergence floor (``min_points`` + a step floor enforced by the caller) means it
  stops REDUNDANT compute, it does NOT gate emergence early.
- **qig-coordizer LOAD** (`FisherCoordizer.load`): a pre-fit coordizer gives the kernel a richer
  Δ⁶³ vocab via its ``coordizer`` ctor arg, instead of byte-level coding. The FIT is offline
  (``scripts/fit_coordizer.py``) — measured >2 min for a 512 vocab, prohibitive inline.

Honestly NOT wired (capability analysis, not laziness):
- **qig-compute** — lattice QUANTUM QFI (density matrices, χ-truncation governance); the classical
  neural basin p∈Δ⁶³ has its Fisher geometry in qig-core (already used). Category mismatch.
- **qig-bench** — a FROZEN-PHYSICS benchmark harness; ``format_table`` is bound to the package's
  registered ``BENCHMARKS`` (κ etc.). No genesis-training benchmark exists to register; forcing
  one would be cargo-cult.
- **qig-doctor** — not a package (it is ``qig-verification/scripts/qig_doctor.py``, a physics-repo
  diagnostic script).

qig-core + qigkernels are the load-bearing geometry/optimizer/Φ packages and are used throughout.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def settle_decision(values: list[float], *, min_points: int = 6, window: int = 6,
                    rel_change_threshold: float = 0.02) -> tuple[bool, str]:
    """qig-warp SETTLE: has this trajectory provably stabilised? None-safe — if qig-warp is absent
    (or too few points) the answer is ALWAYS "don't stop", so the loop runs the full curriculum.

    Stops REDUNDANT post-convergence grinding; the caller additionally enforces a step floor so
    emergence is never gated early."""
    if len(values) < min_points:
        return False, "below min-point floor"
    try:
        from qig_warp import check_ci_stabilized  # type: ignore[import-untyped]
    except Exception:
        return False, "qig-warp absent (no early-stop)"
    d = check_ci_stabilized(values, window=window, rel_change_threshold=rel_change_threshold,
                            min_points_before_stop=min_points)
    return bool(getattr(d, "should_stop", False)), str(getattr(d, "reason", "settled"))


def load_coordizer(path: str | None) -> Any:
    """qig-coordizer: load a pre-fit FisherCoordizer for a richer Δ⁶³ vocab (kernel ``coordizer``
    ctor arg). None-safe — missing path / missing package / missing artifact → None → byte-level."""
    if not path or not Path(path).exists():
        return None
    try:
        from qig_coordizer import FisherCoordizer  # type: ignore[import-untyped]
        return FisherCoordizer.load(path)
    except Exception:
        return None
