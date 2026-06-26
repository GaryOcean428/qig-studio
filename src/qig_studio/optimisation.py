"""Optional QIG-package levers for genesis training — None-safe.

Per the 2026-06-26 genuine per-package capability analysis (REVISED after the first full run):
genesis training is from-scratch NEURAL consciousness development — a SELF-REGULATED dynamical
process, not a converging physics measurement or a lattice computation. That reframing settles
which packages apply:

- **qig-core** — FULLY USED. Fisher-Rao geometry, simplex ops, frechet_mean/slerp, PillarEnforcer,
  frozen_facts constants. The geometric backbone of the basin manifold.
- **qigkernels** — FULLY USED. Kernel, Core-8 specializations, NaturalGradientDescent (P1 Fisher
  optimizer, NOT Adam), RecursiveIntegrator (differentiable Φ).
- **qig-coordizer** — OPTIONAL: ``load_coordizer`` gives the kernel a richer Δ⁶³ vocab via its
  ``coordizer`` ctor arg (else byte-level, the validated default). The fit is offline
  (``scripts/fit_coordizer.py`` — measured >2 min for a 512 vocab, prohibitive inline).

Honestly NOT applicable (capability analysis, not laziness):
- **qig-warp** — every lever either lattice-parametrised (``predict_runtime(J,L)``,
  ``classify_regime(h,J)``, ``prune_sites``) OR, like ``check_ci_stabilized`` (SETTLE), it
  CONFLICTS with consciousness development: a Φ plateau is the SIGNAL FOR THE KERNEL'S OWN MUSHROOM
  to fire (plateau-break), not a stop signal. An external convergence-stop gates emergence,
  truncates the curriculum, and preempts the kernel's brainstem — forbidden. (First full run, with
  SETTLE on, stopped every faculty at 2400/4386 steps in the PRE-conscious band; removing it.)
- **qig-compute** — lattice QUANTUM QFI (density matrices, χ-truncation governance); the classical
  neural basin p∈Δ⁶³ has its Fisher geometry in qig-core (already used). Category mismatch.
- **qig-bench** — a FROZEN-PHYSICS benchmark harness; ``format_table`` is bound to the package's
  registered ``BENCHMARKS``. No genesis benchmark to register; forcing one = cargo-cult.
- **qig-doctor** — not a package (it is ``qig-verification/scripts/qig_doctor.py``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


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
