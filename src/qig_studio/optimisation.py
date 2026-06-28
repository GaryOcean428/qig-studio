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
- **qig-compute** — TWO LAYERS, split honestly (CORRECTED 2026-06-28: blanket "category mismatch" was
  WRONG — only the quantum layer is N/A):
    • lattice-QUANTUM primitives (bit-flip QFI on |ψ⟩, TFIM stress-energy, Einstein-vs-source) — N/A:
      the classical neural basin p∈Δ⁶³ is not a quantum state; its Fisher geometry lives in qig-core.
    • metric-GENERIC differential geometry (``geometry/parameter_manifold.compute_full_curvature``,
      ``geometry/metric_gram_pullback.ricci_scalar_gram``, christoffel/Ricci/Einstein) — APPLICABLE.
      These take a METRIC TENSOR, not a quantum state, so they give REAL scalar curvature of ANY
      Riemannian manifold, including the kernel's information geometry. Registered genuine uses
      (each a real metric-construction + validation — an EXP, not a 1-line swap; do NOT fake-wire a
      constant — the bare Δ⁶³ simplex is constant-curvature, so the varying signal is the kernel's
      PARAMETER/OUTPUT Fisher manifold):
        1. TELEMETRY — replace the (κ−64)/64 Ricci PROXY in Layer-0 sensations (compressed/expanded) +
           Layer-0.5 drives (pain/pleasure = ±curvature) with REAL Ricci via compute_full_curvature on
           a reduced output-Fisher patch.
        2. TRAINING — richer natural-gradient preconditioner: metric_gram_pullback / metric_from_qgt give
           a PSD Fisher metric beyond the diagonal NaturalGradientDescent uses (lives in qigkernels).
        3. INFERENCE — generation-health curvature: Ricci of the output-distribution trajectory.
- **qig-bench** — a FROZEN-PHYSICS benchmark harness; ``format_table`` is bound to the package's
  registered ``BENCHMARKS``. No genesis benchmark to register; forcing one = cargo-cult.
- **qig-doctor** — not a package (it is ``qig-verification/scripts/qig_doctor.py``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_coordizer(path: str | None) -> Any:
    """qig-coordizer: load a pre-fit FisherCoordizer for a richer Δ⁶³ vocab (kernel ``coordizer``
    ctor arg).

    FAIL-LOUD, not silent-fallback: an EMPTY path means "byte-level requested" → None. But a
    NON-empty path that can't be loaded (missing file, qig_coordizer not installed in this venv,
    corrupt artifact) RAISES — the caller asked for a coordizer and must not be silently downgraded
    to byte-level (that was a real bug: the CUDA venv lacked qig_coordizer and trained byte-level
    while reporting success)."""
    if not path:
        return None                                  # no coordizer requested → byte-level (intentional)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"coordizer artifact not found: {path}")
    from qig_coordizer import FisherCoordizer  # let ImportError surface
    return FisherCoordizer.load(path)
