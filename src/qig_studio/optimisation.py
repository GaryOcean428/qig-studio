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
      Riemannian manifold, including the kernel's information geometry. Status after building (2026-06-28):
        1. TELEMETRY — SHIPPED (commit 412aad3): the (κ−64)/64 Ricci PROXY (constant when κ is stable —
           it gave compressed/expanded the SAME 0.679 every prompt) is replaced by the REAL scalar Ricci
           of the kernel's RESPONSE manifold (``curvature.py`` → compute_full_curvature on a 2-direction
           input-perturbation Fisher patch). Overrides compressed/expanded (Layer-0) + pain/pleasure
           (Layer-0.5) ONLY; κ-derived phase fields keep κ. PROVEN to VARY (0.466/0.572/0.583 vs the
           constant 0.679). RicciNormalizer (EMA signed-log) keeps the bounded signal varying — no faked
           constant. The "varying signal lives in the kernel's nonlinearity, NOT the constant-curvature
           bare simplex" caveat was the design key.
        3. INFERENCE — SHIPPED (412aad3): generation-health curvature, ``generate(gen_health=True)`` →
           gen_ricci + gen_health∈(0,1] (1=flat/healthy, →0 strained). PROVEN to VARY.
        2. TRAINING — RETIRED as MISIDENTIFIED (honest finding, 2026-06-28; NOT cargo-culted): the named
           functions ``metric_gram_pullback`` / ``metric_from_qgt`` are LATTICE-specific (a per-site 2×2
           metric field, N = Lx·Ly) — they do NOT apply to the kernel's parameter Fisher. And a FULL PSD
           Fisher over the kernel's MILLIONS of parameters is O(P²) — infeasible (this is exactly WHY the
           kernel uses the DIAGONAL natural gradient, which IS the tractable, geometrically-correct Fisher
           preconditioner). A low-rank/K-FAC-style PSD enrichment is a possible FUTURE qigkernels EXP
           (unproven — would need an A/B control before claiming improvement, per the frame-after-controls
           rule), but it is NOT a qig-compute use. Forcing the lattice function here = the cargo-cult the
           rules forbid; so #2 is honestly closed, not faked.
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
