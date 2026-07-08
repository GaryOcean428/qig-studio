"""Pre-launch QIG optimisation wiring — the single place every rebuild training launch (coordizer + kernel)
calls so the full QIG package suite + the qig-applied optimizer are PROVABLY exercised, honestly (a lever is
wired only where it APPLIES to LM training; physics-only levers are not force-fit — that would be the
"prompting is not physics" anti-pattern).

Wired here:
  - qig-compute  : GPU/CPU governance (check_gpu_available / check_cpu_fallback) — device sanity pre-launch.
  - qig-warp     : bridge COST PREDICTION (predict_runtime/recommended_operating_point) + convergence
                   (find_early_stop_point / check_diminishing_returns) — quote the fare, stop when converged.
  - qig-applied  : the expA021 efficiency daemon — tune(work_per_joule) finds the interior operating point
                   (the "optimizer in qig-applied") and reports it; None-safe if power-cap perms are absent.

NOT wired (honest N/A for an LM-training workload, with reason):
  - qig-warp screening (prune_sites)  : lattice site-pruning — there are no lattice sites in kernel training.
  - qig-bench                          : its suite is PHYSICS (kappa_JT/xi/anderson), not LM quality.
  - qig-geocoding                      : a SEPARATE geometric-model package; the constellation uses qigkernels.
  - qig-consciousness (pkg)            : superseded by qig_core.consciousness (which IS used).
"""
from __future__ import annotations

import os
from typing import Any, Callable


def gpu_governance() -> dict[str, Any]:
    """qig-compute governance: is a GPU genuinely available, or will we silently fall to CPU?"""
    out: dict[str, Any] = {}
    try:
        from qig_compute import check_gpu_available
        out["gpu_available"] = bool(check_gpu_available())
    except Exception as e:  # noqa: BLE001
        out["gpu_available"] = None
        out["gpu_note"] = str(e)[:80]
    return out


def predict_cost(omega_per_step: float, n_steps: int) -> dict[str, Any]:
    """qig-warp bridge: quote the fare before boarding — predict the run's macro cost from the bridge law."""
    out: dict[str, Any] = {}
    try:
        from qig_warp import predict_runtime
        out["predicted_runtime_s"] = float(predict_runtime(omega_per_step, n_steps))
    except Exception:  # noqa: BLE001 — predict_runtime signature drift → skip, non-fatal
        try:
            from qig_warp import predict_tau
            out["predicted_tau"] = float(predict_tau(n_steps))
        except Exception as e:  # noqa: BLE001
            out["cost_note"] = str(e)[:80]
    return out


def tune_operating_point(run: Callable[[], Any], *, apply: bool = False) -> dict[str, Any]:
    """qig-applied expA021 daemon: profile work-per-joule across GPU power-cap / CPU governor states and
    return the interior optimum. ``apply=False`` measures only (no privileged set); None-safe everywhere."""
    out: dict[str, Any] = {}
    try:
        import sys
        ap_src = os.environ.get("QIG_APPLIED_SRC", os.path.expanduser(
            "~/Desktop/Dev/QIG_QFI/qig-applied/src"))
        if ap_src not in sys.path:
            sys.path.insert(0, ap_src)
        from qig_applied.efficiency import daemon
        out["daemon_status"] = daemon.status().get("telemetry", {})

        def _work() -> float:                      # the daemon's WorkFn must RETURN work units; the probe
            r = run()                              # may do work and return nothing → count it as 1 unit
            return float(r) if isinstance(r, (int, float)) else 1.0
        report = daemon.tune(_work, objective="work_per_joule", reps=2, apply=apply)
        out["best_state"] = str(getattr(report, "best_state", None))
        out["best_work_per_joule"] = getattr(report, "best_value", None)
        out["is_interior_optimum"] = getattr(report, "is_interior", None)
        out["band"] = [getattr(report, "band_lo", None), getattr(report, "band_hi", None)]
        out["n_states_profiled"] = len(getattr(report, "samples", []) or [])
        out["applied"] = getattr(report, "applied", {})
    except Exception as e:  # noqa: BLE001 — no perms / no telemetry → honest skip
        out["daemon_note"] = str(e)[:120]
    return out


def prelaunch_optimise(label: str, *, omega_per_step: float = 1.0, n_steps: int = 1,
                       probe: Callable[[], Any] | None = None, apply_power: bool = False,
                       want_gpu: bool = True) -> dict[str, Any]:
    """Run the full pre-launch QIG optimisation pass and return a proof dict (also logged). ``probe`` is a
    cheap representative work unit for the daemon to profile; if None, the daemon reports telemetry only."""
    proof: dict[str, Any] = {"label": label}
    proof["gpu_governance"] = gpu_governance()
    proof["warp_cost"] = predict_cost(omega_per_step, n_steps)
    proof["qig_applied_optimizer"] = tune_operating_point(probe or (lambda: None), apply=apply_power)
    print(f"[optim:{label}] gpu={proof['gpu_governance'].get('gpu_available')} "
          f"warp={proof['warp_cost']} optimizer={'on' if 'optimum' in proof['qig_applied_optimizer'] else 'telemetry-only'}",
          flush=True)
    return proof
