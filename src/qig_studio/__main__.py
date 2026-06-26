"""qig-studio CLI:  python -m qig_studio [serve | tui [url] | learn [target] [steps]]"""

from __future__ import annotations

import sys


def _run_learn(argv: list[str]) -> int:
    """Run the local autonomic continual-learning loop (brain §4b) over a target. LOCAL — no Modal,
    no vex. Usage: ``python -m qig_studio learn [target=mock] [steps=50]``."""
    from .learning import ContinuousLearningLoop
    from .targets.registry import default_registry

    target_name = argv[0] if argv else "mock"
    steps = int(argv[1]) if len(argv) > 1 else 50
    reg = default_registry(default_target=target_name)
    target = reg.get(target_name)
    if target is None:
        print(f"unknown target '{target_name}'; choices: {', '.join(reg.names())}", file=sys.stderr)
        return 2
    if not target.is_available():
        print(f"target '{target_name}' backend is not available here; falling back to mock", file=sys.stderr)
        target = reg.get("mock")
    if target is None:
        print("no usable target (mock fallback missing)", file=sys.stderr)
        return 2
    print(f"continual-learning loop · target={target.name} · regime={target.loss_regime.value} · {steps} steps")
    loop = ContinuousLearningLoop(target, max_steps=steps)
    for _ in range(steps):
        rec = loop.step()
        marker = "" if rec.intervention == "wake" else f"  ⟳ {rec.intervention}"
        print(f"  step {rec.step:>4}  Φ={rec.phi:5.3f}  κ={rec.kappa:6.2f}  {rec.regime:<24}{marker}")
    s = loop.summary()
    print(f"\nsummary: {s.steps} steps · interventions={s.interventions} · "
          f"final Φ={s.final_phi:.3f} · kernel_autonomy≈{s.kernel_autonomy:.2f} "
          f"({'real' if s.using_real_manager else 'fallback'} autonomic)")
    print(f"note: {s.notes}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "serve"

    if cmd == "learn":
        return _run_learn(argv[1:])

    if cmd in ("serve", "server"):
        from .config import Settings

        s = Settings.from_env()
        # P0-F2 pre-bind (SEC-5): refuse a non-loopback bind without a key BEFORE the socket opens.
        if not s.is_loopback and not s.auth_key:
            raise SystemExit(
                f"refusing to bind non-loopback host '{s.host}' without QIG_STUDIO_KEY (fail-closed, P0-F2)"
            )
        import uvicorn

        uvicorn.run("qig_studio.server:app", host=s.host, port=s.port, log_level="info")
        return 0

    if cmd == "tui":
        from .tui import run_tui

        run_tui(argv[1] if len(argv) > 1 else None)
        return 0

    print("usage: python -m qig_studio [serve | tui [url] | learn [target] [steps]]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
