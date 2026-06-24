"""qig-studio CLI:  python -m qig_studio [serve | tui [url]]"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "serve"

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

    print("usage: python -m qig_studio [serve | tui [url]]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
