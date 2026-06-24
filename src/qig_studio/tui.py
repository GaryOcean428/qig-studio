"""Textual TUI — live-telemetry console + chat over SSE (design §3.3, client #1).

A separate process bridged to the FastAPI core by SSE JSON (no JS↔Python FFI).
Textual is an OPTIONAL extra: importing this module never requires it; ``run_tui``
imports it lazily and gives a clean message if absent.

Keys:  Enter = send chat · t = train 20 steps (live telemetry) · q = quit
"""

from __future__ import annotations

import json

from .config import Settings


def _fmt_telemetry(t: dict) -> str:
    phi = t.get("phi", 0.0)
    bar = "█" * int(max(0.0, min(1.0, phi)) * 20)
    return (
        f"Φ  {phi:5.3f} |{bar:<20}|\n"
        f"κ  {t.get('kappa', 0.0):6.2f}   regime: {t.get('regime', '—')}\n"
        f"basin_d {t.get('basin_distance', 0.0):5.3f}   loss {t.get('loss') if t.get('loss') is not None else '—'}\n"
        f"step {t.get('step', 0)}   ΔΦ {t.get('delta_phi', 0.0):+.4f}"
    )


def run_tui(url: str | None = None) -> None:
    try:
        import httpx
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal
        from textual.widgets import Footer, Header, Input, RichLog, Static
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            f"Textual TUI needs the 'tui' extra:  uv pip install 'qig-studio[tui]'  ({exc})"
        )

    settings = Settings.from_env()
    base = (url or settings.server_url).rstrip("/")

    class QIGStudioTUI(App):
        CSS = """
        #telemetry { width: 42; border: round $accent; padding: 1; }
        #log { border: round $primary; }
        """
        BINDINGS = [("q", "quit", "Quit"), ("t", "train", "Train 20")]

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal():
                yield Static("connecting…", id="telemetry")
                yield RichLog(id="log", highlight=True, markup=True)
            yield Input(placeholder="message the active target… (Enter)", id="msg")
            yield Footer()

        def on_mount(self) -> None:
            self.title = "qig-studio"
            self.sub_title = base
            self.refresh_targets()

        def _log(self, msg: str) -> None:
            self.query_one("#log", RichLog).write(msg)

        def _set_telemetry(self, t: dict) -> None:
            self.query_one("#telemetry", Static).update(_fmt_telemetry(t))

        @staticmethod
        async def _get(path: str) -> dict:
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(base + path)
                return r.json()

        def refresh_targets(self) -> None:
            self.run_worker(self._refresh_targets(), exclusive=False)

        async def _refresh_targets(self) -> None:
            try:
                data = await self._get("/targets")
                active = data.get("active")
                names = ", ".join(
                    f"[b]{t['name']}[/b]" if t["name"] == active else t["name"]
                    for t in data["targets"]
                )
                self.sub_title = f"{base} · active={active} · targets: {names}"
                tele = await self._get("/telemetry")
                self._set_telemetry(tele)
            except Exception as exc:  # noqa: BLE001
                self._log(f"[red]connection error:[/red] {exc}")

        def on_input_submitted(self, event: "Input.Submitted") -> None:
            text = event.value.strip()
            event.input.value = ""
            if not text:
                return
            self._log(f"[cyan]you[/cyan]> {text}")
            self.run_worker(self._chat(text), exclusive=False)

        async def _chat(self, text: str) -> None:
            try:
                async with httpx.AsyncClient(timeout=120.0) as c:
                    r = await c.post(base + "/chat", json={"message": text})
                    if r.status_code != 200:
                        self._log(f"[red]{r.status_code}[/red] {r.text}")
                        return
                    body = r.json()
                    self._log(f"[green]target[/green]> {body['text']}")
                    self._set_telemetry(body["telemetry"])
            except Exception as exc:  # noqa: BLE001
                self._log(f"[red]chat error:[/red] {exc}")

        def action_train(self) -> None:
            self._log("[yellow]· training 20 steps (basin-driving) ·[/yellow]")
            self.run_worker(self._train(20), exclusive=True)

        async def _train(self, steps: int) -> None:
            try:
                async with httpx.AsyncClient(timeout=None) as c:
                    async with c.stream(
                        "POST", base + "/train", json={"steps": steps}
                    ) as r:
                        async for line in r.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            ev = json.loads(line[len("data: "):])
                            if ev.get("type") == "step":
                                self._set_telemetry(ev["telemetry"])
                                self._log(f"  step {ev['step']} [{ev['phase']}] {ev['prompt'][:48]}")
                            elif ev.get("type") == "error":
                                self._log(f"[red]train error:[/red] {ev['error']}")
                            elif ev.get("type") == "done":
                                self._log("[yellow]· done ·[/yellow]")
            except Exception as exc:  # noqa: BLE001
                self._log(f"[red]train error:[/red] {exc}")

    QIGStudioTUI().run()
