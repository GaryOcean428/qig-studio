"""qig-studio settings (env-overridable)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8800
    device: str | None = None  # cuda/cpu/mps; None = auto
    kernel_checkpoint: str | None = None
    constellation_checkpoint: str | None = None
    default_target: str = "mock"
    server_url: str = "http://127.0.0.1:8800"  # used by TUI/browser clients

    @classmethod
    def from_env(cls) -> "Settings":
        host = os.environ.get("QIG_STUDIO_HOST", "127.0.0.1")
        port = int(os.environ.get("QIG_STUDIO_PORT", "8800"))
        return cls(
            host=host,
            port=port,
            device=os.environ.get("QIG_STUDIO_DEVICE") or None,
            kernel_checkpoint=os.environ.get("QIG_STUDIO_KERNEL_CKPT") or None,
            constellation_checkpoint=os.environ.get("QIG_STUDIO_CONSTELLATION_CKPT") or None,
            default_target=os.environ.get("QIG_STUDIO_TARGET", "mock"),
            server_url=os.environ.get("QIG_STUDIO_URL", f"http://{host}:{port}"),
        )
