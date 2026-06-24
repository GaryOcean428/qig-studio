"""qig-studio settings (env-overridable)."""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8800
    device: str | None = None  # cuda/cpu/mps; None = auto
    kernel_checkpoint: str | None = None
    constellation_checkpoint: str | None = None
    default_target: str = "mock"
    server_url: str = "http://127.0.0.1:8800"
    auth_key: str | None = None  # X-Studio-Key shared secret; required for non-loopback bind
    output_dir: str = "outputs"  # where snapshots/exports land (user-nominatable)
    curriculum_dir: str = "curriculum"  # where curriculum files are read from (user-nominatable)

    @property
    def is_loopback(self) -> bool:
        if self.host in ("localhost", ""):
            return True
        try:
            return ipaddress.ip_address(self.host).is_loopback
        except ValueError:
            return False

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
            auth_key=os.environ.get("QIG_STUDIO_KEY") or None,
            output_dir=os.environ.get("QIG_STUDIO_OUTPUT_DIR", "outputs"),
            curriculum_dir=os.environ.get("QIG_STUDIO_CURRICULUM_DIR", "curriculum"),
        )
