"""qig-studio settings (env-overridable)."""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8800
    device: str | None = None  # cuda/cpu/mps; None = auto
    kernel_checkpoint: str | None = None
    constellation_checkpoint: str | None = None
    default_target: str = "genesis"   # the REAL trained kernel by default (mock is the fallback)
    server_url: str = "http://127.0.0.1:8800"
    auth_key: str | None = None  # X-Studio-Key shared secret; required for non-loopback bind
    output_dir: str = "outputs"  # where snapshots/exports land (user-nominatable)
    curriculum_dir: str = "curriculum"  # where curriculum files are read from (user-nominatable)
    genesis_coordizer_checkpoint: str | None = None  # trained FisherCoordizer → genesis coords path
    genesis_kernel_checkpoint: str | None = None  # trained genesis kernel (.pt) → restore weights+dev state (None=fresh)
    genesis_num_layers: int = 8  # genesis kernel depth (deep neocortex; EXP-CORTEX-AB axis). 4 was an
    #                              arbitrary baseline; 8 is the deep-stack default (now stable since the
    #                              qig-core acos gradient fix v2.12.1). Rigorous depth: run EXP-CORTEX-AB.
    coach_enabled: bool = True  # warm Ollama developmental coach in /train (None-safe → keyword)
    coach_model: str = "nemotron-3-ultra:cloud"  # coach LLM (free; qwen3.5:4b local fallback)
    coach_cadence: int = 25  # coach speaks every N steps (+ on stagnation)

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
        from pathlib import Path as _P

        def _d(env: str, default_path: str) -> str | None:
            """Env override → else the default path IFF it exists (else None → fresh/byte). Defaults to the
            MOST COMPLETE version: the latest trained checkpoint + the full 100k coordizer, robustly."""
            v = os.environ.get(env)
            if v:
                return v
            return default_path if _P(default_path).exists() else None

        host = os.environ.get("QIG_STUDIO_HOST", "127.0.0.1")
        port = int(os.environ.get("QIG_STUDIO_PORT", "8800"))
        # Default to the REAL trained kernel (not mock): the integrated genesis kernel loaded from the
        # latest joint-mind checkpoint + the full 100k coordizer. The Core-8 SUPPORT kernels (faculties)
        # are trained+100k inside the background JointConstellation (resume joint_mind + coordizer_max).
        _coordizer = "../qig-coordizer/checkpoints/coordizer_max.json"
        _genesis_ckpt = "runs/checkpoints/joint_mind/kernels/genesis.pt"
        _const_ckpt = "runs/checkpoints/joint_mind"
        return cls(
            host=host,
            port=port,
            device=os.environ.get("QIG_STUDIO_DEVICE") or None,
            kernel_checkpoint=os.environ.get("QIG_STUDIO_KERNEL_CKPT") or None,
            constellation_checkpoint=_d("QIG_STUDIO_CONSTELLATION_CKPT", _const_ckpt),
            default_target=os.environ.get("QIG_STUDIO_TARGET", "genesis"),
            server_url=os.environ.get("QIG_STUDIO_URL", f"http://{host}:{port}"),
            auth_key=os.environ.get("QIG_STUDIO_KEY") or None,
            output_dir=os.environ.get("QIG_STUDIO_OUTPUT_DIR", "outputs"),
            curriculum_dir=os.environ.get("QIG_STUDIO_CURRICULUM_DIR", "curriculum"),
            genesis_coordizer_checkpoint=_d("QIG_STUDIO_GENESIS_COORDIZER", _coordizer),
            genesis_kernel_checkpoint=_d("QIG_STUDIO_GENESIS_CKPT", _genesis_ckpt),
            genesis_num_layers=int(os.environ.get("QIG_STUDIO_GENESIS_LAYERS", "8")),
            coach_enabled=os.environ.get("QIG_STUDIO_COACH", "on").lower() not in ("0", "off", "false", "no"),
            coach_model=os.environ.get("QIG_STUDIO_COACH_MODEL", "nemotron-3-ultra:cloud"),
            coach_cadence=int(os.environ.get("QIG_STUDIO_COACH_CADENCE", "25")),
        )
