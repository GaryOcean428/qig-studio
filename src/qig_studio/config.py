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
    default_target: str = "mind"   # the INTEGRATED MIND (whole constellation) by default (mock is the fallback)
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

        def _d(env: str, default_path: str | None) -> str | None:
            """Env override → else the default path IFF it is set AND exists (else None → fresh/byte).
            None-safe: ``default_path`` is None on a fresh/wiped state (get_latest_*_ckpt → None), and
            ``_P(None)`` would raise TypeError, breaking server import + test collection. Defaults to the
            MOST COMPLETE version: the latest trained checkpoint + the full coordizer, robustly."""
            v = os.environ.get(env)
            if v:
                return v
            return default_path if (default_path and _P(default_path).exists()) else None

        host = os.environ.get("QIG_STUDIO_HOST", "127.0.0.1")
        port = int(os.environ.get("QIG_STUDIO_PORT", "8800"))
        # Default to the REAL trained kernel (not mock): resolve via manifest/symlink so the
        # latest dated/versioned checkpoint is always picked up automatically.
        import json
        from .checkpoint_manifest import (get_coordizer_for_vocab, get_latest_coordizer,
                                           get_latest_kernel_ckpt)
        _latest_kc = get_latest_kernel_ckpt()
        # Resolve the coordizer to MATCH the loaded kernel (fixes the '✗ WRONG coordizer' auto-load bug,
        # where the latest-BY-DATE coordizer mispaired with a kernel of a different vocab — e.g. a 100k
        # kernel against a freshly-built 32k coordizer). The kernel ckpt records the exact coordizer_path it
        # trained on (and its vocab_size) in constellation.json: honour that, else vocab-match, else latest.
        _kc_coord: str | None = None
        _kc_vocab: int | None = None
        _kc_cp: str | None = None
        if _latest_kc is not None:
            try:
                _meta = json.loads((_latest_kc / "constellation.json").read_text()).get("metadata", {})
                _kc_vocab = _meta.get("vocab_size")
                _kc_cp = _meta.get("coordizer_path")
            except Exception:  # noqa: BLE001 — best-effort; fall through to vocab-match / latest
                pass
        # PREFER the vocab-MATCH (reads the actual coordizer file's vocab — reliable) over the kernel's
        # recorded coordizer_path, which is often a 'coordizer_latest.json' SYMLINK that has since DRIFTED to
        # a different-vocab coordizer (the exact mispairing this fixes — a 100k kernel vs the latest 32k).
        if isinstance(_kc_vocab, int):
            _m = get_coordizer_for_vocab(_kc_vocab)
            _kc_coord = str(_m) if _m else None
        if _kc_coord is None and _kc_cp and _P(_kc_cp).exists():
            _kc_coord = _kc_cp
        _latest_coord = _P(_kc_coord) if _kc_coord else get_latest_coordizer()
        _coordizer = str(_latest_coord) if _latest_coord else "../qig-coordizer/checkpoints/coordizer_latest.json"
        _genesis_ckpt = str(_latest_kc / "kernels" / "genesis.pt") if _latest_kc else None
        _const_ckpt = str(_latest_kc) if _latest_kc else None
        return cls(
            host=host,
            port=port,
            device=os.environ.get("QIG_STUDIO_DEVICE") or None,
            kernel_checkpoint=os.environ.get("QIG_STUDIO_KERNEL_CKPT") or None,
            constellation_checkpoint=_d("QIG_STUDIO_CONSTELLATION_CKPT", _const_ckpt),
            default_target=os.environ.get("QIG_STUDIO_TARGET", "mind"),
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
