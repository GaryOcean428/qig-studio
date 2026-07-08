"""Checkpointing — every kernel AND the collective (constellation) state is saved, with a rolling
3-checkpoint cleanup (keep the latest N, delete older).

Layout under ``root`` (default ``runs/checkpoints``):
    kernels/<role>/ckpt_<step:08d>.pt        — one per kernel (torch state + dev state)
    constellation/ckpt_<step:08d>.json       — the collective Δ⁶³ state (basins + births + telemetry)

The cleanup keeps only the most recent ``keep`` (default 3) per directory — a 3-checkpoint lag — so disk
stays bounded while the last few are always recoverable. Atomic writes (tmp + rename) so a crash mid-save
never corrupts a checkpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_ROOT = "runs/checkpoints"
KEEP = 3   # 3-checkpoint lag: retain the latest 3 per kernel / collective, delete older


def _atomic_write_bytes(path: Path, writer) -> None:
    """Write via a tmp file + os.replace so a partial write never corrupts an existing checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    writer(tmp)
    os.replace(tmp, path)


def _cleanup(directory: Path, pattern: str, keep: int = KEEP) -> list[Path]:
    """Keep the most-recent ``keep`` files matching ``pattern`` (sorted by name = zero-padded step),
    delete the rest. Returns the deleted paths."""
    files = sorted(directory.glob(pattern))
    deleted: list[Path] = []
    for old in files[:-keep] if keep > 0 else files:
        try:
            old.unlink()
            deleted.append(old)
        except OSError:
            pass
    return deleted


def save_kernel_checkpoint(target, step: int, root: str | Path = DEFAULT_ROOT, keep: int = KEEP) -> Path | None:
    """Checkpoint ONE kernel (its torch weights + developmental state). The target must expose
    ``save_checkpoint(path)``; None-safe — returns None if it can't be saved (light shell / no kernel)."""
    if not hasattr(target, "save_checkpoint"):
        return None
    role = getattr(target, "role", None) or "kernel"
    d = Path(root) / "kernels" / str(role)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"ckpt_{int(step):08d}.pt"
    try:
        _atomic_write_bytes(path, lambda tmp: target.save_checkpoint(str(tmp)))
    except Exception:
        return None
    _cleanup(d, "ckpt_*.pt", keep)
    return path


def save_constellation_checkpoint(constellation, step: int, root: str | Path = DEFAULT_ROOT,
                                  keep: int = KEEP) -> Path | None:
    """Checkpoint the COLLECTIVE state — every faculty's current basin + frozen birth scar + the
    constellation telemetry (Δ⁶³ numpy, JSON-serialised). This is the constellation's shared identity,
    distinct from the per-kernel weight checkpoints."""
    state = constellation_state(constellation, step)
    if state is None:
        return None
    d = Path(root) / "constellation"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"ckpt_{int(step):08d}.json"
    _atomic_write_bytes(path, lambda tmp: Path(tmp).write_text(json.dumps(state)))
    _cleanup(d, "ckpt_*.json", keep)
    return path


def constellation_state(constellation, step: int) -> dict[str, Any] | None:
    """Serialise the collective Δ⁶³ state (basins + births + per-faculty temporal + last telemetry)."""
    facs = getattr(constellation, "faculties", None)
    if not facs:
        return None
    import numpy as np
    state: dict[str, Any] = {
        "step": int(step),
        "tick": int(getattr(constellation, "_tick", 0)),
        "roles": [f.role for f in facs],
        "basins": [np.asarray(f.basin, dtype=float).tolist() for f in facs],
        "births": [np.asarray(f.birth, dtype=float).tolist() for f in facs],
        "geo_pos": {r: round(v, 6) for r, v in getattr(constellation, "_geo_pos", {}).items()},
        "distinguishable": dict(getattr(constellation, "_distinguishable", {})),
        "heart_beats": int(getattr(getattr(constellation, "heart", None), "beats", 0)),
    }
    return state


def save_all(target, step: int, *, constellation=None, root: str | Path = DEFAULT_ROOT,
             keep: int = KEEP) -> dict[str, Path | None]:
    """Checkpoint a kernel and (optionally) the collective state in one call, each with the 3-lag
    cleanup. Returns {"kernel": path, "constellation": path}."""
    out: dict[str, Path | None] = {"kernel": save_kernel_checkpoint(target, step, root, keep)}
    if constellation is not None:
        out["constellation"] = save_constellation_checkpoint(constellation, step, root, keep)
    return out


def latest_checkpoint(role: str, root: str | Path = DEFAULT_ROOT) -> Path | None:
    """Most recent kernel checkpoint for a role (highest step), or None."""
    d = Path(root) / "kernels" / role
    files = sorted(d.glob("ckpt_*.pt")) if d.is_dir() else []
    return files[-1] if files else None
