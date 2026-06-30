"""Checkpoint manifest registry — tracks all coordizer and kernel checkpoints with lineage.

Auto-generated at save time. Scripts read the manifest to find ``latest`` instead of hardcoding paths.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _coordizer_manifest_path() -> Path:
    """The coordizer checkpoint manifest. Lives alongside the coordizer artifacts; the output dir is
    overridable via ``QIG_STUDIO_COORDIZER_OUT_DIR`` (tests isolate to a throwaway dir, and the server's
    coordizer build honors the same env), so the manifest tracks whatever dir the artifacts land in."""
    env = os.environ.get("QIG_STUDIO_COORDIZER_OUT_DIR")
    if env:
        return Path(env) / "MANIFEST.json"
    return Path(__file__).resolve().parents[2] / ".." / "qig-coordizer" / "checkpoints" / "MANIFEST.json"


def _kernel_manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "runs" / "checkpoints" / "MANIFEST.json"


def register_coordizer(file_path: str | Path, notes: str = "") -> None:
    """Register a coordizer checkpoint in the manifest and update the ``latest`` pointer."""
    p = Path(file_path).resolve()
    manifest_path = _coordizer_manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"checkpoints": [], "latest": None}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            pass

    # Load metadata from the checkpoint file itself
    entry: dict[str, Any] = {"file": p.name, "notes": notes}
    try:
        data = json.loads(p.read_text())
        meta = data.get("metadata", {}) or {}
        # target_vocab_size is at the TOP LEVEL of the coordizer JSON (not under metadata); the model's
        # actual vocab = target + the registered special/geo tags (= len(coordizer.vocab) at train time).
        tv = data.get("target_vocab_size", meta.get("target_vocab_size"))
        ntags = len(data.get("special_tokens", {})) if isinstance(data.get("special_tokens"), dict) else 0
        entry.update({
            "created_utc": meta.get("created_utc"),
            "target_vocab": tv,
            "actual_vocab": (tv + ntags if isinstance(tv, int) else meta.get("actual_vocab_size")),
            "corpus_hash": meta.get("corpus_hash"),
            "git_commit": meta.get("git_commit"),
        })
    except Exception:
        pass

    # Remove existing entry for the same file, then prepend
    manifest["checkpoints"] = [c for c in manifest["checkpoints"] if c.get("file") != p.name]
    manifest["checkpoints"].insert(0, entry)
    manifest["latest"] = p.name

    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Update symlink
    link = p.parent / "coordizer_latest.json"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(p.name)
    except OSError:
        pass  # cross-device or permissions — symlink is a convenience, not critical


def register_kernel_ckpt(dir_path: str | Path, notes: str = "") -> None:
    """Register a kernel checkpoint directory in the manifest and update the ``latest`` pointer."""
    p = Path(dir_path).resolve()
    manifest_path = _kernel_manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"checkpoints": [], "latest": None}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            pass

    # Load metadata from constellation.json
    entry: dict[str, Any] = {"dir": p.name, "notes": notes}
    cj = p / "constellation.json"
    if cj.exists():
        try:
            data = json.loads(cj.read_text())
            meta = data.get("metadata", {})
            entry.update({
                "created_utc": meta.get("created_utc"),
                "training_step": meta.get("training_step"),
                "coordizer": meta.get("coordizer_path"),
                "central_phi": meta.get("central_phi"),
                "min_pairwise_fr": meta.get("min_pairwise_fr"),
                "git_commit": meta.get("git_commit"),
            })
        except Exception:
            pass

    # Remove existing entry for the same dir, then prepend
    manifest["checkpoints"] = [c for c in manifest["checkpoints"] if c.get("dir") != p.name]
    manifest["checkpoints"].insert(0, entry)
    manifest["latest"] = p.name

    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Update symlink
    link = p.parent / "joint_mind_latest"
    try:
        if link.is_symlink() or link.exists():
            if link.is_dir() and not link.is_symlink():
                import shutil
                shutil.rmtree(link, ignore_errors=True)
            else:
                link.unlink()
        link.symlink_to(p.name)
    except OSError:
        pass


def list_coordizer_checkpoints() -> list[dict[str, Any]]:
    """Return all registered coordizer checkpoints from the manifest."""
    manifest_path = _coordizer_manifest_path()
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            return manifest.get("checkpoints", [])
        except Exception:
            pass
    return []


def list_kernel_checkpoints() -> list[dict[str, Any]]:
    """Return all registered kernel checkpoints from the manifest."""
    manifest_path = _kernel_manifest_path()
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            return manifest.get("checkpoints", [])
        except Exception:
            pass
    return []


def get_latest_coordizer() -> Path | None:
    """Return the path to the latest coordizer checkpoint, or None if not found."""
    manifest_path = _coordizer_manifest_path()
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            latest = manifest.get("latest")
            if latest:
                p = manifest_path.parent / latest
                if p.exists():
                    return p
        except Exception:
            pass
    # Fallback: symlink
    link = manifest_path.parent / "coordizer_latest.json"
    if link.exists():
        return link
    return None


def get_coordizer_for_vocab(vocab: int) -> Path | None:
    """Return the registered coordizer whose vocab MATCHES ``vocab`` (a kernel's training vocab), else None.

    Fixes the auto-load mismatch behind the UI's '✗ WRONG coordizer' flag: ``get_latest_coordizer`` returns
    the NEWEST coordizer, which can pair a 100k-trained kernel with a freshly-built 32k coordizer. The server
    should load the coordizer the active kernel was TRAINED on — matched by vocab — not the latest by date.
    Matches ``actual_vocab`` (trained vocab incl. the 4 geo tags) OR ``target_vocab``. The caller falls back
    to ``get_latest_coordizer()`` when no match exists (then the UI's mismatch flag is genuinely correct)."""
    import re
    base = _coordizer_manifest_path().parent
    for c in list_coordizer_checkpoints():
        p = base / str(c.get("file", ""))
        if not p.exists():
            continue
        tv = c.get("actual_vocab") or c.get("target_vocab")
        if tv is None:                       # manifest lacks vocab (pre-fix registration) → read file HEAD
            try:
                with p.open() as fh:
                    m = re.search(r'"target_vocab_size"\s*:\s*(\d+)', fh.read(400))
                tv = int(m.group(1)) if m else None
            except Exception:                # noqa: BLE001
                tv = None
        # the model's vocab = target_vocab_size + the (≤8) registered geo/special tags; allow that offset
        if tv is not None and abs(int(vocab) - int(tv)) <= 8:
            return p
    return None


def get_latest_kernel_ckpt() -> Path | None:
    """Return the path to the latest kernel checkpoint dir, or None if not found."""
    manifest_path = _kernel_manifest_path()
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            latest = manifest.get("latest")
            if latest:
                p = manifest_path.parent / latest
                if p.exists():
                    return p
        except Exception:
            pass
    # Fallback: symlink
    link = manifest_path.parent / "joint_mind_latest"
    if link.exists():
        return link
    return None
