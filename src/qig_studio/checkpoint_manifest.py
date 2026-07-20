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

    # Update symlink (same self-alias guard as the kernel variant — never unlink/replace the real file)
    if p.name == "coordizer_latest.json":
        return
    link = p.parent / "coordizer_latest.json"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(p.name)
    except OSError:
        pass  # cross-device or permissions — symlink is a convenience, not critical


def versioned_ckpt_root(stem: str, vocab: int, base_dir: str | Path = "runs/checkpoints") -> str:
    """Build a NAMED / DATED / VERSIONED constellation checkpoint root that CARRIES THE VOCAB, so a saved
    mind is unambiguous about the coordizer it trained on — ``genesis-gk-32004_20260630_v1``. This is the
    structural fix for the '✗ WRONG coordizer' vocab-mismatch class (a 100k kernel mispaired with a 32k
    coordizer): the vocab is in the name, and ``config.from_env`` vocab-matches the coordizer to it.

    Auto-increments ``_v{n}`` so a same-day rebuild never clobbers a prior lineage; the 3-checkpoint
    rotation buffer in :meth:`JointConstellation.save_checkpoint` prunes OLD generations WITHIN a lineage."""
    from datetime import datetime
    date_tag = datetime.now().strftime("%Y%m%d")
    base = f"{stem}-{vocab}_{date_tag}"
    d = Path(base_dir)
    n = 1
    while (d / f"{base}_v{n}").exists():
        n += 1
    return str(d / f"{base}_v{n}")


def prune_lineage(stem: str, vocab: int, keep: int = 3, base_dir: str | Path = "runs/checkpoints") -> int:
    """Delete OLD versioned checkpoint dirs of a lineage (``{stem}-{vocab}_{date}_v{n}``), keeping the newest
    ``keep`` by mtime. Without this, periodic ``save_checkpoint`` calls (each auto-incrementing ``_v{n}`` via
    :func:`versioned_ckpt_root`) grow the disk unboundedly — ~1 GiB every save filled 55 GiB in ~11 h on the
    100k run. The newest ``keep`` are always retained (the ``*_latest`` symlink target is the newest, so it is
    never pruned). Best-effort; returns the count removed. keep<1 is treated as 1 (never delete everything)."""
    import shutil
    keep = max(1, int(keep))
    d = Path(base_dir)
    dirs = sorted((p for p in d.glob(f"{stem}-{vocab}_*_v*") if p.is_dir()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    removed = 0
    for old in dirs[keep:]:
        shutil.rmtree(old, ignore_errors=True)
        removed += 1
    return removed


def _coordizer_pin(coordizer_path: str | None, ckpt_dir: Path) -> dict[str, Any]:
    """Resolve a (possibly mutable-symlink) coordizer reference to an IMMUTABLE pin: the real target
    filename + its sha256. A kernel is meaningless against the wrong coordizer (the basins are
    coordizer-defined), so a saved kernel must be able to prove which coordizer it trained on even
    after ``coordizer_latest.json`` is repointed. Best-effort — records whatever it can resolve.
    See ``qigkernels/20260718-trained-artifact-provenance-1.00W.md``."""
    pin: dict[str, Any] = {"coordizer": coordizer_path, "coordizer_sha256": None}
    if not coordizer_path:
        return pin
    ref = Path(coordizer_path)
    ref = ref.resolve() if ref.is_absolute() else (ckpt_dir / ref).resolve()  # resolve() follows the symlink
    pin["coordizer"] = ref.name  # immutable filename (symlink already dereferenced), not the 'latest' alias
    try:
        import hashlib
        h = hashlib.sha256()
        with ref.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        pin["coordizer_sha256"] = h.hexdigest()
    except Exception:
        pass
    return pin


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
            pin = _coordizer_pin(meta.get("coordizer_path"), p)
            entry.update({
                "created_utc": meta.get("created_utc"),
                "training_step": meta.get("training_step"),
                # PIN the coordizer to its immutable identity + sha256, never the mutable
                # 'coordizer_latest.json' symlink (2026-07-18 fix — a kernel must prove which
                # coordizer defined its basins even after 'latest' is repointed).
                "coordizer": pin["coordizer"],
                "coordizer_sha256": pin["coordizer_sha256"],
                "coordizer_vocab": meta.get("coordizer_vocab"),
                "central_phi": meta.get("central_phi"),
                "min_pairwise_fr": meta.get("min_pairwise_fr"),
                "git_commit": meta.get("git_commit"),
            })
        except Exception:
            pass

    # Remove the existing entry for the same dir; SELF-HEAL phantom rows whose dir no longer exists
    # (prune_lineage deletes dirs but not manifest rows — that left 'latest' pointing at a missing
    # dir, 2026-07-18 fix), then prepend and repoint 'latest' at the just-saved (existing) dir.
    base = manifest_path.parent
    manifest["checkpoints"] = [
        c for c in manifest["checkpoints"]
        if c.get("dir") != p.name and (base / str(c.get("dir", ""))).exists()
    ]
    manifest["checkpoints"].insert(0, entry)
    manifest["latest"] = p.name

    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Update symlink — the "joint_mind_latest" alias points at the newest DATED root.
    # GUARD (2026-07-04 crash fix): if the ckpt root ITSELF is named "joint_mind_latest" (the
    # train_joint_mind.py default --ckpt-root), this block would rmtree the JUST-SAVED checkpoint
    # and replace it with a symlink to its own name (joint_mind_latest -> joint_mind_latest), which
    # (a) destroys the checkpoint and (b) crashes the NEXT save with OSError Errno 40 (symlink loop).
    # The root already IS "latest" — skip the alias entirely.
    if p.name == "joint_mind_latest":
        return
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
