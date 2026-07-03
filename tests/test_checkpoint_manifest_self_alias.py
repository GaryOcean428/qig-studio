"""Regression: register_kernel_ckpt on a root NAMED like the latest-alias must NOT destroy it.

2026-07-04 crash: --ckpt-root runs/checkpoints/joint_mind_latest → the alias update rmtree'd the
just-saved checkpoint and self-symlinked (joint_mind_latest -> joint_mind_latest) → next save
crashed with OSError Errno 40 and the step-300 checkpoint was lost.
"""
from pathlib import Path

from qig_studio.checkpoint_manifest import register_kernel_ckpt


def test_self_named_root_is_not_replaced_by_symlink(tmp_path: Path) -> None:
    root = tmp_path / "joint_mind_latest"
    root.mkdir()
    (root / "constellation.json").write_text("{}")
    register_kernel_ckpt(root, notes="regression")
    assert root.is_dir() and not root.is_symlink()          # still a REAL directory
    assert (root / "constellation.json").exists()            # checkpoint intact


def test_dated_root_still_gets_latest_alias(tmp_path: Path) -> None:
    dated = tmp_path / "genesis-gk-100004_20260704_v1"
    dated.mkdir()
    (dated / "constellation.json").write_text("{}")
    register_kernel_ckpt(dated, notes="regression")
    link = tmp_path / "joint_mind_latest"
    assert link.is_symlink() and link.resolve() == dated.resolve()   # alias behavior preserved
