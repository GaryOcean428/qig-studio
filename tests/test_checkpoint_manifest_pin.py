"""Regression: register_kernel_ckpt must PIN the coordizer to its IMMUTABLE target + sha256 (never the
mutable coordizer_latest.json symlink), and must SELF-HEAL phantom manifest rows whose dir was pruned.

2026-07-18, CCAa — trained-artifact provenance Phase 3 recurrence fix. The live bug: joint_mind_64k
recorded its coordizer as '../qig-coordizer/checkpoints/coordizer_latest.json' (a symlink) + only an
8-hex prefix hash, and a pruned checkpoint left 'latest' pointing at a missing dir."""
from __future__ import annotations

import hashlib
import json

import qig_studio.checkpoint_manifest as cm


def test_coordizer_pin_resolves_symlink_to_immutable_target(tmp_path):
    real = tmp_path / "coordizer_20260705_64k_v1.json"
    real.write_text('{"target_vocab_size": 64000}')
    (tmp_path / "coordizer_latest.json").symlink_to(real.name)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()

    # training record points at the MUTABLE symlink (relative to the ckpt dir)
    pin = cm._coordizer_pin("../coordizer_latest.json", ckpt)
    assert pin["coordizer"] == "coordizer_20260705_64k_v1.json"  # immutable name, symlink dereferenced
    assert pin["coordizer_sha256"] == hashlib.sha256(real.read_bytes()).hexdigest()


def test_register_pins_coordizer_and_heals_phantom(tmp_path, monkeypatch):
    manifest = tmp_path / "checkpoints" / "MANIFEST.json"
    manifest.parent.mkdir(parents=True)
    monkeypatch.setattr(cm, "_kernel_manifest_path", lambda: manifest)

    real = tmp_path / "checkpoints" / "coordizer_x_v1.json"
    real.write_text('{"target_vocab_size": 64000}')
    (tmp_path / "checkpoints" / "coordizer_latest.json").symlink_to(real.name)

    # pre-seed with a PHANTOM latest (dir that does not exist)
    manifest.write_text(json.dumps({
        "checkpoints": [{"dir": "gone_dir", "notes": "phantom"}],
        "latest": "gone_dir",
    }))

    ckpt = tmp_path / "checkpoints" / "mind_ckpt"
    ckpt.mkdir()
    (ckpt / "constellation.json").write_text(json.dumps({
        "metadata": {"training_step": 2400, "coordizer_path": "../coordizer_latest.json",
                     "coordizer_vocab": 64004},
    }))

    cm.register_kernel_ckpt(ckpt)

    m = json.loads(manifest.read_text())
    assert m["latest"] == "mind_ckpt"                      # not the phantom
    assert "gone_dir" not in [c["dir"] for c in m["checkpoints"]]  # phantom self-healed
    entry = m["checkpoints"][0]
    assert entry["coordizer"] == "coordizer_x_v1.json"     # immutable, not 'coordizer_latest.json'
    assert entry["coordizer_sha256"] == hashlib.sha256(real.read_bytes()).hexdigest()
    assert entry["coordizer_vocab"] == 64004
