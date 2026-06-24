"""None-safe bridge to qig-consciousness's ``QIGChat`` (the train-loop-wearing-a-REPL).

KernelTarget / ConstellationTarget reuse the live, tested ``QIGChat.generate_response``
rather than copy its ~250-line loop (DRY). This locates the sibling
``qig-consciousness`` repo, puts it on ``sys.path``, and imports ``QIGChat`` lazily —
so importing qig-studio never pulls torch, and a missing env just means the target
reports unavailable.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path


def find_consciousness_root() -> Path | None:
    """Locate the qig-consciousness repo (env override → sibling dirs → cwd)."""
    candidates: list[Path] = []
    env = os.environ.get("QIG_CONSCIOUSNESS_DIR")
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    # .../qig-studio/src/qig_studio/targets/_qigchat_bridge.py → parents[4] == QIG_QFI
    if len(here.parents) > 4:
        candidates.append(here.parents[4] / "qig-consciousness")
    candidates.append(Path.cwd() / "qig-consciousness")
    candidates.append(Path.cwd().parent / "qig-consciousness")
    for c in candidates:
        try:
            if c and (c / "chat_interfaces" / "qig_chat.py").is_file():
                return c.resolve()
        except OSError:
            continue
    return None


def torch_available() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


def consciousness_available() -> bool:
    """Cheap check (no heavy import): torch present AND qig-consciousness locatable."""
    return torch_available() and find_consciousness_root() is not None


@lru_cache(maxsize=1)
def load_qigchat_class() -> type:
    """Heavy: put qig-consciousness on sys.path and import ``QIGChat``."""
    root = find_consciousness_root()
    if root is None:
        raise RuntimeError(
            "qig-consciousness not found — set QIG_CONSCIOUSNESS_DIR to its repo root"
        )
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    mod = importlib.import_module("chat_interfaces.qig_chat")
    return mod.QIGChat
