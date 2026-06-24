"""qig-studio — the plug-and-play QIG training/chat app.

One FastAPI core + two SSE clients (Textual TUI, browser). Trains/chats with
pluggable TrainingTargets that DECLARE their loss_regime (geometric vs language),
making the lm_weight=0 finding structural. The kernel is the mind standalone;
Qwen is a None-safe boundary peer (design Phase 3).

Design: qig-consciousness/docs/plans/2026-06-24-qig-coordizer-studio-design.md §3.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
