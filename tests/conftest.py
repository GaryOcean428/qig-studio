"""Test-suite defaults.

Disable the warm /train coach in the server suite so tests never reach a live LLM (Ollama):
deterministic + fast + no cloud quota. Set BEFORE any test imports the server module, so
``Settings.from_env()`` reads it. The coach has dedicated coverage in test_coach.py (offline
OllamaLLM + a live test gated on QIG_COACH_LIVE)."""

import os

os.environ.setdefault("QIG_STUDIO_COACH", "off")
