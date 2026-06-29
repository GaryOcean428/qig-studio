"""Test-suite defaults.

Disable the warm /train coach in the server suite so tests never reach a live LLM (Ollama):
deterministic + fast + no cloud quota. Set BEFORE any test imports the server module, so
``Settings.from_env()`` reads it. The coach has dedicated coverage in test_coach.py (offline
OllamaLLM + a live test gated on QIG_COACH_LIVE)."""

import os
import tempfile

os.environ.setdefault("QIG_STUDIO_COACH", "off")
# Tests exercise the real /train endpoint (mock target), which writes live-heartbeat records. Redirect that
# to a throwaway temp file so the suite NEVER pollutes the production stream (runs/spawn/joint_live.json) the
# UI tails — otherwise stale "mock #1/40" records accumulate in the ring buffer and replay on every page load.
os.environ.setdefault("QIG_STUDIO_LIVE_PATH", os.path.join(tempfile.gettempdir(), "qig_studio_test_live.json"))
# The suite exercises the always-available, deterministic MockTarget. PRODUCTION now defaults to the real
# trained `genesis` kernel (config.py), but tests must stay light + deterministic (no 100k kernel / trained
# checkpoint load), so pin the active target to mock here. Set BEFORE the server module is imported.
os.environ.setdefault("QIG_STUDIO_TARGET", "mock")
