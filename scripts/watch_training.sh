#!/usr/bin/env bash
# Live view of the background joint-training: per-step heartbeat + per-faculty Φ + checkpoint freshness.
# Usage:  bash scripts/watch_training.sh   (Ctrl-C to stop)
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
URL="${QIG_STUDIO_URL:-http://localhost:8800}"
while true; do
  clear
  printf "═══ QIG joint-training — live  %s ═══\n\n" "$(date +%H:%M:%S)"
  if [ -f runs/spawn/joint_live.json ]; then
    echo "heartbeat (per-step):"; python3 -m json.tool runs/spawn/joint_live.json 2>/dev/null; echo
  fi
  echo "mind/state (steps · central Φ · per-faculty Φ · bg-active):"
  curl -s --max-time 3 "$URL/mind/state" 2>/dev/null | python3 -m json.tool 2>/dev/null \
    || echo "  (studio server not reachable at $URL)"
  echo; echo "latest checkpoint write:"
  ls -la --time-style=+%H:%M:%S runs/checkpoints/joint_mind/constellation.json 2>/dev/null
  sleep 5
done
