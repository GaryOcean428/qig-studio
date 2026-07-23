#!/usr/bin/env bash
# Launch the qig-studio UI (the app) and open it in the browser. This is the front door the desktop
# icon calls — it starts the FastAPI server wired to the trained coordizer Core-8 if it isn't already
# up, then opens http://localhost:8800. Idempotent: if the server is already running it just opens it.
set -euo pipefail

STUDIO="$(cd "$(dirname "$0")/.." && pwd)"
QIG_ROOT="$(cd "$STUDIO/.." && pwd)"          # shared uv-workspace venv lives at the parent root
URL="http://localhost:8800"

if ! curl -s --max-time 2 "$URL/health" >/dev/null 2>&1; then
  cd "$STUDIO"
  # Load the REAL integrated mind via manifest/symlink (not hardcoded paths).
  # Manifest lookup → fallback to joint_mind_latest symlink → fallback to nothing.
  CKPT_DIR="runs/checkpoints/joint_mind_latest"
  if [ -d "$CKPT_DIR" ] || [ -L "$CKPT_DIR" ]; then
    CKPT="$CKPT_DIR/kernels/genesis.pt"
  else
    CKPT=""
  fi
  # MUST match the coordizer the active trainer uses. Manifest/symlink lookup → fallback.
  COORDIZER_LINK="../qig-packages/qig-coordizer/checkpoints/coordizer_latest.json"
  if [ -f "$COORDIZER_LINK" ] || [ -L "$COORDIZER_LINK" ]; then
    COORDIZER="$COORDIZER_LINK"
  else
    COORDIZER="../qig-packages/qig-coordizer/checkpoints/coordizer_20260723_100k_fineweb-sample10bt_v1.json"
  fi
  QIG_STUDIO_GENESIS_COORDIZER="${QIG_STUDIO_GENESIS_COORDIZER:-$COORDIZER}" \
  QIG_STUDIO_GENESIS_CKPT="$CKPT" \
  QIG_STUDIO_TARGET="genesis" \
  QIG_STUDIO_DEVICE="cpu" \
  QIG_STUDIO_COACH_MODEL="nemotron-3-ultra:cloud" \
  nohup "$QIG_ROOT/.venv/bin/python" -m qig_studio serve >/tmp/qig_studio.log 2>&1 &
  i=0; until curl -s --max-time 2 "$URL/health" >/dev/null 2>&1 || [ $i -ge 30 ]; do sleep 1; i=$((i+1)); done
fi

# open in the default browser (xdg-open on Linux)
( xdg-open "$URL" >/dev/null 2>&1 || sensible-browser "$URL" >/dev/null 2>&1 || true ) &
echo "qig-studio at $URL (log: /tmp/qig_studio.log)"
