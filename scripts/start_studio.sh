#!/usr/bin/env bash
# Launch the qig-studio UI (the app) and open it in the browser. This is the front door the desktop
# icon calls — it starts the FastAPI server wired to the trained coordizer Core-8 if it isn't already
# up, then opens http://localhost:8800. Idempotent: if the server is already running it just opens it.
set -euo pipefail

STUDIO="$(cd "$(dirname "$0")/.." && pwd)"
URL="http://localhost:8800"

if ! curl -s --max-time 2 "$URL/health" >/dev/null 2>&1; then
  cd "$STUDIO"
  # Load the REAL integrated mind: the joint-trained central "I" (genesis-central) if present, else the
  # older core8_coord meta kernel. The joint mind has the trained central Φ + coupled faculties.
  # Load the CURRENT joint-trained central "I" (genesis.pt, written by the live trainer). No stale fallback
  # to the old core8 meta kernel — its vocab differs from the active coordizer and would mismatch on load.
  MIND="runs/checkpoints/joint_mind/kernels/genesis.pt"
  CKPT="$([ -f "$MIND" ] && echo "$MIND" || echo "")"
  # MUST match the coordizer the active trainer uses (currently the fresh 100k rebuild) or the UI shows a
  # different vocab than the kernels are training on. Override with QIG_STUDIO_GENESIS_COORDIZER if needed.
  QIG_STUDIO_GENESIS_COORDIZER="${QIG_STUDIO_GENESIS_COORDIZER:-../qig-coordizer/checkpoints/coordizer_rebuild_100k.json}" \
  QIG_STUDIO_GENESIS_CKPT="$CKPT" \
  QIG_STUDIO_TARGET="genesis" \
  QIG_STUDIO_DEVICE="cpu" \
  QIG_STUDIO_COACH_MODEL="nemotron-3-ultra:cloud" \
  nohup "$STUDIO/.venv/bin/python" -m qig_studio serve >/tmp/qig_studio.log 2>&1 &
  i=0; until curl -s --max-time 2 "$URL/health" >/dev/null 2>&1 || [ $i -ge 30 ]; do sleep 1; i=$((i+1)); done
fi

# open in the default browser (xdg-open on Linux)
( xdg-open "$URL" >/dev/null 2>&1 || sensible-browser "$URL" >/dev/null 2>&1 || true ) &
echo "qig-studio at $URL (log: /tmp/qig_studio.log)"
