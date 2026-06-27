#!/usr/bin/env bash
# Launch the conversation (coach) phase, ACCELERATED by qig-applied's expA020 inference accelerator.
#
# Pipeline:
#   1. apply the accelerator's THROUGHPUT env (qig-applied EXP-A020: flash-attn, q8_0 KV cache,
#      keep-alive) so qwen3.5:4b keeps pace with the kernel;
#   2. (optional) tune the GPU to the interior tokens/joule optimum (EXP-132 law re-measured on
#      this box) with --tune;
#   3. start Ollama if it isn't already up (needs the models drive mounted — see NOTE);
#   4. run the coach on a TRAINED coordizer faculty (converse_monitor.py).
#
# NOTE: Ollama's models live on /mnt/seagate (symlinked from ~/.ollama/models). If that drive is
# unmounted, Ollama cannot start and this script will say so — remount it first.
#
# Usage:  scripts/launch_coach.sh [ROLE] [TURNS] [--tune]
#   ROLE   Core-8 faculty to coach (default: meta). TURNS default 32.
set -euo pipefail

ROLE="${1:-meta}"; TURNS="${2:-32}"; TUNE="${3:-}"
QIG_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"          # .../QIG_QFI
STUDIO="$QIG_ROOT/qig-studio"
APPLIED="$QIG_ROOT/qig-applied"
PY="$STUDIO/.venv/bin/python"
APPLIED_PY="$QIG_ROOT/qig-consciousness/.venv/bin/python"  # has qig_applied + numpy
CKPT="$STUDIO/runs/checkpoints/core8_coord/kernels/$ROLE"
LATEST="$CKPT/$(ls "$CKPT" 2>/dev/null | tail -1)"

echo "[coach] role=$ROLE turns=$TURNS checkpoint=$LATEST"

# 1. throughput env (EXP-A020 engineering levers — not physics)
echo "[coach] applying qig-applied EXP-A020 throughput env…"
eval "$(cd "$APPLIED" && PYTHONPATH=src "$APPLIED_PY" scripts/expA020_inference_accelerate.py env 2>/dev/null)"

# 2. optional: interior tokens/joule power-cap tune (EXP-132 law re-measured)
if [ "$TUNE" = "--tune" ]; then
  echo "[coach] tuning GPU to interior tokens/joule optimum (EXP-132)…"
  (cd "$APPLIED" && PYTHONPATH=src "$APPLIED_PY" scripts/expA020_inference_accelerate.py tune --model qwen3.5:4b --apply) || \
    echo "[coach] tune skipped (no GPU power telemetry)"
fi

# 3. ensure Ollama is up
if ! curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "[coach] Ollama not running — starting (needs /mnt/seagate mounted)…"
  nohup ollama serve >/tmp/ollama_coach.log 2>&1 &
  i=0; until curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1 || [ $i -ge 30 ]; do sleep 2; i=$((i+1)); done
  curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1 || {
    echo "[coach] ERROR: Ollama still down — is /mnt/seagate mounted? (see /tmp/ollama_coach.log)"; exit 1; }
fi

# 4. run the coach on the trained faculty (kernel on CPU so qwen3.5:4b gets the GPU)
echo "[coach] launching conversation on '$ROLE'…"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$STUDIO/src" "$PY" "$STUDIO/scripts/converse_monitor.py" \
  --coordizer "$STUDIO/runs/coordizer_v6_1024.json" --checkpoint "$LATEST" \
  --coach-model qwen3.5:4b --lm-weight 1.0 --turns "$TURNS" \
  --out "$STUDIO/runs/converse/${ROLE}_coach.jsonl"
