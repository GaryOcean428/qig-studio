#!/usr/bin/env bash
# EXP-A020 (qig-applied inference accelerator) — SERVER-SIDE Ollama throughput levers for the qig-studio
# Qwen boundary peer. These are env vars read by `ollama serve` at startup (NOT per-request options — those
# are set studio-side in qwen_local.py: num_gpu=-1 full GPU offload + keep_alive=-1).
#
# Ollama bundles its OWN CUDA runtime, independent of the CPU-only python torch — so these make the idle GPU
# accelerate chat. A 4B q4 model fits a 4GB card; the q8 KV cache frees VRAM for more GPU layers.
#
#   eval "$(scripts/ollama_accelerate.sh env)"   # export the levers into your shell, then `ollama serve`
#   scripts/ollama_accelerate.sh serve            # export + (re)start ollama serve with them
#   scripts/ollama_accelerate.sh tune             # run the EXP-A020 perf/watt operating-point tuner (qig-applied)
#
# Ref: qig-applied/docs/current/20260623-expA020-inference-accelerator-usage-1.00W.md
set -euo pipefail

emit_env() {
  echo "export OLLAMA_FLASH_ATTENTION=1"     # fused attention kernel
  echo "export OLLAMA_KV_CACHE_TYPE=q8_0"    # 8-bit KV cache -> more VRAM for GPU layers
  echo "export OLLAMA_KEEP_ALIVE=-1"         # weights stay resident, no per-call reload
  echo "export OLLAMA_NUM_PARALLEL=1"        # single-stream latency (the studio is one conversation)
  echo "export OLLAMA_MAX_LOADED_MODELS=1"
}

case "${1:-env}" in
  env)
    emit_env
    ;;
  serve)
    eval "$(emit_env)"
    echo "[ollama_accelerate] restarting ollama serve with EXP-A020 levers…" >&2
    pkill -x ollama 2>/dev/null || true
    sleep 1
    OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_KEEP_ALIVE=-1 \
      OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 nohup ollama serve >/tmp/ollama.log 2>&1 &
    sleep 2
    echo "[ollama_accelerate] ollama serve started (log: /tmp/ollama.log)" >&2
    ;;
  tune)
    # the QIG perf/watt operating-point tuner lives in qig-applied (EXP-132 interior-optimum on silicon).
    APPLIED="$(cd "$(dirname "$0")/../../qig-applied" 2>/dev/null && pwd || true)"
    if [ -n "$APPLIED" ] && [ -f "$APPLIED/scripts/expA020_inference_accelerate.py" ]; then
      echo "[ollama_accelerate] running EXP-A020 tuner (dry run; add --apply to set the GPU cap)…" >&2
      ( cd "$APPLIED" && python3 scripts/expA020_inference_accelerate.py tune --model qwen3.5:4b "${@:2}" )
    else
      echo "[ollama_accelerate] qig-applied not found alongside qig-studio; clone it to run the tuner." >&2
      exit 1
    fi
    ;;
  *)
    echo "usage: $0 {env|serve|tune}" >&2; exit 2
    ;;
esac
