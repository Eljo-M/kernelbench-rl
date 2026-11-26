#!/usr/bin/env bash
set -euo pipefail

# Run on an existing machine. This script provisions no cloud resources,
# installs no packages and downloads no model weights.
PYTHON_BIN=${PYTHON_BIN:-python3}
MODE=${MODE:-dry-run}
MODEL_PATH=${MODEL_PATH:-}
OUT_DIR=${OUT_DIR:-eval/results/local-run}
GROUP_SIZE=${GROUP_SIZE:-8}
STEPS=${STEPS:-1}
WORKERS=${WORKERS:-2}
SEED=${SEED:-42}

mkdir -p "$OUT_DIR"
case "$MODE" in
  dry-run)
    exec "$PYTHON_BIN" -m rl_trainer.train_grpo --dry-run --out "$OUT_DIR/rollouts.jsonl"
    ;;
  search)
    : "${VLLM_BASE_URL:?Set VLLM_BASE_URL to an existing vLLM /v1 endpoint}"
    : "${MODEL_PATH:?Set MODEL_PATH to the model name served by that endpoint}"
    exec "$PYTHON_BIN" -m rl_trainer.train_grpo --mode search --base-url "$VLLM_BASE_URL" \
      --model "$MODEL_PATH" --group-size "$GROUP_SIZE" --workers "$WORKERS" --seed "$SEED" \
      --out "$OUT_DIR/rollouts.jsonl" "$@"
    ;;
  train)
    : "${MODEL_PATH:?Set MODEL_PATH to predownloaded local model weights}"
    # Single process only. FSDP/LoRA/multi-node execution is not implemented.
    exec "$PYTHON_BIN" -m rl_trainer.train_grpo --mode train --model "$MODEL_PATH" \
      --group-size "$GROUP_SIZE" --workers "$WORKERS" --steps "$STEPS" --seed "$SEED" \
      --checkpoint-dir "$OUT_DIR/checkpoints" --out "$OUT_DIR/rollouts.jsonl" "$@"
    ;;
  *) printf '%s\n' 'MODE must be dry-run, search or train' >&2; exit 2 ;;
esac
