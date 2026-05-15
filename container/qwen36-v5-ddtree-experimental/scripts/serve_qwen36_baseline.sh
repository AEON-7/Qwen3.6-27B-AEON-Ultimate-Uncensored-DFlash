#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models/aeon-xs}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36-baseline}"
PORT="${PORT:-8000}"

exec vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --load-format safetensors \
  --quantization "${QUANTIZATION:-modelopt}" \
  ${EXTRA_VLLM_ARGS:-}
