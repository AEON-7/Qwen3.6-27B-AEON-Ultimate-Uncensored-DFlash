#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen25-ddtree-m2 qwen25-0.5b}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_MATMUL_PRECISION="${TORCH_MATMUL_PRECISION:-high}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

case "${ENABLE_PREFIX_CACHING,,}" in
  1|true|yes|on)
    echo "ENABLE_PREFIX_CACHING=$ENABLE_PREFIX_CACHING requested, but DDTree research launchers keep prefix caching off; using --no-enable-prefix-caching." >&2
    ;;
  0|false|no|off)
    :
    ;;
  *)
    echo "Invalid ENABLE_PREFIX_CACHING=$ENABLE_PREFIX_CACHING" >&2
    exit 2
    ;;
esac
PREFIX_CACHING_ARGS=(--no-enable-prefix-caching)

EAGER_ARGS=()
case "${ENFORCE_EAGER,,}" in
  1|true|yes|on)
    EAGER_ARGS=(--enforce-eager)
    ;;
  0|false|no|off)
    ;;
  *)
    echo "Invalid ENFORCE_EAGER=$ENFORCE_EAGER" >&2
    exit 2
    ;;
esac

exec vllm serve "$MODEL_ID" \
  --served-model-name $SERVED_MODEL_NAME \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --dtype auto \
  --kv-cache-dtype auto \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-chunked-prefill \
  "${PREFIX_CACHING_ARGS[@]}" \
  "${EAGER_ARGS[@]}" \
  --trust-remote-code \
  --generation-config "$GENERATION_CONFIG" \
  ${EXTRA_VLLM_ARGS:-}
