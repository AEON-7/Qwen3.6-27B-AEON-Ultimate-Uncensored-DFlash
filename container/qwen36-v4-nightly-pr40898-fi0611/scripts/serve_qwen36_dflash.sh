#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models/aeon-xs}"
DFLASH_DIR="${DFLASH_DIR:-/models/dflash-drafter}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-aeon-ultimate qwen36-ultimate aeon-fast aeon-deep qwen36-ultimate-xs}"
PORT="${PORT:-8000}"
PROFILE="${PROFILE:-production}"

case "$PROFILE" in
  production)
    DEFAULT_MAX_MODEL_LEN=200000
    DEFAULT_MAX_NUM_SEQS=16
    DEFAULT_GPU_MEMORY_UTILIZATION=0.85
    DEFAULT_ENABLE_PREFIX_CACHING=1
    DEFAULT_COMPILATION_CONFIG='{"cudagraph_capture_sizes":[1,2,4,8,12,16,20,24,28,32,40,48,56,64],"inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false}}'
    ;;
  gateway)
    DEFAULT_MAX_MODEL_LEN=256000
    DEFAULT_MAX_NUM_SEQS=64
    DEFAULT_GPU_MEMORY_UTILIZATION=0.75
    DEFAULT_ENABLE_PREFIX_CACHING=1
    DEFAULT_COMPILATION_CONFIG='{"cudagraph_capture_sizes":[1,2,4,8,12,16,20,24,28,32,40,48,56,64],"inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false}}'
    ;;
  benchmark)
    DEFAULT_MAX_MODEL_LEN=2048
    DEFAULT_MAX_NUM_SEQS=256
    DEFAULT_GPU_MEMORY_UTILIZATION=0.85
    DEFAULT_ENABLE_PREFIX_CACHING=0
    DEFAULT_COMPILATION_CONFIG='{"inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false}}'
    ;;
  *)
    echo "Unknown PROFILE=$PROFILE (expected production, gateway, or benchmark)" >&2
    exit 2
    ;;
esac

MAX_MODEL_LEN="${MAX_MODEL_LEN:-$DEFAULT_MAX_MODEL_LEN}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$DEFAULT_MAX_NUM_SEQS}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$DEFAULT_GPU_MEMORY_UTILIZATION}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-$DEFAULT_ENABLE_PREFIX_CACHING}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash_attn}"
GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"
MM_SHM_CACHE_MAX_OBJECT_SIZE_MB="${MM_SHM_CACHE_MAX_OBJECT_SIZE_MB:-256}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-$DEFAULT_COMPILATION_CONFIG}"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_MATMUL_PRECISION="${TORCH_MATMUL_PRECISION:-high}"
export NVIDIA_FORWARD_COMPAT="${NVIDIA_FORWARD_COMPAT:-1}"
export NVIDIA_DISABLE_REQUIRE="${NVIDIA_DISABLE_REQUIRE:-1}"
export ENABLE_NVFP4_SM100="${ENABLE_NVFP4_SM100:-0}"
export VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-0}"
export VLLM_TEST_FORCE_FP8_MARLIN="${VLLM_TEST_FORCE_FP8_MARLIN:-0}"
export VLLM_NVFP4_GEMM_BACKEND="${VLLM_NVFP4_GEMM_BACKEND:-flashinfer-cutlass}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCHINDUCTOR_MAX_AUTOTUNE="${TORCHINDUCTOR_MAX_AUTOTUNE:-0}"
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE="${TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE:-0}"
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM="${TORCHINDUCTOR_MAX_AUTOTUNE_GEMM:-0}"

SPEC_CONFIG=$(printf '{"method":"dflash","model":"%s","num_speculative_tokens":%s,"attention_backend":"FLASH_ATTN"}' \
  "$DFLASH_DIR" "$NUM_SPECULATIVE_TOKENS")

PREFIX_CACHING_ARGS=()
case "${ENABLE_PREFIX_CACHING,,}" in
  1|true|yes|on)
    PREFIX_CACHING_ARGS=(--enable-prefix-caching)
    ;;
  0|false|no|off)
    PREFIX_CACHING_ARGS=(--no-enable-prefix-caching)
    ;;
  *)
    echo "Invalid ENABLE_PREFIX_CACHING=$ENABLE_PREFIX_CACHING" >&2
    exit 2
    ;;
esac

exec vllm serve "$MODEL_DIR" \
  --served-model-name $SERVED_MODEL_NAME \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --dtype auto \
  --quantization "${QUANTIZATION:-modelopt}" \
  --kv-cache-dtype auto \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-chunked-prefill \
  "${PREFIX_CACHING_ARGS[@]}" \
  --load-format safetensors \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --attention-backend "$ATTENTION_BACKEND" \
  --generation-config "$GENERATION_CONFIG" \
  --compilation-config "$COMPILATION_CONFIG" \
  --limit-mm-per-prompt '{"image": 4, "video": 2}' \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --mm-shm-cache-max-object-size-mb "$MM_SHM_CACHE_MAX_OBJECT_SIZE_MB" \
  --speculative-config "$SPEC_CONFIG" \
  ${EXTRA_VLLM_ARGS:-}
