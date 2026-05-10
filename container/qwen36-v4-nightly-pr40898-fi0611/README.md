# Qwen3.6 AEON Ultimate DFlash Spark vLLM

Thin DGX Spark image for validating Qwen3.6 AEON Ultimate on the latest official
community `vllm/vllm-openai:nightly` base, with the DFlash sliding-window
attention fix from vLLM PR #40898 overlaid until it lands upstream.

Modes:

- `baseline`: target model only, no DFlash.
- `dflash`: target model plus DFlash drafter.
- `bench`: category benchmark harness.
- `bash`: shell.

The image intentionally keeps the patch surface narrow so benchmark differences
are attributable to DFlash and the SWA fix rather than a custom full-source fork.

## DFlash launch profiles

`dflash` mode accepts a `PROFILE` environment variable so the same image can be
used for production serving, local gateway serving, and short-context stress
benchmarks without rewriting the command line.

| Profile | Max context | Max seqs | GPU util | Prefix cache | Use case |
|---|---:|---:|---:|---|---|
| `production` *(default)* | 200000 | 16 | 0.85 | on | Documented DGX Spark long-context recipe. Preserves prefix caching and Mamba align cache for multi-turn agents. |
| `gateway` | 256000 | 64 | 0.75 | on | Local OpenClaw-style deployment where ASR/TTS or other GPU services need headroom. |
| `benchmark` | 2048 | 256 | 0.85 | off | Short-prompt throughput sweep. Disables prefix caching because unique benchmark prompts do not reuse prefixes and Mamba align cache otherwise caps concurrency. Do not use as the default chat profile. |

All profiles keep the production-critical knobs from the Spark compose recipe:

- CUTLASS NVFP4 selected with `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`,
  `VLLM_TEST_FORCE_FP8_MARLIN=0`, and `ENABLE_NVFP4_SM100=0`.
- FlashAttention selected explicitly with `--attention-backend flash_attn`.
- DFlash k=15 via the z-lab drafter.
- BF16/auto KV cache via `--kv-cache-dtype auto`; FP8 KV is not used with this
  DFlash + FlashAttention path.
- `--generation-config vllm` so model-card sampling defaults do not silently
  override request defaults.
- `--mm-shm-cache-max-object-size-mb 256` so large Qwen3.6 image/video warmup
  objects fit in the multimodal processor cache.
- Production/gateway profiles prune CUDA graph capture sizes to 64 because
  larger graph slots are unused at the documented serving caps and waste boot
  time plus hot memory. The benchmark profile keeps vLLM's broader capture range
  so 128/256-way stress runs are not artificially constrained.

Override any profile value with the matching env var, for example:

```bash
PROFILE=gateway GPU_MEMORY_UTILIZATION=0.75 ./scripts/serve_qwen36_dflash.sh
```
