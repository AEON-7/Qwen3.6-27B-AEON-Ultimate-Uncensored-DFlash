# RTX PRO 6000 Blackwell — NVFP4 deployment recipe

> **Status: validated.** Community-tested on 2026-04-27 by [aptx5497](https://huggingface.co/aptx5497) on a stock vLLM build. Measured numbers (math/code 120 tok/s, long-form 98 tok/s, multi-turn 58–94 tok/s) are documented below. Real-world quirks discovered during validation are captured in the [Known quirks](#known-quirks-on-stock-vllm) section.

This is the recipe for running **[`AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4)** on **NVIDIA RTX PRO 6000 Blackwell** (sm_120, 96 GB GDDR7).

It is **not** the same as the DGX Spark recipe at the repo root. RTX PRO 6000 is a different SM (`sm_120` vs DGX Spark's `sm_121a`) with a different memory architecture (96 GB **dedicated** VRAM at ~1.6 TB/s vs DGX Spark's 128 GB **unified** memory at ~270 GB/s). The flag set and container image differ accordingly.

---

## QuickStart

### 1. Clone this folder and create the local model directories

```bash
git clone https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash.git
cd Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash/other-hardware/rtx6000pro
mkdir -p models
```

### 2. Authenticate to HuggingFace and pull both models

```bash
hf auth login                                    # one time, paste your HF token

hf download AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4 \
  --local-dir ./models/aeon-ultimate-nvfp4

hf download z-lab/Qwen3.6-27B-DFlash \
  --local-dir ./models/dflash-drafter
```

> The DFlash drafter is auto-gated — first download will prompt you to click-accept the terms-of-use page (instant approval). Without this drafter, you lose 40-50 % of the throughput this recipe is tuned for.

### 3. Start

```bash
docker compose up -d
docker compose logs -f vllm    # watch warmup; first boot ~8-10 min cold
```

### 4. Test

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aeon-ultimate",
    "messages": [{"role": "user", "content": "Explain zero-knowledge proofs to a basic-crypto audience."}],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

OpenAI-compatible endpoint at `http://localhost:8000/v1`.

---

## Why this differs from the DGX Spark recipe

| Setting | DGX Spark (`../../docker-compose.yml`) | **RTX PRO 6000 (this folder)** | Why |
|---|---|---|---|
| **Image** | `ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v2.1` | **`vllm/vllm-openai:v0.20.1`** | The AEON-7 patched image is built specifically for sm_121a (DGX Spark). On sm_120 (RTX PRO 6000) it would either fail to import (the `ENABLE_NVFP4_SM100=0` SM121-only build guard) or run sub-optimally. Stock vLLM has full sm_120 NVFP4 support natively — sm_120 is the canonical Blackwell NVFP4 target. |
| `--gpu-memory-utilization` | 0.85 | **0.94** | DGX Spark's unified memory thrashes above 0.88. RTX PRO 6000's dedicated VRAM has no such thrashing concern, but **`0.95` causes the FlashInfer NVFP4 GEMM autotuner to OOM on boot** (it needs ~2.12 GiB of workspace and finds only ~1.32 GiB free at 0.95). `0.94` is the validated sweet spot — autotune still runs (with some "fall back to default tactic" warnings on the largest shapes) and KV capacity is ~98 % of what 0.95 would deliver. |
| `--max-num-seqs` | 16 | **32** | Higher memory bandwidth (~6× DGX Spark) plus dedicated-VRAM headroom allows substantially more concurrent KV. Could potentially push to 48-64 for short-context workloads — measure first. |
| `--max-model-len` | 200000 | **262144** (full) | More memory headroom + absence of the unified-memory KV pressure means the full trained context window fits comfortably. |
| `TORCH_CUDA_ARCH_LIST` | `12.1a` | **`12.0`** | sm_120 is the RTX PRO 6000 target architecture. |
| `VLLM_NVFP4_GEMM_BACKEND` | (default — patched CUTLASS via image) | **`flashinfer-cutlass`** (explicit) | The sm_120 default should pick this anyway, but setting it explicitly removes any ambiguity if vLLM's heuristics change between releases. |
| `ENABLE_NVFP4_SM100`, `NVIDIA_FORWARD_COMPAT`, `NVIDIA_DISABLE_REQUIRE`, `VLLM_TEST_FORCE_FP8_MARLIN`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | required (DGX Spark / GB10 specifics) | **(none of these set)** | All work around DGX Spark / sm_121a-specific quirks (driver mismatch, SM121 build guards, baked-in test defaults). None apply on sm_120. |

What stays the same: the DFlash drafter (`z-lab/Qwen3.6-27B-DFlash`), `--speculative-config` with `num_speculative_tokens=15`, `--max-num-batched-tokens 32768` (the inductor compile-range ceiling — arch-independent), `--no-enable-prefix-caching` (required so DFlash verifier state is not mixed with reused KV/GDN prefixes), tool/reasoning parsers, multimodal hooks, and `--attention-backend flash_attn`.

---

## Measured performance

Validation run on 2026-04-27 by [aptx5497](https://huggingface.co/aptx5497) on:

- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB VRAM, driver 580.126.09
- **Stack**: vLLM `0.19.1rc1.dev102+g1a2c17634`, torch `2.11.0+cu130`, transformers `5.5.4`, flashinfer-python `0.6.7`
- **Sampling per request**: `temperature=0.6, top_p=0.95, top_k=20`
- **GPU memory utilization**: 0.94 (this recipe's default)

| Workload | Prompt tok | Output tok | TTFT | Total tok/s | Decode tok/s after first chunk | DFlash acceptance |
|---|---|---|---|---|---|---|
| math / code | 78 | 512 | 0.065 s | **120.7** | 122.6 | 16.7 % |
| long-form prose | 55 | 512 | 0.061 s | **97.6** | 98.8 | 11.9 % |
| multi-turn 1 (5,155 prompt toks shared context) | 5,155 | 192 | 0.472 s | 65.7 | 78.3 | 9.7 % |
| multi-turn 2 | 5,371 | 192 | 0.512 s | 75.0 | 93.7 | 12.8 % |
| multi-turn 3 (prefix cache hit fired) | 5,586 | 192 | **0.249 s** | 75.8 | 84.1 | 11.4 % (3,456 / 5,586 = 61.9 % prefix cache hit) |

Reference comparison — same model, same DFlash drafter, on DGX Spark (GB10): median 32 tok/s, peak 56 tok/s. RTX PRO 6000 Blackwell runs **2-4× faster** thanks to ~6× higher memory bandwidth and dedicated VRAM.

---

## Known quirks on stock vLLM

These were uncovered during validation. They aren't blockers but they're things you should know.

### `--gpu-memory-utilization 0.95` will OOM the FlashInfer autotuner

Stack trace shows `vllm.utils.flashinfer.flashinfer_mm_fp4 → flashinfer.gemm.mm_fp4` allocating ~2.12 GiB during boot, with only ~1.32 GiB free at 0.95 (after model weights + KV reservation). Three options ranked by quality:

1. **`--gpu-memory-utilization 0.94`** (this recipe's default) — autotune runs to completion with some "OOM detected, falling back to default tactic" warnings on the largest shapes. Server boots cleanly.
2. **`--gpu-memory-utilization 0.92`** — fully clean autotune, no fallback warnings, slightly smaller KV cache.
3. **`--gpu-memory-utilization 0.95 --no-enable-flashinfer-autotune`** — maximum KV cache but loses autotuned kernel selection (typically 5-10 % perf cost).

### Prefix caching stays off for DFlash

Earlier validation measured vLLM prefix-cache behavior for multi-turn requests:

- **Turn 1 of multi-turn**: 0 cached tokens (expected — first request)
- **Turn 2**: 0 cached tokens (**unexpected** — same system prompt + accumulating history should match)
- **Turn 3**: 3,456 / 5,586 = **61.9 % cache hit** with TTFT dropping from 0.51 s → 0.25 s

A separate identical-prompt smoke test (two repeats of a 10,526-token prompt) also reported 0 hits on the second request. This appears to be a real interaction between vLLM's `mamba_cache_mode=align` (auto-enabled when `--enable-prefix-caching` is on for `Qwen3_5ForConditionalGeneration`) and block-boundary alignment of accumulating context.

A/B benchmarks (with vs without prefix caching) show:
- **For short single-turn prompts**: no difference (no cache hits to land).
- **For multi-turn before hits land**: prefix caching adds slight overhead — without it, multi-turn 1 ran 93.8 tok/s; with it, 65.7 tok/s.
- **Once hits land (turn 3+)**: prefix caching wins decisively on TTFT.

**Current recommendation**: keep `--no-enable-prefix-caching` for DFlash and DDTree profiles. The old prefix-cache measurements are useful history, but speculative verifier windows and rejected draft rows make prefix reuse risky for this stack.

### `--mamba-cache-mode none` is overridden when prefix caching is on

vLLM auto-forces `align` for this model whenever `--enable-prefix-caching` is enabled. Both omitting `--mamba-cache-mode` and explicitly setting it to `none` resulted in the same auto-promotion. This is why the DFlash launch recipe disables prefix caching entirely. If prefix caching is accidentally enabled, the boot log will print:

```
Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration by default when prefix caching is enabled
```

This is expected and the warning is informational — not a failure mode.

### Reasoning parser exposes output across two fields

With `--reasoning-parser qwen3`, the model's output stream splits across:

- **Streaming**: `delta.reasoning` for chain-of-thought content, `delta.content` only for the final answer.
- **Non-streaming**: `message.reasoning` while the model is still in the reasoning section, with `message.content: null`. `message.content` populates only when the final answer block begins.

**Client integration footgun**: clients that read only `content` will see empty output for the duration of the reasoning phase. To get the full output stream, read `delta.reasoning` and `delta.content` (or `message.reasoning` and `message.content`) separately and concatenate as needed. To opt out of reasoning entirely on a per-request basis, pass `chat_template_kwargs: {enable_thinking: false}` in the request body.

---

## Other behaviors worth knowing

- **MTP speculative decoding does not work** for this model. Qwen 3.6 was not trained with MTP heads. Acceptance ≈ 0 %. Use DFlash (`num_speculative_tokens=15`) as in this compose. (This was confirmed during validation.)
- **Stochastic sampling at serve time slows decoding.** `--override-generation-config '{"temperature":0.6,...}'` makes every request stochastic and overrides per-request sampling params. Set sampling per-request in the API call instead.
- **`--max-num-batched-tokens` below 32768 is a perf foot-gun.** The inductor compile-range endpoint is `[32768]`; values like 8192 force more chunked-prefill iterations and substantially worse TTFT.
- **Stock vLLM v0.20.x+ lacks the AEON-7 patch series** (#40092 SWA fix, #40454 mamba-cache spec-decode alignment, #40662 unified spec-decode metrics, plus DGX Spark sm_121a-specific patches that don't apply on sm_120). Validation on `0.19.1rc1.dev` worked correctly; if you're on a newer `vllm/vllm-openai` tag and see incoherent output or unusual prefix cache behavior, capture engine logs and open an issue.

---

## Reporting back

If you run this recipe with different hardware, drivers, or vLLM versions and see different numbers — please [open an issue](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash/issues) with your `nvidia-smi` output, vLLM version, and bench results. Validated numbers in this README will be expanded as more reports come in.
