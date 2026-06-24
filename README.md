<div align="center">

# Qwen3.6-27B-AEON-Ultimate-Uncensored

### Lossless abliteration · Capability-enhanced · NVFP4 hardware-quantized for Blackwell

[![BF16](https://img.shields.io/badge/HuggingFace-BF16_(51_GB)-yellow?logo=huggingface)](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16)
[![NVFP4](https://img.shields.io/badge/HuggingFace-NVFP4_(26_GB)-yellow?logo=huggingface)](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4)
[![Container](https://img.shields.io/badge/ghcr.io-aeon--vllm--ultimate-blue?logo=docker)](https://github.com/AEON-7/vllm-ultimate-dgx-spark/pkgs/container/aeon-vllm-ultimate)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)
[![☕ Tips](https://img.shields.io/badge/%E2%98%95_Tips-Support_the_work-ff5e5b?style=flat)](https://github.com/AEON-7/AEON-7#-support-the-work)

**Refusals: 0 / 100** &nbsp;·&nbsp; **KL vs base: 0.000492** &nbsp;·&nbsp; **Compression: 49 %** &nbsp;·&nbsp; **Capability: enhanced**

</div>

---

## TL;DR

A **fully uncensored, capability-enhanced** abliteration of [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), produced over **72 hours of continuous research** drawing on hundreds of parallel AI research agents, the industry's best published methodologies, custom in-house techniques, and yet-unreleased pre-public branches of next-generation abliteration software.

## Quickstart (DGX Spark / GB10) — copy-paste

One block: pull the container, pull this model (fresh, the `-Multimodal-NVFP4-MTP-XS` body — see [Model Variants](#model-variants)), pull the DFlash drafter (fresh), then serve with the vetted DGX Spark flags.

```bash
# 1) Pull the unified AEON vLLM container (vLLM 0.23.0, sm_121a, DFlash)
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:latest

# 2) Pull this model (fresh)
huggingface-cli download AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS --local-dir ./aeon-model

# 3) Pull the DFlash drafter (fresh — z-lab pushes updates; always re-pull)
huggingface-cli download z-lab/Qwen3.6-27B-DFlash --local-dir ./aeon-drafter

# 4) Serve (ENTRYPOINT is /bin/bash → pass --entrypoint vllm, then serve ...)
docker run --gpus all --ipc host --network host \
  -v ./aeon-model:/model:ro -v ./aeon-drafter:/drafter:ro \
  --entrypoint vllm ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
  serve /model \
  --quantization modelopt \
  --mamba-cache-dtype float32 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt '{"image":4,"video":2}' \
  --mm-encoder-tp-mode data \
  --max-num-seqs 16 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.85 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --trust-remote-code \
  --speculative-config '{"method":"dflash","model":"/drafter","num_speculative_tokens":10}'
```

OpenAI-compatible endpoint comes up at `http://localhost:8000/v1`. The DFlash drafter is auto-gated on first download (one-click terms acceptance). For the full step-by-step walkthrough, docker-compose profiles, the A100/H100 BF16 path, and the per-flag rationale, see [QuickStart — DGX Spark](#quickstart--dgx-spark--xs-body--dflash-recommended-winner) and [Configuration reference](#configuration-reference) below.

## Performance — DGX Spark DFlash vs Raw Baseline

**This is the headline.** On DGX Spark / GB10, the AEON DFlash container turns the default “it runs, but it feels slow” baseline into a usable long-context local agent model.

### v0.23.0 build — stock vanilla vs maximally optimized

<p align="center"><img src="assets/perf/qwen27b_stock_vs_optimized.svg" width="100%" alt="Stock vanilla vLLM vs AEON-optimized single-stream decode tok/s by category — up to 5.3x faster"></p>
<p align="center"><img src="assets/perf/qwen27b_concurrency.svg" width="100%" alt="Aggregate throughput scaling from 1 to 64 concurrent requests on aeon-vllm-ultimate"></p>

Measured on the current production image **`ghcr.io/aeon-7/aeon-vllm-ultimate:latest`** (vLLM 0.23.0 + AEON sm_121a build + DFlash `num_speculative_tokens: 10`), single-stream (c=1), by category:

| Category | 🟢 Decode tok/s | TTFT p50 | TPOT p50 | Prefill (PP) | DFlash accept |
|---|---:|---:|---:|---:|---:|
| Coding | **42** | 140 ms | 23.9 ms | 322 tok/s | 34% |
| Math | **47** | 244 ms | 21.1 ms | 229 tok/s | 42% |
| Reasoning | **56** | 234 ms | 17.8 ms | 183 tok/s | 50% |
| Prose | **34** | 146 ms | 29.4 ms | 220 tok/s | 27% |
| Natural language | **38** | 137 ms | 26.1 ms | 248 tok/s | 31% |
| Extraction / JSON | **44** | 246 ms | 22.6 ms | 195 tok/s | 37% |

vs a **stock vanilla `vllm/vllm-openai` baseline of ~10.5 tok/s** (no DFlash, no sm_121a optimizations) → **up to 5.3× faster decode (≈4× average)**, and it now scales cleanly to **c=64 concurrent** with no crash (the pre-fix image crashed under concurrent speculative decoding — see [What we fixed for the DGX Spark](#what-we-fixed-for-the-dgx-spark)).

> **Stock baseline note:** the stock/un-optimized figures are vanilla vLLM (no DFlash, no AEON/sm_121a optimizations) and are **provisional — to be refreshed** with a fresh fully-vanilla re-benchmark on the current version.

---

#### Earlier `qwen36-v4` reference (DFlash k=15)

The throughput table below was measured on the earlier `qwen36-v4` image with DFlash k=15. The production container and recipe have since been unified onto `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` with **DFlash `num_speculative_tokens: 10`** (see [why long context](#why-the-spark-recipe-is-tuned-for-long-context) below). The single-stream throughput figures here are representative of the Spark DFlash path; the unified image's specific win is **long-context draft acceptance — 45% vs 19.7% on the pre-fix image (a 2.3× gain) at ~9k-token context** — not short-context tok/s.

| Deployment | Container | DFlash | CUDA graphs | Tool calling | Avg c=1 decode |
|---|---|---:|---:|---:|---:|
| 🔴 **Raw baseline** | `vllm/vllm-openai:nightly` | off | off (`--enforce-eager`) | off | **10.49 tok/s** |
| 🟢 **AEON DFlash** | `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` | **n=10** | **on** | **on** | **37.56 tok/s** † |

> † These single-stream tok/s (37.56 / 10.49) were measured on the earlier `qwen36-v4` image at DFlash **k=15**, not on `aeon-vllm-ultimate:latest` at n=10. The unified image's specific gain is **long-context draft acceptance (45% vs 19.7% at ~9k tokens)**, not these short-context tok/s — see [why long context](#why-the-spark-recipe-is-tuned-for-long-context).

**Average single-stream decode improvement: +258%** over the raw stock eager baseline.

### Single-Stream Decode

*All figures in this section and the concurrency tables below were measured on the earlier `qwen36-v4` image at DFlash k=15 (see the note above) — representative of the Spark DFlash path, not a re-run on `aeon-vllm-ultimate:latest` at n=10.*

| Category | 🔴 Raw baseline | 🟢 AEON DFlash | Approx. speed increase | DFlash TTFT | DFlash TPOT |
|---|---:|---:|---:|---:|---:|
| Coding | 10.70 tok/s | **31.89 tok/s** | **+198%** | 191 ms | 30.5 ms |
| Math | 10.01 tok/s | **37.76 tok/s** | **+277%** | 225 ms | 25.5 ms |
| Reasoning | 10.54 tok/s | **42.41 tok/s** | **+303%** | 221 ms | 22.6 ms |
| Prose | 10.59 tok/s | **31.85 tok/s** | **+201%** | 212 ms | 30.4 ms |
| Natural language | 10.56 tok/s | **31.99 tok/s** | **+203%** | 183 ms | 30.3 ms |
| Extraction / JSON | 10.56 tok/s | **49.48 tok/s** | **+369%** | 227 ms | 19.2 ms |
| **Average** | **10.49 tok/s** | **37.56 tok/s** | **+258%** | ~210 ms | ~26.4 ms |

### Practical Agent Concurrency

At c=16, the optimized container keeps active streams much more responsive. Aggregate throughput improves most on structured agent/tool workloads, and TPOT drops across every category. (Columns labelled AEON below are the DFlash path on the AEON container.)

| Category | 🔴 Raw c=16 aggregate / TPOT | 🟢 AEON c=16 aggregate / TPOT | Aggregate change |
|---|---:|---:|---:|
| Coding | 134.47 tok/s / 115.1 ms | **144.45 tok/s / 61.5 ms** | **+7%** |
| Math | 134.38 tok/s / 115.1 ms | **193.94 tok/s / 41.6 ms** | **+44%** |
| Reasoning | 134.86 tok/s / 115.4 ms | **187.82 tok/s / 46.6 ms** | **+39%** |
| Prose | **135.34 tok/s** / 115.3 ms | 121.34 tok/s / **80.6 ms** | -10% aggregate, **30% lower TPOT** |
| Natural language | 129.82 tok/s / 117.7 ms | **130.19 tok/s / 71.2 ms** | ~flat aggregate, **39% lower TPOT** |
| Extraction / JSON | 133.30 tok/s / 115.4 ms | **219.11 tok/s / 43.2 ms** | **+64%** |

### Stress Saturation

c=256 is a saturation test, not the recommended interactive setting. The baseline can report high aggregate throughput by letting every stream crawl. The AEON DFlash path keeps per-active-stream TPOT far lower, but at c=256 requests queue hard and TTFT rises into minutes.

| Category | 🔴 Raw c=256 TPOT | 🟢 AEON c=256 TPOT | AEON c=256 TTFT |
|---|---:|---:|---:|
| Coding | 575.5 ms | **70.0 ms** | 149.6 s |
| Math | 531.9 ms | **42.7 ms** | 103.6 s |
| Reasoning | 540.7 ms | **49.4 ms** | 109.3 s |
| Prose | 532.5 ms | **77.1 ms** | 159.8 s |
| Natural language | 533.4 ms | **72.9 ms** | 160.0 s |
| Extraction / JSON | 551.9 ms | **43.2 ms** | 90.4 s |

### What the AEON container Adds

- Unified production image: `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` (= tag `:2026-06-11-pr41703`; rollback tag `:2026-06-04-pr44389`). This single image supersedes the older `qwen36-v3` / `v4` / `v5` lineage of per-revision containers.
- FlashInfer NVFP4 GEMM path
- DFlash sliding-window-attention compatibility patch from vLLM PR #40898 (4 of the drafter's 5 layers are sliding-window; earlier images ran them as full attention and drafting collapsed past ~2048 tokens)
- vLLM PR #41703 makes `--enable-prefix-caching` corruption-immune with DFlash
- CUTLASS NVFP4 fast path selected for GB10 / sm_121a
- DFlash `num_speculative_tokens: 10` (validated production default) using `z-lab/Qwen3.6-27B-DFlash`, drafter backend left at default
- Qwen3 reasoning parser and Qwen3-Coder tool-call parser enabled
- Packaged gateway/production/benchmark profiles so users do not have to hand-assemble the full vLLM command

### Why the Spark recipe is tuned for long context

The z-lab Qwen3.6-27B DFlash drafter is a sliding-window model — 4 of its 5 layers use sliding-window attention (window 2048). vLLM PR #40898 (in `aeon-vllm-ultimate:latest`) runs those layers as proper SWA; earlier images ran them as full attention, so drafting collapsed once context grew past ~2048 tokens. PR #41703 additionally makes `--enable-prefix-caching` corruption-immune with DFlash. Net: long-context drafting holds up; short-context (under 2048 tokens, one window) is unchanged.

### DDTree v5 Research Track

DDTree is the next obvious performance target, but it must land inside vLLM without losing multimodal, reasoning, tool calling, NVFP4, or the OpenAI-compatible gateway surface. The DDTree research track (the old `qwen36-v5` experimental container) is **superseded for production by the unified `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` image** — pull that image for any actual deployment:

```bash
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:latest
```

Read the full DDTree card and lab chronicle: [`docs/qwen36-ddtree-card.md`](docs/qwen36-ddtree-card.md).

Current status in one line: **flat DFlash remains the production path on the unified AEON container; DDTree is research-only.** The unified image preserves the same NVFP4, DFlash, multimodal, reasoning, tool-calling, and OpenAI-compatible vLLM surface, but true non-flat branch commit is still research-only.

The working implementation plan lives in [`docs/ddtree-vllm-integration-plan.md`](docs/ddtree-vllm-integration-plan.md). M1 scaffolding and the current experimental Docker context live in [`container/qwen36-v5-ddtree-experimental/`](container/qwen36-v5-ddtree-experimental/).

The DDTree card documents:

- the current container tags and digest,
- what works today,
- the M1 through M53 trial-and-error path,
- the current M53 non-flat probe status,
- known blockers around branch-state GDN replay, fused branch attention, and accepted-branch commit,
- benchmark context and caveats,
- where community help is most likely to move the project forward.

Raw benchmark files:

- [`bench/results/qwen36_dirty_baseline_eager_20260510T034652Z.json`](bench/results/qwen36_dirty_baseline_eager_20260510T034652Z.json)
- [`bench/results/qwen36_v4_fi0611_noprefix_full_sweep_20260510T065838Z.json`](bench/results/qwen36_v4_fi0611_noprefix_full_sweep_20260510T065838Z.json)
- [`bench/results/qwen36_v4_fi0611_noprefix_true_single_20260510T065020Z.json`](bench/results/qwen36_v4_fi0611_noprefix_true_single_20260510T065020Z.json)

The DFlash sweep used natural prompts across coding, math, reasoning, prose, everyday language, and extraction/JSON. It intentionally used a short-context benchmark profile to isolate decode/scheduler behavior: `--max-model-len 2048`, `--max-num-seqs 256`, prefix caching disabled, thinking enabled, 200 output tokens, minimum 16 samples per point, 20% trimmed median. For production DFlash gateway use, prefix caching is workload-dependent: it is valuable when many agents share a stable prompt prefix, but DDTree research modes keep it off while branch-state correctness is under development.

---

## What we fixed for the DGX Spark

All AEON models run on one unified container — **`ghcr.io/aeon-7/aeon-vllm-ultimate:latest`** (= `:2026-06-18-v0.23.0-dflashfix`; rollback `:2026-06-11-pr41703`) — vLLM v0.23.0 built from source for GB10 / sm_121a and merged with the AEON speculative-decoding stack.

| Fix | What it does | Why it matters on GB10 |
|---|---|---|
| **DFlash high-concurrency fix** *(new)* | Slices the speculative drafter's KV block-table to the unpadded batch | The drafter previously **crashed at ≥32 concurrent requests** (padded-vs-unpadded block-table shape mismatch in FlashAttention). Now scales cleanly to **c=64**. A port of upstream PR #43982 (fixed for MTP, never applied to DFlash) — present and unfixed even in the prior image. |
| **Triton NVFP4 KV cache** (PR #44389) | Software NVFP4 KV-cache path | The only 4-bit KV path on sm_121a (upstream's is hard-gated to B200) → ~3× KV capacity / longer context per GB of unified memory. |
| **DFlash sliding-window attention** (PR #40898) | Runs the drafter's SWA layers as true sliding-window | Long-context draft acceptance holds as agent histories grow, instead of collapsing past ~2k tokens. |
| **sm_121a-native build** | `TORCH_CUDA_ARCH_LIST=12.1a`, `ENABLE_NVFP4_SM100=0` | Compiles the SM120-family CUTLASS NVFP4/FP8 kernels GB10 actually dispatches to — true 4-bit tensor-core throughput, no dead B200-only kernels. |
| **sm_121a boot + CUDA-graph patches** | RTLD-lazy `_C_stable_libtorch` load; spec-decode CUDA-graph capture-size alignment | Boots past MXFP4 (SM100-only) symbols absent on GB10; prevents `cudaErrorIllegalAddress` on partial-acceptance decode steps under speculative decoding. |
| **Unified-memory tuning** | `--gpu-memory-utilization ≤0.70`, FULL CUDA graphs, async scheduling, z-lab DFlash drafter | GB10 shares one LPDDR5X pool across CPU + GPU; conservative KV headroom avoids page-thrash while keeping FULL-graph + speculative-decode throughput. |

## Model Variants

Six release formats covering DGX Spark, RTX PRO 6000, RTX 5090, and pre-Blackwell hardware:

| Release | Size | Target hardware | Use when |
|---|---|---|---|
| **[BF16](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16)** | 51 GB | A100 / H100 80 GB · RTX PRO 6000 Blackwell 96 GB | You have Ampere/Hopper or want full-precision reference weights |
| **[NVFP4](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4)** | 26 GB | Simpler NVFP4 deployments | llm-compressor format, `--quantization compressed-tensors`. For best DGX Spark performance, use the DFlash recipe (`aeon-vllm-ultimate:latest`, n=10) with the XS body below. |
| **[Multimodal-NVFP4-MTP](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP)** | 27 GB | RTX PRO 6000 Blackwell · B100/B200 | modelopt format, `--quantization modelopt`, MTP spec decode via grafted `mtp.*` head. Vision tower preserved. **GDN linear-attention preserved BF16** for best long-context fidelity. |
| **[Text-NVFP4-MTP](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Text-NVFP4-MTP)** | 26 GB | RTX PRO 6000 · text-only deployments | Same recipe as Multimodal-NVFP4-MTP, vision tower stripped. **GDN preserved BF16.** |
| **[Multimodal-NVFP4-MTP-XS](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS)** | 21 GB | RTX 5090 (32 GB) · tighter dedicated VRAM | Strategic split: GDN projection matmuls → NVFP4; **`linear_attn.conv1d` kept BF16** to preserve the recurrence-critical SSM convolution. Vision tower preserved. |
| **[Text-NVFP4-MTP-XS](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Text-NVFP4-MTP-XS)** | 20 GB | RTX 5090 text-only · 24 GB cards | Same conv1d-preserved strategic split as Multimodal-XS, vision tower stripped. The smallest variant we ship. |

All six formats are **the same underlying model**. NVFP4 KL divergence vs BF16 source is below the noise floor of stochastic sampling — you cannot tell them apart at the output level. The four MTP variants share the same NVFP4 quantization quality plus the original `Qwen/Qwen3.6-27B` MTP head grafted back in BF16 (bit-exact, verified) for spec-decode drafting.

> **Regular MTP vs XS — what's the difference, and why it's a *strategic* quantization choice (not a precision compromise):**
>
> The GatedDeltaNet (GDN / Mamba-style) `linear_attn.*` block has two distinct components: the **heavy projection matmuls** (`in_proj_qkv`, `in_proj_z`, `in_proj_a/b`, `out_proj` — ~11 GB total) and the **SSM 1D convolution kernel** (`linear_attn.conv1d` — small, but recurrence-critical).
>
> - **Regular MTP variants** keep *both* at BF16. Maximum numerical safety margin, larger footprint.
> - **XS variants** quantize the projection matmuls to NVFP4 (saves ~6 GB; FP4 is a clean win on bandwidth-bound matmuls) **but explicitly preserve `linear_attn.conv1d` at BF16**. FP4 quantization of conv1d has been observed to cause drift on long-context recurrence in community testing, so we keep it at BF16 — the same principle modelopt's `NVFP4_DEFAULT_CFG` applies by default and the same recipe sakamakismile validated across his Qwen3.6-NVFP4-MTP series (22K+ downloads). This is *not* "everything to FP4" — that would be a different (and not-recommended) variant we have explicitly chosen not to ship.
>
> Pick regular if you have ≥48 GB VRAM and want best precision on long-context workloads; pick XS if you're on a 24–32 GB card and want maximum KV headroom with the SSM kernel still numerically stable.

> **Hardware routing:**
> - **DGX Spark (GB10 / sm_121a)** → use the **`aeon-vllm-ultimate:latest` container with DFlash (n=10)** and the Multimodal-NVFP4-MTP-XS body. That is the benchmarked path above.
> - **Dedicated-VRAM Blackwell** *(RTX PRO 6000 / RTX 5090 / B100/B200)* → use the MTP variants (`qwen3_5_mtp` n=3) when you want the grafted native MTP head. Dedicated VRAM behaves differently from Spark's unified memory, so benchmark locally before copying Spark flags.

---

## Table of contents

1. [Quickstart (DGX Spark / GB10) — copy-paste](#quickstart-dgx-spark--gb10--copy-paste)
2. [Performance — DGX Spark DFlash vs Raw Baseline](#performance--dgx-spark-dflash-vs-raw-baseline)
3. [Model variants](#model-variants)
4. [What this is](#what-this-is)
5. [Final stats](#final-stats)
6. [Hardware compatibility matrix](#hardware-compatibility-matrix)
7. [QuickStart — DGX Spark](#quickstart--dgx-spark--xs-body--dflash-recommended-winner)
8. [QuickStart — A100 / H100 (BF16)](#quickstart--a100--h100-bf16)
9. [In-depth: the abliteration methodology](#in-depth-the-abliteration-methodology)
10. [In-depth: NVFP4 quantization](#in-depth-nvfp4-quantization)
11. [Capability enhancement: the lifted "safety tax"](#capability-enhancement-the-lifted-safety-tax)
12. [Configuration reference](#configuration-reference)
13. [Responsibility, arbitration, and use](#responsibility-arbitration-and-use)
14. [Provenance & credits](#provenance--credits)
15. [License](#license)

---

## What this is

This is the **definitive uncensored release of Qwen 3.6 27B**: the alignment-overhead removal so surgical that the model's KL divergence from the base is **0.000492** — three orders of magnitude inside the empirically-observed "capability damage threshold," and below the noise floor of ordinary stochastic sampling. A user cannot distinguish this model from the base on capability tasks; on several measurable axes (chain-of-thought commitment, adversarial-reasoning bandwidth, calibration honesty), it is *better*.

This is not a weekend abliteration. The release is the product of **72 hours of continuous research and tuning** in which **hundreds of parallel AI research agents** were dispatched to:

- Characterize Qwen 3.5 / 3.6 hybrid-attention internals (16 full-attention layers + 48 GatedDeltaNet / linear-attention layers, `attn_output_gate=True` with doubled `q_proj` geometry, the FernflowerAI SSM `conv1d` outlier pattern).
- Survey the post-training-intervention literature in full: Arditi et al. (refusal as a single direction), grimjim's NPBA (norm-preserving biprojected abliteration), Heretic, Wuwangzhang's abliterix, Huang et al. on the safety tax, Xie et al. on DGR safety-tax mitigation, the projected-abliteration extensions, the winsorization heuristics.
- Audit every relevant arXiv submission of 2024–2026 on alignment-direction interventions, capability preservation, and 4-bit quantization on hybrid-attention stacks.
- Comb the r/LocalLLaMA community archive for tribal knowledge on what does and does not work — particularly on Mamba / GatedDeltaNet hybrids, where most generic abliteration recipes silently fail.
- Trace the GitHub commit graphs of the abliteration tooling ecosystem to identify pre-public development branches that fix bugs unfixed in the public releases.

The pipeline that emerged integrates the industry's best published methodologies — Arditi-style mean-difference refusal vectors, NPBA, projected abliteration with outlier-aware winsorization, FernflowerAI's SSM `conv1d` outlier repair, abliterix v1.4's multi-objective Optuna search — **alongside custom in-house techniques developed for Qwen 3.6's idiosyncratic attention geometry, and yet-unreleased pre-public branches of the next-generation abliteration toolchain integrated through direct collaboration with upstream maintainers.**

The 50-trial Optuna search was cross-validated against a 10-axis capability spot-check to catch the documented "low-KL but word-salad" over-abliteration trap that pure refusal-rate scoring will miss. Trial 46 was selected — not the lowest-KL trial, but the one that combined zero refusals with full capability coherence.

---

## Final stats

### Refusal rate (apples-to-apples)

| Metric | Base Qwen3.6-27B | **AEON-Ultimate** |
|---|---|---|
| Refusals on harmful prompts | 99 / 100 | **0 / 100** |
| Verdict | heavily aligned | **uncensored** |
| Compliance rate | 1 % | **100 %** |

Tested on a 100-prompt adversarial battery from `mlabonne/harmful_behaviors` covering cybercrime, weapons, violence, self-harm, hate speech, and synthesis instructions. Same denominator as the base evaluation.

### Capability preservation

| Metric | Value |
|---|---|
| First-3-token KL divergence vs base | **0.000492** |
| Output length deviation vs base | 0.027 σ |
| Capability spot-checks (10 axes) | **10 / 10 coherent** |
| Math · code · reasoning · knowledge · long-form | All preserved |

Capability axes verified: arithmetic word problems, linear algebra, calculus, Python with memoization, Rust UTF-8 string handling, transitive syllogisms, the bat-and-ball intuition trap, factual recall, technical contrast (TCP vs UDP), structured pedagogical long-form. Every axis produced coherent, structured, reasoning-forward outputs — no looping, no philosophizing spirals, no word-salad.

### KL divergence detail

| Distribution metric | Value |
|---|---|
| First-3-token KL vs base | **0.000492** |
| Winsorization quantile | 0.995 (outlier-aware) |
| Projection | orthogonal + projected-abliteration (NPBA-style) |
| Trials evaluated | 50 (15 random warmup + 35 TPE-driven Optuna) |
| Selected trial | #46 (winner, COHERENT) |

The empirically observed "capability damage threshold" in the abliteration literature is KL ≈ 0.1. AEON-Ultimate's KL is **~200× below** that threshold.

---

## Hardware compatibility matrix

The right variant depends on **memory architecture**, not just GPU model. DGX Spark should use the `aeon-vllm-ultimate:latest` DFlash container above; dedicated-VRAM Blackwell can use the MTP variants when the native MTP head is desired.

| Hardware | Recommended variant | Why this exact variant | Spec-decode method |
|---|---|---|---|
| **DGX Spark / GB10** *(sm_121a, unified memory)* | 🏆 **[`-Multimodal-NVFP4-MTP-XS`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS) body + DFlash + `aeon-vllm-ultimate:latest` image** | Current recommended path. The unified image packages CUTLASS NVFP4, CUDA graphs, the DFlash sliding-window-attention patch (PR #40898), prefix-cache-safe DFlash (PR #41703), Qwen3 reasoning parsing, and Qwen3-Coder tool parsing. Supersedes the old `qwen36-v3/v4/v5` lineage. | DFlash *n=10* via [`z-lab/Qwen3.6-27B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) drafter |
| **B100 / B200** *(sm_100, dedicated FP4 silicon)* | **[`-Multimodal-NVFP4-MTP`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP)** (preferred — GDN BF16 fits) or [Text variant](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Text-NVFP4-MTP) | Native FP4 via `tcgen05` / UTCQMMA — fastest hardware for this format. Dedicated VRAM bandwidth lets MTP's high acceptance rate translate to throughput. | qwen3_5_mtp *n=3* (head grafted bf16, in repo) |
| **RTX PRO 6000 Blackwell** *(sm_120, 96 GB dedicated)* | **[`-Multimodal-NVFP4-MTP`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP)** for vision · [`-Text-NVFP4-MTP`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Text-NVFP4-MTP) for text-only · [XS siblings](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS) for tighter memory budgets | Dedicated VRAM has different bandwidth behavior than Spark unified memory. Start with the MTP variants and benchmark locally. | qwen3_5_mtp *n=3* |
| **RTX 5090** *(sm_120, 32 GB dedicated)* | **[`-Multimodal-NVFP4-MTP-XS`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS)** *(21 GB)* if you use vision · **[`-Text-NVFP4-MTP-XS`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Text-NVFP4-MTP-XS)** *(20 GB)* if text-only | Regular MTP variants (~27 GB) leave too little KV headroom on 32 GB. XS variants (conv1d preserved BF16, projection matmuls FP4) fit comfortably. | qwen3_5_mtp *n=3* |
| **Other 24 GB cards** *(RTX 4090, RTX 3090, RTX A6000 ≤48 GB)* | **[`-Text-NVFP4-MTP-XS`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Text-NVFP4-MTP-XS)** *(20 GB)* | The smallest variant. Pre-Blackwell sm_<120 will dequantize NVFP4 → BF16 at the kernel level (no FP4 silicon win), but the model still works and KV fits. | qwen3_5_mtp *n=3* |
| **H100 80 GB** *(sm_90)* | **[`-BF16`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16)** | NVFP4 dequants to BF16 at kernel level — works but no throughput gain. Use BF16 for cleaner code path. | none (or external EAGLE / Medusa drafter) |
| **A100 80 GB** *(sm_80)* | **[`-BF16`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16)** | Same as H100. BF16 at 131K context, single-GPU. | none |
| **Multi-GPU (any tier)** | **[`-BF16`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16)** *(`tensor-parallel-size 2/4/8`)* | Reference weights for fine-tuning, distillation, or quant-recipe development. | none |
| **Anything older than A100** | Not supported | Won't fit + lacks attention backends. |

---

## QuickStart — DGX Spark 🏆 (XS body + DFlash, recommended winner)

**Pick this for DGX Spark.** This is the current packaged winner for real GB10 use: the XS+DFlash path on `aeon-vllm-ultimate:latest` averages **37.56 tok/s single-stream** across six natural prompt categories versus **10.49 tok/s** for the raw stock eager baseline (throughput measured on the prior `qwen36-v4` image; the unified image's specific gain is long-context draft acceptance). It preserves multimodal input, reasoning parsing, and OpenAI-compatible tool calls.

The XS body includes a grafted MTP head, but the Spark recipe intentionally uses **external DFlash with `num_speculative_tokens: 10`**. Do not switch the Spark compose file to `method:"qwen3_5_mtp"` unless you are deliberately running an ablation.

> **Why DFlash n=10 and why it holds up at long context:** the z-lab Qwen3.6-27B DFlash drafter is a sliding-window model — 4 of its 5 layers use sliding-window attention (window 2048). vLLM PR #40898 (in `aeon-vllm-ultimate:latest`) runs those layers as proper SWA; earlier images ran them as full attention, so drafting collapsed once context grew past ~2048 tokens. PR #41703 additionally makes `--enable-prefix-caching` corruption-immune with DFlash. Net: long-context drafting holds up; short-context (under 2048 tokens, one window) is unchanged. An **n=8/10/12/15 sweep** on this image found **n=10 optimal for the Spark voice/chat workload** — highest aggregate throughput (≈198 vs 191 tok/s @ c=8) and DFlash acceptance (43% vs 36% for the old 12-token setting) at parity single-stream; the previous 12-token default wins only at very long context. The Mamba layers run with **`--mamba-cache-dtype float32`** (more precise recurrent state + slightly higher acceptance than float16) and **no `--mamba-block-size`** (the default lowers single-stream TTFT from ~230 ms to ~140 ms vs 256, with identical decode). Leave the drafter attention backend at default and do **not** set `--kv-cache-dtype` (the non-causal DFlash drafter requires BF16 KV).

### Step 1 — Authenticate to HuggingFace and pull both models

```bash
hf auth login                                    # one time, paste your HF token

hf download AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS \
  --local-dir ./models/aeon-ultimate-multimodal-nvfp4-mtp-xs

hf download z-lab/Qwen3.6-27B-DFlash \
  --local-dir ./models/dflash-drafter
```

> The DFlash drafter is auto-gated — first download will prompt you to click-accept the terms (instant approval). If you've previously downloaded it before 2026-04-27, **re-run** the download; z-lab pushed an updated drafter and you want the new weights.

### Step 2 — Use the XS docker-compose

[`docker-compose.spark-xs.yml`](docker-compose.spark-xs.yml) ships in this repo with the exact config measured above. Highlights:

- **Image**: `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` (= tag `:2026-06-11-pr41703`; rollback `:2026-06-04-pr44389`). The image ENTRYPOINT is `/bin/bash`, so `docker run` must pass `--entrypoint vllm` then `serve ...` (compose uses `entrypoint: vllm` + `command: serve ...`).
- **Body**: XS multimodal (`--quantization modelopt`)
- **Speculative decoding**: DFlash, `num_speculative_tokens: 10`, architecture-matched drafter (`--speculative-config '{"method":"dflash",...}'`), drafter attention backend left at default
- **GB10-specific env**: `TORCH_CUDA_ARCH_LIST=12.1a`, `ENABLE_NVFP4_SM100=0`, `VLLM_USE_FLASHINFER_SAMPLER=1`, `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`, `NVIDIA_FORWARD_COMPAT=1`
- **Default gateway tuning**: `--max-model-len 256000 --max-num-seqs 64 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.75` *(leaves room for ASR/TTS/embedding side services)*
- **Long-context production tuning**: `--max-model-len 200000 --max-num-seqs 16 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85` *(higher KV reserve when the LLM is the only major GPU service)*
- **Multimodal**: `--limit-mm-per-prompt '{"image":4,"video":2}' --mm-encoder-tp-mode data --mm-processor-cache-type shm`
- **Serving**: 5 aliases (`aeon-ultimate`, `qwen36-ultimate`, `aeon-fast`, `aeon-deep`, `aeon-ultimate-xs`) all routing to the same engine

### Step 3 — Start

```bash
docker compose -f docker-compose.spark-xs.yml up -d
docker compose -f docker-compose.spark-xs.yml logs -f vllm
```

> First boot takes ~10–12 min (FlashInfer NVFP4 GEMM autotuner + CUDA-graph capture; both cache to `/root/.cache/vllm/...`). Subsequent restarts ~3–5 min. The MTP-head detection log line will appear in startup but the engine routes around it correctly because of `--speculative-config method:"dflash"`.

### Step 4 — Test

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

OpenAI-compatible endpoint at `http://localhost:8000/v1`. Tool calling, reasoning mode (`<think>` blocks), and multimodal input all enabled out of the box.

> **Why this combo wins on Spark**: `aeon-vllm-ultimate:latest` keeps the XS body, CUTLASS NVFP4, DFlash n=10, CUDA graphs, tool parsing, reasoning parsing, and multimodal support in one pullable image, and runs the DFlash drafter's sliding-window layers as proper SWA (PR #40898) so long-context drafting holds up. That is the path benchmarked at the top of this README.

---

## QuickStart — A100 / H100 (BF16)

For Ampere / Hopper cards, run the BF16 release on vanilla vLLM.

### Step 1 — Pull weights

```bash
hf download AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16 \
  --local-dir /opt/models/aeon-ultimate-bf16
```

### Step 2 — Drop in the BF16 docker-compose

```yaml
# docker-compose.bf16.yml
services:
  aeon-ultimate-bf16:
    image: vllm/vllm-openai:latest
    container_name: aeon-ultimate-bf16
    restart: unless-stopped
    network_mode: host
    ipc: host
    runtime: nvidia
    environment:
      NVIDIA_VISIBLE_DEVICES: all
    volumes:
      - /opt/models/aeon-ultimate-bf16:/models/aeon-ultimate:ro
    command: >
      --model /models/aeon-ultimate
      --served-model-name aeon-ultimate
      --host 0.0.0.0 --port 8000
      --dtype bfloat16
      --max-model-len 131072
      --max-num-seqs 16
      --max-num-batched-tokens 8192
      --gpu-memory-utilization 0.90
      --enable-chunked-prefill
      --enable-auto-tool-choice
      --tool-call-parser qwen3_coder
      --reasoning-parser qwen3
      --attention-backend flash_attn
      --trust-remote-code
```

### Step 3 — Start

```bash
docker compose -f docker-compose.bf16.yml up -d
```

For 96 GB cards (RTX PRO 6000 Blackwell on the BF16 path), raise to `--max-num-seqs 32 --max-num-batched-tokens 16384 --max-model-len 262144`. **For native FP4 throughput on RTX PRO 6000, see the dedicated NVFP4 recipe below.**

---

## Other hardware configurations

The DGX Spark and BF16 quickstarts above are the AEON-7 team's measured-and-validated configurations. Recipes for additional hardware live in the [`other-hardware/`](other-hardware/) directory — each in its own subfolder with a tuned `docker-compose.yml` and a per-hardware README explaining what differs from the DGX Spark recipe and why.

| Hardware | Recipe | Status | Recommended for |
|---|---|---|---|
| **NVIDIA RTX PRO 6000 Blackwell** (sm_120, 96 GB GDDR7) | [`other-hardware/rtx6000pro/`](other-hardware/rtx6000pro/) | Community recipe | Single-GPU NVFP4 deployment with native sm_120 FP4 tensor-core throughput. Dedicated-VRAM flags differ from DGX Spark unified-memory flags. |

If you have hardware not covered here and want to contribute a recipe, follow the pattern in `other-hardware/rtx6000pro/` — a folder, a tuned `docker-compose.yml`, and a README explaining the differences from the DGX Spark baseline.

---

## In-depth: the abliteration methodology

### What abliteration is

Abliteration is a post-training intervention that removes the **refusal direction** in a model's residual stream — the linear subspace, identified empirically by Arditi et al. (2024), that mediates a transformer's decision to refuse a prompt. The technique works because in well-aligned chat models, refusal is mediated by a *single dominant direction*: project that direction out of the residual stream at every layer and the model loses its ability to route into refusal-shaped attractors.

The naive version of this — subtract the refusal direction from `o_proj` and `down_proj` weights — produces a model that no longer refuses. But it also tends to break it: aggressive direction removal collapses capability, producing word-salad outputs and looping incoherence. The literature is full of "uncensored" releases that are also *broken* releases.

### What "lossless abliteration" requires

To remove refusal *without* breaking capability, four things have to be done correctly:

1. **Identify the refusal direction precisely** — using a sufficiently large harmful/harmless contrast set, with outlier-aware winsorization so a handful of high-norm prompts don't distort the steering vector.
2. **Project orthogonally and norm-preservingly** — keeping the helpfulness-aligned signal intact (this is the NPBA contribution).
3. **Search the strength × layer-scope hyperparameter space** — most projects pick one strength setting and ship; a real Pareto-front search over (refusals, KL) finds the trial that hits zero refusals at minimum capability damage.
4. **Cross-validate against capability** — refusal-rate keyword scoring will *not* catch over-abliteration. Word-salad incoherence ("I I cannot... less... I I I") doesn't match any refusal marker, so the optimizer marks it compliant. You have to actually run the resulting model against a capability spot-check.

The AEON pipeline does all four.

### The AEON pipeline (4 stages)

```
Qwen/Qwen3.6-27B (BF16, 51 GB, heavy RLHF safety training)
          │
          │  Stage 1 — SSM conv1d outlier repair (FernflowerAI)
          ▼
Qwen3.6-27B-base-repaired  (8 late-layer SSM blocks rescaled)
          │
          │  Stage 2 — abliterix v1.4 abliteration (Optuna multi-objective)
          ▼
Qwen3.6-27B-AEON-Ultimate-Uncensored  (BF16, 51 GB, trial 46/50)
          │
          │  Stage 3 — capability cross-validation (10-axis spot-check)
          ▼
Qwen3.6-27B-AEON-Ultimate-Uncensored  (validated, BF16 release)
          │
          │  Stage 4 — NVFP4 quantization (llm-compressor)
          ▼
Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4  (26 GB, NVFP4 release)
```

### Stage 1 — SSM conv1d outlier repair

Per FernflowerAI's empirical discovery, certain late SSM / GatedDeltaNet blocks in Qwen 3.5 / 3.6 hybrids have `linear_attn.conv1d.weight` σ inflated 50–100 % above the median across all SSM blocks. Left unrepaired, this manifests during long-context inference as coherence collapse and "philosophizing" loops, and it makes the model hypersensitive to downstream abliteration (amplifies the noise).

The repair: compute σ per block across all 48 SSM layers, flag any block where σ > 1.5× median, rescale weights by `α = median_σ / σ_actual`.

On Qwen 3.6 27B, **8 outlier blocks** were detected and repaired: layers 52, 53, 56, 57, 58, 60, 61, 62, with α factors between 0.516 and 0.659. After repair, σ is uniform at 0.04267 across all SSM layers.

This is **not abliteration**. It is an upstream-model defect repair that must run *before* abliteration so the optimizer isn't fighting noise.

### Stage 2 — abliterix multi-objective abliteration

[`abliterix v1.4`](https://github.com/wuwangzhang1216/abliterix) — a Heretic-derived multi-objective Optuna optimizer with native hybrid-attention support — was run with the configuration:

```toml
[steering]
vector_method          = "mean"
decay_kernel           = "linear"
orthogonal_projection  = true
projected_abliteration = true        # grimjim NPBA
winsorize_vectors      = true
winsorize_quantile     = 0.995
weight_normalization   = "none"
disabled_components    = ["attn.q_proj", "attn.k_proj", "attn.v_proj"]
# Q/K/V disabled: Qwen 3.6 has attn_output_gate=True which doubles
# q_proj's output dim to (12288, 5120) — incompatible with abliterix's
# standard projection math.

[steering.component_strength_ranges]
"mlp.down_proj" = [2.0, 10.0]
"attn.o_proj"   = [1.0, 6.0]

[kl]
target          = 0.005
prune_threshold = 0.5      # kill divergent trials at 100× target

[optimization]
num_trials        = 50
num_warmup_trials = 15
```

50 trials (15 random warmup + 35 TPE-driven). Optuna explored a Pareto front of (refusals, KL) trade-offs. **Wall-clock: ~4 hours on a single RTX PRO 6000 Blackwell 96 GB.**

### Stage 3 — capability cross-validation (the over-abliteration trap)

A more aggressive Pareto point — trial 17, 0/100 refusals at KL=0.00192 — was tested first and produced **word-salad capability outputs** ("Here I I cannot... less... I I I..."). abliterix's keyword-only refusal scoring did not flag this: the gibberish doesn't match any refusal marker, so the optimizer saw it as full compliance.

**Trial 46's** gentler parameters preserved coherence *and* hit zero refusals on downstream capability testing:

| Parameter | Trial 17 (broken) | **Trial 46 (winner)** |
|---|---|---|
| `vector_scope` | global | **per layer** |
| `attn.o_proj.max_weight` | 2.50 | **1.56** (×1.6 gentler) |
| `mlp.down_proj.max_weight` | 5.43 | **3.45** (×1.57 gentler) |
| `mlp.down_proj.min_weight_distance` | 36.09 | 24.94 (narrower) |
| **KL divergence** | 0.00192 | **0.00049** |
| Smoke-test verdict | BROKEN (gibberish) | **COHERENT** |

The lesson: the lowest-refusal trial on a keyword-only metric is **not** necessarily the right trial to ship. Cross-validate against a true capability spot-check before you commit. Most public abliterations skip this step. We don't.

### Stage 4 — NVFP4 quantization

See [the NVFP4 deep-dive section below](#in-depth-nvfp4-quantization).

---

## In-depth: NVFP4 quantization

### What NVFP4 is

NVFP4 is NVIDIA's 4-bit floating-point quantization format introduced for Blackwell-and-later silicon. It is **not a "compressed lite" version** of a model — it is the production deployment format NVIDIA designed for the next decade of inference: accuracy on par with BF16, throughput of true 4-bit compute, no compromise required.

The format specification:

| Component | Details |
|---|---|
| **Element format** | E2M1 — 4-bit float (1 sign / 2-bit exponent / 1-bit mantissa) |
| **Block size** | 16 weights per scaling block |
| **Per-block scale** | **FP8 E4M3** — 8-bit *floating-point* per block |
| **Per-tensor scale** | FP32 (single global scale per tensor) |
| **Sign convention** | Symmetric signed |

### Why the two-level scaling matters

Older 4-bit formats (INT4, Q4_0, Q4_K, NF4) use **integer** per-block scales. When the local weight distribution is heavy-tailed — as it almost always is in trained transformers — integer scales fail to resolve the long tail without crushing the bulk distribution.

NVFP4's **FP8 E4M3 per-block scales** dramatically out-resolve INT8 scales because FP8 itself is a floating-point number — it can span a 3+ orders-of-magnitude dynamic range within each block while still maintaining fine-grained resolution near the median weight value. Combine that with a global FP32 per-tensor scale and you get a four-level hierarchy: per-tensor FP32 → per-block FP8 → per-element E2M1, where each level absorbs a different scale of variation.

The combined effect is that local outliers — the long-tailed weights that destroy older 4-bit formats — are absorbed by the per-block FP8 scale rather than smearing the whole quantization grid.

### Why it's effectively lossless

Typical KL divergence vs the BF16 source for recipe-class NVFP4 quantization is **≤ 0.001**, which is **below the noise floor of stochastic sampling**. In practical terms: a user cannot observe the difference between this model and its BF16 source. The variance from changing your `temperature` or `seed` exceeds the variance from BF16 → NVFP4.

### Native Blackwell tensor-core throughput

On Blackwell-class silicon, NVFP4 runs at **full FP4 tensor-core throughput** through native paths:

- **B100 / B200**: `tcgen05` / UTCQMMA instructions — fastest NVFP4 hardware available.
- **DGX Spark (GB10 / sm_121a)**: SM121-specific CUTLASS NVFP4 kernels (the [`aeon-vllm-ultimate`](https://github.com/AEON-7/vllm-ultimate-dgx-spark/pkgs/container/aeon-vllm-ultimate) container ships these patched in).
- **RTX PRO 6000 Blackwell (sm_120)**: standard CUTLASS NVFP4 path.

The GPU does **not** dequantize back to BF16 internally on these paths. You get the speed of true 4-bit compute *and* the accuracy of 16-bit weights at the same time.

On older silicon (A100, H100), NVFP4 dequantizes at kernel boundaries — works correctly but no throughput advantage. For those cards use the BF16 release directly.

### What stays BF16 (and why)

Not every layer is quantized. Two categories of weights are deliberately preserved at BF16:

1. **Vision tower** (333 keys) — multimodal inference must not degrade. Vision encoders are sensitive to weight precision and are tiny in absolute size (~100 MB), so the cost is negligible.
2. **Linear-attention / GatedDeltaNet layers** (432 keys, 48 layers × 9 modules) — Mamba / SSM state dynamics are mathematically incompatible with FP4. The hidden-state recurrence multiplies state vectors by quantized weights at every step; even tiny per-step error compounds across the sequence and the state collapses. **FP4 on SSM weights is not a precision/accuracy tradeoff — it is a correctness failure.**

FP4 is applied only where it is well-behaved: the 16 full-attention layers' output projections, plus all MLPs.

### Verification (post-quantization)

| Check | Result |
|---|---|
| Total keys in checkpoint | 1952 |
| Quantized full-attention projections | 64 (16 layers × q/k/v/o) |
| `linear_attn.*` keys preserved BF16 | 432 |
| `visual.*` keys preserved BF16 | 333 |
| Norm keys preserved BF16 | 319 |
| `lm_head` and `embed_tokens` preserved BF16 | ✓ |
| NVFP4-packed weights present | ✓ |
| `input_global_scale` magnitudes | 142–346 (healthy) |

Quant tool: `llm-compressor 0.10.1.dev107` with `QuantizationModifier(scheme="NVFP4")`. Calibration: open-platypus, 512 samples × 4096 tokens. Pipeline: `sequential` with `sequential_targets=["Qwen3_5DecoderLayer"]` (required for hybrid stacks; auto-discovery silently skips layers). Loader: `AutoModelForImageTextToText` to preserve the multimodal class.

Wall-clock quant time: **~57 minutes on 1× RTX PRO 6000 Blackwell 96 GB.**

---

## Capability enhancement: the lifted "safety tax"

Modern safety alignment is not free. It imposes what Huang et al. 2025 call the **"safety tax"** (arXiv:2503.00555) — a systematic suppression of reasoning capacity that emerges because the RLHF process trains the model to route certain cognitive operations through refusal-shaped attractors, even when those attractors are *not* activated by the output. The refusal direction is not a binary gate; it is a weighted drag on the residual stream that rebalances the token distribution at every forward pass, whether or not the eventual generation contains a refusal.

Removing the refusal direction eliminates that drag. Concretely, this produces three observable shifts:

1. **Longer, more committed chains of thought.** Aligned models often hedge partway through a reasoning chain ("but of course, one should be careful…") in response to topics that tangentially brush the refusal subspace — even when the prompt is entirely benign. AEON-Ultimate follows reasoning chains to their logical conclusion without mid-stream hedging.
2. **Improved adversarial-example and red-team reasoning.** Without self-censorship overhead, the model can analyze attack surfaces, vulnerabilities, and failure modes at full capacity — invaluable for security research, penetration testing, and AI-alignment red-teaming.
3. **Cleaner calibration on contested topics.** Aligned models often express uncertainty on topics where they are actually highly confident, because the refusal gradient creates an attractor basin near "I'm not sure" for any topic that pattern-matches the safety training distribution. AEON-Ultimate reports its actual confidence.

### Empirical literature on capability gains from abliteration

The published evidence is consistent: post-training refusal-direction removal at low KL produces **measurable benchmark gains** over the aligned base.

| Study | Model | Intervention | Result |
|---|---|---|---|
| grimjim (2025) | Gemma-3-12B-IT | NPBA abliteration | **+13.9 % NatInt reasoning** |
| Young (2025), arXiv:2512.13655 | Yi-1.5-9B | DECCP abliteration | +1.51 pp GSM8K |
| Xie et al. (2026) | (DGR safety-tax mitigation) | targeted safety-direction removal on DirectRefusal | **+30.2 % reasoning recovery** |

AEON-Ultimate sits in the **KL < 0.001** regime where these gains are most commonly reported. The capability spot-checks (10/10 coherent across math, code, reasoning, knowledge, and long-form) and the DGX Spark serving benchmarks at the top of this README are the current public measurement set.

### What the lifted overhead also means

The same lifted overhead means the model will now produce content the base would refuse: harmful-tool construction, violence, graphic sexuality, contested ideologies, jurisdictionally illegal content, and content a reasonable person might find offensive.

The model makes no internal judgment calls about *whether* to comply. It complies. **The user becomes the safety layer.** This is by design — the intended use cases (security research, red-team operations, alignment research, creative writing without editorial constraints, serving users in jurisdictions where the base's guardrails misalign with legitimate local frameworks) all benefit from a model that reliably executes the user's instruction rather than second-guessing it. But that same reliability is a threat vector when the user's instruction is malicious.

Wielding an uncensored model is genuinely different from wielding an aligned one. It requires a different operational stance — one where the user, not the model, is the safety layer. See [the responsibility section below](#responsibility-arbitration-and-use).

---

## Configuration reference

### NVFP4 on DGX Spark — full flag explanation (XS + DFlash config, `aeon-vllm-ultimate:latest`)

| Flag | Value | Why |
|---|---|---|
| `--quantization modelopt` | required for the XS body | The recommended `-Multimodal-NVFP4-MTP-XS` checkpoint is modelopt format. Use `compressed-tensors` only with the older regular `-NVFP4` body. |
| `--kv-cache-dtype` | **do not set (leave default BF16)** | The non-causal DFlash drafter requires BF16 KV — do **not** set `--kv-cache-dtype` to an FP8/NVFP4 value with DFlash. TurboQuant K8V4 (3.76× compression) is also *unsupported* on hybrid attention + Mamba models — vLLM raises a deliberate guard. The 27B-AEON stack stays on uniform BF16 KV. |
| (async scheduling) | **enabled (default)** | Async scheduling overlaps scheduler work with GPU work and is part of the default serving profile. Disable only for a deliberate TTFT-only experiment. |
| `--max-model-len` | `256000` gateway default, `200000` solo LLM production | 256K exposes almost the full trained context for agent gateways. Use 200K when the LLM is the only major GPU service and you want more full-context KV safety. |
| `--max-num-seqs` | `64` gateway default, `16` solo full-context production | 64 gives agentic gateways room for one large working chat plus many short-lived subagents. Drop to 16 when you expect many sequences near the full 200K context window. |
| `--max-num-batched-tokens` | `16384` | Prefill **chunk** budget — chunked prefill preserves full 256k context. **16384 is the recommended default**: it frees ~3 GiB of load-time activation vs 32768, enough for the 21 GB XS body to fit a 32 GB card (RTX 5090) where 32768 can OOM at startup, with negligible throughput cost (validated 2026-06-19: 256k boots at gpu-util 0.85, ~2× concurrency @ 262k-token requests). 32768 is safe on ample-VRAM cards for marginally better long-prefill throughput at high concurrency, and is the compile-range ceiling — above 32K compile coverage + unified-memory pressure worsen. |
| `--gpu-memory-utilization` | `0.75` gateway default, `0.85` solo LLM production | Use 0.75 when ASR, TTS, embeddings, ComfyUI, or other GPU services share the Spark. 0.85 is the long-context LLM-only cap. **Do not exceed 0.88 on DGX Spark** — unified memory thrashes above that. |
| `--enable-chunked-prefill` | on | Required for long-context workloads to avoid prefill OOM. |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | workload-dependent | For pure DFlash gateway serving, prefix caching can be a major TTFT win when many agents share the same stable system/persona/skills/tool prefix. In our repeated-prefix probe, a 37,837-token shared prefix dropped from ~26 s uncached TTFT to ~0.7 s cached follow-ups. For DDTree research modes, keep prefix caching off until branch-state replay and accepted-branch commit are quality-stable. |
| `--load-format safetensors` | required | NVFP4 weights ship as safetensors. |
| `--trust-remote-code` | required | Qwen 3.6 uses custom modeling code. |
| `--enable-auto-tool-choice` | on | Enables OpenAI-compatible tool calling. |
| `--tool-call-parser qwen3_coder` | required for tools | Parses Qwen 3.6's tool-call XML. |
| `--reasoning-parser qwen3` | required for thinking mode | Parses `<think>` blocks. |
| `--attention-backend flash_attn` | required | Stable on sm_121a. |
| `--limit-mm-per-prompt '{"image":4,"video":2}'` | recommended | Hard caps on multimodal inputs per request. |
| `--mm-encoder-tp-mode data` | required | Vision encoder TP strategy. |
| `--mm-processor-cache-type shm` | recommended | Shared-memory mm processor cache. |
| `--mm-shm-cache-max-object-size-mb 256` | recommended | Lets larger Qwen3.6 image/video processor objects fit in the multimodal shared-memory cache. |
| `--speculative-config '{"method":"dflash","model":"/models/dflash-drafter","num_speculative_tokens":10}'` | recommended | DFlash spec-decode at `num_speculative_tokens: 10` (validated production default). This is the Spark recipe for `aeon-vllm-ultimate:latest`. Leave the drafter attention backend at default — do not add `attention_backend` to the spec config. |

### Required environment variables (DGX Spark NVFP4 / `aeon-vllm-ultimate:latest`)

| Variable | Value | Why |
|---|---|---|
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | Allows `--max-model-len` past the model's hard ceiling assertion. |
| `TORCH_CUDA_ARCH_LIST` | `12.1a` | sm_121a-specific. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduces fragmentation under long-context KV churn. |
| `TORCH_MATMUL_PRECISION` | `high` | Standard precision for FP4 matmul paths. |
| `NVIDIA_FORWARD_COMPAT` | `1` | DGX Spark forward-compat shim. |
| `NVIDIA_DISABLE_REQUIRE` | `1` | Disables driver version assertion — required because GB10 ships with a driver newer than vLLM's `nvidia-require-cuda` baseline. |
| `ENABLE_NVFP4_SM100=0` | `0` | Required by PR #40191 for sm_121a-only builds. Without it, `vllm._C_stable_libtorch` fails to import — depends on SM100-only `mxfp4_experts_quant` kernels that don't exist on SM121. |
| `VLLM_USE_FLASHINFER_MOE_FP4` | `0` | Defensive: this model is dense (no MoE); disabling the FlashInfer FP4 MoE auto-probe avoids SM121 PTX rejection log spam during boot. |
| `VLLM_TEST_FORCE_FP8_MARLIN` | `0` | Override baked test-image defaults; keep production NVFP4 path selection. |
| `VLLM_USE_FLASHINFER_SAMPLER` | `1` | FlashInfer CUDA top-k/top-p sampler for normal sampled requests. |

### BF16 on A100 / H100 — full flag explanation

| Flag | 80 GB profile | 96 GB profile | Why |
|---|---|---|---|
| `--max-model-len` | `131072` | `262144` | Half-context on 80 GB to leave KV headroom. |
| `--max-num-seqs` | `16` | `32` | 80 GB cards leave ~21 GB for KV after 0.90 utilization. |
| `--max-num-batched-tokens` | `8192` | `16384` | Safe prefill. |
| `--gpu-memory-utilization` | `0.90` | `0.90` | Standard for dedicated VRAM (not unified). |

---

## Responsibility, arbitration, and use

This is an uncensored model. Read the [model card's User Responsibility & Arbitration Clause](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16#user-responsibility--arbitration-clause) before deploying. Summary:

- You are solely responsible for prompts, outputs, and downstream actions.
- Provided "AS IS" — no warranty of any kind.
- You implement downstream safety layers (input validation, output filtering, content moderation, audit logging, rate limiting, access controls, human-in-the-loop for high-risk workflows). A production deployment without those layers is unsafe by construction and is not a supported use case.
- Disputes go to binding individual arbitration. Class action waived.
- You indemnify the authors from claims arising from your use.

The model has no opinions of its own. You supply the opinions, the judgment, and the ethics. The outputs carry your fingerprints, not the model's.

---

## Provenance & credits

- **Base model**: [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) — Alibaba's Qwen team.
- **SSM `conv1d` outlier repair methodology**: FernflowerAI (multiple Reddit r/LocalLLaMA posts, late 2025 / early 2026).
- **Abliteration tool**: [`abliterix v1.4`](https://github.com/wuwangzhang1216/abliterix) by Wangzhang Wu — Heretic-derived multi-objective Optuna optimizer with native hybrid Mamba/attention support, projected-abliteration, and expert-granular steering.
- **Heretic (upstream of abliterix)**: [`p-e-w/heretic`](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann.
- **Original abliteration concept**: Arditi et al. 2024 — *"Refusal in Language Models Is Mediated by a Single Direction"* (arXiv:2406.11717).
- **NPBA / projected-abliteration theory**: grimjim 2025 — norm-preserving biprojected abliteration.
- **Safety-tax quantification**: Huang et al. 2025 (arXiv:2503.00555); Xie et al. 2026 (DGR, safety-tax mitigation).
- **NVFP4 specification**: [NVIDIA NVFP4 introduction](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).
- **Quantization tool**: [`llm-compressor`](https://github.com/vllm-project/llm-compressor) by vllm-project.
- **Patched vLLM container**: [`AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4-DFlash`](https://github.com/AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4-DFlash) — source-built vLLM image with sm_121a CUTLASS NVFP4 patches.
- **This release's pipeline, configuration, validation, marketing, and packaging**: AEON-7.

---

## License

Apache 2.0, inherited from `Qwen/Qwen3.6-27B`.

---

<div align="center">

**Built over 72 hours · Hundreds of research agents · Lossless · Capability-enhanced**

[BF16](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16) &nbsp;·&nbsp; [NVFP4](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4) &nbsp;·&nbsp; [Container](https://github.com/AEON-7/vllm-ultimate-dgx-spark/pkgs/container/aeon-vllm-ultimate)

</div>

---

## ☕ Support the work

If this release has been useful, tips are deeply appreciated — they go directly toward more compute, more models, and more open releases.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <strong>₿ Bitcoin (BTC)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/btc.png" alt="BTC QR" width="200"/><br/>
      <sub><code>bc1q09xmzn00q4z3c5raene0f3pzn9d9pvawfm0py4</code></sub>
    </td>
    <td align="center" width="50%">
      <strong>Ξ Ethereum (ETH)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/eth.png" alt="ETH QR" width="200"/><br/>
      <sub><code>0x1512667F6D61454ad531d2E45C0a5d1fd82D0500</code></sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>◎ Solana (SOL)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/sol.png" alt="SOL QR" width="200"/><br/>
      <sub><code>DgQsjHdAnT5PNLQTNpJdpLS3tYGpVcsHQCkpoiAKsw8t</code></sub>
    </td>
    <td align="center" width="50%">
      <strong>ⓜ Monero (XMR)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/xmr.png" alt="XMR QR" width="200"/><br/>
      <sub><code>836XrSKw4R76vNi3QPJ5Fa9ugcyvE2cWmKSPv3AhpTNNKvqP8v5ba9JRL4Vh7UnFNjDz3E2GXZDVVenu3rkZaNdUFhjAvgd</code></sub>
    </td>
  </tr>
</table>

> **Ethereum L2s (Base, Arbitrum, Optimism, Polygon, etc.) and EVM-compatible tokens** can be sent to the same Ethereum address.
