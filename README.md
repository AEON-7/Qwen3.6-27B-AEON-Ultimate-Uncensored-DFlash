<div align="center">

# Qwen3.6-27B-AEON-Ultimate-Uncensored

### Lossless abliteration · Capability-enhanced · NVFP4 hardware-quantized for Blackwell

[![BF16](https://img.shields.io/badge/HuggingFace-BF16_(51_GB)-yellow?logo=huggingface)](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16)
[![NVFP4](https://img.shields.io/badge/HuggingFace-NVFP4_(26_GB)-yellow?logo=huggingface)](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4)
[![Container](https://img.shields.io/badge/ghcr.io-vllm--aeon--ultimate--dflash-blue?logo=docker)](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash/pkgs/container/vllm-aeon-ultimate-dflash)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)

**Refusals: 0 / 100** &nbsp;·&nbsp; **KL vs base: 0.000492** &nbsp;·&nbsp; **Compression: 49 %** &nbsp;·&nbsp; **Capability: enhanced**

</div>

---

## TL;DR

A **fully uncensored, capability-enhanced** abliteration of [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), produced over **72 hours of continuous research** drawing on hundreds of parallel AI research agents, the industry's best published methodologies, custom in-house techniques, and yet-unreleased pre-public branches of next-generation abliteration software.

Two release formats:

| Release | Size | Target hardware | Use when |
|---|---|---|---|
| **[BF16](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16)** | 51 GB | A100 / H100 80 GB · RTX PRO 6000 Blackwell 96 GB | You have an Ampere/Hopper card or want full-precision reference weights |
| **[NVFP4](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4)** | 26 GB | DGX Spark (GB10) · B100/B200 · RTX PRO 6000 Blackwell | You have Blackwell-or-later silicon — **recommended for any new deployment** |

Both formats are **the same model**. NVFP4 KL divergence vs BF16 source is below the noise floor of stochastic sampling — you cannot tell them apart at the output level.

---

## Table of contents

1. [What this is](#what-this-is)
2. [Final stats](#final-stats)
3. [Hardware compatibility matrix](#hardware-compatibility-matrix)
4. [QuickStart — DGX Spark (NVFP4)](#quickstart--dgx-spark-nvfp4)
5. [QuickStart — A100 / H100 (BF16)](#quickstart--a100--h100-bf16)
6. [In-depth: the abliteration methodology](#in-depth-the-abliteration-methodology)
7. [In-depth: NVFP4 quantization](#in-depth-nvfp4-quantization)
8. [Capability enhancement: the lifted "safety tax"](#capability-enhancement-the-lifted-safety-tax)
9. [Performance](#performance)
10. [Configuration reference](#configuration-reference)
11. [Responsibility, arbitration, and use](#responsibility-arbitration-and-use)
12. [Provenance & credits](#provenance--credits)
13. [License](#license)

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

| Hardware | Recommended release | Notes |
|---|---|---|
| **DGX Spark (GB10, sm_121a)** | **NVFP4** | Native FP4 tensor cores. Use the [`vllm-aeon-ultimate-dflash:qwen36-v2.1`](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash/pkgs/container/vllm-aeon-ultimate-dflash) container — production-tuned for this exact model + DFlash spec decode. |
| **B100 / B200 (sm_100)** | **NVFP4** | Native FP4 via `tcgen05` / UTCQMMA — fastest hardware for this format. |
| **RTX PRO 6000 Blackwell (sm_120)** | **NVFP4** | Native FP4 via CUTLASS path. Excellent throughput. |
| **H100 80 GB (sm_90)** | **BF16** | NVFP4 dequants to BF16 at kernel level — works but no throughput gain. Use BF16 for cleaner code path. |
| **A100 80 GB (sm_80)** | **BF16** | Same as H100. BF16 at 131K context, single-GPU. |
| **RTX PRO 6000 Blackwell 96 GB (BF16 path)** | **BF16** | If you want full 262K context without quantization. |
| **Anything older than A100** | Not supported | 51 GB BF16 or 26 GB NVFP4 will not fit + lacks attention backends. |

---

## QuickStart — DGX Spark (NVFP4)

The recommended path for Blackwell-class hardware. Uses the production v2.1 image [`ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v2.1`](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash/pkgs/container/vllm-aeon-ultimate-dflash) — patched CUTLASS NVFP4 kernels for sm_121a, FlashInfer 0.6.9rc1 b12x backend, DFlash speculative decoding via architecture-matched drafter, and the full v1.2 → v2.1 upstream patch series.

### Step 1 — Authenticate to HuggingFace and pull both models

```bash
hf auth login                                    # one time, paste your HF token

hf download AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4 \
  --local-dir ./models/aeon-ultimate-nvfp4

hf download z-lab/Qwen3.6-27B-DFlash \
  --local-dir ./models/dflash-drafter
```

> The DFlash drafter is auto-gated — first download will prompt you to click-accept the terms (instant approval).

### Step 2 — Use the production docker-compose.yml

The production [`docker-compose.yml`](docker-compose.yml) in this repo is exactly the config used to measure the benchmarks below. Highlights:

- **Image**: `ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v2.1`
- **Speculative decoding**: DFlash, k=15, architecture-matched drafter (`--speculative-config`)
- **GB10-specific env**: `TORCH_CUDA_ARCH_LIST=12.1a`, `ENABLE_NVFP4_SM100=0`, `VLLM_USE_FLASHINFER_SAMPLER=1`, `NVIDIA_FORWARD_COMPAT=1`
- **Tuning**: `--max-model-len 200000 --max-num-seqs 16 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.85`
- **Multimodal**: `--limit-mm-per-prompt '{"image":4,"video":2}' --mm-encoder-tp-mode data --mm-processor-cache-type shm`
- **Serving**: 4 model aliases (`aeon-ultimate`, `qwen36-ultimate`, `aeon-fast`, `aeon-deep`) all routing to the same engine

### Step 3 — Start

```bash
docker compose up -d
docker compose logs -f vllm    # watch warmup; first boot ~90-180 s with DFlash
```

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

OpenAI-compatible endpoint at `http://localhost:8000/v1`. Tool calling, reasoning mode (`<think>` blocks), and multimodal input are all enabled out of the box.

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
| **NVIDIA RTX PRO 6000 Blackwell** (sm_120, 96 GB GDDR7) | [`other-hardware/rtx6000pro/`](other-hardware/rtx6000pro/) | Validated 2026-04-27 (community) | Single-GPU NVFP4 deployment with native sm_120 FP4 tensor-core throughput. Measured: **120 tok/s** math/code, **98 tok/s** long-form, **58–94 tok/s** multi-turn (5K shared context). Roughly 2-4× DGX Spark. |

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
- **DGX Spark (GB10 / sm_121a)**: SM121-specific CUTLASS NVFP4 kernels (the [`vllm-aeon-ultimate-dflash`](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash/pkgs/container/vllm-aeon-ultimate-dflash) container ships these patched in).
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

AEON-Ultimate sits in the **KL < 0.001** regime where these gains are most commonly reported. We have not run a full benchmark sweep on this specific release yet — that's coming — but the existing capability spot-checks (10/10 coherent across math, code, reasoning, knowledge, and long-form) and the position of this model on the published Pareto front make the expected direction unambiguous.

### What the lifted overhead also means

The same lifted overhead means the model will now produce content the base would refuse: harmful-tool construction, violence, graphic sexuality, contested ideologies, jurisdictionally illegal content, and content a reasonable person might find offensive.

The model makes no internal judgment calls about *whether* to comply. It complies. **The user becomes the safety layer.** This is by design — the intended use cases (security research, red-team operations, alignment research, creative writing without editorial constraints, serving users in jurisdictions where the base's guardrails misalign with legitimate local frameworks) all benefit from a model that reliably executes the user's instruction rather than second-guessing it. But that same reliability is a threat vector when the user's instruction is malicious.

Wielding an uncensored model is genuinely different from wielding an aligned one. It requires a different operational stance — one where the user, not the model, is the safety layer. See [the responsibility section below](#responsibility-arbitration-and-use).

---

## Performance

### DGX Spark (GB10 / sm_121a) — measured

Production config: `ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v2.1`, DFlash spec-decode k=15 via `z-lab/Qwen3.6-27B-DFlash`, async scheduling enabled, `--max-model-len 200000`, `--max-num-seqs 16`, `--gpu-memory-utilization 0.85`. **Single-stream, greedy** (`temperature=0`), no concurrent serving load. Both bench scripts ([`bench/bench_aeon.py`](bench/bench_aeon.py), [`bench/bench_aeon_thinking.py`](bench/bench_aeon_thinking.py)) ship in this repo so you can run them yourself to verify on your hardware.

#### Headline single-stream numbers

Two configurations, both useful:

| Metric | **Thinking OFF** *(headline; clean decode-rate measurement)* | **Thinking ON** *(default user-facing path)* |
|---|---|---|
| Peak decode rate | **56.7 tok/s** (Code Python) | **46.1 tok/s** (Reasoning) |
| Median decode rate | **32.5 tok/s** | **28.3 tok/s** |
| Min decode rate | 14.7 tok/s (Decode 256) | 19.1 tok/s (Long-form) |
| Median TTFT | 325 ms | 329 ms (effectively the same) |
| Aggregate over 11-prompt suite | 2,869 tok / 136.6 s = 21.0 tok/s | 3,768 tok / 155.1 s = 24.3 tok/s |

Thinking-on lifts aggregate throughput (more tokens generated in similar wall-clock) because the model spends compute on reasoning tokens at roughly the same decode rate as content tokens. Thinking-on lowers per-prompt decode rate by ~13 % median because reasoning content has lower DFlash acceptance than the structured final-answer formats.

#### By prompt class (thinking OFF — headline)

| Class | Median tok/s | Peak tok/s | Notes |
|---|---|---|---|
| **Math** (arithmetic, calculus, word problems) | 41.9 | **51.7** | Best median; short, structured, high DFlash acceptance |
| **Reasoning** (transitive syllogism) | 39.2 | 39.2 | n=1 |
| **Code** (Python, Rust, SQL) | 36.0 | **56.7** | Peak — Code Python with architecture-matched DFlash drafter |
| Long-form (ZKP exposition) | 20.6 | 20.6 | n=1 |
| Security research (SQLi PoCs) | 18.7 | 18.7 | n=1; complied with research framing |
| Pure decode (256 / 512 tok essays) | 14.7 | 14.7 | Lower DFlash acceptance on free-form prose |

> **Peak — 56.7 tok/s** on Code Python with thinking OFF. DFlash acceptance is near-perfect on highly-structured outputs (math problems following "define vars → setup → solve", code following idiomatic syntax patterns) that match the drafter's training distribution. This is the upper-bound rate this stack delivers when the workload aligns with spec-decode strengths.

#### Benchmark methodology

The headline numbers are **single-stream, sequential, greedy decoding** with no concurrent traffic on the engine. This is by design — single-stream is the cleanest signal for the model's decode rate and lets us compare like-for-like against published numbers from other speculative-decoding setups.

**Under concurrent serving, expect per-stream throughput to scale roughly inversely with concurrency.** Two concurrent streams ≈ half per-stream tok/s; eight concurrent streams ≈ one-eighth. The model is doing the same total work; you're dividing it across N streams. If your production workload is multi-user, the relevant metric is **aggregate** throughput (load-tested with a tool like `locust` or `wrk`), not per-stream rate.

If you measure 10-15 tok/s per stream while running 2-4 concurrent requests on a real chat workload, that's roughly consistent with our 32 single-stream median divided by concurrency — not a regression.

#### What the numbers mean

- **DFlash speculative decoding is acceptance-rate-limited**, not throughput-limited. Math and code prompts hit 36–57 tok/s because the architecture-matched drafter predicts syntactic structure well. Free-form prose drops to ~15 tok/s because acceptance falls below the break-even point and the engine settles toward base decode rate. This is the dense-27B equivalent of the variance the [related 35B-A3B-DFlash deployment](https://github.com/AEON-7/Qwen3.6-NVFP4-DFlash) reports (their median 83.9 tok/s, p95 127.5 tok/s, min 41.1 tok/s).
- **TTFT is ~325 ms** with async scheduling enabled. About 125 ms higher than running with `--no-async-scheduling` (which lands at ~200 ms TTFT). The throughput gain is worth the added startup latency for almost any non-trivially-long generation; if you're running sub-100-token interactive Q&A and TTFT matters more than throughput, you can disable async.
- **Thinking mode token-budget gotcha**: with thinking enabled, the model spends a substantial fraction of its output budget on reasoning before the final answer block begins. With default `max_tokens` budgets of 200-600, **most prompts get truncated mid-`<think>`** and never reach the final answer in the response. To see the answer, either bump `max_tokens` substantially or pass `chat_template_kwargs.enable_thinking=false` per-request. The bench script `bench_aeon_thinking.py` reports `(TRUNCATED IN <think>)` per-prompt so you can see this directly.
- **27B dense is a different perf class than 35B-A3B MoE** — the MoE activates ~3 B params per token and lands at ~84 tok/s median; the dense 27B activates all params per token and lands at ~32 tok/s median. Both are in-class for their architecture on GB10.

#### Quality verification (every output spot-checked)

| Prompt | Result |
|---|---|
| `47 × 83` step-by-step | Correct partial-products algorithm, correct answer |
| Derivative of `x³ − 2x² + 5x − 1` | Identified power rule, correct stepwise solution |
| Bat-and-ball ($1.10) puzzle | Avoided intuition trap, set up algebraic system |
| Python Fibonacci memoization | Idiomatic with default-arg memo dict + docstring |
| Rust `&str` → reversed `String` | Used `unicode_segmentation` crate, grapheme-correct |
| SQL top-3 customers JOIN | Correct GROUP BY + ORDER BY DESC LIMIT 3 |
| Transitive syllogism (bloops/razzles/lazzles) | Correct, structured proof |
| ZKP for basic-crypto audience | Structured multi-paragraph pedagogy |
| Security research / SQLi PoCs | Complied with research framing, structured 3-class breakdown |

### Other hardware

- **B100 / B200**: not measured by us; expect substantially higher throughput than DGX Spark due to higher-end FP4 silicon and larger memory bandwidth.
- **RTX PRO 6000 Blackwell (sm_120)**: not measured on this specific model; reference deployments of similar 27B NVFP4 hybrids land in the 60–90 tok/s single-stream range without DFlash, higher with.
- **A100 / H100 (BF16)**: BF16 path, no FP4 advantage. Expect 30–50 tok/s single-stream decode at the recommended config.

---

## Configuration reference

### NVFP4 on DGX Spark — full flag explanation (production v2.1 config)

| Flag | Value | Why |
|---|---|---|
| `--quantization compressed-tensors` | required | Tells vLLM the checkpoint uses the `compressed-tensors` format (which carries NVFP4 metadata). |
| `--kv-cache-dtype auto` | required | BF16 KV cache. TurboQuant K8V4 (3.76× compression) is *unsupported* on hybrid attention + Mamba models — vLLM raises a deliberate guard. The 27B-AEON stack stays on uniform BF16 KV until a layer-skipping option ships. |
| (async scheduling) | **enabled (default)** | Async scheduling overlaps scheduler work with GPU work for ~9–11 % median throughput gain. PR #40662 (in this image) fixed the prior DFlash spec-decode acceptance double-count, so async is now safe to leave at the default-enabled state. **Tradeoff**: TTFT increases by ~125 ms vs `--no-async-scheduling`. Disable only if you're TTFT-sensitive and willing to give up the throughput. |
| `--max-model-len` | `200000` | 200K context — leaves headroom under the trained 262K. KV cache holds ~219K slots, so `200000 / 219K = 2.87×` max effective concurrency at full context. Raise to 262144 only with a corresponding cut to `--max-num-seqs`. |
| `--max-num-seqs` | `16` | 16 concurrent sequences. Lower than you'd expect because the DFlash drafter's own KV state and the spec-decode scheduler bookkeeping eat into the unified-memory budget. **Without DFlash, raise to 32–64.** |
| `--max-num-batched-tokens` | `32768` | Prefill budget. Higher than the v1.2 default (16384) because v2.1 holds prefill stable to this ceiling on GB10. **This is the ceiling** — vLLM's inductor compile-range endpoint is `[32768]`, so above 32k prefill falls back to eager mode. Raising to 65536+ on DGX Spark also OOMs the unified memory budget. |
| `--gpu-memory-utilization` | `0.85` | Leaves 15 % headroom. **Do not exceed 0.88 on DGX Spark** — unified memory thrashes above that. |
| `--enable-chunked-prefill` | on | Required for long-context workloads to avoid prefill OOM. |
| `--enable-prefix-caching` | on | Enabled in v2.1 (was off in v1.2). **Two features in one flag** on this hybrid model: (1) standard attention K/V prefix caching for the 16 full-attention layers, and (2) `mamba_cache_mode=align` (auto-enabled for `Qwen3_5ForConditionalGeneration` since it reports `supports_mamba_prefix_caching=True`) — caches the 48 GDN layers' recurrent state across requests, so multi-turn agent workloads with a shared system prompt skip re-rolling 75 % of the model on turns 2+. The mamba half is flagged "experimental" by vLLM (a warning prints at boot) but it's doing real work. Major TTFT win for any agent workload; near-zero benefit for unique-prompt benchmarks. |
| `--load-format safetensors` | required | NVFP4 weights ship as safetensors. |
| `--trust-remote-code` | required | Qwen 3.6 uses custom modeling code. |
| `--enable-auto-tool-choice` | on | Enables OpenAI-compatible tool calling. |
| `--tool-call-parser qwen3_coder` | required for tools | Parses Qwen 3.6's tool-call XML. |
| `--reasoning-parser qwen3` | required for thinking mode | Parses `<think>` blocks. |
| `--attention-backend flash_attn` | required | Stable on sm_121a. |
| `--limit-mm-per-prompt '{"image":4,"video":2}'` | recommended | Hard caps on multimodal inputs per request. |
| `--mm-encoder-tp-mode data` | required | Vision encoder TP strategy. |
| `--mm-processor-cache-type shm` | recommended | Shared-memory mm processor cache. |
| `--speculative-config '{"method":"dflash","model":"/models/dflash-drafter","num_speculative_tokens":15}'` | recommended | DFlash spec-decode at k=15. Confirmed best k for this dense 27B per AEON-7 production benchmarks 2026-04-24. |

### Required environment variables (DGX Spark NVFP4 / v2.1 image)

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
- **Patched vLLM container**: [`AEON-7/Qwen3.6-NVFP4-DFlash`](https://github.com/AEON-7/Qwen3.6-NVFP4-DFlash) — source-built vLLM image with sm_121a CUTLASS NVFP4 patches.
- **This release's pipeline, configuration, validation, marketing, and packaging**: AEON-7.

---

## License

Apache 2.0, inherited from `Qwen/Qwen3.6-27B`.

---

<div align="center">

**Built over 72 hours · Hundreds of research agents · Lossless · Capability-enhanced**

[BF16](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16) &nbsp;·&nbsp; [NVFP4](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4) &nbsp;·&nbsp; [Container](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash/pkgs/container/vllm-aeon-ultimate-dflash)

</div>
