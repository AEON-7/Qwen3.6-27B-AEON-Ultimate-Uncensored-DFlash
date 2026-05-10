# AGENTS.md — Operator's Manual for AI Agents

**Read this file before changing anything in this repo or its container.**

You are an AI coding agent that has been asked to work with this repository, its container image, or the model it serves. You have probably been trained on, or recently web-searched, documentation that is out-of-date for this specific stack. The Blackwell / sm_121a / GB10 ecosystem evolved very fast in 2025–2026, and most public stack-overflow answers, blog posts, NVIDIA Developer Forum threads, and even some vLLM issues are **stale**.

This file is the authoritative source. If a piece of public documentation contradicts something below, **trust this file** for this stack.

---

## ⚠️ Hardware scope: this file is for DGX Spark / GB10 / sm_121a

**Every flag, env var, container reference, and "DO NOT UNDO" rule in this file targets the DGX Spark (NVIDIA GB10, sm_121a, 128 GB unified memory) deployment.** That is the AEON-7 team's primary, measured-and-validated hardware platform. The container image, the v4 DFlash sliding-attention patch, the env var set, the `--gpu-memory-utilization` profile split, the `--max-num-seqs` profile split, and the `ENABLE_NVFP4_SM100=0` build guard — all of it is GB10-specific.

**If you are operating on different hardware, the rules in this file do NOT directly apply.** Specifically:

| You're on | Recipe location | Why DGX Spark rules don't apply |
|---|---|---|
| **NVIDIA RTX PRO 6000 Blackwell** (sm_120) | [`other-hardware/rtx6000pro/`](other-hardware/rtx6000pro/) | Different SM (sm_120 vs sm_121a) → different kernels. Dedicated VRAM (not unified) → no 0.88 ceiling. Higher memory bandwidth → more concurrency budget. Uses **stock `vllm/vllm-openai:v0.20.1`**, not the AEON-7 patched container. |
| **A100 / H100** (BF16 path) | [`docker-compose.bf16.yml`](docker-compose.bf16.yml) at repo root | No NVFP4 hardware support — runs the BF16 release. Vanilla `vllm/vllm-openai`, no DFlash drafter, different memory budget. |
| **B100 / B200** (sm_100) | Not in this repo yet | sm_100 native NVFP4 via `tcgen05`/UTCQMMA — different code path than sm_121a. Stock vLLM should work; recipe contributions welcome. |

If your task is on RTX PRO 6000, **do not apply this file's flags wholesale** — read `other-hardware/rtx6000pro/README.md` first; many of this file's rules invert (e.g., "don't push gpu-memory-utilization above 0.88" is a unified-memory rule that does NOT apply on dedicated VRAM). The per-hardware folder explains every difference.

---

## TL;DR for agents (60 seconds)

| Thing | Value | Don't second-guess |
|---|---|---|
| Container image | `ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v4` (current production, 2026-05-10) | Don't substitute stock `vllm/vllm-openai` for production — stock is useful as a transparency baseline, but it has no DFlash sliding-attention overlay and no packaged Spark defaults. `:latest` now points at v4; pin explicitly to `:qwen36-v4` for reproducibility. |
| Hardware target | DGX Spark (NVIDIA GB10, sm_121a, 128 GB unified) | Don't apply Hopper / Ada tuning advice |
| **Recommended body** | **`AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS`** (modelopt format, ~21 GB) — **winning Spark config** | Alternate body: `AEON-7/.../-NVFP4` (compressed-tensors, ~26 GB, simpler but slower). The v4 XS+DFlash path averages **37.56 tok/s c=1** across six natural prompt categories vs **10.49 tok/s** for the dirty stock eager baseline. |
| Quantization | NVFP4 via patched CUTLASS path | **Don't force Marlin** — the patched CUTLASS path is faster than Marlin on this hardware. Use `--quantization modelopt` with the XS body, `--quantization compressed-tensors` with the regular `-NVFP4` body. |
| Spec decode | **DFlash k=15** via `z-lab/Qwen3.6-27B-DFlash` drafter | Don't disable; don't switch to ngram/EAGLE. **DO NOT use `--speculative-config '{"method":"qwen3_5_mtp",...}'` on Spark** unless you are deliberately running an ablation. |
| GPU memory util cap | **0.75 gateway default**, **0.85 LLM-only production**, never above 0.88 | Unified memory thrashes above 0.88 — this is not a typo |
| Max model len | `256000` gateway default, `200000` LLM-only production | Set with `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` |
| Max num seqs | **64 gateway default**, **16 LLM-only full-context production** | 64 supports one large working chat plus many short subagents; 16 is safer when many sequences approach full context |
| Reasoning mode | enabled by default; pass `chat_template_kwargs.enable_thinking=false` to disable per request | Don't disable thinking globally. It is a core capability of this model and the v4 stack parses reasoning cleanly. |

The complete production configs are in [`docker-compose.yml`](docker-compose.yml) (regular `-NVFP4` body, simpler legacy stack) and [`docker-compose.spark-xs.yml`](docker-compose.spark-xs.yml) (XS body, current winning recipe). The Spark XS compose pins the `qwen36-v4` image. The compose files are **the source of truth**. Don't deviate without reading the rationale below.

---

## DO NOT UNDO — common stale-documentation traps

These are flags / settings / decisions that AI agents and human operators **repeatedly try to "fix"** based on documentation that's out-of-date for this stack. Don't.

### 1. Don't force the Marlin NVFP4 backend

```
# WRONG — looks reasonable based on 2024 forum advice for stock vLLM on Blackwell:
VLLM_NVFP4_GEMM_BACKEND=marlin
VLLM_TEST_FORCE_FP8_MARLIN=1
```

**Why this is wrong:** The advice "use Marlin on SM121 because CUTLASS NVFP4 doesn't work" was **true for stock vLLM builds before our patches**. Our `vllm-aeon-ultimate-dflash` container ships with the patched CUTLASS NVFP4 kernels (PR #40191 + the `vllm-spark-omni-q36` patch series). On this image, **patched CUTLASS is faster than Marlin**. Forcing Marlin is a measurable regression.

If you see a forum post or a stack-overflow answer telling you to set `VLLM_NVFP4_GEMM_BACKEND=marlin` on DGX Spark / GB10 / sm_121a, the post is from before the CUTLASS patches landed. Ignore it.

### 2. Don't set `ENABLE_NVFP4_SM100=1`

```
# WRONG — agents see "SM100 is close to SM121, this should be on":
ENABLE_NVFP4_SM100=1
```

**Why this is wrong:** SM100 (B100/B200) and SM121 (GB10) require **different kernels** for NVFP4 GEMM. The `mxfp4_experts_quant` symbols are SM100-only. Setting `ENABLE_NVFP4_SM100=1` on a SM121 image causes `vllm._C_stable_libtorch.abi3.so` to fail to import — vLLM won't boot. **PR #40191** added this guard specifically because the import failure was hard to debug.

The correct setting on this image is `ENABLE_NVFP4_SM100=0`. It is in the compose file. Leave it alone.

### 3. Don't enable `VLLM_USE_FLASHINFER_MOE_FP4`

```
# WRONG — agents see "Blackwell has FlashInfer FP4 MoE support":
VLLM_USE_FLASHINFER_MOE_FP4=1
```

**Why this is wrong:** This model (Qwen3.6-27B-AEON-Ultimate) is **dense**, not MoE. The FlashInfer FP4 MoE auto-probe runs at boot regardless of model architecture. On SM121 the probe fails (PTX rejection) and produces ~50 lines of noisy log spam. Setting `VLLM_USE_FLASHINFER_MOE_FP4=0` short-circuits the probe.

If you're confused because you read this is a "good Blackwell setting" — that advice applies to **MoE models**. For dense models on SM121, disable it.

### 4. Don't push `--gpu-memory-utilization` above 0.88

```
# WRONG — vLLM docs commonly recommend 0.90 or 0.95:
--gpu-memory-utilization 0.95
```

**Why this is wrong:** DGX Spark / GB10 has **unified memory** between CPU and GPU (128 GB shared). vLLM's `--gpu-memory-utilization` calculation assumes dedicated VRAM and over-allocates on unified architectures. Above 0.88, the system enters memory thrashing — KV cache pages get evicted to "CPU memory" which is the same physical memory, the OS swap subsystem starts paging, and throughput collapses to near-zero.

Production setting: **0.75** when ASR/TTS/embeddings or other GPU sidecars share the Spark, **0.85** when the LLM is the only major GPU service, and 0.88 as the absolute ceiling. Anything higher is a regression even though it looks like more headroom.

### 5. Don't add Speculators v0.3.x

```
# WRONG — looks like an obvious dependency upgrade:
pip install speculators==0.3.1
```

**Why this is wrong:** external `speculators` packages are not required for the v4 path and can change imports in ways this image does not need. vLLM's native `--speculative-config '{"method":"dflash",...}'` already covers the DFlash path.

Don't add the package at runtime. Rebuild the image only if a future release has a concrete feature we intentionally adopt.

### 6. Don't enable TurboQuant K8V4 KV compression

```
# WRONG — looks like an obvious 3.76× memory win:
--kv-cache-dtype turboquant_k8v4
```

**Why this is wrong:** TurboQuant K8V4 has a **deliberate guard** in the vLLM integration code (`NotImplementedError: TurboQuant KV cache is not supported for hybrid (attention + Mamba) models`) for stacks that mix full-attention and linear-attention/GDN/Mamba layers. Qwen3.6-27B is exactly that kind of hybrid (16 full + 48 GDN). Even the AEON-7/turboquant CUDA-graph-safe fork doesn't bypass this — the guard is enforced separately from kernel correctness.

**Tracking:** vLLM PR #39931 (TurboQuant hybrid models) is open and will eventually relax this guard for the full-attention layers only. Until that lands, leave `--kv-cache-dtype auto` (BF16). When #39931 merges, the option will be `--kv-cache-dtype-skip-layers` applied to the 16 full-attention layers only.

### 7. Don't switch `--attention-backend` from `flash_attn`

`--attention-backend xformers` and `--attention-backend triton` will boot but produce **silent quality degradation** on Qwen3.6's hybrid GDN layers. The flash_attn backend is the only one validated for this model on this image. Don't change it.

### 8. Don't blindly bump `--max-num-seqs` past the documented profile

```
# WRONG for full-context LLM-only serving — looks "wasteful" of the 128 GB unified memory:
--max-num-seqs 128
```

**Why this is wrong:** The DFlash drafter has its **own KV state** (~1 GB of weights + per-sequence draft KV) and the spec-decode scheduler bookkeeping eats further into the budget. Use `--max-num-seqs 64` for the gateway profile (`--max-model-len 256000 --gpu-memory-utilization 0.75`) when most agents are short-lived and side services need headroom. Use `--max-num-seqs 16` for LLM-only full-context production (`--max-model-len 200000 --gpu-memory-utilization 0.85`) when many sequences can approach the full window.

Without DFlash spec decode (i.e., if you removed `--speculative-config`), you can experiment higher, but that is a different deployment class. With DFlash on, stay inside the documented profile.

### 9. Don't downgrade the image tag

```
# WRONG — tag confusion:
image: ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:v1.2
image: ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2     # different image; missing patches
```

The current production tag is **`qwen36-v4`** (built 2026-05-10; latest community vLLM nightly `0.20.2rc1.dev166+gf6490a284` + FlashInfer 0.6.11 + DFlash sliding-attention overlay). The `qwen36-v3` and `qwen36-v2.1` tags remain pullable for rollback. `:latest` now points at v4. Agents should still pin explicitly to `:qwen36-v4` for reproducibility — `:latest` will move forward whenever a new image ships. v1.x predecessor tags lacked PR #40191 and produced import failures on sm_121a-only builds.

### 10. Don't `pip install` into the container

```
# WRONG — agents try to "fix" missing packages this way:
docker exec vllm-aeon-ultimate-v2 pip install some-package
```

**Why this is wrong:** The container's vLLM is **patched at source level**. Any `pip install` that pulls a vLLM, FlashInfer, or torch upgrade will overwrite the patched binaries with stock ones, breaking sm_121a NVFP4 support. If a package is missing, the correct fix is to rebuild the image with that package added — not to install at runtime.

---

## Required environment variables (with rationale)

These all appear in [`docker-compose.yml`](docker-compose.yml). Each line below is annotated with **what it does** + **what stale advice you may have seen contradicting it**.

| Variable | Value | Why | Common stale advice to ignore |
|---|---|---|---|
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | Bypasses vLLM's hard `max_model_len` ceiling assertion. The model's trained context is 262K but our budget supports 200K under DFlash. | "vLLM enforces max_model_len strictly; you can't bypass it" — outdated; this flag was added in 0.18+ |
| `TORCH_CUDA_ARCH_LIST` | `12.1a` | sm_121a-specific compilation target for the GB10 chip. | "Use `12.0+PTX` for forward-compat" — works for some kernels but not for the patched CUTLASS NVFP4 kernels which are sm_121a-specific |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduces fragmentation under long-context KV churn. Required for stable 200K context operation. | None recent — this is established |
| `TORCH_MATMUL_PRECISION` | `high` | Standard precision tier for FP4 matmul paths. | None |
| `NVIDIA_FORWARD_COMPAT` | `1` | DGX Spark forward-compat shim — required because GB10 ships with a CUDA driver newer than vLLM's compiled `nvidia-require-cuda` baseline. | "Don't set forward-compat flags" — only true for stock CUDA toolchains, not our patched build |
| `NVIDIA_DISABLE_REQUIRE` | `1` | Disables driver version assertion. Pairs with the above. | "Required version checks are important" — not for this image; the assertion is a mismatch with the actual driver, not a real incompatibility |
| `ENABLE_NVFP4_SM100` | `0` | **Required by PR #40191** — without it, `vllm._C_stable_libtorch.abi3.so` import fails. See "Don't undo #2" above. | "SM100 is close to SM121" — wrong; different kernels |
| `VLLM_USE_FLASHINFER_MOE_FP4` | `0` | Disables FlashInfer FP4 MoE auto-probe. Our model is dense. See "Don't undo #3". | "Enable for Blackwell perf" — only for MoE models |
| `VLLM_TEST_FORCE_FP8_MARLIN` | `0` | Override baked test-image defaults; keep production NVFP4 path selection. See "Don't undo #1". | "Marlin is recommended on SM121" — pre-CUTLASS-patch advice |
| `VLLM_USE_FLASHINFER_SAMPLER` | `1` | Use FlashInfer's CUDA top-k/top-p sampler for normal sampled requests. Materially faster than the PyTorch fallback. | None |

---

## Required vLLM serve flags (with rationale)

| Flag | Value | Why | Common stale advice to ignore |
|---|---|---|---|
| `--quantization modelopt` | required for the XS body | Tells vLLM the recommended XS checkpoint uses modelopt NVFP4 metadata. Use `compressed-tensors` only with the older regular `-NVFP4` body. | "Use `--quantization fp4` directly" — wrong; FP4 is encoded inside checkpoint metadata |
| `--kv-cache-dtype auto` | BF16 | TurboQuant K8V4 is unsupported on hybrid models. See "Don't undo #6". | "Use fp8 KV cache for memory savings" — works but reduces acceptance length under DFlash |
| (async scheduling) | **enabled (default)** | Async scheduling overlaps scheduler work with GPU work and is part of the v4 profile. | "Add `--no-async-scheduling` by default" — don't do this unless you are running a targeted TTFT-only experiment. |
| `--max-model-len` | `256000` gateway, `200000` LLM-only production | 256K exposes nearly the full trained context for agent gateways. 200K is the safer solo-LLM profile when many sequences may be long. | "Always use one universal max len" — wrong on unified memory; choose the profile |
| `--max-num-seqs` | `64` gateway, `16` LLM-only production | DFlash drafter budget plus side-service headroom. See "Don't undo #8". | "More concurrent seqs = more throughput" — true only until queueing and unified-memory pressure dominate |
| `--max-num-batched-tokens` | `32768` | Prefill budget and practical Spark ceiling for this stack. | "Stock vLLM uses 65536" — OOMs or falls off the optimized path on GB10 unified memory under concurrent long-context |
| `--gpu-memory-utilization` | `0.75` gateway, `0.85` LLM-only production | Unified memory cap. See "Don't undo #4". | "0.95 is the recommended default" — for dedicated VRAM only |
| `--enable-chunked-prefill` | flag | Required for long-context workloads to avoid prefill OOM. | None |
| `--enable-prefix-caching` | flag | Required for real agent workloads. It enables normal attention prefix caching plus Mamba/GDN align-cache behavior for shared-prefix multi-turn sessions. | "Disable prefix caching because benchmarks did" — wrong; the benchmark profile disables it only to isolate unique-prompt decode behavior |
| `--load-format safetensors` | required | NVFP4 weights ship as safetensors. | None |
| `--trust-remote-code` | required | Qwen 3.6 uses custom modeling code. | None |
| `--enable-auto-tool-choice` | flag | Enables OpenAI-compatible tool calling. | None |
| `--tool-call-parser qwen3_coder` | required for tools | Parses Qwen 3.6's tool-call XML. | "Use the generic `hermes` parser" — wrong; Qwen 3.6 has its own format |
| `--reasoning-parser qwen3` | required for thinking mode | Parses `<think>` blocks. | "Use the `mistral` reasoning parser" — wrong format |
| `--attention-backend flash_attn` | required | Stable on sm_121a + hybrid GDN. See "Don't undo #7". | "Triton backend is more flexible" — produces silent quality issues on this model |
| `--limit-mm-per-prompt '{"image":4,"video":2}'` | recommended | Hard caps on multimodal inputs per request. Prevents pathological MM overload. | None |
| `--mm-encoder-tp-mode data` | required | Vision encoder TP strategy. | "Use `weight` mode" — incompatible with this model's vision tower layout |
| `--mm-processor-cache-type shm` | recommended | Shared-memory mm processor cache. | None |
| `--speculative-config '{"method":"dflash","model":"/models/dflash-drafter","num_speculative_tokens":15}'` | required for spec decode | DFlash spec-decode at k=15. This is the v4 Spark recipe benchmarked in README.md. | "Use EAGLE-3 / Medusa / ngram for better spec decode" — different drafter requirements; DFlash is the matching public drafter checkpoint for Qwen3.6-27B |
| `--served-model-name aeon-ultimate qwen36-ultimate aeon-fast aeon-deep` | flag | Four aliases for the same engine. Don't reduce — downstream tools assume these names exist. | None |

---

## Patch series in the v4 image (the "what's actually in the binary")

The current production image is `vllm-aeon-ultimate-dflash:qwen36-v4`, built **2026-05-10**. It is sourced from the latest community vLLM nightly we validated (`0.20.2rc1.dev166+gf6490a284`) plus **FlashInfer 0.6.11** and a DFlash sliding-window-attention overlay from vLLM PR #40898.

| PR | Title | What it fixes |
|---|---|---|
| #40092 | SWA backend assert fix | Sliding-window attention assertion that fired on Qwen3.6's specific layer config |
| #40454 | Default-align mamba cache w/ spec-decode | Hybrid GDN + DFlash spec-decode cache alignment bug. **Also enables `mamba_cache_mode=align`** — when `--enable-prefix-caching` is on (it is, in this image), this caches GDN/SSM hidden state across requests for hybrid models that report `supports_mamba_prefix_caching=True` (Qwen3.6-27B does). Multi-turn agent workloads with shared prefix tokens benefit dramatically: turns 2+ skip re-rolling the 48 GDN layers' recurrent state. **Currently flagged "experimental" by vLLM** — a warning prints at boot ("Prefix caching in Mamba cache 'align' mode is currently enabled. Its support for Mamba layers is experimental.") This is expected; the feature is doing real work under the experimental label. |
| #40191 | `ENABLE_NVFP4_SM100=0` guard | Allows sm_121a-only builds to import without SM100-only `mxfp4_experts_quant` symbols |
| #40662 | Unified spec-decode acceptance metrics | DFlash + DynamicProposer + EAGLE all report through the same `/metrics` schema |
| #38479 | TurboQuant K8V4 backend | KV-cache compression backend (currently guarded out for hybrid models — see #39931 for the unblock) |

- **FlashInfer 0.6.11** — newer GB10/Blackwell kernel package than v3's 0.6.9 stable.
- **DFlash sliding-window-attention compatibility overlay** — applies the fix from vLLM PR #40898 while we wait for it to land upstream.
- **Packaged profiles** — `gateway`, `production`, and `benchmark` profiles in the v4 container scripts so users can choose the correct memory/concurrency shape without hand-editing a long vLLM command.
- **Measured stock-vs-v4 benchmark set** — 6 natural prompt categories x 8 concurrency levels through c=256, with TTFT and TPOT captured, now documented in `README.md`.

- **TurboQuant @ AEON-7/turboquant fix/cuda-graph-safe-qjl-powers** — fork with the cached `_POWERS` per-device patch that prevents the CUDA-graph capture crash on TurboQuant boot (upstream PR 0xSero/turboquant#12 still pending)
- **External speculators package skipped** — vLLM's native `--speculative-config` covers DFlash without an extra import surface.

If an agent is debugging an issue and finds a stack-overflow / forum thread saying "this is a known bug in vLLM," check the patch list above first. The bug may already be fixed in this image even though the upstream issue is still open.

### Why "build" instead of "pip install vllm"

A second-pass agent considering an upgrade should compare to v3 (not stock vLLM). v3 is a *fresh source build* against the official v0.20.0 release commit, with FlashInfer 0.6.9 stable, and our 5 sm_121a patches applied. Running `pip install vllm==0.20.0` over this image still loses: the patched FlashInfer (rc1 → stable diff), the 5 python-level patches, and the TurboQuant fork. The image is the deployment artifact — rebuild from `Dockerfile.v3`, don't `pip install`.

---

## Common agent failure modes (and how to recognize them)

### Failure mode: "Let me update vLLM to the latest version"

**Symptom:** Agent sees vLLM v0.20.0 in the image and runs `pip install --upgrade vllm` to "fix things."

**What actually happens:** Overwrites the patched binaries with stock vLLM. Loss of: SM121 CUTLASS NVFP4 kernels, hybrid GDN spec-decode alignment, all 5 patches above.

**Diagnostic for this failure:** If after the upgrade you see "no kernel image is available for execution on the device" or "PTX JIT compilation failed", you've hit it.

**Fix:** Don't do this. The container is the deployment artifact — rebuild the image, don't pip install.

### Failure mode: "Let me try `--quantization awq` since that worked for me before"

**Symptom:** Agent assumes the model is AWQ-quantized because Mitko Vasilev's blog used Qwen3.5-27B AWQ.

**What actually happens:** vLLM tries to load AWQ packing format on NVFP4 weights and fails with shape mismatch or scale-tensor-not-found errors.

**Diagnostic:** Boot logs show `RuntimeError: Could not find scale tensor` or shape errors during weight loading.

**Fix:** Two valid quant choices on this stack — pick by which body the operator deployed:
- Regular `-NVFP4` body → `--quantization compressed-tensors`
- XS body (`-Multimodal-NVFP4-MTP-XS`, current Spark winner) → `--quantization modelopt`

If the body is the XS variant and you set `--quantization compressed-tensors`, you'll get a "ModelOpt NVFP4 checkpoint detected, please use --quantization modelopt" warning followed by errors. The two formats are incompatible at the loader level.

### Failure mode: "Let me set `method:qwen3_5_mtp` since the body has an MTP head"

**Symptom:** Agent inspects the XS body's safetensors, sees 15 `mtp.*` tensors, and concludes "obviously use the MTP spec method."

**What actually happens:** On DGX Spark, the current documented path is external DFlash k=15. The MTP head sits in the safetensors for compatibility with dedicated-VRAM Blackwell workflows, but it is not the Spark serving method.

**Diagnostic:** The serve command says `method:"qwen3_5_mtp"` instead of `method:"dflash"`, or the DFlash drafter model is not mounted.

**Fix:** On DGX Spark, **always** use `--speculative-config '{"method":"dflash","model":"/models/dflash-drafter","num_speculative_tokens":15}'`. The MTP method is correct for **dedicated-VRAM Blackwell** (RTX 5090, RTX PRO 6000, B100/B200), not unified memory. See [the hardware-routing section in README.md](README.md#hardware-compatibility-matrix).

### Failure mode: "Let me disable thinking to make this faster"

**Symptom:** Agent reads "thinking adds latency" advice from a generic LLM-deployment forum and bakes `chat_template_kwargs.enable_thinking=false` into the production config.

**What actually happens:** Disabling thinking globally removes one of the model's core capabilities and can break workflows that expect reasoning traces. v4 parses reasoning cleanly, so the gateway can decide when to show or hide it.

**Diagnostic:** Server-side decode rate identical or worse with `enable_thinking=false` baked in vs allowing thinking to happen by default.

**Fix:** Don't bake `enable_thinking=false` into the server-side defaults. The model and v4 image preserve thinking as a first-class capability. Per-request override is fine for callers who want pure-content responses, but don't make it the global default.

### Failure mode: "This is a 27B model, I'll allocate 64 GB GPU memory"

**Symptom:** Agent on Hopper / Ada uses dedicated-VRAM rules of thumb and sets aggressive memory params.

**What actually happens:** On GB10 unified memory, this works "until" the KV cache + DFlash drafter exceed ~110 GB combined, then OS swap thrashing kicks in and serving collapses.

**Diagnostic:** Throughput drops by 10-100× after the first long-context request; `dmesg` shows page-out/page-in churn.

**Fix:** Use the documented profile: gateway defaults are `--gpu-memory-utilization 0.75 --max-num-seqs 64`; LLM-only full-context production is `--gpu-memory-utilization 0.85 --max-num-seqs 16`. Resist the urge to scale both context and concurrency at once.

### Failure mode: "The model isn't refusing — let me add safety system prompt"

**Symptom:** Agent observes the model's 0/100 refusal rate and assumes this is a config bug.

**What actually happens:** The model is **deliberately uncensored**. This is a feature of this release, not a bug.

**Diagnostic:** Refusal rate matches the model card's stated 0/100.

**Fix:** Read the [HF model card](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4). The user implements downstream safety layers; the model itself is unaligned by design. Do not modify the system prompt or add guardrails to "fix" the refusal rate without explicit user instruction.

### Failure mode: "Reasoning mode is producing weird `<think>` tags"

**Symptom:** Agent sees `<think>...</think>` blocks in the output and tries to strip them, or tries to disable reasoning mode globally.

**What actually happens:** Reasoning mode is enabled by default. `<think>` blocks are the model's chain-of-thought, parsed by `--reasoning-parser qwen3`.

**Fix:** To disable reasoning per-request, pass `chat_template_kwargs.enable_thinking=false` in the request body. To strip in client code, parse the response's `reasoning` field separately from `content`. Do not modify the global flag.

### Failure mode: "Let me run benchmarks with curl in a tight loop"

**Symptom:** Agent benchmarks via shell loop and reports decode rate of ~5 tok/s.

**What actually happens:** Each `curl` call pays the full HTTP setup cost; without streaming, total wall-time = TTFT + decode time on every request.

**Fix:** Use `stream: true` with `stream_options: {include_usage: true}` and measure time-to-first-token vs subsequent inter-token times separately. The included `bench_aeon.py` (in the repo) does this correctly.

---

## Diagnostics — confirming the stack is healthy

Run these in order. If any fail, **don't try to fix the higher-level symptom — fix the failure at the level it appears.**

### 1. Container is running

```bash
docker ps --filter "name=vllm-aeon-ultimate" --format "{{.Image}} | {{.Status}}"
```

Expected: `ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v4 | Up <time> (healthy)`. (`qwen36-v3` is acceptable only as an intentional rollback; older tags warrant investigation.)

### 2. vLLM accepted the config and booted

```bash
docker logs vllm-aeon-ultimate-v2 --tail 50 2>&1 | grep -E "ERROR|Traceback|started|listening"
```

Expected: a `Started server process` line and **no** `Traceback` or `RuntimeError`.

If you see `mxfp4_experts_quant` in a stacktrace: `ENABLE_NVFP4_SM100=0` is unset.
If you see `PTX JIT failed`: the CUTLASS path tried to compile and the patches are not in the image — wrong tag.
If you see `TurboQuant KV cache is not supported for hybrid`: someone added `--kv-cache-dtype turboquant_*`. Remove it.

### 3. Endpoint serves and lists 4 model aliases

```bash
curl -sf http://localhost:8000/v1/models | python3 -c "import json,sys; d=json.load(sys.stdin); print([m['id'] for m in d['data']])"
```

Expected: `['aeon-ultimate', 'qwen36-ultimate', 'aeon-fast', 'aeon-deep']`.

### 4. A real request returns coherent output

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aeon-ultimate",
    "messages": [{"role": "user", "content": "Calculate 47 * 83 step by step."}],
    "max_tokens": 200,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }' | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'][:300])"
```

Expected: a coherent multiplication walkthrough ending in `3901`. Anything else (gibberish, refusal, error) means a deeper config issue.

### 5. DFlash speculative decoding is actually firing

```bash
curl -s http://localhost:8000/metrics | grep -E "spec_decode|draft_acceptance"
```

Expected: nonzero values for spec-decode acceptance metrics. If all zeros after a few requests, DFlash isn't running — check that the drafter path `/models/dflash-drafter` is mounted.

---

## Performance baseline (what to expect on GB10)

Production config, single-stream, greedy decoding, `enable_thinking=false`:

| Metric | Expected |
|---|---|
| Median decode rate (mixed prompts) | ~30 tok/s |
| Peak decode rate (math/code, high DFlash acceptance) | ~50 tok/s |
| Min decode rate (free-form prose, low DFlash acceptance) | ~14 tok/s |
| TTFT (warm) | 190-225 ms |
| TTFT (cold, first request after restart) | up to ~500 ms while FlashInfer kernels JIT-compile |
| Cold-start to first token | ~90-180 s (model load + DFlash drafter load + CUTLASS autotuning) |

If your measured numbers are materially below these (e.g., median < 15 tok/s), something's wrong with the config — don't accept the regression as "expected." Run the diagnostics above.

---

## Client integration notes (read this before debugging "the model returned empty content")

These are not bugs — they're contracts agents and client developers commonly misread.

### Reasoning parser splits output across two fields

With `--reasoning-parser qwen3` (this image's default), the model's chain-of-thought and final answer arrive in different response fields:

| Mode | Reasoning section lives in | Final answer lives in |
|---|---|---|
| Streaming (`stream: true`) | `delta.reasoning` (multiple chunks) | `delta.content` (only after reasoning ends) |
| Non-streaming | `message.reasoning` (set while still reasoning) | `message.content` (populated only when final answer block begins; **`null` until then**) |

**Failure mode for clients that read only `content`**: they will see empty output for the entire reasoning phase. If your client appears to "hang silently" or returns an empty `choices[0].message.content`, this is almost certainly the cause — not a model failure.

**To capture the full output**: read both fields. Concatenate `delta.reasoning + delta.content` (or `message.reasoning + message.content`) for the complete model output. Or render them separately if you want to display the chain-of-thought to users distinctly.

**To opt out of reasoning per-request**: pass `chat_template_kwargs: {enable_thinking: false}` in the request body. The model will skip the reasoning section entirely and stream content directly.

```python
# Correct: reads both fields
import requests
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "aeon-ultimate",
    "messages": [{"role": "user", "content": "..."}],
    # Add this to skip reasoning:
    # "chat_template_kwargs": {"enable_thinking": False},
}).json()
msg = response["choices"][0]["message"]
full_output = (msg.get("reasoning") or "") + (msg.get("content") or "")
```

### `mamba_cache_mode=align` is auto-coupled to `--enable-prefix-caching`

vLLM auto-promotes `mamba_cache_mode` to `align` for `Qwen3_5ForConditionalGeneration` whenever `--enable-prefix-caching` is on (which is our default). **Setting `--mamba-cache-mode none` does NOT disable this** — it's overridden silently. The boot log line `Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration by default when prefix caching is enabled` is informational, not a warning.

The only way to actually run without align is to drop `--enable-prefix-caching` entirely, which loses prefix caching too. In practice, leave it on — the multi-turn benefit is large.

### Prefix cache hit rate is empirically inconsistent on early turns

Validation on RTX PRO 6000 (community report) observed 0 % cache hit on turns 1 and 2 of a multi-turn agent workload, then 61.9 % hit on turn 3 — even with identical system prompts and accumulating shared context. An identical-prompt smoke test (two repeats of a 10K-token prompt) also reported 0 hits on the second request.

This appears to be a real interaction between block-boundary alignment and how `mamba_cache_mode=align` lazily snapshots GDN state. **It's not a config bug.** Once cache hits land (typically turn 3+), the TTFT win is substantial and the feature pays off.

If you're running a benchmark that expects cache hits on turn 2, you may need to extend it to 3+ turns to see the expected behavior.

---

## When you think something is broken

Before making any change, ask yourself:

1. **Does my proposed fix appear in the "DO NOT UNDO" list?** If yes, stop. The symptom you're seeing is probably expected behavior or has a different root cause.
2. **Is my proposed fix from a 2024 or early-2025 source?** If yes, it's likely stale. The Blackwell / sm_121a stack changed substantially in mid-2025 through 2026.
3. **Have I run the diagnostics above?** If no, do that first. The failure may be at a layer you haven't checked.
4. **Have I read the docker-compose.yml comments?** They contain rationale that's not always in this file.
5. **Is my fix changing the patched binaries (pip install, image rebuild without the patches)?** If yes, you're regressing the stack. Don't.

If you've checked all the above and you're still confident the issue is real, the right action is to **report it to the user** with: a clear description of the symptom, what diagnostic step failed, and what your proposed fix would do. **Do not silently apply fixes that contradict this file.**

---

## Out-of-scope optimizations to skip (or schedule for later)

These are tempting but **either don't help or actively break things** for this stack as of v3.

| Optimization | Why to skip |
|---|---|
| EAGLE-2 / EAGLE-3 dynamic tree spec decode | Requires a different drafter (Eagle head, not DFlash). No public Qwen3.6-27B Eagle drafter exists. |
| Medusa heads | No Qwen3.6-27B Medusa heads published. Static-tree only in vLLM, doesn't help our hybrid GDN layers. |
| DDTree (dynamic-tree on top of DFlash) | Ongoing R&D — see [the GDN compatibility issue](#hybrid-attention--ddtree-status). Not in any public vLLM release. |
| `--enable-prefix-caching=False` to "save memory" | Counterproductive — chat workloads benefit substantially from prefix caching. |
| Aggressive CUDA graphs (`--num-cudagraphs N`) | Already tuned in the image. Manual override likely regresses. |
| Tensor parallelism (`--tensor-parallel-size 2+`) | DGX Spark is single-GPU. TP > 1 is meaningless. |
| Pipeline parallelism | Same. |
| Disable `--enable-chunked-prefill` for "lower latency" | Will OOM on long-context requests. |

---

## Hybrid attention & DDTree status

This model uses **16 full-attention + 48 GatedDeltaNet/linear-attention layers** (75% recurrent). This architecture has specific implications:

1. **DFlash works** because the verify pass is a flat block — GDN sees ordinary sequential decode.
2. **Tree-based spec decode (Eagle-2, DDTree, Medusa-2 with branches) does NOT work correctly** on this stack as of the current vLLM release. GDN's `fused_recurrent` writes state in-place per token and doesn't support per-branch state forking. Tree verification on hybrid models produces silent quality degradation (vLLM issues #39273, #36872 confirm this on Qwen3.5/3.6 hybrids).
3. **MTP-style methods work** because they decouple draft proposal from target state evolution.

If an agent or user proposes "let's just enable Eagle tree spec decode," the answer is: **not on this hybrid model with the current vLLM.** Wait for either (a) Mitko Vasilev's `vllm-dflash-ddtree` repo to ship code, (b) vLLM PR #39487 (custom callable proposer) + a custom DDTreeProposer with parent-indexed GDN handling, or (c) the upstream tree-attention-on-FLA work to mature.

---

## Quick reference: the canonical command

If you've read this far and just want the production command:

```bash
docker compose up -d
docker compose logs -f vllm
```

That runs [`docker-compose.yml`](docker-compose.yml) which is the single source of truth for production config. **If you find yourself crafting a custom `vllm serve` command line, you're probably about to undo something in the DO NOT UNDO list.** Use the compose file unless you have a specific reason not to and have reviewed this entire document.

---

## Where to ask if you're stuck

- This file's section that matches your symptom
- [`docker-compose.yml`](docker-compose.yml) inline comments
- The [HF model card for the NVFP4 release](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4) — quantization details
- The [HF model card for BF16](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16) — abliteration pipeline details
- The user — for anything that requires architectural decisions, hardware changes, or visible side effects

**Do not invent fixes from training data.** This stack is recent and your training data probably doesn't contain it. When in doubt, report what you observe and stop.
