# Qwen3.6-27B-AEON-Ultimate-Uncensored+DDTree

> **WARNING: experimental research container.**
>
> This image is published so the community can inspect, reproduce, benchmark,
> and extend the current DDTree-on-vLLM work for Qwen3.6 on DGX Spark / GB10.
> It is **not** the recommended production serving image yet. For reliable
> serving today, use the stable DFlash path. Use this DDTree image when you are
> testing tree verification, branch-state replay, Gated DeltaNet state handling,
> fused branch attention, or helping push hybrid-model speculative decoding
> forward.

This card is intentionally candid. We have spent days building and testing this
path, including many failed approaches. The goal is to give other researchers a
real map of what has already been tried, what appears to work, what still breaks,
and where the next breakthrough likely lives.

## Container

```bash
docker pull ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental
```

Published tags:

| Tag | Purpose |
|---|---|
| `qwen36-v5-m53-experimental` | Milestone tag for the current public DDTree research image. |
| `experimental` | Moving tag for the newest published DDTree build. |

Published digest:

```text
sha256:baddf917bbc8f547bd70bbd09d122157d44f545bb9597a54145aa6795704d552
```

The DDTree package is intentionally separate from the stable DFlash package:

| Stable production path | DDTree research path |
|---|---|
| `ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v4` | `ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental` |

Both images are built from the same validated vLLM base commit:

```text
vLLM 0.20.2rc1.dev166+gf6490a284
base commit f6490a28413526adefae8dd7af32906d060e0436
```

So the DDTree image is not a random fork. It is the stable Qwen3.6 DFlash /
Blackwell stack plus experimental DDTree overlays.

## What This Preserves

The DDTree effort is not a stripped-down text-only experiment. The design goal is
to keep the same user-facing capabilities as the working Qwen3.6 DFlash runtime:

- Qwen3.6-27B AEON Ultimate Multimodal NVFP4-MTP-XS body.
- ModelOpt NVFP4 format with GB10 / sm_121a CUTLASS hardware path.
- DFlash drafter integration using `z-lab/Qwen3.6-27B-DFlash`.
- OpenAI-compatible vLLM `/v1/chat/completions` endpoint.
- Qwen3 reasoning parser for `<think>...</think>`.
- Qwen3-Coder tool-call parser for structured `message.tool_calls[]`.
- Vision/multimodal request path through vLLM's Qwen3.6 model class.
- JSON / structured-output compatibility inherited from vLLM.
- CUDA graph capable DFlash serving path when running in pure `dflash` mode.

## What DDTree Is Trying To Do

Flat speculative decoding spends its verifier budget on one draft chain:

```text
prefix -> draft_1 -> draft_2 -> draft_3 -> ...
```

If the target model rejects an early draft token, most later verifier work is
wasted.

DDTree spends the same budget on a tree of alternatives:

```text
prefix
  -> token A
       -> token A1
       -> token A2
  -> token B
       -> token B1
```

In principle, this lets the verifier recover when the drafter's top-1 token is
wrong but another high-probability branch is right. For a local long-context
agent model, that is the prize: more accepted tokens per expensive target
forward pass without losing reasoning, tool calling, vision, or long-context
behavior.

## Why Qwen3.6 Is A Hard Target

Qwen3.6 is not a plain transformer. It is a hybrid architecture with:

- full-attention layers,
- Gated DeltaNet / Mamba-style recurrent layers,
- recurrent convolution state,
- long-context position handling,
- multimodal serving hooks,
- reasoning and tool-call parsing on top.

For a flat chain, recurrent state is simple:

```text
state[t + 1] = layer(token[t], state[t])
```

For a tree, every branch must fork from its own parent:

```text
state[node] = layer(token[node], state[parent(node)])
```

That turns DDTree into a three-surface problem:

1. **Tree attention**: each verifier row may attend to the prefix plus its own
   ancestors, not sibling branches.
2. **Branch-local recurrent state**: GDN and conv state must be read from the
   parent branch and written into scratch state for the child branch.
3. **Accepted-branch commit**: after sampling, only the accepted path may be
   committed back into vLLM's normal contiguous KV and recurrent state.

If any of those three is wrong, the benchmark can look alive while quality
quietly collapses.

## How To Run It

### Safe Research Smoke Mode

This is the default DDTree research entrypoint. It uses conservative limits and
guards unsafe full-branch commit paths.

```bash
docker run --gpus all --ipc host --network host --rm \
  -v /path/to/aeon-xs:/models/aeon-xs:ro \
  -v /path/to/dflash-drafter:/models/dflash-drafter:ro \
  ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental \
  ddtree
```

### Pure DFlash Mode From The Same Image

The same image can run the stable DFlash method surface:

```bash
docker run --gpus all --ipc host --network host --rm \
  -e PROFILE=gateway \
  -e SPEC_METHOD=dflash \
  -e ENABLE_PREFIX_CACHING=1 \
  -v /path/to/aeon-xs:/models/aeon-xs:ro \
  -v /path/to/dflash-drafter:/models/dflash-drafter:ro \
  ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental \
  dflash
```

Pure DFlash mode is expected to behave like the stable DFlash lineage, but the
production-recommended image remains `qwen36-v4` until this exact path has been
A/B validated.

### Unsafe Full-Branch Research Mode

This exists for kernel and commit-path developers. It is **not** production safe:

```bash
docker run --gpus all --ipc host --network host --rm \
  -v /path/to/aeon-xs:/models/aeon-xs:ro \
  -v /path/to/dflash-drafter:/models/dflash-drafter:ro \
  ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental \
  ddtree-full
```

## Prefix Cache Caveat

For **DDTree research modes**, keep prefix caching off. DDTree verifier work
owns temporary KV and recurrent state, and prefix reuse can hide state-alignment
bugs while the branch commit path is still experimental.

For **pure DFlash production serving**, prefix caching can be valuable. In an
OpenClaw-style repeated-agent-prefix test, a shared 37,837-token system prompt
dropped from about 26 seconds uncached TTFT to about 0.7 seconds on cached
follow-up requests. That is a real gateway win, but it is separate from DDTree
correctness.

Practical rule:

| Mode | Prefix cache |
|---|---|
| `dflash` production/gateway | Worth testing, often useful for repeated agent prefixes. |
| `ddtree` smoke/research | Off by default. |
| `ddtree-full` branch commit research | Off unless you are explicitly testing prefix-cache interaction. |

## Current Operating Modes

| Entry point | Purpose | Status |
|---|---|---|
| `dflash` | Stable DFlash serving surface from this image. | Should be close to v4; still validate before replacing v4. |
| `ddtree` | Guarded DDTree research mode with safe defaults. | Useful for payload, sampler, and flat-prefix validation. |
| `ddtree-full` | Full non-flat branch commit experiments. | Research only; quality not production safe. |
| `qwen25-m2` | Small full-attention target for DDTree correctness work. | Recommended proving ground before Qwen3.6 hybrid integration. |
| `bench` | Natural-prompt benchmark harness. | Used for c=1..256 category sweeps. |
| `ddtree-*test` | Unit/smoke tests for tree metadata, sampler, payload, and GDN reference code. | Useful for patch validation. |

## Chronicle Of The DDTree Work So Far

This section compresses the trial-and-error path into what future contributors
actually need to know.

### Phase 1: M1 to M3, vLLM Method Bridge

What worked:

- Added `method="dflash_ddtree"` as a vLLM speculative method surface.
- Preserved the existing DFlash path while plumbing tree metadata beside it.
- Proved the packaging path could boot, serve, and benchmark without breaking
  normal DFlash behavior.

What it taught us:

- The container and launch surface are viable.
- A flat-safe bridge can benchmark well, but that is not yet true DDTree.

### Phase 2: M4, Payload Generation

What worked:

- Converted DFlash top-k logits into DDTree parent metadata.
- Built root/child payloads and compact verifier row descriptions.
- Added tests around tree node ordering, parent IDs, and compact logits indices.

What failed or got complicated:

- Naive root-wide top-k alternatives were toxic for Qwen3.6. DFlash depth logits
  are useful for flat continuation, but not automatically good branch-conditioned
  root alternatives.

What it taught us:

- DDTree needs branch-conditioned proposal logic, not just "take the top-k at
  each flat depth and call it a tree."

### Phase 3: M5 to M6, GDN State And Tree Attention

What worked:

- Built a slow reference GDN replay path.
- Added parent metadata into Qwen3.6 GDN-layer execution.
- Tested FlexAttention ancestor masks and FlashAttention/Triton branch-correction
  ideas.

What failed or got complicated:

- Slow Python GDN replay was not a clean quality oracle. It could degrade even
  flat-chain probes.
- Dynamic tree masks and non-contiguous hybrid KV layouts are easy to make
  correct-looking but hard to make fast and graph-safe.

What it taught us:

- The core blocker is not just "make an attention mask." The recurrent branch
  state must be first-class.

### Phase 4: M7 to M8, Accepted Branch Compaction

What worked:

- Added accepted-branch KV/state compaction experiments.
- Added safeguards so flat-prefix rows could be compacted without corrupting the
  base path.
- Added drafter-context compaction and empty-context fixes.

What failed or got complicated:

- Full non-flat commits could produce plausible token streams but degrade quality.
- Bonus-token handling is subtle: the accepted draft path and target bonus row
  must not be fed back into the drafter as if they were the same thing.

What it taught us:

- "Accepted tokens" and "emitted tokens" need explicit, separate plumbing.
- The drafter context has to be rebuilt from accepted draft states, not guessed
  from target verifier rows.

### Phase 5: M8 to M10, Fused Paths And CUDA-Graph Safety

What worked:

- Added opt-in Triton GDN replay experiments.
- Added branch-only attention correction paths.
- Added CUDA-int32 parent metadata normalization and graph-safety probes.
- Explored paged-KV verifier support and FlashAttention branch mask correction.

What failed or got complicated:

- Graph-safe tree attention is not the same as fast tree attention.
- The branch-row path can be technically wired but still lose the performance
  benefit if it falls back to dynamic or eager work too often.

What it taught us:

- The final production path likely needs fused branch attention plus fused
  branch-state GDN replay, not a stack of Python-side corrections.

### Phase 6: M10 to M11/M53, Non-Flat Commit And DFlash Context Repair

What worked:

- Added accepted-count plumbing.
- Restored accepted+bonus scheduler contracts.
- Added branch GDN state mirroring and recurrent-state cursor synchronization.
- Added DFlash hidden-state compaction and one-step drafter suppression tests.
- Fixed several concrete crash/shape issues:
  - empty context KV handling,
  - zero-context position handling,
  - query input padding,
  - current vLLM sampler output signature compatibility.

What remains unresolved:

- Non-flat branch commit is not yet quality-equivalent.
- Branch-state replay still needs scratch buffers that never mutate committed
  recurrent state before acceptance.
- We need a clean full-attention proving target before trusting Qwen3.6 hybrid
  results.

## Current Benchmark State

The numbers below are included for transparency. They should not be read as
"DDTree is production faster now." The current production winner remains flat
DFlash.

### Stable Production Reference: v4 DFlash

Natural prompts, thinking enabled, Qwen3.6 AEON Ultimate XS, DFlash k=15.

| Category | Decode tok/s | TTFT p50 | TPOT p50 |
|---|---:|---:|---:|
| Coding | 31.12 | 222 ms | 30.3 ms |
| Math | 41.09 | 222 ms | 23.4 ms |
| Reasoning | 43.41 | 233 ms | 22.2 ms |
| Prose | 29.42 | 211 ms | 33.3 ms |
| Natural language | 31.08 | 227 ms | 31.3 ms |
| Extraction / JSON | 45.36 | 219 ms | 21.3 ms |
| **Average** | **36.91** | **223 ms** | **27.0 ms** |

### DFlash With Prefix Cache, Agent-Style Repeated Prefix

This is not DDTree, but it matters for real OpenClaw/gateway serving. The test
used a stable 37,837-token shared system prefix followed by short per-agent
tasks.

| Request | TTFT |
|---:|---:|
| First uncached request | 26,045 ms |
| Cached follow-up 1 | 682 ms |
| Cached follow-up 2 | 694 ms |
| Cached follow-up 3 | 672 ms |
| Cached follow-up 4 | 669 ms |
| Cached follow-up 5 | 683 ms |

Conclusion: prefix caching is a production DFlash optimization for repeated
agent prefixes, but it should not be used to hide DDTree state correctness bugs.

### DDTree M1 Bridge Benchmark

M1 validates the vLLM method bridge and flat-safe DDTree packaging. It is not
true non-flat branch acceleration.

Single stream, thinking disabled in this particular run:

| Category | Decode tok/s | Peak tok/s | TTFT p50 | TPOT p50 |
|---|---:|---:|---:|---:|
| Coding | 34.13 | 37.13 | 185 ms | 28.4 ms |
| Math | 47.32 | 54.54 | 208 ms | 20.1 ms |
| Reasoning | 31.67 | 42.53 | 208 ms | 30.7 ms |
| Prose | 18.52 | 20.46 | 180 ms | 53.2 ms |
| Natural language | 21.59 | 23.16 | 154 ms | 45.5 ms |
| Extraction / JSON | 67.15 | 72.42 | 212 ms | 12.8 ms |
| **Average** | **36.73** | **41.71** | **191 ms** | **31.8 ms** |

Full c=1..256 sweep, thinking enabled:

| Category | c=1 tok/s | Peak aggregate tok/s | c=256 aggregate tok/s | c=256 TTFT p50 |
|---|---:|---:|---:|---:|
| Coding | 32.00 | 185.86 @ c=256 | 185.86 | 126.6 s |
| Math | 40.95 | 270.29 @ c=256 | 270.29 | 87.6 s |
| Reasoning | 45.88 | 254.06 @ c=128 | 252.65 | 94.1 s |
| Prose | 31.23 | 170.12 @ c=256 | 170.12 | 139.4 s |
| Natural language | 33.60 | 177.55 @ c=256 | 177.55 | 134.5 s |
| Extraction / JSON | 54.77 | 306.75 @ c=128 | 304.64 | 77.0 s |

### Current M53 Non-Flat Probe Status

The current published research image can build and pass DDTree payloads through
vLLM. In a debug probe profile, observed acceptance looked like this:

| Metric | Observed range |
|---|---:|
| Mean accepted length | about 2.7 tokens |
| Draft acceptance rate | about 21% |
| Position 0 acceptance | about 50-67% |
| Position 1 acceptance | about 44-50% |
| Position 2 acceptance | about 33% |
| Position 3 acceptance | about 28-33% |
| Later branch positions | mostly 0% |

That is enough to prove that the plumbing is not dead. It is not enough to call
DDTree operational.

## Current Outstanding Blockers

### 1. Scratch Branch-State Buffers

GDN and conv layers need per-branch scratch state. A verifier row should load
state from its parent, write state to its own branch slot, and leave the
committed recurrent cursor untouched until a branch is accepted.

### 2. Explicit Accepted-Branch Commit

The runtime needs first-class accepted node IDs, parent IDs, accepted count, and
bonus-token ownership. Only the accepted branch should be copied into normal
vLLM KV/recurrent state.

### 3. Fused Branch Attention

Ancestor-only attention masks need a fast fused implementation. Dynamic fallback
paths can be useful for correctness, but they erase much of the speed benefit.

### 4. DFlash Drafter Context Compaction

The drafter context must be compacted from accepted draft hidden states. Rejected
branches and target bonus rows must not leak into the drafter's next context.

### 5. Position And RoPE Correctness

Verifier rows need logical sequence positions based on branch depth and accepted
history, not just compact row order.

### 6. Small Full-Attention Proving Ground

Before declaring Qwen3.6 hybrid DDTree solved, the non-flat algorithm should
produce clean outputs on a smaller full-attention model such as
`Qwen/Qwen2.5-0.5B-Instruct`. That separates tree-verifier bugs from hybrid GDN
bugs.

## Important Caveats

- **Experimental means experimental.** This image is for research and
  reproducibility, not production SLAs.
- **DDTree mode is not faster yet.** The published image preserves the work so
  others can help push it over the line.
- **The v5 image shares the same base vLLM commit as v4.** Most v5 changes are
  DDTree overlays, not generic DFlash speedups.
- **Root-wide alternatives were not enough.** Naive top-k branch expansion can
  hurt quality on Qwen3.6.
- **Greedy-only testing can mislead.** Some anomalies changed when native
  sampling was restored, so both deterministic and normal sampling tests matter.
- **Prefix cache is mode-dependent.** Useful for repeated DFlash agent prefixes,
  risky for DDTree correctness work.
- **Thermals matter on DGX Spark.** Long c=256 sweeps can heat-soak the box and
  shift results. Cooldown cycles are recommended before final benchmark runs.
- **c=256 is a saturation map, not a suggested user setting.** Most real agent
  gateways should care more about c=1, c=4, c=8, c=16, and c=32 behavior.

## What Help Would Be Most Useful

If you want to help unlock production DDTree for Qwen3.6, the highest-value
contributions are:

1. A clean small-model DDTree correctness harness with token-by-token parity
   against baseline decoding.
2. A fused ancestor-mask branch attention kernel that works with vLLM's paged KV
   layout.
3. Scratch-buffer GDN/conv branch-state replay that never mutates committed
   recurrent state before acceptance.
4. A robust accepted-branch commit contract for vLLM speculative decoding.
5. DFlash branch-conditioned proposal logic, rather than naive root-wide top-k
   expansion.
6. Quality canaries covering text, long-form reasoning, tool calls, JSON,
   vision, and long context.

## Bottom Line

The fruit of this work so far is a public, reproducible DDTree research
container that preserves the full Qwen3.6 AEON Ultimate serving stack and exposes
the exact places where true tree acceleration still needs engineering.

Use it as a map, a testbed, and a starting point.

Do **not** mistake it for the final production breakthrough yet.

