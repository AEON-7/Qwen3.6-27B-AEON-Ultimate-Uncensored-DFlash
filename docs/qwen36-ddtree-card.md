# Qwen3.6-27B AEON Ultimate DDTree Card

> **Status: experimental research image.** This container publishes the current
> Qwen3.6 DDTree-on-vLLM implementation so the community can inspect, reproduce,
> and extend it. Use the production DFlash v4 container for reliable serving
> today. Use this DDTree image when you are testing tree verification,
> parent-state replay, or helping push hybrid-model speculative decoding forward.

## Container

```bash
docker pull ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental
```

Tags:

| Tag | Purpose |
|---|---|
| `qwen36-v5-m53-experimental` | Immutable-ish milestone tag for the current DDTree research image. |
| `experimental` | Moving experimental tag for the newest published DDTree build. |

Published digest:

```text
sha256:baddf917bbc8f547bd70bbd09d122157d44f545bb9597a54145aa6795704d552
```

The package is separate from the stable DFlash image on purpose:

| Stable serving | DDTree research |
|---|---|
| `ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v4` | `ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental` |

## What This Image Preserves

The point of the AEON DDTree track is not to abandon the working vLLM stack.
The image keeps the same user-facing capabilities as the Qwen3.6 DFlash runtime:

- Qwen3.6-27B AEON Ultimate XS NVFP4 modelopt weights.
- FlashInfer CUTLASS NVFP4 path for GB10 / sm_121a.
- DFlash drafter integration with `num_speculative_tokens=15`.
- OpenAI-compatible vLLM server.
- Qwen3 reasoning parser for `<think>...</think>`.
- Qwen3-Coder tool-call parser for structured `message.tool_calls[]`.
- Multimodal request support through vLLM's Qwen3.6 model path.
- `--no-enable-prefix-caching` enforced for DFlash/DDTree correctness.

## Why DDTree Exists

Flat speculative decoding asks the drafter for one linear chain:

```text
target prefix -> draft token 1 -> draft token 2 -> draft token 3 -> ...
```

The target model then verifies that chain. If the first wrong token appears
early, the rest of the verifier work is mostly wasted.

DDTree changes the proposal shape. Instead of spending the whole budget on one
chain, it builds a small tree of likely alternatives:

```text
prefix
  -> best token
       -> best continuation
       -> alternate continuation
  -> alternate token
       -> best continuation
```

In principle, this gives the verifier more useful paths per target forward
pass. If the drafter's top-1 token misses but another high-probability branch
matches the target, DDTree can still accept useful tokens instead of throwing
away the whole window.

That is why DDTree is attractive for Qwen3.6 on DGX Spark:

- The target model is expensive enough that every verifier pass matters.
- DFlash already produces rich draft logits we can mine for branches.
- Blackwell/GB10 has enough memory bandwidth and FP4 throughput that a fused
  tree verifier could become a real speed multiplier.
- Long-thinking agent workloads often have local ambiguity where top-k branch
  recovery should help acceptance.

## Why Qwen3.6 Is Hard

Qwen3.6 is not a plain full-attention transformer. It is a hybrid model with
full-attention layers plus Gated DeltaNet / Mamba-style recurrent layers.

For a normal chain, recurrent state is simple:

```text
state[t + 1] = layer(token[t], state[t])
```

For a tree, every branch must fork state from its own parent:

```text
state[node] = layer(token[node], state[parent(node)])
```

That means true DDTree acceleration needs three pieces to be correct at the
same time:

1. **Tree attention**: every verifier row may attend only to the prefix plus
   its ancestors, not to sibling branches.
2. **Branch-local recurrent state**: GDN/conv state must be loaded from the
   node's parent and written into a branch scratch slot.
3. **Commit replay**: after sampling, only the accepted branch is copied back
   into the model's normal contiguous KV/recurrent state.

The current image contains prototypes and guarded implementations for all
three surfaces. The unresolved work is making the non-flat branch commit path
quality-equivalent and fast enough for production.

## Implementation Breakdown

Current milestone: **M53 temp non-flat GDN**, published as
`qwen36-v5-m53-experimental`.

The image includes:

| Area | What is implemented |
|---|---|
| vLLM method | `method="dflash_ddtree"` is accepted by speculative config. |
| DFlash payload | DFlash top-k logits are converted into DDTree parent metadata. |
| Scheduler bridge | Tree payloads move through vLLM beside the legacy flat draft-token path. |
| Runtime sampler | Target logits can be interpreted as root+tree rows and sampled as an accepted branch plus bonus token. |
| Attention verifier | Ancestor-mask experiments exist for FlexAttention and FlashAttention/Triton branch correction. |
| GDN replay | Slow reference replay plus Triton parent-state reload experiments exist. |
| State compaction | Accepted flat-prefix rows can be compacted safely; non-flat commit is guarded as research-only. |
| DFlash context repair | Empty-context, target-position, query-padding, and drafter-context compaction fixes are included through M52/M53. |
| Safety defaults | Prefix caching is off; unsafe full-branch commit requires explicit research env vars. |

## Current Operating Modes

| Mode | Use it for | Status |
|---|---|---|
| `dflash` | Normal DFlash serving from the same image. | Stable path inherited from v4. |
| `ddtree` | Safe DDTree method surface and flat-prefix validation. | Coherent, research/validation only. |
| `ddtree-full` | Non-flat branch commit, recurrent compaction, Triton GDN replay. | Research only; quality not production-safe yet. |
| `qwen25-m2` | Small full-attention DDTree oracle target. | Useful for verifier logic without hybrid GDN complexity. |

## Quick Start

```bash
docker run --gpus all --ipc host --network host --rm \
  -v /path/to/aeon-xs:/models/aeon-xs:ro \
  -v /path/to/dflash-drafter:/models/dflash-drafter:ro \
  ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental \
  ddtree
```

For manual vLLM launches, keep this invariant:

```bash
--no-enable-prefix-caching
```

Prefix caching conflicts with DFlash/DDTree verifier ownership of KV and GDN
state. The packaged launchers now force it off even if
`ENABLE_PREFIX_CACHING=1` is accidentally set.

## Research Environment Knobs

These are for isolated experiments, not default serving:

| Env var | Meaning |
|---|---|
| `DDTREE_ROOT_LEAF_ONLY=0` | Enables heap-shaped non-flat DDTree construction. |
| `DDTREE_CHAIN_SEED_LIMIT=4` | Builds a short chain before adding branch alternatives. |
| `DDTREE_TOP_K=4` | Top-k width used when expanding tree candidates. |
| `DDTREE_BUDGET=8` | Number of draft nodes in the tree verifier window. |
| `DDTREE_TRITON_TREE_GDN=1` | Uses the Triton/FLA GDN parent replay path. |
| `DDTREE_BRANCH_ONLY_ATTN=1` | Applies branch-row attention correction only where needed. |
| `DDTREE_FULL_BRANCH_COMMIT=1` | Research-only branch commit path; not production safe. |
| `DDTREE_UNSAFE_FULL_BRANCH_RESEARCH=1` | Required guard for unsafe full-branch experiments. |

## Benchmarks

The benchmark numbers below are included to make the status clear, not to claim
DDTree is already faster than v4. The production speed target remains the v4
DFlash image until non-flat branch commit is quality-equivalent.

### Production Reference: v4 DFlash

Natural-prompt single stream, thinking enabled, prefix caching disabled for
DFlash correctness:

| Category | Decode tok/s | TTFT p50 | TPOT p50 |
|---|---:|---:|---:|
| Coding | 31.12 | 222 ms | 30.3 ms |
| Math | 41.09 | 222 ms | 23.4 ms |
| Reasoning | 43.41 | 233 ms | 22.2 ms |
| Prose | 29.42 | 211 ms | 33.3 ms |
| Natural language | 31.08 | 227 ms | 31.3 ms |
| Extraction / JSON | 45.36 | 219 ms | 21.3 ms |
| **Average** | **36.91** | **223 ms** | **27.0 ms** |

### DDTree M1 Bridge Benchmark

M1 validates the vLLM `dflash_ddtree` method surface and bridge while retaining
flat-safe behavior. It is useful as a regression baseline because it shows the
DDTree packaging path can preserve DFlash-like throughput before enabling
unsafe non-flat branch commit.

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

### M53 Non-Flat Research Status

M53 is the currently published research image. It includes the latest fixes for
DFlash context compaction, empty context KV handling, query input padding, and
temporary non-flat GDN experiments.

Observed status:

- Flat-chain / flat-prefix behavior remains coherent.
- Root-wide alternatives are toxic for Qwen3.6 because native DFlash depth
  logits are not branch-conditioned root alternatives.
- Heap-shaped non-flat payloads can be built and carried through vLLM.
- Non-flat branch rows still degrade quality when branch-state replay and
  commit are enabled.
- Slow Python GDN replay is not a reliable quality oracle; it can degrade even
  on flat-chain tests.
- Fused/Triton GDN replay is structurally present, but still needs scratch
  branch-state buffers and an explicit commit path before it can be called
  production acceleration.

Practical conclusion:

```text
Use v4 DFlash for serving.
Use v5 DDTree for research, validation, and kernel/commit-path development.
```

## What Needs To Land Next

True full DDTree acceleration for Qwen3.6 needs:

1. Scratch branch-state buffers for GDN/conv layers.
2. Fused branch attention that handles ancestor-only masks without falling back
   to slow dynamic paths.
3. Branch-state GDN replay that never mutates the committed recurrent cursor
   until a branch is accepted.
4. Accepted-branch commit that writes only the sampled branch back to the
   normal vLLM KV/recurrent state.
5. DFlash drafter-context compaction that keeps accepted draft hidden states
   aligned without accidentally feeding the target bonus row back to the
   drafter.
6. Full capability regression: text, vision, reasoning, tool calls, structured
   output, long context, and guided JSON.

When those land, DDTree should move from a research container to a production
`qwen36-v5` release candidate.

