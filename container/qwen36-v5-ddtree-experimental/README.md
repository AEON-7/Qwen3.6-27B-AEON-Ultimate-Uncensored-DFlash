# Qwen3.6 AEON Ultimate DFlash / DDTree Experimental Spark vLLM

Thin DGX Spark image for validating Qwen3.6 AEON Ultimate on the latest official
community `vllm/vllm-openai:nightly` base, with the DFlash sliding-window
attention fix from vLLM PR #40898 overlaid until it lands upstream.

This v5 experimental image adds the DDTree integration milestones needed to
move from paper prototype toward live Qwen3.6 verification:

- M1: `method="dflash_ddtree"` is accepted by vLLM and routed through the
  existing flat DFlash verifier.
- M2: DDTree builder, verifier metadata, compact-logit sampler, Qwen2.5 oracle,
  and one-pass ancestor-mask verifier prototypes are installed and tested.
- M3: vLLM now has an optional tree payload bridge beside the existing flat
  speculative-token path: `DraftTokenIds.draft_trees` -> `Request.spec_tree` ->
  `SchedulerOutput.scheduled_spec_decode_trees` -> GPU model-runner hook.
- M4A: Qwen2.5 full-attention multi-step tree decode-loop oracle. It repeatedly
  builds a DDTree, verifies all branches with an ancestor-only mask, emits the
  accepted branch plus bonus token, and checks the whole generated response
  against ordinary greedy decoding.
- M4B: DFlash now builds a real top-k DDTree payload from its parallel draft
  logits when `method="dflash_ddtree"` is selected. The payload is carried beside
  the old flat top-1 chain through the M3 bridge, so flat DFlash remains
  available as the fallback verifier path.
- M5A: tree-aware Qwen3.6 GDN/conv reference math. This proves the hybrid
  recurrent state update we need to port into vLLM kernels: every sibling reloads
  convolution history and Gated DeltaNet state from its parent, then writes its
  own intermediate state for descendants.
- M5B: guarded runtime tree sampler. When enabled, target logits are interpreted
  as compact root+tree rows and sampled by walking the accepted branch plus the
  bonus/recovery token.
- M6A-M6B: Qwen3.6 parent metadata plumbing plus a correctness-first slow GDN
  replay path. This makes linear-attention/Mamba siblings reload state from
  their actual tree parent rather than the previous flat token.
- M6C-M6F: FlexAttention ancestor-mask bring-up. This proved the full-attention
  mask shape but is too heavy for live Spark serving and remains a diagnostic
  path only.
- M6G: guarded FlashAttention eager verifier path for full-attention layers.
  The normal FlashAttention backend stays selected, but the tiny root+tree
  verifier window uses an explicit ancestor mask when DDTree is active.
- M7A: accepted-branch KV/state compaction. Accepted tree nodes are copied back
  into the contiguous committed spine and hybrid recurrent state uses the
  accepted compact branch node instead of the flat accepted-count assumption.
- M8A: opt-in Triton DDTree conv/GDN fast path. This replaces the slow Python
  parent-state replay for Qwen3.6 hybrid layers with tree-parent-aware Triton
  conv and Gated DeltaNet state reload kernels while retaining the M7B slow path
  as a runtime fallback.
- M8B: sequential DFlash DDTree payload generation. Some live DFlash configs use
  vLLM's sequential proposer path rather than the parallel-drafting early exit,
  so this retains each draft step's hidden state and builds the same DDTree
  payload after the sequential chain completes.
- M8C: flat-chain metadata fallback. If a live path schedules flat DFlash tokens
  before a true branch payload arrives, the runner represents those tokens as a
  chain-shaped DDTree. That preserves flat behavior and still exercises
  parent-aware Qwen3.6 GDN state handling.
- M8D: opt-in inline attention parent-id fallback. If
  `DDTREE_INLINE_PARENT_FALLBACK=1` is set and no branch parent tensor is
  available when attention metadata is built, GDN/Flash metadata receives a
  chain-shaped parent tensor directly. This is a diagnostic switch, not the
  default serving path, because it proved the Triton parent-state replay can run
  live but regressed quality/latency when forced globally.
- M8E: DDTree payload handoff tracing. The DFlash proposer, model runner, and
  `take_draft_token_ids` bridge each log one concise payload count so we can
  identify where true branch metadata disappears without forcing fallback parent
  ids into every attention window.
- M8F: live parent metadata install. The runner now builds
  `ddtree_parent_ids` directly from the real proposer payload immediately after
  `drafter.propose()`, so the verifier GDN window can use true branch parent
  state before the next scheduler round exposes `scheduled_spec_decode_trees`.
- M8G: cached attention metadata parent refresh. When vLLM reuses a cached
  attention metadata object via `update_block_table()`, the runner refreshes its
  mutable `ddtree_parent_ids` field so GDN/Flash layers do not see stale `None`
  parent metadata.
- M8H: pending proposer payload rehydration. If the scheduler bridge does not
  carry `draft_trees` yet, the runner preserves the real DFlash DDTree payload
  from the prior `drafter.propose()` and rehydrates it before building the next
  target attention metadata.
- M8I: current vLLM sampler compatibility. The DDTree runtime sampler now
  returns the current `SamplerOutput(sampled_token_ids, logprobs_tensors)` shape
  used by the vLLM nightly base.
- M8J-M10T: live Qwen3.6 branch verification research. These milestones added
  flat-chain attention bypasses, branch diagnostics, branch-mask correction,
  graph-safe Triton verifier experiments, accepted-branch KV/state alignment,
  branch-state compaction experiments, DFlash drafter-context compaction, and
  the vLLM-safe non-flat DDTree walk adaptation.
- M11A-M11D: full-branch commit hardening. These milestones proved that
  committing non-flat branches through Qwen3.6 hybrid recurrent state is still a
  research problem, then added an explicit guard so stale full-branch env vars
  cannot silently corrupt output.
- M11E: deployable DDTree-safe defaults. The `ddtree` entrypoint now defaults
  to the coherent vLLM-compatible tree mode: flat accepted prefix, root-leaf
  bonus branches, DFlash drafter-context compaction, no recurrent branch-state
  compaction, and no Triton GDN replay by default.
- M11F: smoke-profile compile tightening. The `ddtree` entrypoint now also
  caps `MAX_NUM_BATCHED_TOKENS=2048` by default so the operational smoke server
  does not spend boot time and memory compiling a 32K graph for a 2K context.
- M11G: full-branch accepted+bonus contract restoration. The research path now
  commits the non-flat accepted branch while still emitting the target bonus
  token, and relies on M11A's explicit accepted-count channel to tell the
  scheduler how many draft nodes were accepted.
- M11H: fused GDN replay hardening. Triton DDTree GDN parent loads are
  width-guarded so padded/guided decode windows cannot read past the parent-id
  tensor.

Flat DFlash behavior is preserved unless `SPEC_METHOD=dflash_ddtree` and the
guarded DDTree runtime env vars are enabled. The `ddtree` entrypoint enables the
validated deployable-safe recipe by default: `MAX_MODEL_LEN=2048`,
`MAX_NUM_SEQS=1`, `NUM_SPECULATIVE_TOKENS=15`, `DDTREE_BUDGET=15`,
`MAX_NUM_BATCHED_TOKENS=2048`, `DDTREE_TOP_K=8`, `DDTREE_ROOT_LEAF_ALT_COUNT=4`,
`DDTREE_MIN_ROOT_BRANCHES=5`, `DDTREE_TRITON_TREE_GDN=0`, and CUDA graphs on.
Override these env vars only after the safe path is healthy.

Modes:

- `baseline`: target model only, no DFlash.
- `dflash`: target model plus DFlash drafter.
- `ddtree`: experimental live DDTree verifier. It enables target-side tree
  scheduling, runtime branch sampling, slow GDN parent replay, FlashAttention
  ancestor masking, and accepted-branch KV/state compaction.
- `ddtree-full`: research-only full acceleration profile. It enables non-flat
  branch commit, recurrent branch-state compaction, Triton GDN replay, and
  Triton branch attention. Use only for isolated validation; keep the normal
  `ddtree` profile for quality-safe serving until full validation passes.
- `qwen25-m2`: small full-attention M2 correctness target using
  `Qwen/Qwen2.5-0.5B-Instruct`.
- `ddtree-tree-test`: runs the local DDTree builder unit tests.
- `ddtree-metadata-test`: runs the vLLM-shaped tree metadata unit tests.
- `ddtree-vllm-import-test`: verifies the M2 metadata modules are importable
  from `vllm.v1.spec_decode`.
- `ddtree-m3-bridge-test`: verifies the M3 scheduler/model-runner tree payload
  carrier is present while flat DFlash remains optional.
- `ddtree-m4b-payload-test`: verifies DFlash top-k tree payload source and
  payload shape are present.
- `ddtree-gdn-reference-test`: verifies the tree-aware GDN/conv reference math
  against path replay.
- `qwen25-tree-oracle`: runs a standalone Qwen2.5 greedy DDTree oracle.
- `qwen25-tree-oracle-suite`: runs the oracle across six prompt categories.
- `qwen25-tree-mask-verifier`: runs the first one-pass ancestor-mask verifier
  prototype for Qwen2.5.
- `qwen25-tree-decode-loop`: runs the multi-step Qwen2.5 DDTree decode-loop
  oracle for one prompt.
- `qwen25-tree-decode-suite`: runs the multi-step decode-loop oracle across six
  prompt categories.
- `ddtree-gdn-reference`: runs the tree-aware GDN/conv reference check directly.
- `bench`: category benchmark harness.
- `bash`: shell.

Build:

```bash
docker build \
  -t ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental \
  container/qwen36-v5-ddtree-experimental
```

The image intentionally keeps the patch surface narrow so benchmark differences
are attributable to DFlash and the SWA fix rather than a custom full-source fork.

## DFlash launch profiles

`dflash` mode accepts a `PROFILE` environment variable so the same image can be
used for production serving, local gateway serving, and short-context stress
benchmarks without rewriting the command line.

| Profile | Max context | Max seqs | GPU util | Prefix cache | Use case |
|---|---:|---:|---:|---|---|
| `production` *(default)* | 200000 | 16 | 0.85 | off | Documented DGX Spark long-context recipe with DFlash-compatible KV ownership. |
| `gateway` | 256000 | 64 | 0.75 | off | Local OpenClaw-style deployment where ASR/TTS or other GPU services need headroom. |
| `benchmark` | 2048 | 256 | 0.85 | off | Short-prompt throughput sweep. Keeps prefix caching off so DFlash/DDTree verifier state is never mixed with reused KV prefixes. |

All profiles keep the production-critical knobs from the Spark compose recipe:

- CUTLASS NVFP4 selected with `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`,
  `VLLM_TEST_FORCE_FP8_MARLIN=0`, and `ENABLE_NVFP4_SM100=0`.
- FlashAttention selected explicitly with `--attention-backend flash_attn`.
- DFlash k=15 via the z-lab drafter.
- Prefix caching disabled with `--no-enable-prefix-caching`; vLLM prefix-cache
  reuse conflicts with DFlash/DDTree's target/drafter KV and recurrent-state
  handoff assumptions on Qwen3.6 hybrid models.
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

## Experimental DDTree live mode

Run the deployable-safe method surface:

```bash
docker run --gpus all --rm -p 8000:8000 \
  -v /path/to/aeon-xs:/models/aeon-xs:ro \
  -v /path/to/dflash-drafter:/models/dflash-drafter:ro \
  ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental ddtree
```

This launches the experimental DDTree profile. It uses the real DFlash DDTree
payload bridge and runtime sampler, but only commits the flat accepted prefix
back into vLLM. Additional root-leaf branches are still verified as bonus
candidates. That keeps output coherent while preserving the full multimodal,
reasoning, structured-output, and tool-calling surface.

Do not enable `DDTREE_TRITON_TREE_GDN=1`,
`DDTREE_COMPACT_RECURRENT_STATE=1`, `DDTREE_ALLOW_BRANCH_STATE_COMPACTION=1`,
or `DDTREE_FULL_BRANCH_COMMIT=1` for normal serving. Those switches are
research-only. In live tests, Triton GDN replay survived normal text but hit a
CUDA illegal memory access during guided JSON decoding, while full branch commit
corrupted Qwen3.6 hybrid recurrent-state output. M11D-M11F require
`DDTREE_UNSAFE_FULL_BRANCH_RESEARCH=1` before full-branch commit can activate.

Live DGX Spark validation for `qwen36-v5-ddtree-m11d/m11e/m11f`:

- Booted from the safe `ddtree` profile with CUDA graphs enabled.
- Confirmed launch flags: `method=dflash_ddtree`,
  `num_speculative_tokens=15`, `ddtree_budget=15`, `ddtree_top_k=8`,
  `DDTREE_ROOT_LEAF_ALT_COUNT=4`, `DDTREE_MIN_ROOT_BRANCHES=5`,
  `DDTREE_TRITON_TREE_GDN=0`, `--tool-call-parser qwen3_coder`,
  `--reasoning-parser qwen3`, `--attention-backend flash_attn`.
- Confirmed backend: `FlashInferCutlassNvFp4LinearKernel` for NVFP4 GEMM and
  FlashAttention v2 for full-attention layers.
- Smoke-tested exact response, prose, chat, reasoning, code, math, guided JSON,
  structured tool calls, and OpenAI-style image input.
- Quality stayed coherent. Tool calls returned OpenAI `message.tool_calls[]`.
  Guided JSON returned valid JSON. Vision correctly identified an opaque red
  PNG.
- DDTree logs showed real tree parent metadata with a flat chain plus root
  branches, for example `first_parents=[-1,0,1,2,3,4,5,6,7,8,9,-1,-1,-1,-1]`.
- Speculative metrics stayed live, with observed mean acceptance length ranging
  from about 3.3 to 5.9 on the smoke prompts. The measured ratio is a
  correctness signal, not a production throughput benchmark.

Equivalent explicit env form:

```bash
SPEC_METHOD=dflash_ddtree \
DDTREE_TARGET_VERIFY=1 \
DDTREE_USE_RUNTIME_SAMPLER=1 \
DDTREE_FORCE_GREEDY_TREE_SAMPLER=1 \
DDTREE_TRITON_TREE_GDN=0 \
DDTREE_INLINE_PARENT_FALLBACK=0 \
DDTREE_ROOT_LEAF_ONLY=1 \
DDTREE_ROOT_LEAF_ALT_COUNT=4 \
DDTREE_MIN_ROOT_BRANCHES=5 \
DDTREE_COMPACT_RECURRENT_STATE=0 \
DDTREE_ALLOW_BRANCH_STATE_COMPACTION=0 \
DDTREE_FULL_BRANCH_COMMIT=0 \
DDTREE_COMPACT_DRAFTER_CONTEXT=1 \
PROFILE=benchmark \
MAX_MODEL_LEN=2048 \
MAX_NUM_BATCHED_TOKENS=2048 \
MAX_NUM_SEQS=1 \
GPU_MEMORY_UTILIZATION=0.65 \
NUM_SPECULATIVE_TOKENS=15 \
DDTREE_BUDGET=15 \
DDTREE_TOP_K=8 \
EXTRA_VLLM_ARGS=--no-async-scheduling \
./scripts/serve_qwen36_dflash.sh
```

For first live validation keep `DDTREE_BUDGET <= NUM_SPECULATIVE_TOKENS` and
use a short `MAX_MODEL_LEN` smoke profile. The safe path is operational for
quality validation, but it is not the final performance target. The next
performance milestone is making branch-state GDN replay and branch attention
safe under guided decoding, tool calls, vision, and CUDA graphs before enabling
full non-flat branch commit.

## M2 bring-up target: Qwen2.5-0.5B

M2 uses `Qwen/Qwen2.5-0.5B-Instruct` as the first correctness model. It is a
small dense causal transformer with full 32K context, RoPE, SwiGLU, RMSNorm,
and GQA. That keeps the first tree-verifier work focused on attention masks,
tree sampling, and KV commit behavior instead of Qwen3.6 hybrid recurrent state.

Start the small-model server:

```bash
docker run --gpus all --rm -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental \
  qwen25-m2
```

Useful overrides:

```bash
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=32768
GPU_MEMORY_UTILIZATION=0.35
ENFORCE_EAGER=1       # optional for debugger-friendly correctness work
```

Run the local tree-builder tests:

```bash
docker run --rm \
  ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental \
  ddtree-tree-test
```

Run the vLLM-shaped metadata tests:

```bash
docker run --rm \
  ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental \
  ddtree-metadata-test
```

Run the vLLM M3 bridge test:

```bash
docker run --rm \
  ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental \
  ddtree-m3-bridge-test
```

Run the DFlash tree payload test:

```bash
docker run --rm \
  ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental \
  ddtree-m4b-payload-test
```

Run the tree-aware GDN/conv reference test:

```bash
docker run --rm \
  ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental \
  ddtree-gdn-reference-test
```

Run the Qwen2.5 greedy oracle:

```bash
docker run --gpus all --rm \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental \
  qwen25-tree-oracle \
  --budget 8 \
  --depth 6 \
  --top-k 4 \
  --prompt "Write one vivid sentence about sunrise over the ocean."
```

Run the six-category oracle suite:

```bash
docker run --gpus all --rm \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD/bench/results:/results" \
  ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental \
  qwen25-tree-oracle-suite \
  --output /results/qwen25_tree_oracle_suite.json
```

Run the one-pass ancestor-mask verifier:

```bash
docker run --gpus all --rm \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental \
  qwen25-tree-mask-verifier \
  --budget 8 \
  --depth 6 \
  --top-k 4
```

Run the multi-step decode-loop oracle:

```bash
docker run --gpus all --rm \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental \
  qwen25-tree-decode-loop \
  --max-new-tokens 32 \
  --budget 8 \
  --depth 6 \
  --top-k 4
```

Run the six-category multi-step decode-loop suite:

```bash
docker run --gpus all --rm \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD/bench/results:/results" \
  ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental \
  qwen25-tree-decode-suite \
  --max-new-tokens 32 \
  --output /results/qwen25_tree_decode_suite.json
```

## Implementation Map

The standalone correctness prototypes live in:

```text
prototypes/ddtree_tree.py
prototypes/ddtree_vllm_metadata.py
prototypes/qwen25_tree_oracle.py
prototypes/qwen25_tree_oracle_suite.py
prototypes/qwen25_tree_mask_verifier.py
prototypes/qwen25_tree_decode_loop.py
prototypes/qwen25_tree_decode_suite.py
tests/test_ddtree_tree.py
tests/test_ddtree_vllm_metadata.py
tests/test_dflash_ddtree_m3_bridge.py
tests/test_dflash_ddtree_m4b_payload.py
tests/test_ddtree_runtime_sampler.py
tests/test_ddtree_parent_metadata.py
tests/test_ddtree_gdn_reference.py
```

The build applies these patches in order:

- `apply_pr40898_dflash_swa.py`: overlays the upstream DFlash/SWA support.
- `apply_dflash_ddtree_m1.py` through `m3.py`: register `dflash_ddtree` and
  carry tree payloads through scheduler/model-runner boundaries.
- `apply_dflash_ddtree_m4b.py`: makes the DFlash drafter emit top-k tree
  payloads with a flat fallback.
- `apply_dflash_ddtree_m5b.py`: installs the compact-logit runtime sampler.
- `apply_dflash_ddtree_m6a.py` and `m6b.py`: plumb parent ids into Qwen3.6 GDN
  metadata and replay hybrid recurrent state from each node's parent.
- `apply_dflash_ddtree_m6g.py`: verifies full-attention root+tree rows with an
  explicit ancestor mask while keeping FlashAttention as the selected backend.
- `apply_dflash_ddtree_m7a.py`: compacts accepted branch KV/state back into the
  committed spine after sampling.

The current live path is correctness-first. It is meant to prove quality and
state semantics on Qwen3.6 before replacing the eager tree verifier and slow GDN
replay with fused CUDA/Triton kernels.




