# DDTree vLLM Integration Plan

This document is the working plan for a future `qwen36-v5-ddtree` image. The goal is to bring DDTree into vLLM without giving up the current AEON v4 production capabilities:

- Qwen3.6-27B multimodal XS body in modelopt NVFP4.
- CUTLASS NVFP4 fast path on DGX Spark / GB10 / sm_121a.
- DFlash drafter with Qwen3.6 sliding-attention compatibility.
- Qwen3 reasoning parsing via `--reasoning-parser qwen3`.
- Qwen3-Coder XML tool parsing via `--tool-call-parser qwen3_coder`.
- OpenAI-compatible chat, tools, vision, streaming, and gateway behavior.

DDTree must be integrated as a vLLM speculative decoding method. It should not replace vLLM's HTTP server, tokenizer, multimodal processors, tool parsers, reasoning parser, scheduler, or model loading path.

## Current State

The current production image is:

```text
ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v4
```

The current production speculative configuration is flat DFlash:

```json
{
  "method": "dflash",
  "model": "/models/dflash-drafter",
  "num_speculative_tokens": 15,
  "attention_backend": "FLASH_ATTN"
}
```

This is the stable baseline. DDTree work must preserve this path and fall back to it cleanly.

## Implementation Status

M1 scaffolding now exists in:

```text
container/qwen36-v5-ddtree-experimental/
```

It adds a build-time overlay script:

```text
scripts/apply_dflash_ddtree_m1.py
```

M1 behavior:

- vLLM accepts `method="dflash_ddtree"`.
- `SpeculativeConfig` accepts `ddtree_budget`, `ddtree_top_k`, `ddtree_temperature`, and `ddtree_chain_seed`.
- `DFlashProposer` allows both `dflash` and `dflash_ddtree`.
- The base proposer treats `dflash_ddtree` like DFlash for hidden-state handling, tuple returns, and parallel drafting.
- The serve script supports `SPEC_METHOD=dflash_ddtree`.
- The `ddtree` container entrypoint mode sets `SPEC_METHOD=dflash_ddtree`.
- Actual verification still uses the flat DFlash path.

Expected M1 benchmark result: identical to v4 flat DFlash within normal noise. Any measurable regression in M1 is a plumbing bug.

Build command:

```bash
docker build \
  -t ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental \
  container/qwen36-v5-ddtree-experimental
```

## Why DDTree Is Not Just a Flag

Flat DFlash proposes a single chain of draft tokens. vLLM's current speculative metadata and rejection sampler are shaped around that chain:

- `draft_token_ids`
- `num_draft_tokens`
- `cu_num_draft_tokens`
- `target_logits_indices`
- `bonus_logits_indices`

DDTree changes the verify problem from one chain to a tree:

1. DFlash produces per-position draft distributions.
2. A best-first tree builder expands likely sibling branches under a fixed node budget.
3. The target verifies the entire flattened tree in one pass.
4. Attention uses an ancestor-only mask so each node only sees its path.
5. The sampler walks one accepted path and commits only that path.
6. Hybrid recurrent state must persist only the accepted path, not discarded siblings.

The last point is the hard part for Qwen3.6. Qwen3.6 is a hybrid model with full-attention layers plus GatedDeltaNet / Mamba-style recurrent layers. DDTree cannot be production-safe until the GDN state update is tree-aware.

## Reference Sources

- Lucebox Hub: <https://github.com/Luce-Org/lucebox-hub>
- Lucebox DFlash source: <https://github.com/Luce-Org/lucebox-hub/tree/main/dflash>
- DDTree paper: <https://arxiv.org/abs/2604.12989>
- DFlash paper: <https://arxiv.org/abs/2602.06036>
- DDTree reference implementation: <https://github.com/liranringel/ddtree>
- vLLM repository: <https://github.com/vllm-project/vllm>
- vLLM speculative decoding docs: <https://docs.vllm.ai/en/latest/features/speculative_decoding/>
- vLLM tree-attention feature request: <https://github.com/vllm-project/vllm/issues/18327>
- vLLM DFlash KV-quantization compatibility issue: <https://github.com/vllm-project/vllm/issues/41559>

Lucebox already demonstrates the key GB10-relevant pieces in a ggml/llama.cpp-style runtime:

- DDTree best-first tree construction.
- Ancestor-only tree attention masks.
- Tree-aware SSM/GDN rollback and persist logic.
- `ggml_ssm_conv_tree`.
- `ggml_gated_delta_net_tree`.
- `ggml_gated_delta_net_tree_persist`.
- Blackwell / GB10 build detection for `sm_121`.

Those pieces are directly useful as algorithm and kernel references, but they are not a drop-in vLLM backend. vLLM needs native integration.

## Upstream vLLM Status

As of the current vLLM docs and main branch:

- vLLM documents `dflash` as a supported speculative decoding method.
- vLLM's public speculative config also mentions suffix decoding tree-depth knobs, but those belong to suffix/prompt-cache speculation and are not DDTree verifier support.
- vLLM issue #18327 requested tree-attention support for speculative decoding, but it is closed as not planned. Do not assume upstream already has a generic tree verifier we can simply enable.
- vLLM's current DFlash path requires non-causal draft cross-attention. There is an open compatibility issue around DFlash plus KV-cache quantization. For the AEON v4/v5 path, keep `--kv-cache-dtype auto` unless a later upstream patch explicitly fixes non-causal quantized KV support.
- vLLM internals already contain useful hints for future tree work: attention metadata comments mention tree-based attention indexing, and `llm_base_proposer` contains a FIXME about tree-based speculative decoding pass depth. These are hooks, not a complete implementation.

Conclusion: DDTree support must be implemented as a first-class vLLM speculative method, not as a launch flag, and not by switching to suffix decoding.

## Proposed User-Facing Config

Keep v4 behavior as the default. Add DDTree as an explicit experimental method:

```json
{
  "method": "dflash_ddtree",
  "model": "/models/dflash-drafter",
  "num_speculative_tokens": 15,
  "ddtree_budget": 22,
  "ddtree_top_k": 8,
  "ddtree_temperature": 1.0,
  "ddtree_chain_seed": true,
  "fallback_method": "dflash"
}
```

Recommended first sweep on DGX Spark:

| Setting | Initial value | Notes |
|---|---:|---|
| `num_speculative_tokens` | 15 | Preserve current DFlash sweet spot. |
| `ddtree_budget` | 22 | Lucebox RTX 3090 sweet spot; GB10 may prefer larger. |
| `ddtree_top_k` | 8 | Only needed when budget exceeds chain length. |
| `ddtree_chain_seed` | true | Keeps chain behavior strong while adding sibling recovery. |
| `ddtree_budget` sweep | 15, 22, 32, 40, 48, 64 | Stop when verifier bandwidth or queueing dominates. |

## vLLM Patch Surface

### 1. Speculative Config

Files:

- `vllm/config/speculative.py`
- Any method registry that currently accepts `dflash`.

Add:

- `method="dflash_ddtree"`.
- `ddtree_budget`.
- `ddtree_top_k`.
- `ddtree_temperature`.
- `ddtree_chain_seed`.
- validation that `ddtree_budget >= num_speculative_tokens` when tree mode is active.
- clean fallback to `method="dflash"` when tree mode is disabled.

### 2. Tree Draft Data Model

Files:

- `vllm/v1/spec_decode/metadata.py`
- new `vllm/v1/spec_decode/tree_metadata.py` if separating the dataclass is cleaner.

Add a tree metadata structure rather than overloading flat chain metadata too far:

```python
class SpecDecodeTreeMetadata:
    tree_token_ids: torch.Tensor          # [total_tree_nodes]
    tree_parent_indices: torch.Tensor     # [total_tree_nodes]
    tree_depths: torch.Tensor             # [total_tree_nodes]
    tree_node_offsets: torch.Tensor       # [batch + 1]
    tree_target_logits_indices: torch.Tensor
    tree_bonus_logits_indices: torch.Tensor
    tree_visibility: torch.Tensor | None
    tree_draft_log_probs: torch.Tensor | None
```

Do not remove or mutate `SpecDecodeMetadata`. The flat chain path must remain stable.

### 3. DDTree Proposer

Files:

- `vllm/v1/spec_decode/dflash.py`
- new `vllm/v1/spec_decode/dflash_ddtree.py`
- `vllm/v1/spec_decode/utils.py`

Implementation:

- Reuse `DFlashProposer` to run the drafter and obtain draft distributions.
- Extend the DFlash pass to expose top-k token ids and log-probs for positions `1..k`.
- Build a per-request DDTree using a best-first heap:
  - root is the previous committed token.
  - chain seed uses top-1 for each depth.
  - siblings are added by cumulative log-prob under budget.
- Return both:
  - the tree metadata for verifier mode.
  - a canonical flat top-1 draft chain for fallback compatibility.

### 4. Scheduler and Model Runner

Files:

- `vllm/v1/worker/gpu/model_runner.py`
- `vllm/v1/worker/gpu_model_runner.py` on older branches.
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/outputs.py`

Required behavior:

- The scheduler still exposes a linear committed sequence to users.
- Tree sibling nodes must never be appended to request history.
- Only the accepted path is committed.
- Metrics must distinguish verified tree nodes from committed tokens.
- CUDA graph capture sizes must account for `1 + ddtree_budget` verifier tokens per request.

### 5. Tree Attention Verifier

Files:

- `vllm/v1/attention/backend.py`
- active attention backend used by Qwen3.6 on GB10.
- possibly FlashAttention / FlashInfer wrapper code for custom masks.

Required behavior:

- Flatten each request's tree into one verifier block.
- Full-attention layers use an ancestor-only mask.
- Past KV is visible according to the model's normal attention/sliding-window rules.
- Tree nodes only see their ancestors, never siblings.
- Padding nodes must be invisible and graph-stable.

MVP options:

1. Safe but slower: use an existing mask-capable attention path for tree verify only.
2. Production: add a FlashAttention/FlashInfer-compatible packed tree mask path.
3. Fallback: if a backend cannot support tree masks, disable DDTree and use flat DFlash.

### 6. Qwen3.6 GDN / Mamba State

Files:

- `vllm/model_executor/models/qwen3_5.py`
- `vllm/model_executor/layers/mamba/gdn_linear_attn.py`
- `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`
- `vllm/v1/attention/backends/gdn_attn.py`
- `vllm/v1/worker/mamba_utils.py`
- `vllm/v1/worker/gpu/model_states/mamba_hybrid.py`

This is the production blocker.

Current vLLM hybrid spec decode tracks accepted chain length through `num_accepted_tokens`. DDTree needs more information:

- accepted tree path indices.
- tree parent indices.
- per-node GDN intermediate states.
- selected final recurrent state for each GDN layer.
- selected conv state for each GDN layer.

The required kernel behavior mirrors Lucebox:

- tree convolution reads from each node's parent chain instead of DFS neighbor order.
- tree GDN computes each node from its parent state.
- tree persist writes per-node intermediate state.
- commit copies only the selected accepted path state into persistent cache.

Until this exists, DDTree on Qwen3.6 should not be marked production-ready. A wrong state commit may look fast while corrupting long-context behavior.

### 7. Tree Sampler

Files:

- `vllm/v1/sample/rejection_sampler.py`
- `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`
- `vllm/v1/worker/gpu/spec_decode/probabilistic_rejection_sampler_utils.py`

Add a separate tree sampler:

- Walk from root.
- At each node, use target logits to select or sample the next token.
- If the selected token exists as a child, accept that child and continue.
- If not, emit the target token as the recovery/bonus token and stop.
- Return a linear list of committed token ids.

For first production pass, support greedy and temperature-0 correctness first. Add probabilistic rejection sampling after the tree path is stable.

### 8. Metrics

Add metrics for:

- `ddtree_budget`.
- `ddtree_nodes_verified`.
- `ddtree_nodes_wasted`.
- `ddtree_accept_depth`.
- `ddtree_accept_depth_by_position`.
- `ddtree_sibling_branch_taken`.
- `ddtree_fallback_count`.
- DFlash acceptance with tree enabled vs flat DFlash.
- target verify time.
- tree build time.
- GDN tree state commit time.

These are necessary to know whether DDTree is actually helping on GB10 or just adding verifier pressure.

## Milestones

### M0: Design and Reference Capture

No production image changes.

- Keep v4 stable.
- Document Lucebox tree builder, tree mask, and GDN persist design.
- Extract minimal pseudocode from Lucebox into internal notes.
- Confirm vLLM branch base and current DFlash patch status.

### M1: No-Op Tree Metadata Path

Goal: `method="dflash_ddtree"` boots and behaves exactly like flat DFlash when `ddtree_budget == num_speculative_tokens`.

- Add config.
- Add proposer wrapper.
- Add metadata object.
- No custom tree attention yet.
- Fallback to flat DFlash sampler.

Pass criteria:

- text generation matches v4 distribution within sampling noise.
- tool calls still return `message.tool_calls[]`.
- reasoning still lands in `reasoning_content`.
- vision request still works.
- no CUDA graph capture regression.

### M2: Attention-Only Tree Verify

Goal: tree verify works for full-attention layers on a non-hybrid or controlled model.

Chosen bring-up model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Reason:

- Qwen-family tokenizer and transformer conventions.
- Small enough for rapid rebuild/test cycles.
- Dense causal transformer without Qwen3.6 hybrid recurrent state.
- Full 32K context, so context behavior remains representative enough for
  speculative decoding integration.

Initial M2 files:

```text
container/qwen36-v5-ddtree-experimental/prototypes/ddtree_tree.py
container/qwen36-v5-ddtree-experimental/prototypes/ddtree_vllm_metadata.py
container/qwen36-v5-ddtree-experimental/prototypes/qwen25_tree_oracle.py
container/qwen36-v5-ddtree-experimental/prototypes/qwen25_tree_oracle_suite.py
container/qwen36-v5-ddtree-experimental/prototypes/qwen25_tree_mask_verifier.py
container/qwen36-v5-ddtree-experimental/scripts/apply_dflash_ddtree_m2.py
container/qwen36-v5-ddtree-experimental/tests/test_ddtree_tree.py
container/qwen36-v5-ddtree-experimental/tests/test_ddtree_vllm_metadata.py
container/qwen36-v5-ddtree-experimental/scripts/serve_qwen25_m2.sh
```

- Build DDTree from DFlash top-k.
- Convert DDTree into vLLM-shaped flattened metadata:
  - token ids,
  - parent indices,
  - depth/position ids,
  - compact verifier-logit rows,
  - edge-to-parent logit mapping.
- Add the greedy compact-logit tree sampler that emits accepted tokens plus a
  bonus/recovery token.
- Install the tree and metadata modules into `vllm.v1.spec_decode` at image
  build time, without routing production traffic through them yet.
- Validate greedy tree walking against Qwen2.5 target logits using the oracle.
- Apply ancestor-only attention mask.
- Compare one-pass tree logits against per-path replay logits.
- Add tree sampler.
- Do not claim Qwen3.6 production support yet.

Pass criteria:

- `budget=1` equals ordinary one-token greedy verification.
- tree sampler commits valid linear paths.
- no sibling leakage in attention mask tests.
- Qwen2.5-0.5B output matches baseline when tree mode is reduced to a single
  top-1 chain.

### M3: Qwen3.6 Hybrid Safe Mode

Goal: run Qwen3.6 with DDTree without corrupting GDN state, even if slower.

Possible strategy:

- Tree verify attention logits.
- Commit only accepted path.
- Replay accepted path through the existing flat GDN path to update recurrent state.

This may reduce speedup, but it gives a correctness bridge while tree GDN kernels are built.

Pass criteria:

- long-context regression tests pass.
- repeated multi-turn prompts remain stable.
- DFlash acceptance and output quality do not degrade.

### M4: Tree-Aware GDN Kernel Path

Goal: eliminate replay and commit GDN state directly from the tree verify pass.

Implement or port:

- tree conv1d update.
- tree gated delta net update.
- per-node intermediate state storage.
- selected-path persistent state commit.

This is the milestone that makes DDTree worth shipping as a performance feature for Qwen3.6.

### M5: v5 Experimental Container

Image:

```text
ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental
```

Keep v4 as stable. Publish v5 as experimental until benchmarks and capability tests pass.

Required launch profile:

- same model body as v4.
- same NVFP4 CUTLASS env.
- same multimodal limits.
- same Qwen3 reasoning parser.
- same Qwen3-Coder tool parser.
- same gateway and benchmark profiles.
- new `PROFILE=ddtree` or `SPEC_METHOD=dflash_ddtree`.

## Capability Regression Suite

DDTree cannot ship if any of these fail:

1. Plain chat response.
2. Long thinking response with `reasoning_content` separated.
3. Tool call prompt returns structured `tool_calls[]`.
4. Forced tool call works with `tool_choice`.
5. Vision input works.
6. Multimodal plus tool metadata still works.
7. Streaming works.
8. JSON schema / structured output still works.
9. Prefix cache does not corrupt multi-turn state.
10. Long-context prompt with GDN recurrence remains stable.

## Benchmark Plan

Compare four modes:

| Mode | Purpose |
|---|---|
| Raw baseline eager | Transparency baseline. |
| v4 flat DFlash | Current production winner. |
| v5 DDTree safe replay | Correctness bridge. |
| v5 DDTree tree-GDN | True target. |

Prompt categories:

- coding.
- math.
- reasoning.
- prose.
- natural language.
- extraction / JSON.
- tool calling.
- vision reasoning.

Concurrency:

```text
1, 4, 8, 16, 32, 64, 128, 256
```

Metrics:

- TTFT p50 / p90.
- TPOT p50 / p90.
- median decode tok/s.
- peak sample tok/s.
- prefill throughput.
- DFlash acceptance by position.
- DDTree accepted depth.
- DDTree sibling branch rate.
- request errors.
- thermal state and cooldown notes.

Use at least 16 samples per category/concurrency point and report trimmed medians. Preserve raw JSON files.

## Go / No-Go Rules

Ship as v5 experimental if:

- all capability regression tests pass.
- c=1 decode improves meaningfully over v4 flat DFlash or matches it with better acceptance/depth.
- c=16 practical agent profile is not worse than v4.
- no recurring CUDA graph capture failure.
- no GDN long-context corruption.

Promote as default only if:

- tree-GDN path is correct.
- multimodal/tool/reasoning behavior is unchanged.
- benchmarks beat v4 in the common c=1..16 range.
- fallback to flat DFlash is automatic on unsupported backends.

Do not promote if:

- GDN state requires replay and the replay erases the speedup.
- tree masks force a slow attention backend.
- long-context behavior drifts.
- tool calls or reasoning parsing regress.

## Immediate Next Work

1. Run the Qwen2.5 tree oracle across a small prompt suite and preserve the
   JSON outputs as correctness fixtures.
2. Add a Qwen2.5 one-pass tree-verifier prototype that consumes the flattened
   tree, parent indices, and ancestor mask.
3. Compare one-pass verifier logits against the oracle's per-path replay logits.
4. Add tree metadata to vLLM without touching the flat DFlash fast path.
5. Wire Qwen2.5-0.5B through the metadata path first.
6. Prototype safe replay mode for Qwen3.6 GDN state after full-attention M2 is
   correct.
7. Only then start CUDA/Triton/FlashInfer kernel work for tree-aware GDN.

The shortest safe path is:

```text
v4 flat DFlash
  -> v5 method alias with exact flat fallback
  -> DDTree builder + tree sampler tests
  -> attention-only tree verify
  -> Qwen3.6 safe replay mode
  -> tree-aware GDN kernels
  -> v5 experimental image
  -> benchmark/promotion decision
```
