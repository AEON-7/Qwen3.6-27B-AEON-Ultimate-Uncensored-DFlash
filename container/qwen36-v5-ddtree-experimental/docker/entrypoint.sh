#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/qwen36-aeon-dflash-spark-vllm"
MODE="${1:-dflash}"

if [[ $# -gt 0 ]]; then
  shift
fi

case "$MODE" in
  baseline)
    exec bash "$ROOT/scripts/serve_qwen36_baseline.sh" "$@"
    ;;
  dflash)
    exec bash "$ROOT/scripts/serve_qwen36_dflash.sh" "$@"
    ;;
  ddtree)
    export SPEC_METHOD="${SPEC_METHOD:-dflash_ddtree}"
    export PROFILE="${PROFILE:-benchmark}"
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
    export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"
    export DDTREE_BUDGET="${DDTREE_BUDGET:-15}"
    export DDTREE_TOP_K="${DDTREE_TOP_K:-8}"
    export DDTREE_TARGET_VERIFY="${DDTREE_TARGET_VERIFY:-1}"
    export DDTREE_USE_RUNTIME_SAMPLER="${DDTREE_USE_RUNTIME_SAMPLER:-1}"
    export DDTREE_FORCE_GREEDY_TREE_SAMPLER="${DDTREE_FORCE_GREEDY_TREE_SAMPLER:-1}"
    export DDTREE_TRITON_TREE_GDN="${DDTREE_TRITON_TREE_GDN:-0}"
    export DDTREE_SLOW_TREE_GDN="${DDTREE_SLOW_TREE_GDN:-0}"
    export DDTREE_SLOW_TREE_CONV="${DDTREE_SLOW_TREE_CONV:-0}"
    export DDTREE_EAGER_TREE_ATTN="${DDTREE_EAGER_TREE_ATTN:-0}"
    export DDTREE_BRANCH_ONLY_ATTN="${DDTREE_BRANCH_ONLY_ATTN:-0}"
    export DDTREE_BYPASS_FLAT_CHAIN_ATTN="${DDTREE_BYPASS_FLAT_CHAIN_ATTN:-1}"
    export DDTREE_TRITON_BRANCH_ATTN="${DDTREE_TRITON_BRANCH_ATTN:-0}"
    export DDTREE_INLINE_PARENT_FALLBACK="${DDTREE_INLINE_PARENT_FALLBACK:-0}"
    export DDTREE_CHAIN_SEED="${DDTREE_CHAIN_SEED:-true}"
    export DDTREE_MIN_ROOT_BRANCHES="${DDTREE_MIN_ROOT_BRANCHES:-5}"
    export DDTREE_ROOT_LEAF_ONLY="${DDTREE_ROOT_LEAF_ONLY:-1}"
    export DDTREE_ROOT_LEAF_ALT_COUNT="${DDTREE_ROOT_LEAF_ALT_COUNT:-4}"
    export DDTREE_COMPACT_RECURRENT_STATE="${DDTREE_COMPACT_RECURRENT_STATE:-0}"
    export DDTREE_ALLOW_BRANCH_STATE_COMPACTION="${DDTREE_ALLOW_BRANCH_STATE_COMPACTION:-0}"
    export DDTREE_FULL_BRANCH_COMMIT="${DDTREE_FULL_BRANCH_COMMIT:-0}"
    export DDTREE_FULL_BRANCH_STATE_COUNT_BIAS="${DDTREE_FULL_BRANCH_STATE_COUNT_BIAS:-1}"
    export DDTREE_COMPACT_DRAFTER_CONTEXT="${DDTREE_COMPACT_DRAFTER_CONTEXT:-1}"
    export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash_attn}"
    export EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:---no-async-scheduling}"
    exec bash "$ROOT/scripts/serve_qwen36_dflash.sh" "$@"
    ;;
  ddtree-full)
    export SPEC_METHOD="${SPEC_METHOD:-dflash_ddtree}"
    export PROFILE="${PROFILE:-benchmark}"
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
    export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"
    export DDTREE_BUDGET="${DDTREE_BUDGET:-15}"
    export DDTREE_TOP_K="${DDTREE_TOP_K:-8}"
    export DDTREE_TARGET_VERIFY="${DDTREE_TARGET_VERIFY:-1}"
    export DDTREE_USE_RUNTIME_SAMPLER="${DDTREE_USE_RUNTIME_SAMPLER:-1}"
    export DDTREE_FORCE_GREEDY_TREE_SAMPLER="${DDTREE_FORCE_GREEDY_TREE_SAMPLER:-1}"
    export DDTREE_CHAIN_SEED="${DDTREE_CHAIN_SEED:-true}"
    export DDTREE_MIN_ROOT_BRANCHES="${DDTREE_MIN_ROOT_BRANCHES:-5}"
    export DDTREE_ROOT_LEAF_ONLY="${DDTREE_ROOT_LEAF_ONLY:-1}"
    export DDTREE_ROOT_LEAF_ALT_COUNT="${DDTREE_ROOT_LEAF_ALT_COUNT:-4}"
    export DDTREE_FULL_BRANCH_COMMIT="${DDTREE_FULL_BRANCH_COMMIT:-1}"
    export DDTREE_ALLOW_BRANCH_STATE_COMPACTION="${DDTREE_ALLOW_BRANCH_STATE_COMPACTION:-1}"
    export DDTREE_UNSAFE_FULL_BRANCH_RESEARCH="${DDTREE_UNSAFE_FULL_BRANCH_RESEARCH:-1}"
    export DDTREE_COMPACT_RECURRENT_STATE="${DDTREE_COMPACT_RECURRENT_STATE:-1}"
    export DDTREE_FULL_BRANCH_STATE_COUNT_BIAS="${DDTREE_FULL_BRANCH_STATE_COUNT_BIAS:-1}"
    export DDTREE_COMPACT_DRAFTER_CONTEXT="${DDTREE_COMPACT_DRAFTER_CONTEXT:-1}"
    export DDTREE_EAGER_TREE_ATTN="${DDTREE_EAGER_TREE_ATTN:-1}"
    export DDTREE_BRANCH_ONLY_ATTN="${DDTREE_BRANCH_ONLY_ATTN:-1}"
    export DDTREE_BYPASS_FLAT_CHAIN_ATTN="${DDTREE_BYPASS_FLAT_CHAIN_ATTN:-1}"
    export DDTREE_TRITON_BRANCH_ATTN="${DDTREE_TRITON_BRANCH_ATTN:-1}"
    export DDTREE_TRITON_TREE_GDN="${DDTREE_TRITON_TREE_GDN:-1}"
    export DDTREE_SLOW_TREE_GDN="${DDTREE_SLOW_TREE_GDN:-0}"
    export DDTREE_SLOW_TREE_CONV="${DDTREE_SLOW_TREE_CONV:-0}"
    export DDTREE_INLINE_PARENT_FALLBACK="${DDTREE_INLINE_PARENT_FALLBACK:-0}"
    export DDTREE_ROOT_SIBLING_STATE_OFFSET="${DDTREE_ROOT_SIBLING_STATE_OFFSET:-zero}"
    export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash_attn}"
    export EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:---no-async-scheduling}"
    exec bash "$ROOT/scripts/serve_qwen36_dflash.sh" "$@"
    ;;
  qwen25-m2)
    exec bash "$ROOT/scripts/serve_qwen25_m2.sh" "$@"
    ;;
  ddtree-tree-test)
    exec python3 "$ROOT/tests/test_ddtree_tree.py" "$@"
    ;;
  ddtree-metadata-test)
    exec python3 "$ROOT/tests/test_ddtree_vllm_metadata.py" "$@"
    ;;
  ddtree-m3-bridge-test)
    exec python3 "$ROOT/tests/test_dflash_ddtree_m3_bridge.py" "$@"
    ;;
  ddtree-m4b-payload-test)
    exec python3 "$ROOT/tests/test_dflash_ddtree_m4b_payload.py" "$@"
    ;;
  ddtree-runtime-sampler-test)
    exec python3 "$ROOT/tests/test_ddtree_runtime_sampler.py" "$@"
    ;;
  ddtree-parent-metadata-test)
    exec python3 "$ROOT/tests/test_ddtree_parent_metadata.py" "$@"
    ;;
  ddtree-gdn-reference-test)
    exec python3 "$ROOT/tests/test_ddtree_gdn_reference.py" "$@"
    ;;
  ddtree-vllm-import-test)
    exec python3 - <<'PY'
from vllm.v1.spec_decode.ddtree_tree import build_ddtree
from vllm.v1.spec_decode.ddtree_metadata import TreeVerifierMetadata
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.core.sched.output import SchedulerOutput

tree = build_ddtree(
    [[(11, -0.1), (12, -0.2)], [(21, -0.1), (22, -0.2)]],
    budget=3,
    top_k=2,
)
metadata = TreeVerifierMetadata.from_tree(prompt_len=5, tree=tree)
assert metadata.compact_logits_indices[0] == 4
draft = DraftTokenIds(["req-a"], [list(metadata.tree_token_ids)], draft_trees={"req-a": {"tree_token_ids": list(metadata.tree_token_ids)}})
assert draft.draft_trees["req-a"]["tree_token_ids"][0] == 11
assert SchedulerOutput.make_empty().scheduled_spec_decode_trees == {}
print("vLLM DDTree metadata import test passed")
PY
    ;;
  qwen25-tree-oracle)
    exec python3 "$ROOT/prototypes/qwen25_tree_oracle.py" "$@"
    ;;
  qwen25-tree-oracle-suite)
    exec python3 "$ROOT/prototypes/qwen25_tree_oracle_suite.py" "$@"
    ;;
  qwen25-tree-mask-verifier)
    exec python3 "$ROOT/prototypes/qwen25_tree_mask_verifier.py" "$@"
    ;;
  qwen25-tree-decode-loop)
    exec python3 "$ROOT/prototypes/qwen25_tree_decode_loop.py" "$@"
    ;;
  qwen25-tree-decode-suite)
    exec python3 "$ROOT/prototypes/qwen25_tree_decode_suite.py" "$@"
    ;;
  ddtree-gdn-reference)
    exec python3 "$ROOT/prototypes/ddtree_gdn_reference.py" "$@"
    ;;
  bench)
    exec python3 "$ROOT/scripts/bench_categories_stream.py" "$@"
    ;;
  bash|shell)
    exec bash "$@"
    ;;
  *)
    exec "$MODE" "$@"
    ;;
esac
