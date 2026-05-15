#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototypes"))

from ddtree_tree import build_ddtree  # noqa: E402
from ddtree_vllm_metadata import (  # noqa: E402
    TreeVerifierMetadata,
    greedy_sample_from_compact_logits,
    make_batched_metadata,
    make_prefill_tree_attention_mask,
)


def assert_equal(left, right) -> None:
    if left != right:
        raise AssertionError(f"{left!r} != {right!r}")


def make_tree():
    return build_ddtree(
        [
            [(11, -0.01), (12, -0.02)],
            [(21, -0.01), (22, -0.02)],
            [(31, -0.01), (32, -0.02)],
        ],
        budget=5,
        top_k=2,
        chain_seed=True,
    )


def test_single_request_metadata_indices() -> None:
    tree = make_tree()
    metadata = TreeVerifierMetadata.from_tree(prompt_len=7, tree=tree)

    assert_equal(metadata.tree_token_ids[:3], (11, 21, 31))
    assert_equal(metadata.compact_logits_indices[0], 6)
    assert_equal(metadata.compact_logits_indices[1], 7)
    assert_equal(metadata.node_compact_indices, tuple(range(1, tree.non_root_nodes[-1].index + 1)))
    assert_equal(metadata.edge_parent_compact_indices[0], 0)
    assert_equal(metadata.edge_parent_compact_indices[1], 1)
    assert_equal(metadata.tree_position_ids[:3], (7, 8, 9))


def test_attention_mask_hides_siblings() -> None:
    tree = make_tree()
    prompt_len = 5
    mask = make_prefill_tree_attention_mask(
        prompt_len=prompt_len,
        tree=tree,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )[0, 0]

    paths = {node.index: tree.path_token_ids(node.index) for node in tree.nodes}
    node_11 = next(index for index, path in paths.items() if path == (11,))
    node_12 = next(index for index, path in paths.items() if path == (12,))
    row_11 = prompt_len + node_11 - 1
    col_12 = prompt_len + node_12 - 1
    assert mask[row_11, col_12].item() < -1e20

    child_1121 = next(index for index, path in paths.items() if path == (11, 21))
    row_child = prompt_len + child_1121 - 1
    col_parent = prompt_len + node_11 - 1
    assert_equal(mask[row_child, col_parent].item(), 0.0)


def test_batched_metadata_offsets() -> None:
    left = make_tree()
    right = build_ddtree(
        [
            [(101, -0.01), (102, -0.02)],
            [(201, -0.01), (202, -0.02)],
        ],
        budget=3,
        top_k=2,
        chain_seed=True,
    )
    batch = make_batched_metadata(
        prompt_lens=[5, 3],
        trees=[left, right],
        device=torch.device("cpu"),
    )

    left_nodes = len(left.non_root_nodes)
    assert_equal(batch.cu_num_tree_nodes.tolist(), [left_nodes, left_nodes + 3])

    right_parent_indices = batch.parent_indices[left_nodes:].tolist()
    assert_equal(right_parent_indices[0], -1)
    assert_equal(right_parent_indices[1], left_nodes)

    # First request compact logits rows are [root + tree nodes] from rows 4..9.
    # Second request starts after prompt_len + tree nodes = 5 + left_nodes.
    compact = batch.compact_logits_indices.tolist()
    assert_equal(compact[0], 4)
    assert_equal(compact[left_nodes + 1], 5 + left_nodes + 2)

    edge_compact = batch.edge_parent_compact_indices.tolist()
    assert_equal(edge_compact[0], 0)
    assert_equal(edge_compact[left_nodes], left_nodes + 1)


def test_greedy_sample_from_compact_logits() -> None:
    tree = make_tree()
    vocab_size = 128
    logits = torch.full((len(tree.non_root_nodes) + 1, vocab_size), -10.0)
    logits[0, 11] = 10.0
    logits[1, 21] = 10.0
    logits[2, 99] = 10.0

    walk = greedy_sample_from_compact_logits(tree=tree, compact_logits=logits)
    assert_equal(walk.accepted_token_ids, (11, 21))
    assert_equal(walk.bonus_token_id, 99)
    assert_equal(walk.output_token_ids, (11, 21, 99))


def main() -> int:
    test_single_request_metadata_indices()
    test_attention_mask_hides_siblings()
    test_batched_metadata_offsets()
    test_greedy_sample_from_compact_logits()
    print("ddtree_vllm_metadata tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
