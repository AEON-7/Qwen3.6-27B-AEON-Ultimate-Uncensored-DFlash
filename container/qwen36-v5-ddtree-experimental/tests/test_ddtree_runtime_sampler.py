#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototypes"))

from ddtree_runtime_sampler import (  # noqa: E402
    DDTreeRuntimeMetadata,
    PLACEHOLDER_TOKEN_ID,
    greedy_sample_ddtree,
)


def test_walks_sibling_branch_as_vllm_bonus() -> None:
    # Tree:
    # root --10--> node 1
    # root --21--> node 3 --30--> node 4
    #      \-20--> node 2
    payload_by_req_id = {
        "req-a": {
            "tree_token_ids": [10, 20, 21, 30],
            "parent_indices": [-1, 0, -1, 2],
        }
    }
    metadata = DDTreeRuntimeMetadata.from_payloads(["req-a"], payload_by_req_id)
    logits = torch.zeros((5, 64), dtype=torch.float32)
    logits[0, 21] = 1.0  # root chooses sibling node 3.
    logits[3, 30] = 1.0  # node 3 could choose its child.
    logits[4, 42] = 1.0

    sample = greedy_sample_ddtree(metadata, logits)

    # vLLM cannot yet roll Qwen3.6 recurrent state back to arbitrary sibling
    # nodes. The first non-flat accepted token is emitted as the vLLM bonus
    # instead, so its state is computed normally on the next decode step.
    assert sample.output_token_ids.tolist() == [
        [21, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]
    ]
    assert sample.accepted_compact_indices == [[]]
    assert sample.bonus_parent_compact_indices == [0]


def test_batches_variable_tree_sizes() -> None:
    payload_by_req_id = {
        "req-a": {
            "tree_token_ids": [10, 20],
            "parent_indices": [-1, 0],
        },
        "req-b": {
            "tree_token_ids": [31],
            "parent_indices": [-1],
        },
    }
    metadata = DDTreeRuntimeMetadata.from_payloads(["req-a", "req-b"], payload_by_req_id)
    logits = torch.zeros((5, 80), dtype=torch.float32)
    logits[0, 10] = 1.0
    logits[1, 77] = 1.0
    logits[3, 44] = 1.0

    sample = greedy_sample_ddtree(metadata, logits)

    assert sample.output_token_ids.tolist() == [
        [10, 77, PLACEHOLDER_TOKEN_ID],
        [44, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
    ]
    assert sample.accepted_compact_indices == [[1], []]
    assert sample.bonus_parent_compact_indices == [1, 0]


def test_flat_prefix_then_sibling_bonus() -> None:
    payload_by_req_id = {
        "req-a": {
            "tree_token_ids": [10, 20, 21, 30],
            "parent_indices": [-1, 0, 0, 2],
        }
    }
    metadata = DDTreeRuntimeMetadata.from_payloads(["req-a"], payload_by_req_id)
    logits = torch.zeros((5, 64), dtype=torch.float32)
    logits[0, 10] = 1.0
    logits[1, 21] = 1.0
    logits[3, 30] = 1.0

    sample = greedy_sample_ddtree(metadata, logits)

    assert sample.output_token_ids.tolist() == [
        [10, 21, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]
    ]
    assert sample.accepted_compact_indices == [[1]]
    assert sample.bonus_parent_compact_indices == [1]


def test_full_branch_commit_requires_research_flag() -> None:
    old_full = os.environ.get("DDTREE_FULL_BRANCH_COMMIT")
    old_allow = os.environ.get("DDTREE_ALLOW_BRANCH_STATE_COMPACTION")
    old_unsafe = os.environ.get("DDTREE_UNSAFE_FULL_BRANCH_RESEARCH")
    os.environ["DDTREE_FULL_BRANCH_COMMIT"] = "1"
    os.environ["DDTREE_ALLOW_BRANCH_STATE_COMPACTION"] = "1"
    os.environ.pop("DDTREE_UNSAFE_FULL_BRANCH_RESEARCH", None)
    try:
        payload_by_req_id = {
            "req-a": {
                "tree_token_ids": [10, 20, 21, 30],
                "parent_indices": [-1, 0, 0, 2],
            }
        }
        metadata = DDTreeRuntimeMetadata.from_payloads(["req-a"], payload_by_req_id)
        logits = torch.zeros((5, 64), dtype=torch.float32)
        logits[0, 10] = 1.0
        logits[1, 21] = 1.0
        logits[3, 30] = 1.0
        logits[4, 42] = 1.0

        sample = greedy_sample_ddtree(metadata, logits)

        assert sample.output_token_ids.tolist() == [
            [10, 21, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]
        ]
        assert sample.accepted_compact_indices == [[1]]
        assert sample.bonus_parent_compact_indices == [1]

        os.environ["DDTREE_UNSAFE_FULL_BRANCH_RESEARCH"] = "1"
        sample = greedy_sample_ddtree(metadata, logits)

        assert sample.output_token_ids.tolist() == [
            [10, 21, 30, 42, PLACEHOLDER_TOKEN_ID]
        ]
        assert sample.accepted_compact_indices == [[1, 3, 4]]
        assert sample.bonus_parent_compact_indices == [4]
    finally:
        if old_full is None:
            os.environ.pop("DDTREE_FULL_BRANCH_COMMIT", None)
        else:
            os.environ["DDTREE_FULL_BRANCH_COMMIT"] = old_full
        if old_allow is None:
            os.environ.pop("DDTREE_ALLOW_BRANCH_STATE_COMPACTION", None)
        else:
            os.environ["DDTREE_ALLOW_BRANCH_STATE_COMPACTION"] = old_allow
        if old_unsafe is None:
            os.environ.pop("DDTREE_UNSAFE_FULL_BRANCH_RESEARCH", None)
        else:
            os.environ["DDTREE_UNSAFE_FULL_BRANCH_RESEARCH"] = old_unsafe


def test_rejects_malformed_payload() -> None:
    try:
        DDTreeRuntimeMetadata.from_payloads(
            ["req-a"],
            {"req-a": {"tree_token_ids": [1, 2], "parent_indices": [-1]}},
        )
    except ValueError as exc:
        assert "mismatched" in str(exc)
    else:
        raise AssertionError("expected malformed payload to fail")


def main() -> int:
    test_walks_sibling_branch_as_vllm_bonus()
    test_batches_variable_tree_sizes()
    test_flat_prefix_then_sibling_bonus()
    test_full_branch_commit_requires_research_flag()
    test_rejects_malformed_payload()
    print("DDTree runtime sampler tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
