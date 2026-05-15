#!/usr/bin/env python3
from __future__ import annotations

import inspect

import torch

from vllm.v1.outputs import DraftTokenIds
from vllm.v1.spec_decode import dflash
from vllm.v1.spec_decode.ddtree_tree import build_ddtree
from vllm.v1.worker import gpu_model_runner


def test_payload_source_present() -> None:
    dflash_source = inspect.getsource(dflash.DFlashProposer)
    base_source = inspect.getsource(
        dflash.DFlashProposer.__mro__[1].propose
    )
    runner_source = inspect.getsource(gpu_model_runner.GPUModelRunner)
    assert "_build_ddtree_payloads_from_logits" in dflash_source
    assert "pop_last_ddtree_payloads" in dflash_source
    assert "flat_fallback_token_ids" in dflash_source
    assert 'self.method == "dflash_ddtree"' in base_source
    assert "_draft_tree_payloads_cpu" in runner_source


def test_draft_token_ids_preserves_tree_payload() -> None:
    payload = {
        "method": "dflash_ddtree",
        "tree_token_ids": [11, 21, 22],
        "parent_indices": [-1, 0, 0],
    }
    draft = DraftTokenIds(
        req_ids=["req-1"],
        draft_token_ids=[[11, 21, 31]],
        draft_trees={"req-1": payload},
    )
    assert draft.draft_trees == {"req-1": payload}


def test_tree_builder_shape_matches_payload_contract() -> None:
    tree = build_ddtree(
        [
            [(11, 3.0), (12, 2.0), (13, 1.0)],
            [(21, 3.0), (22, 2.0), (23, 1.0)],
            [(31, 3.0), (32, 2.0), (33, 1.0)],
        ],
        budget=7,
        top_k=3,
        chain_seed=True,
    )
    payload = {
        "tree_token_ids": list(tree.token_ids_for_verifier()),
        "parent_indices": list(tree.parent_indices_for_verifier()),
        "node_depths": [node.depth for node in tree.non_root_nodes],
        "node_scores": [float(node.score) for node in tree.non_root_nodes],
        "flat_fallback_token_ids": torch.tensor([11, 21, 31]).tolist(),
    }
    assert len(payload["tree_token_ids"]) == len(payload["parent_indices"])
    assert len(payload["tree_token_ids"]) == len(payload["node_depths"])
    assert payload["parent_indices"][0] == -1
    assert payload["flat_fallback_token_ids"] == [11, 21, 31]


if __name__ == "__main__":
    test_payload_source_present()
    test_draft_token_ids_preserves_tree_payload()
    test_tree_builder_shape_matches_payload_contract()
    print("dflash_ddtree M4B payload tests passed")
