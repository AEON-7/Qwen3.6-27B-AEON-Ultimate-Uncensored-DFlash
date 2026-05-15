#!/usr/bin/env python3
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototypes"))

from ddtree_parent_metadata import (  # noqa: E402
    ROOT_PARENT,
    build_padded_parent_ids,
    full_parent_ids_from_payload,
)


def test_full_parent_ids_from_payload() -> None:
    payload = {
        "tree_token_ids": [10, 20, 21, 30],
        "parent_indices": [-1, 0, 0, 2],
    }
    assert full_parent_ids_from_payload(payload) == [
        ROOT_PARENT,
        ROOT_PARENT,
        1,
        1,
        3,
    ]


def test_padded_parent_ids_preserve_request_order() -> None:
    payloads = {
        "b": {
            "tree_token_ids": [5, 6],
            "parent_indices": [-1, 0],
        },
        "a": {
            "tree_token_ids": [1],
            "parent_indices": [-1],
        },
    }
    metadata = build_padded_parent_ids(["a", "missing", "b"], payloads)
    assert metadata is not None
    assert metadata.request_ids == ("a", "missing", "b")
    assert metadata.num_tree_tokens == (1, 0, 2)
    assert metadata.parent_ids.dtype == torch.int32
    assert metadata.parent_ids.tolist() == [
        [-1, -1, 0],
        [0, 0, 0],
        [-1, -1, 1],
    ]


def test_vllm_patch_source_present() -> None:
    from vllm.v1.attention.backends import gdn_attn
    from vllm.v1.worker import gpu_model_runner

    gdn_source = inspect.getsource(gdn_attn)
    runner_source = inspect.getsource(gpu_model_runner.GPUModelRunner)
    assert "ddtree_parent_ids" in gdn_source
    assert "build_padded_parent_ids" in runner_source
    assert "_last_ddtree_parent_ids_gpu" in runner_source


if __name__ == "__main__":
    test_full_parent_ids_from_payload()
    test_padded_parent_ids_preserve_request_order()
    test_vllm_patch_source_present()
    print("DDTree parent metadata tests passed")
