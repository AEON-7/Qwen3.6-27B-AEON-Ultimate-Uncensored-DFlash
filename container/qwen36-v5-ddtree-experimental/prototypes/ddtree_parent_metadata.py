#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


ROOT_PARENT = -1
PADDING_PARENT = 0


@dataclass(frozen=True)
class DDTreeParentMetadata:
    parent_ids: torch.Tensor
    request_ids: tuple[str, ...]
    num_tree_tokens: tuple[int, ...]


def full_parent_ids_from_payload(payload: dict[str, Any]) -> list[int]:
    """Return root+tree parent ids for model-state replay.

    Payloads store non-root nodes only:
    - `tree_token_ids[i]` is compact node `i + 1`
    - `parent_indices[i] == -1` means read the pre-tree cached state
    - otherwise `parent_indices[i]` is a non-root node index and maps to
      compact parent `parent_indices[i] + 1`

    This is intentionally different from sampler traversal, where root
    children are looked up under compact cursor 0. Attention/GDN replay needs
    the real state parent, and row 0 is the root-logits row, not an accepted
    draft node.
    """

    tree_token_ids = payload.get("tree_token_ids", ())
    parent_indices = payload.get("parent_indices", ())
    if len(tree_token_ids) != len(parent_indices):
        raise ValueError(
            "DDTree payload has mismatched tree_token_ids/parent_indices: "
            f"{len(tree_token_ids)} != {len(parent_indices)}"
        )

    parents = [ROOT_PARENT]
    for parent in parent_indices:
        parent_int = int(parent)
        parents.append(ROOT_PARENT if parent_int < 0 else parent_int + 1)
    return parents


def build_padded_parent_ids(
    req_ids: list[str],
    payload_by_req_id: dict[str, Any] | None,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.int32,
    pad_to: int | None = None,
) -> DDTreeParentMetadata | None:
    """Build [num_reqs, max_tree_tokens + 1] parent-id tensor for vLLM.

    Rows without a tree payload are filled with `PADDING_PARENT` and reported
    with length 0. GDN metadata later filters to spec-decode rows.
    """

    if not payload_by_req_id:
        return None

    parents_by_req: list[list[int]] = []
    lengths: list[int] = []
    max_len = 0
    found = False
    for req_id in req_ids:
        payload = payload_by_req_id.get(req_id)
        if isinstance(payload, dict) and payload.get("tree_token_ids"):
            parents = full_parent_ids_from_payload(payload)
            found = True
        else:
            parents = []
        parents_by_req.append(parents)
        lengths.append(max(0, len(parents) - 1))
        max_len = max(max_len, len(parents))

    if not found:
        return None

    if pad_to is not None:
        max_len = max(max_len, int(pad_to))
    if max_len < 1:
        max_len = 1

    tensor = torch.full(
        (len(req_ids), max_len),
        PADDING_PARENT,
        dtype=dtype,
        device=device,
    )
    for row, parents in enumerate(parents_by_req):
        if parents:
            if len(parents) > max_len:
                raise ValueError(
                    f"DDTree parent row for {req_ids[row]} has length "
                    f"{len(parents)} > pad_to {max_len}"
                )
            tensor[row, : len(parents)] = torch.tensor(
                parents,
                dtype=dtype,
                device=device,
            )

    return DDTreeParentMetadata(
        parent_ids=tensor,
        request_ids=tuple(req_ids),
        num_tree_tokens=tuple(lengths),
    )


def demo() -> dict[str, object]:
    metadata = build_padded_parent_ids(
        ["req-a", "req-b"],
        {
            "req-a": {
                "tree_token_ids": [11, 21, 22, 31],
                "parent_indices": [-1, 0, 0, 2],
            }
        },
    )
    assert metadata is not None
    return {
        "parent_ids": metadata.parent_ids.tolist(),
        "request_ids": list(metadata.request_ids),
        "num_tree_tokens": list(metadata.num_tree_tokens),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(demo(), indent=2))
