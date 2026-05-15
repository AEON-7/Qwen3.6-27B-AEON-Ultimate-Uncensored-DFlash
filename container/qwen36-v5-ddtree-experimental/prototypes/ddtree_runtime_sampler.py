#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import os

import torch


PLACEHOLDER_TOKEN_ID = -1


@dataclass(frozen=True)
class DDTreeRequestRuntime:
    """One request's flattened DDTree verifier payload.

    Compact verifier rows use the Lucebox convention:
    row 0 is root logits, rows 1..N are logits after non-root tree nodes.
    `parent_indices` is indexed by non-root node id and uses -1 for root.
    """

    req_id: str
    tree_token_ids: tuple[int, ...]
    parent_indices: tuple[int, ...]

    @property
    def num_nodes(self) -> int:
        return len(self.tree_token_ids)

    def child_maps(self) -> list[dict[int, int]]:
        children: list[dict[int, int]] = [dict() for _ in range(self.num_nodes + 1)]
        for node_index, (token_id, parent_index) in enumerate(
            zip(self.tree_token_ids, self.parent_indices, strict=True)
        ):
            parent_compact = 0 if parent_index < 0 else parent_index + 1
            children[parent_compact][int(token_id)] = node_index + 1
        return children


@dataclass(frozen=True)
class DDTreeGreedySample:
    output_token_ids: torch.Tensor
    accepted_compact_indices: list[list[int]]
    bonus_parent_compact_indices: list[int]


@dataclass(frozen=True)
class DDTreeRuntimeMetadata:
    requests: tuple[DDTreeRequestRuntime, ...]

    @property
    def max_num_nodes(self) -> int:
        return max((request.num_nodes for request in self.requests), default=0)

    @property
    def num_requests(self) -> int:
        return len(self.requests)

    @classmethod
    def from_payloads(
        cls,
        req_ids: list[str],
        payload_by_req_id: dict[str, Any],
    ) -> "DDTreeRuntimeMetadata":
        requests: list[DDTreeRequestRuntime] = []
        for req_id in req_ids:
            payload = payload_by_req_id.get(req_id)
            if not isinstance(payload, dict):
                continue
            tree_token_ids = tuple(int(x) for x in payload.get("tree_token_ids", ()))
            parent_indices = tuple(int(x) for x in payload.get("parent_indices", ()))
            if not tree_token_ids:
                continue
            if len(tree_token_ids) != len(parent_indices):
                raise ValueError(
                    f"DDTree payload for {req_id} has mismatched token/parent lengths: "
                    f"{len(tree_token_ids)} != {len(parent_indices)}"
                )
            requests.append(
                DDTreeRequestRuntime(
                    req_id=req_id,
                    tree_token_ids=tree_token_ids,
                    parent_indices=parent_indices,
                )
            )
        return cls(requests=tuple(requests))


def _walk_one_tree(
    request: DDTreeRequestRuntime,
    target_argmax: torch.Tensor,
) -> tuple[list[int], list[int], int, int]:
    children = request.child_maps()
    accepted_tokens: list[int] = []
    accepted_compact: list[int] = []
    cursor_compact = 0

    while True:
        next_token = int(target_argmax[cursor_compact].item())
        child_compact = children[cursor_compact].get(next_token)
        if child_compact is None:
            return accepted_tokens, accepted_compact, next_token, cursor_compact
        node_index = child_compact - 1
        accepted_tokens.append(request.tree_token_ids[node_index])
        accepted_compact.append(child_compact)
        cursor_compact = child_compact


def _adapt_tree_walk_to_vllm_contract(
    accepted_tokens: list[int],
    accepted_compact: list[int],
    bonus_token: int,
    bonus_parent: int,
) -> tuple[list[int], list[int], int]:
    """Return emitted tokens, accepted compact ids, and bonus parent.

    vLLM's speculative state update can safely keep recurrent state for a flat
    accepted prefix plus one uncomputed target bonus. Lucebox-style DDTree can
    walk to an arbitrary sibling branch and then roll recurrent state back to
    that sibling using custom tree kernels. Until vLLM has that same rollback
    primitive for Qwen3.6's GDN layers, commit only the contiguous flat prefix
    and turn the first non-flat accepted branch token into vLLM's bonus token.

    Example: accepted compact path [1, 2, 7, 8] becomes accepted [1, 2] plus
    bonus token for compact node 7. The next decode step computes node 7's
    state normally, preserving quality while still letting DDTree choose a
    better branch token at the divergence point.
    """

    flat_prefix_len = 0
    for index, compact in enumerate(accepted_compact):
        if compact == index + 1:
            flat_prefix_len += 1
        else:
            break

    if flat_prefix_len == len(accepted_compact):
        return (
            accepted_tokens + [bonus_token],
            accepted_compact,
            bonus_parent,
        )

    if (
        os.environ.get("DDTREE_FULL_BRANCH_COMMIT", "0") == "1"
        and os.environ.get("DDTREE_ALLOW_BRANCH_STATE_COMPACTION", "0") == "1"
        and os.environ.get("DDTREE_UNSAFE_FULL_BRANCH_RESEARCH", "0") == "1"
    ):
        # aeon_dflash_ddtree_m11a
        # Full branch commit is only safe when the caller also patches vLLM's
        # speculative scheduler with an explicit accepted-token count. In that
        # mode, a non-flat DDTree branch is emitted as computed accepted tokens
        # plus the normal target bonus token while the accepted-count side
        # channel tells the scheduler how many draft nodes were actually
        # accepted. This preserves vLLM's recurrent-state cursor convention:
        # emitted_count == accepted_count + 1, and postprocess_mamba keeps
        # state through the last computed accepted draft node.
        # aeon_dflash_ddtree_m11d
        # Qwen3.6's hybrid GDN recurrent state is not yet safe for arbitrary
        # branch-state commit in vLLM. Keep this path behind an explicit
        # research-only flag so deployable DDTree always falls back to the
        # quality-preserving branch-as-bonus contract.
        # aeon_dflash_ddtree_m11g
        # M10J's no-bonus full-branch experiment broke the stock vLLM
        # accepted+bonus shape. M11G restores that contract while keeping
        # explicit non-flat accepted_compact metadata for branch-state
        # compaction.
        bonus_parent = accepted_compact[-1] if accepted_compact else 0
        if os.environ.get("DDTREE_FULL_BRANCH_SUPPRESS_BONUS", "0") == "1":
            return accepted_tokens, accepted_compact, bonus_parent
        return accepted_tokens + [bonus_token], accepted_compact, bonus_parent

    safe_accepted = accepted_compact[:flat_prefix_len]
    safe_bonus_parent = safe_accepted[-1] if safe_accepted else 0
    emitted = accepted_tokens[:flat_prefix_len] + [accepted_tokens[flat_prefix_len]]
    return emitted, safe_accepted, safe_bonus_parent


def greedy_sample_ddtree(
    metadata: DDTreeRuntimeMetadata,
    compact_logits: torch.Tensor,
) -> DDTreeGreedySample:
    """Greedy DDTree walk over compact target logits.

    Args:
        metadata: Batched tree payloads.
        compact_logits: [sum(1 + request.num_nodes), vocab] target logits in
            request order, each request storing root logits followed by one row
            per non-root tree node.
    """

    if compact_logits.ndim != 2:
        raise ValueError("compact_logits must be [root_plus_tree_nodes, vocab]")
    expected_rows = sum(1 + request.num_nodes for request in metadata.requests)
    if compact_logits.shape[0] != expected_rows:
        raise ValueError(
            f"compact_logits row mismatch: expected {expected_rows}, "
            f"got {compact_logits.shape[0]}"
        )

    max_output_len = metadata.max_num_nodes + 1
    output_token_ids = torch.full(
        (metadata.num_requests, max_output_len),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=compact_logits.device,
    )
    target_argmax = compact_logits.argmax(dim=-1)

    accepted_compact_by_req: list[list[int]] = []
    bonus_parent_by_req: list[int] = []
    offset = 0
    for req_index, request in enumerate(metadata.requests):
        rows = target_argmax[offset : offset + request.num_nodes + 1]
        accepted_tokens, accepted_compact, bonus_token, bonus_parent = _walk_one_tree(
            request, rows
        )
        emitted, reported_accepted, reported_bonus_parent = (
            _adapt_tree_walk_to_vllm_contract(
                accepted_tokens,
                accepted_compact,
                bonus_token,
                bonus_parent,
            )
        )
        if emitted:
            output_token_ids[req_index, : len(emitted)] = torch.tensor(
                emitted,
                dtype=torch.int32,
                device=compact_logits.device,
            )
        accepted_compact_by_req.append(reported_accepted)
        bonus_parent_by_req.append(reported_bonus_parent)
        offset += request.num_nodes + 1

    return DDTreeGreedySample(
        output_token_ids=output_token_ids,
        accepted_compact_indices=accepted_compact_by_req,
        bonus_parent_compact_indices=bonus_parent_by_req,
    )


def demo() -> dict[str, object]:
    payload = {
        "req-a": {
            "tree_token_ids": [10, 20, 21, 30],
            "parent_indices": [-1, 0, 0, 2],
        }
    }
    metadata = DDTreeRuntimeMetadata.from_payloads(["req-a"], payload)
    logits = torch.zeros((5, 64), dtype=torch.float32)
    logits[0, 21] = 1.0
    logits[3, 30] = 1.0
    logits[4, 42] = 1.0
    sample = greedy_sample_ddtree(metadata, logits)
    return {
        "output_token_ids": sample.output_token_ids.tolist(),
        "accepted_compact_indices": sample.accepted_compact_indices,
        "bonus_parent_compact_indices": sample.bonus_parent_compact_indices,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(demo(), indent=2))
