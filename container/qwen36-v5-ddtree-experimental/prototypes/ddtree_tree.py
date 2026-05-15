#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import count
from typing import Callable, Iterable


@dataclass(frozen=True)
class DraftCandidate:
    token_id: int
    logprob: float


@dataclass(frozen=True)
class TreeNode:
    index: int
    parent_index: int | None
    token_id: int
    depth: int
    score: float


@dataclass(frozen=True)
class DDTree:
    nodes: tuple[TreeNode, ...]

    @property
    def non_root_nodes(self) -> tuple[TreeNode, ...]:
        return self.nodes[1:]

    def parent_indices_for_verifier(self) -> tuple[int, ...]:
        """Parent indices for non-root verifier nodes, with -1 meaning root."""
        parents: list[int] = []
        for node in self.non_root_nodes:
            if node.parent_index is None or node.parent_index == 0:
                parents.append(-1)
            else:
                parents.append(node.parent_index - 1)
        return tuple(parents)

    def token_ids_for_verifier(self) -> tuple[int, ...]:
        return tuple(node.token_id for node in self.non_root_nodes)

    def path_token_ids(self, node_index: int) -> tuple[int, ...]:
        path: list[int] = []
        cursor: int | None = node_index
        while cursor is not None and cursor != 0:
            node = self.nodes[cursor]
            path.append(node.token_id)
            cursor = node.parent_index
        path.reverse()
        return tuple(path)

    def ancestor_indices(self, node_index: int, *, include_self: bool = True) -> set[int]:
        ancestors: set[int] = set()
        cursor: int | None = node_index if include_self else self.nodes[node_index].parent_index
        while cursor is not None:
            ancestors.add(cursor)
            cursor = self.nodes[cursor].parent_index
        return ancestors

    def visibility_mask(self) -> tuple[tuple[bool, ...], ...]:
        """Ancestor-only visibility mask, including root and self columns."""
        rows: list[tuple[bool, ...]] = []
        for node in self.nodes:
            visible = self.ancestor_indices(node.index)
            rows.append(tuple(col in visible for col in range(len(self.nodes))))
        return tuple(rows)

    def child_by_token(self, parent_index: int) -> dict[int, int]:
        children: dict[int, int] = {}
        for node in self.non_root_nodes:
            if node.parent_index == parent_index:
                children[node.token_id] = node.index
        return children


@dataclass(frozen=True)
class GreedyTreeWalk:
    accepted_node_indices: tuple[int, ...]
    accepted_token_ids: tuple[int, ...]
    bonus_token_id: int
    visited_node_indices: tuple[int, ...]

    @property
    def output_token_ids(self) -> tuple[int, ...]:
        return self.accepted_token_ids + (self.bonus_token_id,)


def greedy_tree_walk(
    tree: DDTree,
    next_token_for_path: Callable[[tuple[int, ...]], int],
) -> GreedyTreeWalk:
    """Walk the tree with greedy target logits.

    `next_token_for_path` is the verifier oracle: given the accepted path tokens
    up to the current node, return the target model's greedy next token.
    """
    cursor = 0
    accepted_nodes: list[int] = []
    accepted_tokens: list[int] = []
    visited_nodes: list[int] = [0]

    while True:
        path = tree.path_token_ids(cursor)
        next_token = int(next_token_for_path(path))
        child_index = tree.child_by_token(cursor).get(next_token)
        if child_index is None:
            return GreedyTreeWalk(
                accepted_node_indices=tuple(accepted_nodes),
                accepted_token_ids=tuple(accepted_tokens),
                bonus_token_id=next_token,
                visited_node_indices=tuple(visited_nodes),
            )

        accepted_nodes.append(child_index)
        accepted_tokens.append(next_token)
        visited_nodes.append(child_index)
        cursor = child_index


def _normalize_candidates(
    candidates_by_depth: Iterable[Iterable[DraftCandidate | tuple[int, float]]],
    top_k: int,
) -> list[list[DraftCandidate]]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    normalized: list[list[DraftCandidate]] = []
    for depth, raw_candidates in enumerate(candidates_by_depth, start=1):
        candidates: list[DraftCandidate] = []
        for raw in raw_candidates:
            if isinstance(raw, DraftCandidate):
                candidate = raw
            else:
                token_id, logprob = raw
                candidate = DraftCandidate(int(token_id), float(logprob))
            candidates.append(candidate)

        if not candidates:
            raise ValueError(f"depth {depth} has no draft candidates")

        candidates.sort(key=lambda item: (item.logprob, -item.token_id), reverse=True)
        normalized.append(candidates[:top_k])

    if not normalized:
        raise ValueError("candidates_by_depth must contain at least one depth")
    return normalized


def build_ddtree(
    candidates_by_depth: Iterable[Iterable[DraftCandidate | tuple[int, float]]],
    *,
    budget: int,
    top_k: int = 8,
    chain_seed: bool = True,
    min_root_branches: int = 0,
    root_token_id: int = -1,
) -> DDTree:
    """Build a best-first speculative tree from per-depth draft candidates.

    The budget counts non-root verifier nodes. The output shape is intentionally
    close to what vLLM needs later: a flattened tree, parent indices, token ids,
    path scores, and an ancestor-only visibility mask.
    """
    if budget < 1:
        raise ValueError("budget must be >= 1")

    candidates = _normalize_candidates(candidates_by_depth, top_k)
    nodes: list[TreeNode] = [
        TreeNode(index=0, parent_index=None, token_id=root_token_id, depth=0, score=0.0)
    ]
    child_edges: set[tuple[int, int, int]] = set()

    def add_child(parent_index: int, candidate: DraftCandidate) -> TreeNode:
        parent = nodes[parent_index]
        depth = parent.depth + 1
        if depth > len(candidates):
            raise ValueError("cannot add child beyond available candidate depths")
        edge = (parent_index, depth, candidate.token_id)
        if edge in child_edges:
            raise ValueError("duplicate child edge")
        node = TreeNode(
            index=len(nodes),
            parent_index=parent_index,
            token_id=candidate.token_id,
            depth=depth,
            score=parent.score + candidate.logprob,
        )
        child_edges.add(edge)
        nodes.append(node)
        return node

    if chain_seed:
        cursor = 0
        while len(nodes) - 1 < budget and nodes[cursor].depth < len(candidates):
            node = add_child(cursor, candidates[nodes[cursor].depth][0])
            cursor = node.index
    elif min_root_branches > 0:
        for candidate in candidates[0][: min(min_root_branches, len(candidates[0]))]:
            if len(nodes) - 1 >= budget:
                break
            add_child(0, candidate)

    order = count()
    heap: list[tuple[float, int, int, DraftCandidate]] = []

    def push_children(parent_index: int) -> None:
        parent = nodes[parent_index]
        if parent.depth >= len(candidates):
            return
        depth = parent.depth + 1
        for candidate in candidates[parent.depth]:
            edge = (parent_index, depth, candidate.token_id)
            if edge in child_edges:
                continue
            score = parent.score + candidate.logprob
            heapq.heappush(heap, (-score, next(order), parent_index, candidate))

    for node in tuple(nodes):
        push_children(node.index)

    while len(nodes) - 1 < budget and heap:
        _, _, parent_index, candidate = heapq.heappop(heap)
        parent = nodes[parent_index]
        edge = (parent_index, parent.depth + 1, candidate.token_id)
        if edge in child_edges:
            continue
        node = add_child(parent_index, candidate)
        push_children(node.index)

    return DDTree(nodes=tuple(nodes))


def demo_tree() -> DDTree:
    return build_ddtree(
        [
            [(101, -0.05), (102, -0.80), (103, -1.20)],
            [(201, -0.10), (202, -0.35), (203, -1.30)],
            [(301, -0.25), (302, -0.40), (303, -0.90)],
            [(401, -0.20), (402, -0.55), (403, -1.10)],
        ],
        budget=8,
        top_k=3,
    )


if __name__ == "__main__":
    tree = demo_tree()
    print("idx parent depth token score path")
    for node in tree.nodes:
        print(
            node.index,
            node.parent_index,
            node.depth,
            node.token_id,
            round(node.score, 4),
            tree.path_token_ids(node.index),
        )
