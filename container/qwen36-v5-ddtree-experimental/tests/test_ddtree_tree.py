#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototypes"))

from ddtree_tree import build_ddtree, greedy_tree_walk  # noqa: E402


def assert_equal(left, right) -> None:
    if left != right:
        raise AssertionError(f"{left!r} != {right!r}")


def test_budget_one_matches_top1() -> None:
    tree = build_ddtree(
        [
            [(11, -0.1), (12, -0.2)],
            [(21, -0.1), (22, -0.2)],
        ],
        budget=1,
    )
    assert_equal(tree.token_ids_for_verifier(), (11,))
    assert_equal(tree.parent_indices_for_verifier(), (-1,))


def test_chain_seed_keeps_top1_path() -> None:
    tree = build_ddtree(
        [
            [(11, -0.1), (12, -0.2)],
            [(21, -0.1), (22, -0.2)],
            [(31, -0.1), (32, -0.2)],
        ],
        budget=3,
        chain_seed=True,
    )
    deepest = max(tree.nodes, key=lambda node: node.depth)
    assert_equal(tree.path_token_ids(deepest.index), (11, 21, 31))


def test_best_first_adds_sibling_branch() -> None:
    tree = build_ddtree(
        [
            [(11, -0.01), (12, -0.02)],
            [(21, -0.01), (22, -2.0)],
            [(31, -0.01), (32, -2.0)],
        ],
        budget=4,
        top_k=2,
        chain_seed=True,
    )
    paths = {tree.path_token_ids(node.index) for node in tree.non_root_nodes}
    assert (12,) in paths


def test_visibility_is_ancestor_only() -> None:
    tree = build_ddtree(
        [
            [(11, -0.01), (12, -0.02)],
            [(21, -0.01), (22, -0.02)],
        ],
        budget=4,
        top_k=2,
        chain_seed=True,
    )
    mask = tree.visibility_mask()
    path_by_index = {node.index: tree.path_token_ids(node.index) for node in tree.nodes}

    sibling_indices = [
        index for index, path in path_by_index.items() if path in {(11,), (12,)}
    ]
    assert_equal(len(sibling_indices), 2)
    left, right = sibling_indices
    assert_equal(mask[left][right], False)
    assert_equal(mask[right][left], False)

    child_11 = next(index for index, path in path_by_index.items() if path == (11, 21))
    node_11 = next(index for index, path in path_by_index.items() if path == (11,))
    assert_equal(mask[child_11][node_11], True)


def test_greedy_tree_walk_accepts_path_and_bonus() -> None:
    tree = build_ddtree(
        [
            [(11, -0.01), (12, -0.02)],
            [(21, -0.01), (22, -0.02)],
            [(31, -0.01), (32, -0.02)],
        ],
        budget=5,
        top_k=2,
        chain_seed=True,
    )

    next_tokens = {
        (): 11,
        (11,): 22,
        (11, 22): 99,
    }
    walk = greedy_tree_walk(tree, lambda path: next_tokens[path])
    assert_equal(walk.accepted_token_ids, (11, 22))
    assert_equal(walk.bonus_token_id, 99)
    assert_equal(walk.output_token_ids, (11, 22, 99))


def main() -> int:
    test_budget_one_matches_top1()
    test_chain_seed_keeps_top1_path()
    test_best_first_adds_sibling_branch()
    test_visibility_is_ancestor_only()
    test_greedy_tree_walk_accepts_path_and_bonus()
    print("ddtree_tree tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
