#!/usr/bin/env python3
"""M11N: build internal DDTree branches without root-sibling commits.

Qwen3.6 quality probes isolated root-sibling branch commits as a failure mode.
This patch adds two guarded builder controls:

* ``DDTREE_CHAIN_SEED_LIMIT``: seed only the first N trunk nodes, then let the
  best-first heap spend the remaining verifier rows on branches.
* ``DDTREE_DISABLE_EXTRA_ROOT_CHILDREN``: after the trunk has a root child,
  skip additional root children so the remaining budget becomes internal
  branches rather than root alternatives.
"""

from __future__ import annotations

from pathlib import Path

import vllm


ROOT = Path(vllm.__file__).resolve().parent
TREE = ROOT / "v1" / "spec_decode" / "ddtree_tree.py"


def replace_exact(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        raise SystemExit(f"pattern not found in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1))


replace_exact(
    TREE,
    "from dataclasses import dataclass\nimport heapq\n",
    "from dataclasses import dataclass\nimport heapq\nimport os\n",
)

replace_exact(
    TREE,
    """    if chain_seed:
        cursor = 0
        while len(nodes) - 1 < budget and nodes[cursor].depth < len(candidates):
            node = add_child(cursor, candidates[nodes[cursor].depth][0])
            cursor = node.index
""",
    """    if chain_seed:
        cursor = 0
        chain_seed_limit = int(os.environ.get("DDTREE_CHAIN_SEED_LIMIT", "0"))
        while len(nodes) - 1 < budget and nodes[cursor].depth < len(candidates):
            if chain_seed_limit > 0 and len(nodes) - 1 >= chain_seed_limit:
                break
            node = add_child(cursor, candidates[nodes[cursor].depth][0])
            cursor = node.index
""",
)

replace_exact(
    TREE,
    """        for candidate in candidates[parent.depth]:
            edge = (parent_index, depth, candidate.token_id)
            if edge in child_edges:
                continue
            score = parent.score + candidate.logprob
            heapq.heappush(heap, (-score, next(order), parent_index, candidate))
""",
    """        for candidate in candidates[parent.depth]:
            if (
                os.environ.get("DDTREE_DISABLE_EXTRA_ROOT_CHILDREN", "0") == "1"
                and parent_index == 0
                and any(node.parent_index == 0 for node in nodes[1:])
            ):
                continue
            edge = (parent_index, depth, candidate.token_id)
            if edge in child_edges:
                continue
            score = parent.score + candidate.logprob
            heapq.heappush(heap, (-score, next(order), parent_index, candidate))
""",
)

text = TREE.read_text()
for needle in (
    "DDTREE_CHAIN_SEED_LIMIT",
    "DDTREE_DISABLE_EXTRA_ROOT_CHILDREN",
):
    if needle not in text:
        raise SystemExit(f"M11N verification failed: missing {needle}")

print("Applied AEON DFlash DDTree M11N internal-branch topology controls")
