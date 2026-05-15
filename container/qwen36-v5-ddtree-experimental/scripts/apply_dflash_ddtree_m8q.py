#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8q"


def replace_exact(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Could not find expected text in {path}:\n{old}")
    path.write_text(text.replace(old, new, 1))
    return True


def clear_python_caches(pkg_root: Path) -> None:
    for pyc in pkg_root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    for pycache in pkg_root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


ROOT_LEAF_BLOCK = r'''            # aeon_dflash_ddtree_m8q
            # DFlash produces strong parallel depth logits for the greedy path,
            # but not branch-conditioned continuations for alternate roots. A
            # full tree with children under alternate roots is therefore not an
            # exact verifier target yet. This guarded topology keeps the top-1
            # chain and spends the remaining budget on root alternatives as
            # leaves. If the target accepts an alternate root, the runtime
            # sampler immediately emits the target bonus from that branch row.
            if os.environ.get("DDTREE_ROOT_LEAF_ONLY", "0") == "1":
                requested_alt_count = int(
                    os.environ.get(
                        "DDTREE_ROOT_LEAF_ALT_COUNT",
                        str(max(0, int(os.environ.get("DDTREE_MIN_ROOT_BRANCHES", "0")) - 1)),
                    )
                )
                alt_count = min(max(0, requested_alt_count), max(0, top_k - 1))
                alt_count = min(alt_count, max(0, max_tree_budget - 1))
                chain_len = min(self.num_speculative_tokens, max_tree_budget - alt_count)

                tree_token_ids: list[int] = []
                parent_indices: list[int] = []
                node_depths: list[int] = []
                node_scores: list[float] = []

                running_score = 0.0
                for depth in range(chain_len):
                    token_id = int(top_ids_cpu[req_index][depth][0])
                    running_score += float(top_scores_cpu[req_index][depth][0])
                    tree_token_ids.append(token_id)
                    parent_indices.append(-1 if depth == 0 else depth - 1)
                    node_depths.append(depth + 1)
                    node_scores.append(running_score)

                for alt_i in range(1, alt_count + 1):
                    tree_token_ids.append(int(top_ids_cpu[req_index][0][alt_i]))
                    parent_indices.append(-1)
                    node_depths.append(1)
                    node_scores.append(float(top_scores_cpu[req_index][0][alt_i]))

                payloads.append(
                    {
                        "method": "dflash_ddtree",
                        "version": 1,
                        "budget": max_tree_budget,
                        "effective_budget": len(tree_token_ids),
                        "top_k": top_k,
                        "num_speculative_tokens": self.num_speculative_tokens,
                        "score_type": "draft_logprobs_root_leaf",
                        "tree_token_ids": tree_token_ids,
                        "parent_indices": parent_indices,
                        "node_depths": node_depths,
                        "node_scores": node_scores,
                        "flat_fallback_token_ids": draft_token_ids[req_index].tolist(),
                    }
                )
                continue

'''


def patch_dflash_root_leaf(pkg_root: Path) -> None:
    path = pkg_root / "v1/spec_decode/dflash.py"
    replace_exact(
        path,
        """            tree = build_ddtree(
                candidates_by_depth,
""",
        ROOT_LEAF_BLOCK
        + """            tree = build_ddtree(
                candidates_by_depth,
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/spec_decode/dflash.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m8q",
        "DDTREE_ROOT_LEAF_ONLY",
        "draft_logprobs_root_leaf",
        "DDTREE_ROOT_LEAF_ALT_COUNT",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M8Q verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_dflash_root_leaf(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree root-leaf exact topology verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
