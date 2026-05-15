#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8k"


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


def patch_dflash_tree_scores(pkg_root: Path) -> None:
    path = pkg_root / "v1/spec_decode/dflash.py"
    replace_exact(
        path,
        """        top_k = min(self.ddtree_top_k, logits.shape[-1])
        top_values, top_ids = torch.topk(logits, k=top_k, dim=-1)
        draft_token_ids = top_ids[:, 0].view(batch_size, self.num_speculative_tokens)
""",
        """        top_k = min(self.ddtree_top_k, logits.shape[-1])
        # aeon_dflash_ddtree_m8k
        # Use normalized log-probabilities for tree expansion scores. Raw
        # logits make cumulative deeper paths dominate and collapse the
        # best-first tree back into a chain.
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        top_values, top_ids = torch.topk(log_probs, k=top_k, dim=-1)
        draft_token_ids = top_ids[:, 0].view(batch_size, self.num_speculative_tokens)
        max_tree_budget = self.ddtree_budget
        if os.environ.get("DDTREE_ALLOW_OVERSIZED_TREE", "0") != "1":
            max_tree_budget = min(max_tree_budget, self.num_speculative_tokens)
""",
    )
    replace_exact(
        path,
        """            tree = build_ddtree(
                candidates_by_depth,
                budget=min(self.ddtree_budget, self.num_speculative_tokens * top_k),
                top_k=top_k,
                chain_seed=True,
            )
""",
        """            tree = build_ddtree(
                candidates_by_depth,
                budget=min(max_tree_budget, self.num_speculative_tokens * top_k),
                top_k=top_k,
                chain_seed=os.environ.get("DDTREE_CHAIN_SEED", "true").lower()
                in ("1", "true", "yes", "on"),
            )
""",
    )
    replace_exact(
        path,
        """                    "budget": self.ddtree_budget,
""",
        """                    "budget": max_tree_budget,
""",
    )


def patch_scheduler_oversize_guard(pkg_root: Path) -> None:
    path = pkg_root / "v1/core/sched/scheduler.py"
    replace_exact(
        path,
        """                tree_token_ids = request.spec_tree.get("tree_token_ids")
                if isinstance(tree_token_ids, list) and tree_token_ids:
                    spec_token_ids = [int(token_id) for token_id in tree_token_ids]
""",
        """                tree_token_ids = request.spec_tree.get("tree_token_ids")
                if isinstance(tree_token_ids, list) and tree_token_ids:
                    # aeon_dflash_ddtree_m8k
                    # vLLM's existing speculative prep kernels size internal
                    # buffers from num_speculative_tokens. Until the tree path
                    # owns those kernels end-to-end, never schedule more
                    # flattened tree nodes than the original draft buffer.
                    if len(tree_token_ids) <= len(spec_token_ids):
                        spec_token_ids = [int(token_id) for token_id in tree_token_ids]
                    else:
                        request.spec_tree = None
""",
    )


def verify_static(pkg_root: Path) -> None:
    checks = {
        "v1/spec_decode/dflash.py": (
            "aeon_dflash_ddtree_m8k",
            "torch.log_softmax",
            "DDTREE_ALLOW_OVERSIZED_TREE",
            "DDTREE_CHAIN_SEED",
            "budget\": max_tree_budget",
        ),
        "v1/core/sched/scheduler.py": (
            "aeon_dflash_ddtree_m8k",
            "len(tree_token_ids) <= len(spec_token_ids)",
        ),
    }
    for rel, needles in checks.items():
        text = (pkg_root / rel).read_text()
        for needle in needles:
            if needle not in text:
                raise RuntimeError(f"Static M8K verification failed: {rel} missing {needle}")


def main() -> int:
    pkg_root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if pkg_root_override:
        pkg_root = Path(pkg_root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_dflash_tree_scores(pkg_root)
    patch_scheduler_oversize_guard(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree log-prob scoring and speculative buffer guard verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
