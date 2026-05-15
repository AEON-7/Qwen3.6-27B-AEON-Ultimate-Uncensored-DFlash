#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8l"


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


def patch_dflash_builder_call(pkg_root: Path) -> None:
    path = pkg_root / "v1/spec_decode/dflash.py"
    replace_exact(
        path,
        """            tree = build_ddtree(
                candidates_by_depth,
                budget=min(max_tree_budget, self.num_speculative_tokens * top_k),
                top_k=top_k,
                chain_seed=os.environ.get("DDTREE_CHAIN_SEED", "true").lower()
                in ("1", "true", "yes", "on"),
            )
""",
        """            tree = build_ddtree(
                candidates_by_depth,
                budget=min(max_tree_budget, self.num_speculative_tokens * top_k),
                top_k=top_k,
                chain_seed=os.environ.get("DDTREE_CHAIN_SEED", "true").lower()
                in ("1", "true", "yes", "on"),
                # aeon_dflash_ddtree_m8l
                # Validation knob: force a small number of root alternatives so
                # branch verifier plumbing can be tested even on high-confidence
                # prompts where best-first log-prob scoring naturally forms a
                # pure chain.
                min_root_branches=int(os.environ.get("DDTREE_MIN_ROOT_BRANCHES", "0")),
            )
""",
    )


def verify_static(pkg_root: Path) -> None:
    checks = {
        "v1/spec_decode/ddtree_tree.py": (
            "min_root_branches",
            "elif min_root_branches > 0",
        ),
        "v1/spec_decode/dflash.py": (
            "aeon_dflash_ddtree_m8l",
            "DDTREE_MIN_ROOT_BRANCHES",
            "min_root_branches=int",
        ),
    }
    for rel, needles in checks.items():
        text = (pkg_root / rel).read_text()
        for needle in needles:
            if needle not in text:
                raise RuntimeError(f"Static M8L verification failed: {rel} missing {needle}")


def main() -> int:
    pkg_root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if pkg_root_override:
        pkg_root = Path(pkg_root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_dflash_builder_call(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree forced-root-branch validation knob verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
