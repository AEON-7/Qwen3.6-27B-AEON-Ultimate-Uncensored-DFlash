#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m10m"


def replace_exact(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Could not find expected text in {path}:\n{old}")
    path.write_text(text.replace(old, new, 1))


def clear_python_caches(pkg_root: Path) -> None:
    for pyc in pkg_root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    for pycache in pkg_root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


def patch_gpu_model_runner(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """        if (
            os.environ.get("DDTREE_USE_RUNTIME_SAMPLER", "0") == "1"
            and ddtree_payload
            and sampling_metadata.all_greedy
        ):
""",
        """        if (
            os.environ.get("DDTREE_USE_RUNTIME_SAMPLER", "0") == "1"
            and ddtree_payload
            and (
                sampling_metadata.all_greedy
                or os.environ.get("DDTREE_FORCE_GREEDY_TREE_SAMPLER", "1") == "1"
            )
        ):
            # aeon_dflash_ddtree_m10m
            # The stock rejection sampler assumes flat speculative rows. A
            # DDTree verifier batch is root+tree rows, so falling through here
            # on temperature/top-p requests corrupts output. Until stochastic
            # tree sampling lands, route all DDTree verifier batches through the
            # tree-aware greedy sampler by default. This preserves text quality
            # and avoids silently treating a tree as a flat chain.
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m10m",
        "DDTREE_FORCE_GREEDY_TREE_SAMPLER",
        "The stock rejection sampler assumes flat speculative rows",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M10M verification failed: missing {needle}")


def main() -> int:
    root_override = __import__("os").environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_gpu_model_runner(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree non-greedy fallback guard verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
