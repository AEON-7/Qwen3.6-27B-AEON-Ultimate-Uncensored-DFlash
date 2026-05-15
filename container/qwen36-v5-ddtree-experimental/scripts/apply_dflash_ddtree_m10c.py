#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m10c"


def clear_python_caches(pkg_root: Path) -> None:
    for pyc in pkg_root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    for pycache in pkg_root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


def patch_gpu_model_runner(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    text = path.read_text()
    old_count = text.count('os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "1")')
    if old_count == 0:
        if 'os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "0")' in text:
            return
        raise RuntimeError("Could not find DDTREE_COMPACT_RECURRENT_STATE default")
    text = text.replace(
        'os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "1")',
        'os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "0")',
    )
    path.write_text(text)


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    if 'os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "1")' in text:
        raise RuntimeError("Static M10C verification failed: compaction still defaults on")
    if 'os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "0")' not in text:
        raise RuntimeError("Static M10C verification failed: missing opt-in compaction default")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_gpu_model_runner(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree recurrent compaction default is opt-in")
    return 0


if __name__ == "__main__":
    sys.exit(main())
