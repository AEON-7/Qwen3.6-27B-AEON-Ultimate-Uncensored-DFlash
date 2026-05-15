#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m11b"


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


def patch_state_counts(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """                if req_i is not None and accepted_compact:
                    state_counts[req_i] = len(accepted_compact) + 1
            return state_counts
""",
        """                if req_i is not None and accepted_compact:
                    if os.environ.get("DDTREE_FULL_BRANCH_COMMIT", "0") == "1":
                        # aeon_dflash_ddtree_m11b
                        # M11A gives the scheduler an explicit accepted count,
                        # so full-branch mode no longer needs to pretend that a
                        # target bonus was emitted. Keep a bias knob for fast
                        # A/B testing: 0 means exact accepted branch count, 1
                        # restores the old accepted+bonus convention.
                        state_bias = int(
                            os.environ.get("DDTREE_FULL_BRANCH_STATE_COUNT_BIAS", "0")
                        )
                        state_counts[req_i] = len(accepted_compact) + state_bias
                    else:
                        state_counts[req_i] = len(accepted_compact) + 1
            return state_counts
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m11b",
        "DDTREE_FULL_BRANCH_STATE_COUNT_BIAS",
        "state_counts[req_i] = len(accepted_compact) + state_bias",
    ):
        if needle not in text:
            raise RuntimeError(f"M11B static verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_state_counts(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] full-branch state-count bias knob installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
