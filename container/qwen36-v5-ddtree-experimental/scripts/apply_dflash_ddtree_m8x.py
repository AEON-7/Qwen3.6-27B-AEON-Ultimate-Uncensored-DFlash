#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8x"


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


def patch_recurrent_compaction(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """                if (
                    src_compact == dst_compact
                    or src_compact < 0
                    or dst_compact < 0
                    or src_compact >= row.numel()
                    or dst_compact >= row.numel()
                ):
                    continue
""",
        """                # aeon_dflash_ddtree_m8x
                # Even when src_compact == dst_compact, the tree verifier has
                # written the accepted conv window into the compact source
                # block. vLLM's next speculative step reads conv history from
                # the base rolling block at dst_compact, so always perform the
                # conv copy. SSM can still skip the same-block copy below.
                if (
                    src_compact < 0
                    or dst_compact < 0
                    or src_compact >= row.numel()
                    or dst_compact >= row.numel()
                ):
                    continue
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m8x",
        "always perform the",
        "src_compact >= row.numel()",
        "if src_block_id != dst_ssm_block_id:",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M8X verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_recurrent_compaction(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree recurrent conv compaction fix verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
