#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m6e"


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


def patch_flex_power2_blocks(pkg_root: Path) -> None:
    path = pkg_root / "v1/attention/backends/flex_attention.py"
    replace_exact(
        path,
        """        supports_small_blocks = is_torch_equal_or_newer("2.9.0.dev0")
        self.direct_build: bool = supports_small_blocks
        self.q_block_size: int = 16 if supports_small_blocks else 128
        self.kv_block_size: int = self.block_size if supports_small_blocks else 128
""",
        """        supports_small_blocks = is_torch_equal_or_newer("2.9.0.dev0")
        # aeon_dflash_ddtree_m6e
        # Qwen3.6 hybrid cache alignment can set attention pages to 864 tokens.
        # FlexAttention direct-build then feeds 864 into tl.arange, which fails
        # because Triton requires a power-of-two range. Correctness mode uses
        # the generic block-mask path with a power-of-two KV block instead.
        block_size_is_power2 = self.block_size > 0 and (
            self.block_size & (self.block_size - 1)
        ) == 0
        force_generic_blocks = (
            os.environ.get("DDTREE_FLEX_TREE_MASK", "0") == "1"
            or not block_size_is_power2
        )
        self.direct_build: bool = supports_small_blocks and not force_generic_blocks
        self.q_block_size: int = 16 if supports_small_blocks else 128
        self.kv_block_size: int = (
            self.block_size if self.direct_build else 128
        )
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/attention/backends/flex_attention.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m6e",
        "force_generic_blocks",
        "self.block_size if self.direct_build else 128",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M6E verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_flex_power2_blocks(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] FlexAttention power-of-two correctness blocks verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
