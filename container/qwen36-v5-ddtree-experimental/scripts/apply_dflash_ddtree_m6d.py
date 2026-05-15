#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m6d"


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


def patch_flex_noncontiguous_cache(pkg_root: Path) -> None:
    path = pkg_root / "v1/attention/backends/flex_attention.py"
    replace_exact(
        path,
        """            key_cache = key_cache.view(-1, self.num_kv_heads, self.head_size)
            value_cache = value_cache.view(-1, self.num_kv_heads, self.head_size)
""",
        """            # aeon_dflash_ddtree_m6d
            # Hybrid Qwen3.6 cache layouts can be non-contiguous after the
            # attention/mamba page-size alignment. FlexAttention only needs a
            # flattened logical view, so use reshape instead of view.
            key_cache = key_cache.reshape(-1, self.num_kv_heads, self.head_size)
            value_cache = value_cache.reshape(-1, self.num_kv_heads, self.head_size)
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/attention/backends/flex_attention.py").read_text()
    for needle in ("aeon_dflash_ddtree_m6d", "key_cache.reshape", "value_cache.reshape"):
        if needle not in text:
            raise RuntimeError(f"Static M6D verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_flex_noncontiguous_cache(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] FlexAttention non-contiguous KV cache fix verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
