#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8z"


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


def patch_flex_cudagraph_step_markers(pkg_root: Path) -> None:
    path = pkg_root / "v1/attention/backends/flex_attention.py"
    replace_exact(
        path,
        """        # aeon_dflash_ddtree_m8y
        # The M6F eager correctness path materializes dense Flex scores over
        # paged KV slots and OOMs on real Qwen3.6 verifier batches. Keep the
        # ancestor mask but force compiled block-mask creation so Flex remains
        # sparse/fused for DDTree branch verification.
        create_block_mask_fn = create_block_mask_compiled
        return create_block_mask_fn(
""",
        """        # aeon_dflash_ddtree_m8z
        # The M6F eager correctness path materializes dense Flex scores over
        # paged KV slots and OOMs on real Qwen3.6 verifier batches. Keep the
        # ancestor mask but force compiled block-mask creation so Flex remains
        # sparse/fused for DDTree branch verification. Dynamic DDTree masks
        # repeatedly invoke compiled Flex helpers, so explicitly mark graph
        # step boundaries to avoid reusing overwritten CUDA graph outputs.
        if os.environ.get("DDTREE_FLEX_TREE_MASK", "0") == "1":
            torch.compiler.cudagraph_mark_step_begin()
        create_block_mask_fn = create_block_mask_compiled
        return create_block_mask_fn(
""",
    )
    replace_exact(
        path,
        """        # aeon_dflash_ddtree_m8y
        # DDTree branch verification must not fall back to unfused Flex: the
        # unfused math path repeats KV heads and allocates a dense score matrix.
        # Use the compiled Flex kernel with the dynamic ancestor block mask.
        flex_attention_fn = flex_attention_compiled
        out = flex_attention_fn(
""",
        """        # aeon_dflash_ddtree_m8z
        # DDTree branch verification must not fall back to unfused Flex: the
        # unfused math path repeats KV heads and allocates a dense score matrix.
        # Use the compiled Flex kernel with the dynamic ancestor block mask and
        # mark a fresh CUDA graph step before consuming the block mask.
        if os.environ.get("DDTREE_FLEX_TREE_MASK", "0") == "1":
            torch.compiler.cudagraph_mark_step_begin()
        flex_attention_fn = flex_attention_compiled
        out = flex_attention_fn(
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/attention/backends/flex_attention.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m8z",
        "torch.compiler.cudagraph_mark_step_begin()",
        "create_block_mask_fn = create_block_mask_compiled",
        "flex_attention_fn = flex_attention_compiled",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M8Z verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_flex_cudagraph_step_markers(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree compiled Flex CUDA graph step markers verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
