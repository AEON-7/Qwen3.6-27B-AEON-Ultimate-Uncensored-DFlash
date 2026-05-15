#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m6f"


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


def patch_flex_uncompiled_correctness_path(pkg_root: Path) -> None:
    path = pkg_root / "v1/attention/backends/flex_attention.py"
    replace_exact(
        path,
        """        return create_block_mask_compiled(
            mask_mod,
            None,
            None,
            self.num_actual_tokens,
            kv_len,
            device=self.block_table.device,
            BLOCK_SIZE=(self.q_block_size, self.kv_block_size),
        )
""",
        """        # aeon_dflash_ddtree_m6f
        # DDTree correctness mode threads dynamic ancestor masks into Flex.
        # Torch-compiled block-mask creation can retain graph outputs across
        # warmup/smoke invocations on this path, so use the eager helper only
        # while the guarded tree-mask mode is enabled.
        create_block_mask_fn = (
            create_block_mask
            if os.environ.get("DDTREE_FLEX_TREE_MASK", "0") == "1"
            else create_block_mask_compiled
        )
        return create_block_mask_fn(
            mask_mod,
            None,
            None,
            self.num_actual_tokens,
            kv_len,
            device=self.block_table.device,
            BLOCK_SIZE=(self.q_block_size, self.kv_block_size),
        )
""",
    )
    replace_exact(
        path,
        """        out = flex_attention_compiled(
            query,
            key_tensor,
            value_tensor,
            attn_metadata.transformed_score_mod,
            attn_metadata.block_mask,
            self.scale,
            enable_gqa=enable_gqa,
            kernel_options=kernel_options,
        )
""",
        """        # aeon_dflash_ddtree_m6f
        # Keep the normal compiled Flex fast path for every deployment except
        # the experimental DDTree ancestor-mask verifier. That verifier carries
        # per-request tree metadata and is easier to validate without Dynamo's
        # CUDA-graph output lifetime assumptions in the loop.
        flex_attention_fn = (
            flex_attention
            if os.environ.get("DDTREE_FLEX_TREE_MASK", "0") == "1"
            else flex_attention_compiled
        )
        out = flex_attention_fn(
            query,
            key_tensor,
            value_tensor,
            attn_metadata.transformed_score_mod,
            attn_metadata.block_mask,
            self.scale,
            enable_gqa=enable_gqa,
            kernel_options=kernel_options,
        )
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/attention/backends/flex_attention.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m6f",
        "create_block_mask_fn = (",
        "flex_attention_fn = (",
        "else create_block_mask_compiled",
        "else flex_attention_compiled",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M6F verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_flex_uncompiled_correctness_path(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] FlexAttention DDTree uncompiled correctness path verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
