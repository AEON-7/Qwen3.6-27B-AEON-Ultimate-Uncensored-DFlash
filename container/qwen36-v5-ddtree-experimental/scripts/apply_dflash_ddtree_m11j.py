#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m11j"


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


def patch_causal_conv_root_parent(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/mamba/ops/causal_conv1d.py"
    replace_exact(
        path,
        """        src_token = tl.where(parent_t < 0, 0, parent_t)
        src_state_offset = tl.where(parent_t < 0, root_state_offset, 0)
        src_state_idx = tl.load(
""",
        """        # aeon_dflash_ddtree_m11j
        # Compact row 0 is the ordinary flat-chain first draft token and must
        # preserve vLLM's rolling-state cursor. Later parent=-1 rows are root
        # siblings; they are children of the pre-tree state and must not reuse
        # the cursor offset left by the prior accepted/bonus shape.
        src_token = tl.where(parent_t < 0, 0, parent_t)
        is_root_sibling = (parent_t < 0) & (idx_token > 0)
        src_state_offset = tl.where(
            is_root_sibling, 0, tl.where(parent_t < 0, root_state_offset, 0)
        )
        src_state_idx = tl.load(
""",
    )


def patch_fused_gdn_root_parent(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/fla/ops/fused_sigmoid_gating.py"
    replace_exact(
        path,
        """            reload_t = tl.where(parent_t < 0, root_t, parent_t)
            should_reload = (i_t > 0) & (parent_t != i_t - 1)
""",
        """            # aeon_dflash_ddtree_m11j
            # Keep cursor replay for compact row 0, but root siblings at
            # compact rows >0 are children of the pre-tree state. Reload row 0
            # for those sibling branches so fused SSM replay matches Lucebox
            # branch semantics.
            is_root_sibling = (parent_t < 0) & (i_t > 0)
            reload_t = tl.where(
                is_root_sibling, 0, tl.where(parent_t < 0, root_t, parent_t)
            )
            should_reload = (i_t > 0) & (parent_t != i_t - 1)
""",
    )


def verify_static(pkg_root: Path) -> None:
    causal = (pkg_root / "model_executor/layers/mamba/ops/causal_conv1d.py").read_text()
    fused = (
        pkg_root / "model_executor/layers/fla/ops/fused_sigmoid_gating.py"
    ).read_text()
    checks = (
        (causal, "causal_conv1d.py", "is_root_sibling = (parent_t < 0) & (idx_token > 0)"),
        (fused, "fused_sigmoid_gating.py", "is_root_sibling = (parent_t < 0) & (i_t > 0)"),
        (causal, "causal_conv1d.py", MARKER),
        (fused, "fused_sigmoid_gating.py", MARKER),
    )
    for text, rel, needle in checks:
        if needle not in text:
            raise RuntimeError(f"M11J verification failed: {rel} missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_causal_conv_root_parent(pkg_root)
    patch_fused_gdn_root_parent(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree root-sibling GDN replay loads pre-tree root state")
    return 0


if __name__ == "__main__":
    sys.exit(main())
