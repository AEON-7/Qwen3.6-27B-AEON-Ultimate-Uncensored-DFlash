#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m11h"


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


def patch_causal_conv(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/mamba/ops/causal_conv1d.py"
    replace_exact(
        path,
        """    stride_parent_ids_seq: tl.constexpr,
    stride_parent_ids_tok: tl.constexpr,
    stride_o_token: tl.int64,
""",
        """    stride_parent_ids_seq: tl.constexpr,
    stride_parent_ids_tok: tl.constexpr,
    parent_width: tl.constexpr,
    stride_o_token: tl.int64,
""",
    )
    replace_exact(
        path,
        """        parent_t = tl.load(
            parent_ids_ptr
            + idx_seq * stride_parent_ids_seq
            + idx_token * stride_parent_ids_tok
        ).to(tl.int64)
""",
        """        # aeon_dflash_ddtree_m11h
        # vLLM may hand GDN kernels a state-index window wider than the live
        # DDTree parent tensor during CUDA graph capture or guided decoding.
        # Mask padded parent columns instead of reading past parent_ids.
        fallback_parent_t = idx_token - 1
        parent_t = tl.load(
            parent_ids_ptr
            + idx_seq * stride_parent_ids_seq
            + idx_token * stride_parent_ids_tok,
            mask=idx_token < parent_width,
            other=fallback_parent_t,
        ).to(tl.int64)
""",
    )
    replace_exact(
        path,
        """        stride_parent_ids_seq,
        stride_parent_ids_tok,
        stride_o_token,
""",
        """        stride_parent_ids_seq,
        stride_parent_ids_tok,
        parent_ids.shape[1],
        stride_o_token,
""",
    )


def patch_fused_sigmoid_gating(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/fla/ops/fused_sigmoid_gating.py"
    replace_exact(
        path,
        """    stride_parent_ids_seq: tl.constexpr,
    stride_parent_ids_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
""",
        """    stride_parent_ids_seq: tl.constexpr,
    stride_parent_ids_tok: tl.constexpr,
    parent_width: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
""",
    )
    replace_exact(
        path,
        """            parent_t = tl.load(
                ddtree_parent_ids
                + i_n * stride_parent_ids_seq
                + i_t * stride_parent_ids_tok
            ).to(tl.int64)
""",
        """            # aeon_dflash_ddtree_m11h
            # Guard parent-id loads for padded GDN windows. Padded rows are
            # inactive or flat-chain diagnostic rows, so falling back to
            # i_t - 1 preserves stock replay semantics without OOB reads.
            fallback_parent_t = i_t - 1
            parent_t = tl.load(
                ddtree_parent_ids
                + i_n * stride_parent_ids_seq
                + i_t * stride_parent_ids_tok,
                mask=i_t < parent_width,
                other=fallback_parent_t,
            ).to(tl.int64)
""",
    )
    replace_exact(
        path,
        """    if ddtree_parent_ids is None:
        stride_parent_ids_seq, stride_parent_ids_tok = 1, 1
    elif ddtree_parent_ids.ndim == 1:
        stride_parent_ids_seq, stride_parent_ids_tok = ddtree_parent_ids.stride(0), 1
    else:
        stride_parent_ids_seq, stride_parent_ids_tok = ddtree_parent_ids.stride()
""",
        """    if ddtree_parent_ids is None:
        stride_parent_ids_seq, stride_parent_ids_tok = 1, 1
        parent_width = 0
    elif ddtree_parent_ids.ndim == 1:
        stride_parent_ids_seq, stride_parent_ids_tok = ddtree_parent_ids.stride(0), 1
        parent_width = ddtree_parent_ids.shape[0]
    else:
        stride_parent_ids_seq, stride_parent_ids_tok = ddtree_parent_ids.stride()
        parent_width = ddtree_parent_ids.shape[1]
""",
    )
    replace_exact(
        path,
        """        stride_parent_ids_seq=stride_parent_ids_seq,
        stride_parent_ids_tok=stride_parent_ids_tok,
        INPLACE_FINAL_STATE=inplace_final_state,
""",
        """        stride_parent_ids_seq=stride_parent_ids_seq,
        stride_parent_ids_tok=stride_parent_ids_tok,
        parent_width=parent_width,
        INPLACE_FINAL_STATE=inplace_final_state,
""",
    )


def verify_static(pkg_root: Path) -> None:
    causal = (pkg_root / "model_executor/layers/mamba/ops/causal_conv1d.py").read_text()
    fused = (
        pkg_root / "model_executor/layers/fla/ops/fused_sigmoid_gating.py"
    ).read_text()
    for text, rel in ((causal, "causal_conv1d.py"), (fused, "fused_sigmoid_gating.py")):
        for needle in (
            MARKER,
            "parent_width",
            "fallback_parent_t",
            "mask=",
        ):
            if needle not in text:
                raise RuntimeError(f"M11H verification failed: {rel} missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_causal_conv(pkg_root)
    patch_fused_sigmoid_gating(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] Triton DDTree GDN parent-id loads are width-guarded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
