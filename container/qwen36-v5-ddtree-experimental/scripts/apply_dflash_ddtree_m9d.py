#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m9d"


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


def patch_triton_module(pkg_root: Path) -> None:
    path = pkg_root / "v1/attention/backends/ddtree_branch_triton.py"

    replace_exact(
        path,
        """    max_seq_len: tl.constexpr,
    block_size: tl.constexpr,
    num_heads: tl.constexpr,
""",
        """    max_seq_len: tl.constexpr,
    block_size: tl.constexpr,
    num_heads: tl.constexpr,
""",
    )

    replace_exact(
        path,
        """    block_stride0: tl.constexpr,
    block_stride1: tl.constexpr,
    q_stride_t: tl.constexpr,
""",
        """    block_stride0: tl.constexpr,
    block_stride1: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    q_stride_t: tl.constexpr,
""",
    )

    replace_exact(
        path,
        """    for kv_start in range(0, max_seq_len, BLOCK_N):
        offs_n = kv_start + tl.arange(0, BLOCK_N)
""",
        """    # aeon_dflash_ddtree_m9d
    # Use a dynamic loop over the live sequence length instead of the static
    # model max length. This keeps long-context runs from doing empty work.
    kv_start = 0
    while kv_start < seq_len:
        offs_n = kv_start + tl.arange(0, BLOCK_N)
""",
    )

    replace_exact(
        path,
        """        flat_slots = block_ids * block_size + block_offsets

        k_ptrs = (
            key_cache
            + (flat_slots[:, None] * num_kv_heads + kv_head) * head_size
            + offs_d[None, :]
        )
        v_ptrs = (
            value_cache
            + (flat_slots[:, None] * num_kv_heads + kv_head) * head_size
            + offs_d[None, :]
        )
""",
        """        k_ptrs = (
            key_cache
            + block_ids[:, None] * k_stride_b
            + block_offsets[:, None] * k_stride_t
            + kv_head * k_stride_h
            + offs_d[None, :] * k_stride_d
        )
        v_ptrs = (
            value_cache
            + block_ids[:, None] * v_stride_b
            + block_offsets[:, None] * v_stride_t
            + kv_head * v_stride_h
            + offs_d[None, :] * v_stride_d
        )
""",
    )

    replace_exact(
        path,
        """        m_i = m_new
        l_i = l_new

    out = acc / l_i
""",
        """        m_i = m_new
        l_i = l_new
        kv_start += BLOCK_N

    out = acc / l_i
""",
    )

    replace_exact(
        path,
        """def _flat_key_value_cache(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() != 4:
        raise ValueError(f"expected paged KV cache [blocks, block, heads, dim], got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError("DDTree graph-safe branch attention requires contiguous KV cache")
    return tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])
""",
        """def _validate_paged_key_value_cache(tensor: torch.Tensor) -> None:
    if tensor.dim() != 4:
        raise ValueError(f"expected paged KV cache [blocks, block, heads, dim], got {tuple(tensor.shape)}")
    if tensor.stride(-1) <= 0:
        raise ValueError(f"unsupported KV cache stride {tensor.stride()}")
""",
    )

    replace_exact(
        path,
        """    block_size = key_cache.shape[1]
    flat_k = _flat_key_value_cache(key_cache)
    flat_v = _flat_key_value_cache(value_cache)
""",
        """    block_size = key_cache.shape[1]
    _validate_paged_key_value_cache(key_cache)
    _validate_paged_key_value_cache(value_cache)
    if parent_ids.device != query.device or parent_ids.dtype != torch.int32:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise ValueError("DDTree M9D parent_ids must already be CUDA int32 during graph capture")
        parent_ids = parent_ids.to(device=query.device, dtype=torch.int32)
""",
    )

    replace_exact(
        path,
        """        flat_k,
        flat_v,
""",
        """        key_cache,
        value_cache,
""",
    )

    replace_exact(
        path,
        """        block_stride0=block_table.stride(0),
        block_stride1=block_table.stride(1),
        q_stride_t=query.stride(0),
""",
        """        block_stride0=block_table.stride(0),
        block_stride1=block_table.stride(1),
        k_stride_b=key_cache.stride(0),
        k_stride_t=key_cache.stride(1),
        k_stride_h=key_cache.stride(2),
        k_stride_d=key_cache.stride(3),
        v_stride_b=value_cache.stride(0),
        v_stride_t=value_cache.stride(1),
        v_stride_h=value_cache.stride(2),
        v_stride_d=value_cache.stride(3),
        q_stride_t=query.stride(0),
""",
    )


def verify_static(pkg_root: Path) -> None:
    path = pkg_root / "v1/attention/backends/ddtree_branch_triton.py"
    text = path.read_text()
    for needle in (
        "aeon_dflash_ddtree_m9d",
        "while kv_start < seq_len",
        "k_stride_b=key_cache.stride(0)",
        "_validate_paged_key_value_cache",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M9D verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_triton_module(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree paged-KV graph-safe Triton verifier installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
