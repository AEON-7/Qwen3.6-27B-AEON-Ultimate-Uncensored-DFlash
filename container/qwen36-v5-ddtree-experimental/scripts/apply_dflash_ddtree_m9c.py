#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m9c"


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


GRAPH_SAFE_TRITON_MODULE = r'''# SPDX-License-Identifier: Apache-2.0
"""Graph-safe DDTree branch verifier attention correction."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ddtree_window_attention_kernel(
    query,
    key_cache,
    value_cache,
    output,
    block_table,
    parent_ids,
    query_start_loc,
    seq_lens,
    max_q_len: tl.constexpr,
    max_seq_len: tl.constexpr,
    block_size: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    scale: tl.constexpr,
    sliding_left: tl.constexpr,
    sliding_right: tl.constexpr,
    parent_stride0: tl.constexpr,
    parent_stride1: tl.constexpr,
    block_stride0: tl.constexpr,
    block_stride1: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    o_stride_t: tl.constexpr,
    o_stride_h: tl.constexpr,
    o_stride_d: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    req_i = tl.program_id(0)
    row = tl.program_id(1)
    q_head = tl.program_id(2)
    kv_head = q_head // num_queries_per_kv

    q_start = tl.load(query_start_loc + req_i).to(tl.int32)
    q_end = tl.load(query_start_loc + req_i + 1).to(tl.int32)
    q_len = q_end - q_start
    seq_len = tl.load(seq_lens + req_i).to(tl.int32)
    context_len = seq_len - q_len
    active = row < q_len

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_size
    q_ptrs = query + (q_start + row) * q_stride_t + q_head * q_stride_h + offs_d * q_stride_d
    q_vec = tl.load(q_ptrs, mask=d_mask & active, other=0.0).to(tl.float32)

    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    q_abs = context_len + row

    for kv_start in range(0, max_seq_len, BLOCK_N):
        offs_n = kv_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        local = offs_n - context_len

        visible = (offs_n < context_len) & n_mask & active
        cur = row
        for _ in range(max_q_len):
            cur_valid = active & (cur >= 0) & (cur < q_len)
            visible = visible | (
                (local == cur) & (local >= 0) & (local < q_len) & n_mask & cur_valid
            )
            safe_cur = tl.maximum(tl.minimum(cur, max_q_len - 1), 0)
            parent_ptr = parent_ids + req_i * parent_stride0 + safe_cur * parent_stride1
            cur = tl.load(parent_ptr, mask=cur_valid, other=-1).to(tl.int32)

        if sliding_left >= 0:
            visible = visible & (offs_n >= (q_abs - sliding_left))
        if sliding_right >= 0:
            visible = visible & (offs_n <= (q_abs + sliding_right))

        block_ptrs = block_table + req_i * block_stride0 + (offs_n // block_size) * block_stride1
        block_ids = tl.load(block_ptrs, mask=n_mask, other=0).to(tl.int64)
        block_offsets = offs_n - (offs_n // block_size) * block_size
        flat_slots = block_ids * block_size + block_offsets

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
        kv_mask = n_mask[:, None] & d_mask[None, :]
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        scores = tl.sum(k * q_vec[None, :], axis=1) * scale
        scores = tl.where(visible, scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
        l_i = l_new

    out = acc / l_i
    o_ptrs = output + (q_start + row) * o_stride_t + q_head * o_stride_h + offs_d * o_stride_d
    tl.store(o_ptrs, out, mask=d_mask & active)


def _flat_key_value_cache(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() != 4:
        raise ValueError(f"expected paged KV cache [blocks, block, heads, dim], got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError("DDTree graph-safe branch attention requires contiguous KV cache")
    return tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])


def ddtree_branch_attention_correction(
    impl,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    attn_metadata,
    parent_ids: torch.Tensor,
) -> bool:
    """Overwrite every verifier row with an exact DDTree ancestor-mask result.

    This intentionally avoids Tensor.item(), torch.nonzero(), or temporary CUDA
    allocations so the path can be captured in vLLM decode CUDA graphs.
    """

    if impl.alibi_slopes is not None:
        raise ValueError("DDTree Triton branch attention does not support ALiBi")
    if getattr(impl, "logits_soft_cap", None):
        raise ValueError("DDTree Triton branch attention does not support softcap")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported query dtype {query.dtype}")
    if key_cache.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported key cache dtype {key_cache.dtype}")
    if value_cache.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported value cache dtype {value_cache.dtype}")
    if impl.head_size > 256:
        raise ValueError(f"unsupported head size {impl.head_size}")
    if parent_ids is None or parent_ids.ndim != 2 or parent_ids.shape[1] <= 0:
        return True

    query_start_loc = attn_metadata.query_start_loc
    seq_lens = attn_metadata.seq_lens
    block_table = attn_metadata.block_table
    block_size = key_cache.shape[1]
    flat_k = _flat_key_value_cache(key_cache)
    flat_v = _flat_key_value_cache(value_cache)

    num_reqs = min(parent_ids.shape[0], query_start_loc.shape[0] - 1)
    max_q_len = int(parent_ids.shape[1])
    if num_reqs <= 0 or max_q_len <= 0:
        return True

    max_seq_len = int(getattr(attn_metadata, "max_seq_len", 0) or 0)
    if max_seq_len <= 0:
        max_seq_len = int(block_table.shape[1]) * int(block_size)

    sliding_left = -1
    sliding_right = -1
    if impl.sliding_window is not None:
        sliding_left = int(impl.sliding_window[0])
        sliding_right = int(impl.sliding_window[1])

    block_d = triton.next_power_of_2(impl.head_size)
    if block_d > 256:
        raise ValueError(f"unsupported rounded head size {block_d}")

    grid = (num_reqs, max_q_len, impl.num_heads)
    _ddtree_window_attention_kernel[grid](
        query,
        flat_k,
        flat_v,
        output,
        block_table,
        parent_ids.to(device=query.device, dtype=torch.int32),
        query_start_loc,
        seq_lens,
        max_q_len=max_q_len,
        max_seq_len=max_seq_len,
        block_size=block_size,
        num_heads=impl.num_heads,
        num_kv_heads=impl.num_kv_heads,
        head_size=impl.head_size,
        num_queries_per_kv=impl.num_queries_per_kv,
        scale=float(impl.scale),
        sliding_left=sliding_left,
        sliding_right=sliding_right,
        parent_stride0=parent_ids.stride(0),
        parent_stride1=parent_ids.stride(1),
        block_stride0=block_table.stride(0),
        block_stride1=block_table.stride(1),
        q_stride_t=query.stride(0),
        q_stride_h=query.stride(1),
        q_stride_d=query.stride(2),
        o_stride_t=output.stride(0),
        o_stride_h=output.stride(1),
        o_stride_d=output.stride(2),
        BLOCK_N=64,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return True
'''


def patch_triton_module(pkg_root: Path) -> None:
    path = pkg_root / "v1/attention/backends/ddtree_branch_triton.py"
    path.write_text(GRAPH_SAFE_TRITON_MODULE)


def patch_parent_mode(pkg_root: Path) -> None:
    path = pkg_root / "v1/attention/backends/flash_attn.py"
    replace_exact(
        path,
        """    def _ddtree_parent_mode(
        self,
        attn_metadata: FlashAttentionMetadata,
    ) -> str:
        parent_ids = attn_metadata.ddtree_parent_ids
        if parent_ids is None:
            return "none"
        query_start_loc = attn_metadata.query_start_loc
        saw_tree = False
        max_rows = min(parent_ids.shape[0], query_start_loc.shape[0] - 1)
        for req_i in range(max_rows):
            q_start = int(query_start_loc[req_i].item())
            q_end = int(query_start_loc[req_i + 1].item())
            q_len = q_end - q_start
            if q_len <= 0 or q_len > parent_ids.shape[1]:
                continue
            row = parent_ids[req_i, :q_len].detach().cpu()
            if row.numel() == 0 or int(row[0].item()) >= 0:
                continue
            saw_tree = True
            expected = torch.arange(q_len, dtype=row.dtype, device=row.device) - 1
            expected[0] = -1
            if not torch.equal(row, expected):
                return "branch"
        return "flat_chain" if saw_tree else "none"
""",
        """    def _ddtree_parent_mode(
        self,
        attn_metadata: FlashAttentionMetadata,
    ) -> str:
        parent_ids = attn_metadata.ddtree_parent_ids
        if parent_ids is None:
            return "none"
        # aeon_dflash_ddtree_m9c
        # CUDA graph capture cannot tolerate Tensor.item() on CUDA metadata.
        # When the graph-safe Triton verifier is enabled, force the branch path
        # during capture so the kernel launch is recorded; the kernel reads the
        # real parent tensor values at replay time.
        if (
            os.environ.get("DDTREE_TRITON_BRANCH_ATTN", "0") == "1"
            and torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        ):
            return "branch"
        query_start_loc = attn_metadata.query_start_loc
        saw_tree = False
        max_rows = min(parent_ids.shape[0], query_start_loc.shape[0] - 1)
        for req_i in range(max_rows):
            q_start = int(query_start_loc[req_i].item())
            q_end = int(query_start_loc[req_i + 1].item())
            q_len = q_end - q_start
            if q_len <= 0 or q_len > parent_ids.shape[1]:
                continue
            row = parent_ids[req_i, :q_len].detach().cpu()
            if row.numel() == 0 or int(row[0].item()) >= 0:
                continue
            saw_tree = True
            expected = torch.arange(q_len, dtype=row.dtype, device=row.device) - 1
            expected[0] = -1
            if not torch.equal(row, expected):
                return "branch"
        return "flat_chain" if saw_tree else "none"
""",
    )


def verify_static(pkg_root: Path) -> None:
    module_text = (pkg_root / "v1/attention/backends/ddtree_branch_triton.py").read_text()
    flash_text = (pkg_root / "v1/attention/backends/flash_attn.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m9c",
        "torch.cuda.is_current_stream_capturing()",
    ):
        if needle not in flash_text:
            raise RuntimeError(f"Static M9C flash_attn verification failed: missing {needle}")
    for needle in (
        "_ddtree_window_attention_kernel",
        "query_start_loc",
        "max_seq_len",
        "avoids Tensor.item()",
    ):
        if needle not in module_text:
            raise RuntimeError(f"Static M9C module verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_triton_module(pkg_root)
    patch_parent_mode(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree graph-safe Triton verifier installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
