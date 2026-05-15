#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8a"


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


DDTREE_CONV = r'''

@triton.jit()
def _causal_conv1d_update_ddtree_kernel(
    x_ptr,  # (num_tokens, dim)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,  # (num_cache_lines, dim, width - 1)
    conv_state_indices_ptr,  # (batch, max_query_len)
    parent_ids_ptr,  # (batch, max_query_len)
    query_start_loc_ptr,  # (batch + 1)
    o_ptr,  # (num_tokens, dim)
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,
    stride_x_token: tl.int64,
    stride_x_dim: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices_seq: tl.constexpr,
    stride_state_indices_tok: tl.constexpr,
    stride_parent_ids_seq: tl.constexpr,
    stride_parent_ids_tok: tl.constexpr,
    stride_o_token: tl.int64,
    stride_o_dim: tl.constexpr,
    null_block_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_NULL_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # aeon_dflash_ddtree_m8a
    # DDTree variant of causal_conv1d_update. Each compact verifier node loads
    # its convolution window from its actual tree parent and writes the resulting
    # state into that node's own state-index slot. This mirrors the M6B Python
    # replay path and Lucebox's ssm_conv_tree CUDA kernel.
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_feats = idx_feats < dim

    query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
    query_end_index = tl.load(query_start_loc_ptr + idx_seq + 1).to(tl.int64)
    actual_seqlen = query_end_index - query_start_index

    for idx_token in tl.range(seqlen):
        valid_token = idx_token < actual_seqlen

        state_ptr = (
            conv_state_indices_ptr
            + idx_seq * stride_state_indices_seq
            + idx_token * stride_state_indices_tok
        )
        dst_state_idx = tl.load(state_ptr).to(tl.int64)
        active = valid_token
        if HAS_NULL_BLOCK:
            active = active & (dst_state_idx != null_block_id)

        parent_t = tl.load(
            parent_ids_ptr
            + idx_seq * stride_parent_ids_seq
            + idx_token * stride_parent_ids_tok
        ).to(tl.int64)
        src_token = tl.where(parent_t < 0, idx_token, parent_t)
        src_state_idx = tl.load(
            conv_state_indices_ptr
            + idx_seq * stride_state_indices_seq
            + src_token * stride_state_indices_tok
        ).to(tl.int64)
        if HAS_NULL_BLOCK:
            active = active & (src_state_idx != null_block_id)

        x_row = query_start_index + idx_token
        x_base = x_ptr + x_row * stride_x_token + idx_feats * stride_x_dim
        raw = tl.load(x_base, mask=mask_feats & active, other=0.0)

        if HAS_BIAS:
            acc = tl.load(bias_ptr + idx_feats, mask=mask_feats, other=0.0).to(
                tl.float32
            )
        else:
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for j in tl.static_range(0, KERNEL_WIDTH):
            w_j = tl.load(
                w_ptr + idx_feats * stride_w_dim + j * stride_w_width,
                mask=mask_feats,
                other=0.0,
            ).to(tl.float32)
            if j == KERNEL_WIDTH - 1:
                x_j = raw.to(tl.float32)
            else:
                x_j = tl.load(
                    conv_state_ptr
                    + src_state_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim
                    + j * stride_conv_state_tok,
                    mask=mask_feats & active,
                    other=0.0,
                ).to(tl.float32)
            acc += x_j * w_j

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))

        tl.store(
            o_ptr + x_row * stride_o_token + idx_feats * stride_o_dim,
            acc,
            mask=mask_feats & active,
        )

        for j in tl.static_range(0, KERNEL_WIDTH - 1):
            if j == KERNEL_WIDTH - 2:
                new_state = raw
            else:
                new_state = tl.load(
                    conv_state_ptr
                    + src_state_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim
                    + (j + 1) * stride_conv_state_tok,
                    mask=mask_feats & active,
                    other=0.0,
                )
            tl.store(
                conv_state_ptr
                + dst_state_idx * stride_conv_state_seq
                + idx_feats * stride_conv_state_dim
                + j * stride_conv_state_tok,
                new_state,
                mask=mask_feats & active,
            )


def causal_conv1d_update_ddtree(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    parent_ids: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
):
    """Tree-parent-aware causal conv update for DDTree verifier windows."""
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    assert conv_state_indices is not None
    assert parent_ids is not None
    assert query_start_loc is not None
    assert x.dim() == 2
    assert conv_state_indices.dim() == 2
    assert parent_ids.dim() == 2

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    out = torch.empty_like(x)

    batch = conv_state_indices.size(0)
    dim = x.size(1)
    seqlen = conv_state_indices.size(1)
    _, width = weight.shape
    num_cache_lines, _, state_len = conv_state.size()
    assert state_len >= width - 1

    stride_x_token, stride_x_dim = x.stride()
    stride_o_token, stride_o_dim = out.stride()
    stride_w_dim, stride_w_width = weight.stride()
    (
        stride_conv_state_seq,
        stride_conv_state_dim,
        stride_conv_state_tok,
    ) = conv_state.stride()
    stride_state_indices_seq, stride_state_indices_tok = conv_state_indices.stride()
    stride_parent_ids_seq, stride_parent_ids_tok = parent_ids.stride()

    def grid(META):
        return (batch, triton.cdiv(dim, META["BLOCK_N"]))

    _causal_conv1d_update_ddtree_kernel[grid](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        parent_ids,
        query_start_loc,
        out,
        batch,
        dim,
        seqlen,
        width - 1,
        num_cache_lines,
        stride_x_token,
        stride_x_dim,
        stride_w_dim,
        stride_w_width,
        stride_conv_state_seq,
        stride_conv_state_dim,
        stride_conv_state_tok,
        stride_state_indices_seq,
        stride_state_indices_tok,
        stride_parent_ids_seq,
        stride_parent_ids_tok,
        stride_o_token,
        stride_o_dim,
        null_block_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_NULL_BLOCK=null_block_id is not None,
        BLOCK_N=256,
    )
    return out.to(original_x_dtype)
'''


def patch_causal_conv(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/mamba/ops/causal_conv1d.py"
    replace_exact(
        path,
        "\ndef causal_conv1d_update(\n",
        DDTREE_CONV + "\n\ndef causal_conv1d_update(\n",
    )


def patch_fused_gdn(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/fla/ops/fused_sigmoid_gating.py"
    replace_exact(
        path,
        '''        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
    }
)
''',
        '''        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
        "IS_DDTREE": lambda args: args["ddtree_parent_ids"] is not None,
    }
)
''',
    )
    replace_exact(
        path,
        """    ssm_state_indices,
    num_accepted_tokens,
    scale,
""",
        """    ssm_state_indices,
    num_accepted_tokens,
    ddtree_parent_ids,
    scale,
""",
    )
    replace_exact(
        path,
        """    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
""",
        """    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    stride_parent_ids_seq: tl.constexpr,
    stride_parent_ids_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
""",
    )
    replace_exact(
        path,
        """    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
):
""",
        """    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_DDTREE: tl.constexpr,
    IS_KDA: tl.constexpr,
):
""",
    )
    replace_exact(
        path,
        """        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
""",
        """        if IS_CONTINUOUS_BATCHING:
            if IS_DDTREE:
                i_t = 0
            elif IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
""",
    )
    replace_exact(
        path,
        """    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
""",
        """    for i_t in range(0, T):
        # aeon_dflash_ddtree_m8a
        # Parent-aware state reload for tree verifier windows. Stock vLLM
        # assumes the verifier rows form one flat chain; DDTree rows are a DFS
        # tree, so a sibling must reload the recurrent state from its parent
        # compact row before applying the token update.
        if IS_DDTREE:
            parent_t = tl.load(
                ddtree_parent_ids
                + i_n * stride_parent_ids_seq
                + i_t * stride_parent_ids_tok
            ).to(tl.int64)
            reload_t = tl.where(parent_t < 0, i_t, parent_t)
            should_reload = (i_t > 0) & (parent_t != i_t - 1)
            reload_state_idx = tl.load(
                ssm_state_indices
                + i_n * stride_indices_seq
                + reload_t * stride_indices_tok
            ).to(tl.int64)
            p_h_reload = h0 + reload_state_idx * stride_init_state_token
            p_h_reload = p_h_reload + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            reload_h = tl.load(
                p_h_reload,
                mask=mask_h & (reload_state_idx > 0),
                other=0,
            ).to(tl.float32)
            b_h = tl.where(should_reload & (reload_state_idx > 0), reload_h, b_h)

        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
""",
    )
    replace_exact(
        path,
        """    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
):
""",
        """    num_accepted_tokens: torch.Tensor | None = None,
    ddtree_parent_ids: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
):
""",
    )
    replace_exact(
        path,
        """    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * HV)
""",
        """    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    if ddtree_parent_ids is None:
        stride_parent_ids_seq, stride_parent_ids_tok = 1, 1
    elif ddtree_parent_ids.ndim == 1:
        stride_parent_ids_seq, stride_parent_ids_tok = ddtree_parent_ids.stride(0), 1
    else:
        stride_parent_ids_seq, stride_parent_ids_tok = ddtree_parent_ids.stride()

    grid = (NK, NV, N * HV)
""",
    )
    replace_exact(
        path,
        """        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
""",
        """        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        ddtree_parent_ids=ddtree_parent_ids,
        scale=scale,
""",
    )
    replace_exact(
        path,
        """        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        INPLACE_FINAL_STATE=inplace_final_state,
""",
        """        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        stride_parent_ids_seq=stride_parent_ids_seq,
        stride_parent_ids_tok=stride_parent_ids_tok,
        INPLACE_FINAL_STATE=inplace_final_state,
""",
    )


def patch_gdn_linear_attention(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py"
    replace_exact(
        path,
        """logger = init_logger(__name__)
""",
        """logger = init_logger(__name__)
_DDTREE_TRITON_GDN_LOGGED = False
_DDTREE_TRITON_GDN_MISSING_PARENT_LOGGED = False
""",
    )
    replace_exact(
        path,
        """from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
""",
        """from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
    causal_conv1d_update_ddtree,
)
""",
    )
    replace_exact(
        path,
        """            os.environ.get("DDTREE_SLOW_TREE_GDN", "0") == "1"
            and attn_metadata.ddtree_parent_ids is not None
""",
        """            os.environ.get("DDTREE_SLOW_TREE_GDN", "0") == "1"
            and os.environ.get("DDTREE_TRITON_TREE_GDN", "0") != "1"
            and attn_metadata.ddtree_parent_ids is not None
""",
    )
    replace_exact(
        path,
        """            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][  # type: ignore[index]
                    : attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )
""",
        """            if (
                os.environ.get("DDTREE_TRITON_TREE_GDN", "0") == "1"
                and attn_metadata.ddtree_parent_ids is not None
            ):
                global _DDTREE_TRITON_GDN_LOGGED
                if not _DDTREE_TRITON_GDN_LOGGED:
                    logger.info("Using DDTree Triton conv/GDN parent-state replay")
                    _DDTREE_TRITON_GDN_LOGGED = True
                mixed_qkv_spec = causal_conv1d_update_ddtree(
                    mixed_qkv_spec,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=spec_state_indices_tensor,
                    parent_ids=attn_metadata.ddtree_parent_ids[
                        : attn_metadata.num_spec_decodes
                    ],
                    query_start_loc=spec_query_start_loc,
                    null_block_id=0,
                )
            else:
                mixed_qkv_spec = causal_conv1d_update(
                    mixed_qkv_spec,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=spec_state_indices_tensor[:, 0][  # type: ignore[index]
                        : attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
                    ],
                    num_accepted_tokens=num_accepted_tokens,
                    query_start_loc=spec_query_start_loc,
                    max_query_len=spec_state_indices_tensor.size(-1),
                    validate_data=False,
                )
                if (
                    os.environ.get("DDTREE_TRITON_TREE_GDN", "0") == "1"
                    and attn_metadata.ddtree_parent_ids is None
                ):
                    global _DDTREE_TRITON_GDN_MISSING_PARENT_LOGGED
                    if not _DDTREE_TRITON_GDN_MISSING_PARENT_LOGGED:
                        logger.warning(
                            "DDTree Triton GDN requested but parent metadata "
                            "was not present for this speculative GDN window"
                        )
                        _DDTREE_TRITON_GDN_MISSING_PARENT_LOGGED = True
""",
    )
    replace_exact(
        path,
        """                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
""",
        """                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    ddtree_parent_ids=(
                        attn_metadata.ddtree_parent_ids[
                            : attn_metadata.num_spec_decodes
                        ]
                        if os.environ.get("DDTREE_TRITON_TREE_GDN", "0") == "1"
                        and attn_metadata.ddtree_parent_ids is not None
                        else None
                    ),
                    use_qk_l2norm_in_kernel=True,
                )
""",
    )


def verify_static(pkg_root: Path) -> None:
    causal = (pkg_root / "model_executor/layers/mamba/ops/causal_conv1d.py").read_text()
    fused = (
        pkg_root / "model_executor/layers/fla/ops/fused_sigmoid_gating.py"
    ).read_text()
    gdn = (pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py").read_text()
    for needle, text in (
        ("causal_conv1d_update_ddtree", causal),
        ("_causal_conv1d_update_ddtree_kernel", causal),
        ("IS_DDTREE", fused),
        ("ddtree_parent_ids", fused),
        ("DDTREE_TRITON_TREE_GDN", gdn),
        ("causal_conv1d_update_ddtree", gdn),
    ):
        if needle not in text:
            raise RuntimeError(f"Static M8A verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_causal_conv(pkg_root)
    patch_fused_gdn(pkg_root)
    patch_gdn_linear_attention(pkg_root)
    verify_static(pkg_root)
    clear_python_caches(pkg_root)
    print(f"[{MARKER}] Triton DDTree conv/GDN fast path verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
