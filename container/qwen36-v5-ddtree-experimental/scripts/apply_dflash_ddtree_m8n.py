#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8n"


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


def patch_causal_conv_ddtree(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/mamba/ops/causal_conv1d.py"
    replace_exact(
        path,
        """    parent_ids_ptr,  # (batch, max_query_len)
    query_start_loc_ptr,  # (batch + 1)
""",
        """    parent_ids_ptr,  # (batch, max_query_len)
    num_accepted_tokens_ptr,  # (batch,)
    query_start_loc_ptr,  # (batch + 1)
""",
    )
    replace_exact(
        path,
        """    query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
    query_end_index = tl.load(query_start_loc_ptr + idx_seq + 1).to(tl.int64)
    actual_seqlen = query_end_index - query_start_index

    for idx_token in tl.range(seqlen):
""",
        """    query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
    query_end_index = tl.load(query_start_loc_ptr + idx_seq + 1).to(tl.int64)
    actual_seqlen = query_end_index - query_start_index
    # aeon_dflash_ddtree_m8n
    # Stock vLLM's speculative conv update starts from the rolling cache offset
    # selected by num_accepted_tokens - 1. Tree replay must use the same base
    # offset for the root row, then use offset zero for freshly materialized
    # parent rows inside the verifier window.
    root_state_offset = tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64) - 1
    root_state_offset = tl.where(root_state_offset < 0, 0, root_state_offset)

    for idx_token in tl.range(seqlen):
""",
    )
    replace_exact(
        path,
        """        parent_t = tl.load(
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
""",
        """        parent_t = tl.load(
            parent_ids_ptr
            + idx_seq * stride_parent_ids_seq
            + idx_token * stride_parent_ids_tok
        ).to(tl.int64)
        src_token = tl.where(parent_t < 0, 0, parent_t)
        src_state_offset = tl.where(parent_t < 0, root_state_offset, 0)
        src_state_idx = tl.load(
            conv_state_indices_ptr
            + idx_seq * stride_state_indices_seq
            + src_token * stride_state_indices_tok
        ).to(tl.int64)
        if HAS_NULL_BLOCK:
            active = active & (src_state_idx != null_block_id)
""",
    )
    replace_exact(
        path,
        """                x_j = tl.load(
                    conv_state_ptr
                    + src_state_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim
                    + j * stride_conv_state_tok,
                    mask=mask_feats & active,
                    other=0.0,
                ).to(tl.float32)
""",
        """                x_j = tl.load(
                    conv_state_ptr
                    + src_state_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim
                    + (src_state_offset + j) * stride_conv_state_tok,
                    mask=mask_feats & active,
                    other=0.0,
                ).to(tl.float32)
""",
    )
    replace_exact(
        path,
        """                new_state = tl.load(
                    conv_state_ptr
                    + src_state_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim
                    + (j + 1) * stride_conv_state_tok,
                    mask=mask_feats & active,
                    other=0.0,
                )
""",
        """                new_state = tl.load(
                    conv_state_ptr
                    + src_state_idx * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim
                    + (src_state_offset + j + 1) * stride_conv_state_tok,
                    mask=mask_feats & active,
                    other=0.0,
                )
""",
    )
    replace_exact(
        path,
        """    parent_ids: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
):
""",
        """    parent_ids: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
):
""",
    )
    replace_exact(
        path,
        """    assert parent_ids is not None
    assert query_start_loc is not None
""",
        """    assert parent_ids is not None
    assert num_accepted_tokens is not None
    assert query_start_loc is not None
""",
    )
    replace_exact(
        path,
        """        parent_ids,
        query_start_loc,
        out,
""",
        """        parent_ids,
        num_accepted_tokens,
        query_start_loc,
        out,
""",
    )


def patch_fused_gdn_ddtree(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/fla/ops/fused_sigmoid_gating.py"
    replace_exact(
        path,
        """            if IS_DDTREE:
                i_t = 0
            elif IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
""",
        """            if IS_DDTREE:
                # aeon_dflash_ddtree_m8n
                # Match stock speculative decoding's rolling-state entry point.
                # Row 0 materializes the current root state; branch children
                # then reload from compact parent rows produced in this window.
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            elif IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
""",
    )


def patch_gdn_linear_attention(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py"
    replace_exact(
        path,
        """                    parent_ids=attn_metadata.ddtree_parent_ids[
                        : attn_metadata.num_spec_decodes
                    ],
                    query_start_loc=spec_query_start_loc,
""",
        """                    parent_ids=attn_metadata.ddtree_parent_ids[
                        : attn_metadata.num_spec_decodes
                    ],
                    num_accepted_tokens=num_accepted_tokens,
                    query_start_loc=spec_query_start_loc,
""",
    )


def patch_slow_reference(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py"
    replace_exact(
        path,
        """        for req_i in range(attn_metadata.num_spec_decodes):
            start = int(starts_cpu[req_i].item())
            end = int(starts_cpu[req_i + 1].item())
            for compact_i in range(end - start):
                token_row = start + compact_i
                dst_state_idx = int(state_indices_cpu[req_i, compact_i].item())
                if dst_state_idx <= 0:
                    continue
                parent_compact = int(parent_ids_cpu[req_i, compact_i].item())
                if parent_compact < 0:
                    src_state_idx = dst_state_idx
                else:
                    src_state_idx = int(state_indices_cpu[req_i, parent_compact].item())
                if src_state_idx <= 0:
                    continue

                parent_conv = conv_state[src_state_idx].clone()
                raw = mixed_qkv[token_row].to(parent_conv.dtype)
                window = torch.cat([parent_conv, raw.unsqueeze(-1)], dim=-1)
                acc = (
                    window.to(torch.float32)
                    * conv_weights.to(torch.float32)
                ).sum(dim=-1)
""",
        """        for req_i in range(attn_metadata.num_spec_decodes):
            start = int(starts_cpu[req_i].item())
            end = int(starts_cpu[req_i + 1].item())
            root_offset = int(num_accepted_tokens[req_i].item()) - 1
            root_offset = max(0, root_offset)
            width_minus_one = conv_weights.shape[-1] - 1
            for compact_i in range(end - start):
                token_row = start + compact_i
                dst_state_idx = int(state_indices_cpu[req_i, compact_i].item())
                if dst_state_idx <= 0:
                    continue
                parent_compact = int(parent_ids_cpu[req_i, compact_i].item())
                if parent_compact < 0:
                    src_state_idx = int(state_indices_cpu[req_i, 0].item())
                    src_offset = root_offset
                else:
                    src_state_idx = int(state_indices_cpu[req_i, parent_compact].item())
                    src_offset = 0
                if src_state_idx <= 0:
                    continue

                parent_conv_full = conv_state[src_state_idx].clone()
                parent_conv = parent_conv_full[
                    :, src_offset : src_offset + width_minus_one
                ]
                raw = mixed_qkv[token_row].to(parent_conv.dtype)
                window = torch.cat([parent_conv, raw.unsqueeze(-1)], dim=-1)
                acc = (
                    window.to(torch.float32)
                    * conv_weights.to(torch.float32)
                ).sum(dim=-1)
""",
    )
    replace_exact(
        path,
        """                conv_out[token_row] = acc.to(conv_out.dtype)
                conv_state[dst_state_idx].copy_(
                    torch.cat([parent_conv[:, 1:], raw.unsqueeze(-1)], dim=-1)
                )
""",
        """                conv_out[token_row] = acc.to(conv_out.dtype)
                dst_conv_full = conv_state[dst_state_idx]
                dst_conv_full[:, :width_minus_one].copy_(
                    torch.cat([parent_conv[:, 1:], raw.unsqueeze(-1)], dim=-1)
                )
""",
    )
    replace_exact(
        path,
        """        for req_i in range(attn_metadata.num_spec_decodes):
            start = int(starts_cpu[req_i].item())
            end = int(starts_cpu[req_i + 1].item())
            for compact_i in range(end - start):
                token_row = start + compact_i
                dst_state_idx = int(state_indices_cpu[req_i, compact_i].item())
                if dst_state_idx <= 0:
                    continue
                parent_compact = int(parent_ids_cpu[req_i, compact_i].item())
                if parent_compact < 0:
                    src_state_idx = dst_state_idx
                else:
                    src_state_idx = int(state_indices_cpu[req_i, parent_compact].item())
                if src_state_idx <= 0:
                    continue

                h = ssm_state[src_state_idx].to(torch.float32).clone()
""",
        """        for req_i in range(attn_metadata.num_spec_decodes):
            start = int(starts_cpu[req_i].item())
            end = int(starts_cpu[req_i + 1].item())
            root_offset = int(num_accepted_tokens[req_i].item()) - 1
            root_offset = max(0, root_offset)
            for compact_i in range(end - start):
                token_row = start + compact_i
                dst_state_idx = int(state_indices_cpu[req_i, compact_i].item())
                if dst_state_idx <= 0:
                    continue
                parent_compact = int(parent_ids_cpu[req_i, compact_i].item())
                if parent_compact < 0:
                    src_state_idx = int(state_indices_cpu[req_i, root_offset].item())
                else:
                    src_state_idx = int(state_indices_cpu[req_i, parent_compact].item())
                if src_state_idx <= 0:
                    continue

                h = ssm_state[src_state_idx].to(torch.float32).clone()
""",
    )


def verify_static(pkg_root: Path) -> None:
    causal = (pkg_root / "model_executor/layers/mamba/ops/causal_conv1d.py").read_text()
    fused = (
        pkg_root / "model_executor/layers/fla/ops/fused_sigmoid_gating.py"
    ).read_text()
    gdn = (pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py").read_text()
    checks = {
        "causal_conv1d.py": (
            "aeon_dflash_ddtree_m8n",
            "root_state_offset",
            "num_accepted_tokens_ptr",
            "(src_state_offset + j + 1) * stride_conv_state_tok",
        ),
        "fused_sigmoid_gating.py": (
            "aeon_dflash_ddtree_m8n",
            "Match stock speculative decoding's rolling-state entry point",
        ),
        "gdn_linear_attn.py": (
            "num_accepted_tokens=num_accepted_tokens",
            "width_minus_one",
            "src_offset",
        ),
    }
    texts = {
        "causal_conv1d.py": causal,
        "fused_sigmoid_gating.py": fused,
        "gdn_linear_attn.py": gdn,
    }
    for rel, needles in checks.items():
        for needle in needles:
            if needle not in texts[rel]:
                raise RuntimeError(f"Static M8N verification failed: {rel} missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_causal_conv_ddtree(pkg_root)
    patch_fused_gdn_ddtree(pkg_root)
    patch_gdn_linear_attention(pkg_root)
    patch_slow_reference(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree GDN rolling-state offset replay verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
