#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m6b"


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


SLOW_METHOD = r'''
    def _forward_core_ddtree_slow(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        """Correctness-first DDTree verifier path for GDN layers.

        vLLM's stock speculative GDN kernels assume a flat chain. DDTree needs
        each compact verifier node to load conv/SSM state from its tree parent.
        This guarded path mirrors Lucebox's parent replay semantics in PyTorch;
        it is deliberately slower and exists to make quality correct before a
        dedicated fused CUDA/Triton kernel replaces it.
        """

        spec_query_start_loc = attn_metadata.spec_query_start_loc
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
        ddtree_parent_ids = attn_metadata.ddtree_parent_ids
        assert spec_query_start_loc is not None
        assert spec_state_indices_tensor is not None
        assert ddtree_parent_ids is not None
        assert attn_metadata.spec_sequence_masks is not None
        assert attn_metadata.num_prefills == 0
        assert attn_metadata.num_decodes == 0

        self_kv_cache = self.kv_cache
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        conv_bias = self.conv1d.bias
        conv_out = torch.empty_like(mixed_qkv)

        state_indices_cpu = (
            spec_state_indices_tensor[: attn_metadata.num_spec_decodes]
            .detach()
            .cpu()
        )
        parent_ids_cpu = (
            ddtree_parent_ids[: attn_metadata.num_spec_decodes].detach().cpu()
        )
        starts_cpu = (
            spec_query_start_loc[: attn_metadata.num_spec_decodes + 1]
            .detach()
            .cpu()
        )

        for req_i in range(attn_metadata.num_spec_decodes):
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
                if conv_bias is not None:
                    acc = acc + conv_bias.to(torch.float32)
                if self.activation in ("silu", "swish"):
                    acc = torch.nn.functional.silu(acc)
                conv_out[token_row] = acc.to(conv_out.dtype)
                conv_state[dst_state_idx].copy_(
                    torch.cat([parent_conv[:, 1:], raw.unsqueeze(-1)], dim=-1)
                )

        query, key, value = self.rearrange_mixed_qkv(conv_out)
        assert query is not None and key is not None and value is not None
        query = query.squeeze(0)
        key = key.squeeze(0)
        value = value.squeeze(0)

        num_k_heads = query.shape[1]
        num_v_heads = value.shape[1]
        value_per_key = max(1, num_v_heads // num_k_heads)
        hv_to_h = (
            torch.arange(num_v_heads, device=value.device, dtype=torch.long)
            // value_per_key
        )
        scale = self.head_k_dim**-0.5
        recurrent_out = torch.empty_like(value)

        A_log = self.A_log.to(torch.float32)
        dt_bias = self.dt_bias.to(torch.float32)
        for req_i in range(attn_metadata.num_spec_decodes):
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
                q_t = query[token_row].to(torch.float32)
                k_t = key[token_row].to(torch.float32)
                v_t = value[token_row].to(torch.float32)
                q_t = q_t * torch.rsqrt(
                    (q_t * q_t).sum(dim=-1, keepdim=True) + 1e-6
                )
                k_t = k_t * torch.rsqrt(
                    (k_t * k_t).sum(dim=-1, keepdim=True) + 1e-6
                )
                q_hv = q_t.index_select(0, hv_to_h) * scale
                k_hv = k_t.index_select(0, hv_to_h)

                gate = -torch.exp(A_log) * torch.nn.functional.softplus(
                    a[token_row].to(torch.float32) + dt_bias,
                    beta=1.0,
                    threshold=20.0,
                )
                h = h * torch.exp(gate).view(num_v_heads, 1, 1)
                predicted_v = torch.einsum("hvk,hk->hv", h, k_hv)
                delta_v = (v_t - predicted_v) * torch.sigmoid(
                    b[token_row].to(torch.float32)
                ).view(num_v_heads, 1)
                h = h + delta_v.unsqueeze(-1) * k_hv.unsqueeze(1)
                recurrent_out[token_row] = torch.einsum("hvk,hk->hv", h, q_hv).to(
                    recurrent_out.dtype
                )
                ssm_state[dst_state_idx].copy_(h.to(ssm_state.dtype))

        core_attn_out[:num_actual_tokens] = recurrent_out
'''


def patch_gdn_slow_tree_path(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py"
    replace_exact(
        path,
        """import torch
from einops import rearrange
""",
        """import os

import torch
from einops import rearrange
""",
    )
    replace_exact(
        path,
        """    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
""",
        SLOW_METHOD
        + """

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
""",
    )
    replace_exact(
        path,
        """        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
""",
        """        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        # aeon_dflash_ddtree_m6b
        # Guarded correctness-first tree replay for GDN state. This is slow,
        # but it prevents sibling branches from inheriting the wrong conv/SSM
        # state while we build the fused kernel.
        if (
            os.environ.get("DDTREE_SLOW_TREE_GDN", "0") == "1"
            and attn_metadata.ddtree_parent_ids is not None
            and spec_sequence_masks is not None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes == 0
        ):
            return self._forward_core_ddtree_slow(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        mixed_qkv = mixed_qkv[:num_actual_tokens]
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py").read_text()
    needles = (
        "DDTREE_SLOW_TREE_GDN",
        "_forward_core_ddtree_slow",
        "parent_compact",
        "torch.einsum(\"hvk,hk->hv\"",
    )
    for needle in needles:
        if needle not in text:
            raise RuntimeError(f"Static M6B verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_gdn_slow_tree_path(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree slow GDN replay path verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
