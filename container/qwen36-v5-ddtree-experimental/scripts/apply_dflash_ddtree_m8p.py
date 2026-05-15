#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8p"


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


SLOW_CONV_METHOD = r'''
    def _conv1d_ddtree_slow(
        self,
        mixed_qkv_spec: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        conv_bias: torch.Tensor | None,
        spec_state_indices_tensor: torch.Tensor,
        ddtree_parent_ids: torch.Tensor,
        spec_query_start_loc: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # aeon_dflash_ddtree_m8p
        # Diagnostic split path: correctness-first conv replay with the fast
        # Triton SSM replay still enabled. This isolates whether remaining
        # branch-quality errors live in the conv kernel or the SSM kernel.
        conv_out = torch.empty_like(mixed_qkv_spec)
        state_indices_cpu = spec_state_indices_tensor.detach().cpu()
        parent_ids_cpu = ddtree_parent_ids.detach().cpu()
        starts_cpu = spec_query_start_loc.detach().cpu()
        width_minus_one = conv_weights.shape[-1] - 1

        for req_i in range(ddtree_parent_ids.shape[0]):
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
                    src_state_idx = int(state_indices_cpu[req_i, 0].item())
                    src_offset = root_offset
                else:
                    src_state_idx = int(state_indices_cpu[req_i, parent_compact].item())
                    src_offset = 0
                if src_state_idx <= 0:
                    continue
                parent_conv = conv_state[
                    src_state_idx, :, src_offset : src_offset + width_minus_one
                ].clone()
                raw = mixed_qkv_spec[token_row].to(parent_conv.dtype)
                window = torch.cat([parent_conv, raw.unsqueeze(-1)], dim=-1)
                acc = (
                    window.to(torch.float32) * conv_weights.to(torch.float32)
                ).sum(dim=-1)
                if conv_bias is not None:
                    acc = acc + conv_bias.to(torch.float32)
                if self.activation in ("silu", "swish"):
                    acc = torch.nn.functional.silu(acc)
                conv_out[token_row] = acc.to(conv_out.dtype)
                conv_state[dst_state_idx, :, :width_minus_one].copy_(
                    torch.cat([parent_conv[:, 1:], raw.unsqueeze(-1)], dim=-1)
                )

        return conv_out

'''


def patch_gdn_linear_attention(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py"
    replace_exact(
        path,
        "\n    def _forward_core(\n",
        "\n" + SLOW_CONV_METHOD + "    def _forward_core(\n",
    )
    replace_exact(
        path,
        """                mixed_qkv_spec = causal_conv1d_update_ddtree(
                    mixed_qkv_spec,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=spec_state_indices_tensor,
                    parent_ids=attn_metadata.ddtree_parent_ids[
                        : attn_metadata.num_spec_decodes
                    ],
                    num_accepted_tokens=num_accepted_tokens,
                    query_start_loc=spec_query_start_loc,
                    null_block_id=0,
                )
""",
        """                if os.environ.get("DDTREE_SLOW_TREE_CONV", "0") == "1":
                    mixed_qkv_spec = self._conv1d_ddtree_slow(
                        mixed_qkv_spec=mixed_qkv_spec,
                        conv_state=conv_state,
                        conv_weights=conv_weights,
                        conv_bias=self.conv1d.bias,
                        spec_state_indices_tensor=spec_state_indices_tensor,
                        ddtree_parent_ids=attn_metadata.ddtree_parent_ids[
                            : attn_metadata.num_spec_decodes
                        ],
                        spec_query_start_loc=spec_query_start_loc,
                        num_accepted_tokens=num_accepted_tokens,
                    )
                else:
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
                        num_accepted_tokens=num_accepted_tokens,
                        query_start_loc=spec_query_start_loc,
                        null_block_id=0,
                    )
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m8p",
        "_conv1d_ddtree_slow",
        "DDTREE_SLOW_TREE_CONV",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M8P verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_gdn_linear_attention(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree slow-conv/fast-SSM diagnostic split verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
