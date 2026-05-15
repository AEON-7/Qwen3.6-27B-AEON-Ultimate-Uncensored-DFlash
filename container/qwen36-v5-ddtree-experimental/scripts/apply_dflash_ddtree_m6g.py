#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m6g"


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


def patch_gpu_model_runner(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadataBuilder
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
""",
        """from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadataBuilder
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
""",
    )
    replace_exact(
        path,
        """            elif use_spec_decode and isinstance(builder, FlexAttentionMetadataBuilder):
                # aeon_dflash_ddtree_m6c
                # FlexAttention correctness mode uses the same compact parent ids
                # to restrict verifier-window KV visibility to ancestors only.
                extra_attn_metadata_args = dict(
                    ddtree_parent_ids=getattr(
                        self, "_last_ddtree_parent_ids_gpu", None
                    ),
                )
""",
        """            elif use_spec_decode and isinstance(
                builder, (FlexAttentionMetadataBuilder, FlashAttentionMetadataBuilder)
            ):
                # aeon_dflash_ddtree_m6c / m6g
                # Flex and FlashAttention correctness paths use the same compact
                # parent ids to restrict verifier-window KV visibility to
                # ancestors only.
                extra_attn_metadata_args = dict(
                    ddtree_parent_ids=getattr(
                        self, "_last_ddtree_parent_ids_gpu", None
                    ),
                )
""",
    )


def patch_flash_attention(pkg_root: Path) -> None:
    path = pkg_root / "v1/attention/backends/flash_attn.py"
    replace_exact(
        path,
        """import copy
from dataclasses import dataclass
""",
        """import copy
import os
from dataclasses import dataclass
""",
    )
    replace_exact(
        path,
        """    causal: bool = True
""",
        """    causal: bool = True
    # aeon_dflash_ddtree_m6g
    # Optional root+tree parent ids in compact verifier coordinates. Shape is
    # [num_reqs, max_tree_tokens + 1]. Used only by guarded DDTree correctness
    # mode; normal FlashAttention deployments keep this as None.
    ddtree_parent_ids: torch.Tensor | None = None
""",
    )
    replace_exact(
        path,
        """    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
""",
        """    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        ddtree_parent_ids: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
""",
    )
    replace_exact(
        path,
        """            max_num_splits=max_num_splits,
            causal=causal,
        )
""",
        """            max_num_splits=max_num_splits,
            causal=causal,
            ddtree_parent_ids=ddtree_parent_ids,
        )
""",
    )
    replace_exact(
        path,
        """        if is_quantized_kv_cache(self.kv_cache_dtype):
            # queries are quantized in the attention layer
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        if not attn_metadata.use_cascade:
""",
        """        if is_quantized_kv_cache(self.kv_cache_dtype):
            # queries are quantized in the attention layer
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        if (
            os.environ.get("DDTREE_EAGER_TREE_ATTN", "0") == "1"
            and attn_metadata.ddtree_parent_ids is not None
            and not is_quantized_kv_cache(self.kv_cache_dtype)
            and not attn_metadata.use_cascade
            and self.dcp_world_size == 1
        ):
            # aeon_dflash_ddtree_m6g
            # Correctness-first DDTree verifier for full-attention layers. It
            # keeps the normal FlashAttention backend selected for production,
            # but computes the tiny root+tree verifier window with an explicit
            # ancestor mask so sibling draft branches cannot see each other.
            return self._forward_ddtree_eager_attention(
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                output[:num_actual_tokens],
                attn_metadata,
            )

        if not attn_metadata.use_cascade:
""",
    )
    replace_exact(
        path,
        """        return output

    def do_kv_cache_update(
""",
        """        return output

    def _forward_ddtree_eager_attention(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
    ) -> torch.Tensor:
        parent_ids = attn_metadata.ddtree_parent_ids
        assert parent_ids is not None
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        block_size = key_cache.shape[1]

        # Flattening preserves the block-table slot convention:
        # absolute_slot = block_id * block_size + block_offset.
        flat_k = key_cache.reshape(-1, self.num_kv_heads, self.head_size)
        flat_v = value_cache.reshape(-1, self.num_kv_heads, self.head_size)

        for req_i in range(query_start_loc.shape[0] - 1):
            q_start = int(query_start_loc[req_i].item())
            q_end = int(query_start_loc[req_i + 1].item())
            q_len = q_end - q_start
            if q_len <= 0:
                continue
            parent_row = None
            if req_i < parent_ids.shape[0] and q_len <= parent_ids.shape[1]:
                candidate_parent_row = parent_ids[req_i, :q_len].detach().cpu().tolist()
                if candidate_parent_row and int(candidate_parent_row[0]) < 0:
                    parent_row = candidate_parent_row

            seq_len = int(seq_lens[req_i].item())
            context_len = max(0, seq_len - q_len)
            positions = torch.arange(seq_len, device=query.device)
            block_ids = block_table[req_i, positions // block_size].to(torch.long)
            block_offsets = positions % block_size
            slots = block_ids * block_size + block_offsets

            q_req = query[q_start:q_end].to(torch.float32)
            k_req = flat_k[slots].to(torch.float32)
            v_req = flat_v[slots].to(torch.float32)
            if self.num_queries_per_kv != 1:
                k_req = k_req.repeat_interleave(self.num_queries_per_kv, dim=1)
                v_req = v_req.repeat_interleave(self.num_queries_per_kv, dim=1)

            scores = torch.einsum("qhd,khd->hqk", q_req, k_req) * self.scale
            if self.logits_soft_cap:
                scores = self.logits_soft_cap * torch.tanh(
                    scores / self.logits_soft_cap
                )

            visible = torch.ones((q_len, seq_len), device=query.device, dtype=torch.bool)
            if parent_row is None:
                local_q = torch.arange(q_len, device=query.device)
                local_kv = torch.arange(q_len, device=query.device)
                visible[:, context_len : context_len + q_len] = (
                    local_kv.unsqueeze(0) <= local_q.unsqueeze(1)
                )
            else:
                ancestor = torch.zeros(
                    (q_len, q_len), device=query.device, dtype=torch.bool
                )
                for q_local in range(q_len):
                    cur = q_local
                    while cur >= 0:
                        ancestor[q_local, cur] = True
                        cur = int(parent_row[cur])
                visible[:, context_len : context_len + q_len] = ancestor

            if self.sliding_window is not None and self.sliding_window[0] >= 0:
                q_abs = context_len + torch.arange(q_len, device=query.device)
                kv_abs = torch.arange(seq_len, device=query.device)
                visible &= kv_abs.unsqueeze(0) >= (
                    q_abs - self.sliding_window[0]
                ).unsqueeze(1)
                if self.sliding_window[1] >= 0:
                    visible &= kv_abs.unsqueeze(0) <= (
                        q_abs + self.sliding_window[1]
                    ).unsqueeze(1)

            scores = scores.masked_fill(~visible.unsqueeze(0), -float("inf"))
            probs = torch.softmax(scores, dim=-1)
            out_req = torch.einsum("hqk,khd->qhd", probs, v_req)
            output[q_start:q_end].copy_(out_req.to(output.dtype))

        return output

    def do_kv_cache_update(
""",
    )


def verify_static(pkg_root: Path) -> None:
    checks = {
        "v1/worker/gpu_model_runner.py": (
            "FlashAttentionMetadataBuilder",
            "(FlexAttentionMetadataBuilder, FlashAttentionMetadataBuilder)",
        ),
        "v1/attention/backends/flash_attn.py": (
            "DDTREE_EAGER_TREE_ATTN",
            "ddtree_parent_ids",
            "_forward_ddtree_eager_attention",
            "ancestor[q_local, cur] = True",
        ),
    }
    for rel, needles in checks.items():
        text = (pkg_root / rel).read_text()
        for needle in needles:
            if needle not in text:
                raise RuntimeError(f"Static M6G verification failed: {rel} missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_gpu_model_runner(pkg_root)
    patch_flash_attention(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] FlashAttention eager DDTree verifier path verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
