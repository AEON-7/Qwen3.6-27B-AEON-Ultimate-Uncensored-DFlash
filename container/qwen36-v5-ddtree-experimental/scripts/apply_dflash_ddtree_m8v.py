#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8v"


def replace_exact(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Could not find expected text in {path}:\n{old}")
    path.write_text(text.replace(old, new, 1))
    return True


def replace_between(path: Path, start_marker: str, end_marker: str, new: str) -> None:
    text = path.read_text()
    if "aeon_dflash_ddtree_m8v" in text and new.strip() in text:
        return
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"Could not find start marker in {path}: {start_marker}")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"Could not find end marker in {path}: {end_marker}")
    path.write_text(text[:start] + new + text[end:])


def clear_python_caches(pkg_root: Path) -> None:
    for pyc in pkg_root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    for pycache in pkg_root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


BRANCH_OVERLAY_FN = r'''    def _forward_ddtree_eager_attention(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
    ) -> torch.Tensor:
        # aeon_dflash_ddtree_m8v
        parent_ids = attn_metadata.ddtree_parent_ids
        assert parent_ids is not None
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        block_size = key_cache.shape[1]
        branch_only = os.environ.get("DDTREE_BRANCH_ONLY_ATTN", "0") == "1"

        if branch_only:
            # First let the production FlashAttention backend compute the normal
            # chain rows. The small PyTorch verifier below overwrites only rows
            # whose parent differs from the flat-chain parent, e.g. root-leaf
            # alternate branches. This keeps the correctness mask where it is
            # needed without paying the eager cost for every draft row.
            descale_shape = (query_start_loc.shape[0] - 1, self.num_kv_heads)
            q_descale = (
                layer._q_scale.expand(descale_shape)
                if self.supports_quant_query_input
                else None
            )
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)
            sliding_window_size = (
                list(self.sliding_window) if self.sliding_window is not None else None
            )
            flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                out=output,
                cu_seqlens_q=query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=seq_lens,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                scheduler_metadata=attn_metadata.scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                num_splits=attn_metadata.max_num_splits,
                s_aux=self.sinks,
            )

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

            if branch_only:
                if parent_row is None:
                    continue
                manual_rows = [
                    row_i
                    for row_i, parent_i in enumerate(parent_row)
                    if not (
                        (row_i == 0 and int(parent_i) < 0)
                        or int(parent_i) == row_i - 1
                    )
                ]
                if not manual_rows:
                    continue
                row_indices = torch.tensor(
                    manual_rows, device=query.device, dtype=torch.long
                )
            else:
                row_indices = torch.arange(q_len, device=query.device, dtype=torch.long)

            seq_len = int(seq_lens[req_i].item())
            context_len = max(0, seq_len - q_len)
            positions = torch.arange(seq_len, device=query.device)
            block_ids = block_table[req_i, positions // block_size].to(torch.long)
            block_offsets = positions % block_size
            slots = block_ids * block_size + block_offsets

            q_req = query[q_start + row_indices].to(torch.float32)
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

            visible = torch.ones(
                (row_indices.numel(), seq_len), device=query.device, dtype=torch.bool
            )
            if parent_row is None:
                local_kv = torch.arange(q_len, device=query.device)
                visible[:, context_len : context_len + q_len] = (
                    local_kv.unsqueeze(0) <= row_indices.unsqueeze(1)
                )
            else:
                ancestor = torch.zeros(
                    (row_indices.numel(), q_len), device=query.device, dtype=torch.bool
                )
                for out_i, q_local_tensor in enumerate(row_indices.detach().cpu()):
                    cur = int(q_local_tensor.item())
                    while cur >= 0:
                        ancestor[out_i, cur] = True
                        cur = int(parent_row[cur])
                visible[:, context_len : context_len + q_len] = ancestor

            if self.sliding_window is not None and self.sliding_window[0] >= 0:
                q_abs = context_len + row_indices
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
            output[q_start + row_indices].copy_(out_req.to(output.dtype))

        return output

'''


def patch_flash_attention(pkg_root: Path) -> None:
    path = pkg_root / "v1/attention/backends/flash_attn.py"
    replace_exact(
        path,
        """                return self._forward_ddtree_eager_attention(
                    query[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                )
""",
        """                return self._forward_ddtree_eager_attention(
                    layer,
                    query[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                )
""",
    )
    replace_between(
        path,
        "    def _forward_ddtree_eager_attention(\n",
        "    def do_kv_cache_update(\n",
        BRANCH_OVERLAY_FN,
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/attention/backends/flash_attn.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m8v",
        "DDTREE_BRANCH_ONLY_ATTN",
        "flash_attn_varlen_func(",
        "manual_rows",
        "output[q_start + row_indices].copy_",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M8V verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_flash_attention(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree branch-only attention overlay verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
