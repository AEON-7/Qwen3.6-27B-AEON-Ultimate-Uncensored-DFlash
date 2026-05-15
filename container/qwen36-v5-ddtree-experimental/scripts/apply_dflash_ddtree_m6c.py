#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m6c"


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
        """from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
""",
        """from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadataBuilder
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
""",
    )
    replace_exact(
        path,
        """            if use_spec_decode and isinstance(
                builder, (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder)
            ):
                assert ubid is None, "UBatching not supported with GDN yet"
                extra_attn_metadata_args = dict(
                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                        :num_reqs_padded
                    ],
                    # aeon_dflash_ddtree_m6a
                    # Parent ids are root+tree compact coordinates, padded in
                    # active request order. GDN filters them down to spec rows.
                    ddtree_parent_ids=getattr(
                        self, "_last_ddtree_parent_ids_gpu", None
                    ),
                )
""",
        """            if use_spec_decode and isinstance(
                builder, (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder)
            ):
                assert ubid is None, "UBatching not supported with GDN yet"
                extra_attn_metadata_args = dict(
                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                        :num_reqs_padded
                    ],
                    # aeon_dflash_ddtree_m6a
                    # Parent ids are root+tree compact coordinates, padded in
                    # active request order. GDN filters them down to spec rows.
                    ddtree_parent_ids=getattr(
                        self, "_last_ddtree_parent_ids_gpu", None
                    ),
                )
            elif use_spec_decode and isinstance(builder, FlexAttentionMetadataBuilder):
                # aeon_dflash_ddtree_m6c
                # FlexAttention correctness mode uses the same compact parent ids
                # to restrict verifier-window KV visibility to ancestors only.
                extra_attn_metadata_args = dict(
                    ddtree_parent_ids=getattr(
                        self, "_last_ddtree_parent_ids_gpu", None
                    ),
                )
""",
    )


def patch_flex_attention(pkg_root: Path) -> None:
    path = pkg_root / "v1/attention/backends/flex_attention.py"
    replace_exact(
        path,
        """import math
from collections.abc import Callable
""",
        """import math
import os
from collections.abc import Callable
""",
    )
    replace_exact(
        path,
        """    block_sparsity_hint: BlockSparsityHint | None = None
""",
        """    block_sparsity_hint: BlockSparsityHint | None = None
    # aeon_dflash_ddtree_m6c
    # Optional [num_reqs, max_tree_len, max_tree_len] mask where
    # ancestor_mask[req, q, kv] is true iff compact KV node `kv` is on
    # compact query node `q`'s ancestor path. Used only by the guarded
    # correctness mode for DDTree verifier windows.
    ddtree_ancestor_mask: torch.Tensor | None = None
""",
    )
    replace_exact(
        path,
        """            return is_valid & self.logical_mask_mod(b, h, logical_q_idx, logical_kv_idx)
""",
        """            base_mask = is_valid & self.logical_mask_mod(
                b, h, logical_q_idx, logical_kv_idx
            )
            if self.ddtree_ancestor_mask is None:
                return base_mask

            q_req = self.doc_ids[q_idx]
            tree_len = self.query_start_loc[q_req + 1] - self.query_start_loc[q_req]
            tree_start = self.decode_offset[q_req]
            local_q_idx = logical_q_idx - tree_start
            local_kv_idx = logical_kv_idx - tree_start
            in_tree_q = (local_q_idx >= 0) & (local_q_idx < tree_len)
            in_tree_kv = (local_kv_idx >= 0) & (local_kv_idx < tree_len)
            max_tree_len = self.ddtree_ancestor_mask.shape[-1]
            clamped_q = torch.clamp(local_q_idx, 0, max_tree_len - 1).to(torch.long)
            clamped_kv = torch.clamp(local_kv_idx, 0, max_tree_len - 1).to(torch.long)
            tree_visible = self.ddtree_ancestor_mask[
                q_req.to(torch.long), clamped_q, clamped_kv
            ]
            return torch.where(in_tree_q & in_tree_kv, is_valid & tree_visible, base_mask)
""",
    )
    replace_exact(
        path,
        """    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlexAttentionMetadata:
""",
        """    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        ddtree_parent_ids: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> FlexAttentionMetadata:
""",
    )
    replace_exact(
        path,
        """        uses_paged_kv = not isinstance(self.kv_cache_spec, EncoderOnlyAttentionSpec)
        logical_mask_mod = (
""",
        """        ddtree_ancestor_mask = None
        if ddtree_parent_ids is not None and os.environ.get(
            "DDTREE_FLEX_TREE_MASK", "0"
        ) == "1":
            parent_ids_cpu = ddtree_parent_ids[:num_reqs].detach().cpu()
            query_lens_cpu = query_start_loc_cpu[1 : num_reqs + 1] - query_start_loc_cpu[
                :num_reqs
            ]
            max_tree_len = int(parent_ids_cpu.shape[1])
            ancestor_cpu = torch.zeros(
                (num_reqs, max_tree_len, max_tree_len), dtype=torch.bool
            )
            for req_i in range(num_reqs):
                # Rows without DDTree payload have no root marker and should use
                # the normal causal mask.
                if max_tree_len == 0 or int(parent_ids_cpu[req_i, 0].item()) >= 0:
                    continue
                q_len = min(int(query_lens_cpu[req_i].item()), max_tree_len)
                for q_local in range(q_len):
                    cur = q_local
                    while cur >= 0:
                        ancestor_cpu[req_i, q_local, cur] = True
                        cur = int(parent_ids_cpu[req_i, cur].item())
            ddtree_ancestor_mask = ancestor_cpu.to(
                device=query_start_loc.device, non_blocking=True
            )

        uses_paged_kv = not isinstance(self.kv_cache_spec, EncoderOnlyAttentionSpec)
        logical_mask_mod = (
""",
    )
    replace_exact(
        path,
        """            persistent_doc_ids=self.persistent_doc_ids,
        )
""",
        """            persistent_doc_ids=self.persistent_doc_ids,
            ddtree_ancestor_mask=ddtree_ancestor_mask,
        )
""",
    )


def verify_static(pkg_root: Path) -> None:
    checks = {
        "v1/worker/gpu_model_runner.py": (
            "FlexAttentionMetadataBuilder",
            "ddtree_parent_ids=getattr",
        ),
        "v1/attention/backends/flex_attention.py": (
            "DDTREE_FLEX_TREE_MASK",
            "ddtree_ancestor_mask",
            "ancestor_cpu",
            "tree_visible",
        ),
    }
    for rel, needles in checks.items():
        text = (pkg_root / rel).read_text()
        for needle in needles:
            if needle not in text:
                raise RuntimeError(f"Static M6C verification failed: {rel} missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_gpu_model_runner(pkg_root)
    patch_flex_attention(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree FlexAttention ancestor mask path verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
