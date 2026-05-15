#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8j"


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


def patch_flash_attention(pkg_root: Path) -> None:
    path = pkg_root / "v1/attention/backends/flash_attn.py"
    replace_exact(
        path,
        """        if (
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
        """        if (
            os.environ.get("DDTREE_EAGER_TREE_ATTN", "0") == "1"
            and attn_metadata.ddtree_parent_ids is not None
            and not is_quantized_kv_cache(self.kv_cache_dtype)
            and not attn_metadata.use_cascade
            and self.dcp_world_size == 1
        ):
            # aeon_dflash_ddtree_m8j
            # If the DDTree payload is actually a flat chain, native
            # FlashAttention is already the mathematically correct verifier.
            # Bypassing the eager PyTorch path here makes top_k=1 a clean
            # control case and keeps branch-specific debugging isolated.
            ddtree_parent_mode = self._ddtree_parent_mode(attn_metadata)
            if (
                ddtree_parent_mode == "flat_chain"
                and os.environ.get("DDTREE_BYPASS_FLAT_CHAIN_ATTN", "1") == "1"
            ):
                if not getattr(self, "_ddtree_flat_chain_bypass_logged", False):
                    logger.warning(
                        "DDTree eager attention bypassed for flat-chain payload; "
                        "using native FlashAttention verifier parent_preview=%s",
                        self._ddtree_parent_preview(attn_metadata),
                    )
                    self._ddtree_flat_chain_bypass_logged = True
            else:
                if not getattr(self, "_ddtree_eager_branch_logged", False):
                    logger.warning(
                        "DDTree eager attention active parent_mode=%s "
                        "parent_preview=%s key_shape=%s value_shape=%s",
                        ddtree_parent_mode,
                        self._ddtree_parent_preview(attn_metadata),
                        tuple(key_cache.shape),
                        tuple(value_cache.shape),
                    )
                    self._ddtree_eager_branch_logged = True
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
        """    def _forward_ddtree_eager_attention(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
    ) -> torch.Tensor:
""",
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

    def _ddtree_parent_preview(
        self,
        attn_metadata: FlashAttentionMetadata,
    ) -> list[int] | None:
        parent_ids = attn_metadata.ddtree_parent_ids
        if parent_ids is None or parent_ids.shape[0] == 0:
            return None
        query_start_loc = attn_metadata.query_start_loc
        if query_start_loc.shape[0] < 2:
            return None
        q_len = int((query_start_loc[1] - query_start_loc[0]).item())
        q_len = max(0, min(q_len, parent_ids.shape[1], 24))
        if q_len == 0:
            return None
        return [int(v) for v in parent_ids[0, :q_len].detach().cpu().tolist()]

    def _forward_ddtree_eager_attention(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
    ) -> torch.Tensor:
""",
    )


def patch_payload_preview(pkg_root: Path) -> None:
    path = pkg_root / "v1/spec_decode/dflash.py"
    replace_exact(
        path,
        """            first_nodes = (
                len(payloads[0].get("tree_token_ids", ())) if payloads else 0
            )
            logger.warning(
                "DDTree proposer built payloads=%s first_nodes=%s",
                len(payloads),
                first_nodes,
            )
""",
        """            first_nodes = (
                len(payloads[0].get("tree_token_ids", ())) if payloads else 0
            )
            first_parents = (
                payloads[0].get("parent_indices", [])[:8] if payloads else []
            )
            logger.warning(
                "DDTree proposer built payloads=%s first_nodes=%s "
                "first_parents=%s",
                len(payloads),
                first_nodes,
                first_parents,
            )
""",
    )


def verify_static(pkg_root: Path) -> None:
    checks = {
        "v1/attention/backends/flash_attn.py": (
            "aeon_dflash_ddtree_m8j",
            "DDTREE_BYPASS_FLAT_CHAIN_ATTN",
            "_ddtree_parent_mode",
            "_ddtree_parent_preview",
            "DDTree eager attention active parent_mode",
        ),
        "v1/spec_decode/dflash.py": (
            "first_parents",
            "DDTree proposer built payloads=%s first_nodes=%s ",
        ),
    }
    for rel, needles in checks.items():
        text = (pkg_root / rel).read_text()
        for needle in needles:
            if needle not in text:
                raise RuntimeError(f"Static M8J verification failed: {rel} missing {needle}")


def main() -> int:
    pkg_root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if pkg_root_override:
        pkg_root = Path(pkg_root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_flash_attention(pkg_root)
    patch_payload_preview(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree flat-chain attention bypass and branch diagnostics verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
