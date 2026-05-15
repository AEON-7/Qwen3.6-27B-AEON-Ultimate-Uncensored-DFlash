#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8s"


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


def patch_imports(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """    NULL_BLOCK_ID,
    create_fast_prefill_custom_backend,
""",
        """    NULL_BLOCK_ID,
    create_fast_prefill_custom_backend,
    mamba_get_block_table_tensor,
""",
    )


def patch_recurrent_compaction(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """    def _compact_ddtree_recurrent_states(
        self,
        num_reqs: int,
    ) -> None:
        # aeon_dflash_ddtree_m8r
        # Full-attention KV and GDN recurrent caches use different speculative
        # layouts. SSM states are stored in per-compact blocks, while conv
        # states are read by stock vLLM from offsets inside the current running
        # state block. After DDTree accepts a non-flat branch, materialize that
        # branch's final recurrent state back into the flat locations vLLM
        # expects: conv at base_block[offset=output_count-1], SSM at
        # block[base_idx + output_count - 1]. Then the normal accepted-token
        # count can flow through postprocess_mamba unchanged.
        if os.environ.get("DDTREE_USE_RUNTIME_SAMPLER", "0") != "1":
            return
        if os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "1") != "1":
            return
        runtime_req_ids = getattr(self, "_last_ddtree_runtime_req_ids", None)
        bonus_parent_by_req = getattr(
            self, "_last_ddtree_bonus_parent_compact_indices", None
        )
        if not runtime_req_ids or not bonus_parent_by_req:
            return
        if not self.mamba_state_idx:
            return

        try:
            mamba_group_ids, _ = mamba_utils.get_mamba_groups(self.kv_cache_config)
        except Exception:
            return

        accepted_counts = self.num_accepted_tokens.gpu[:num_reqs].detach().cpu()
        req_to_batch_index = {
            req_id: idx for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs])
        }

        for req_id, bonus_parent in zip(
            runtime_req_ids, bonus_parent_by_req, strict=False
        ):
            req_i = req_to_batch_index.get(req_id)
            base_state_idx = self.mamba_state_idx.get(req_id)
            if req_i is None or base_state_idx is None:
                continue
            output_count = int(accepted_counts[req_i].item())
            if output_count <= 0:
                continue
            src_compact = int(bonus_parent)
            dst_compact = output_count - 1
            if src_compact == dst_compact:
                continue

            req_state = self.requests.get(req_id)
            if req_state is None:
                continue

            for mamba_group_id in mamba_group_ids:
                block_ids = req_state.block_ids[mamba_group_id]
                src_block_idx = base_state_idx + src_compact
                dst_ssm_block_idx = base_state_idx + dst_compact
                if (
                    base_state_idx < 0
                    or src_block_idx < 0
                    or dst_ssm_block_idx < 0
                    or base_state_idx >= len(block_ids)
                    or src_block_idx >= len(block_ids)
                    or dst_ssm_block_idx >= len(block_ids)
                ):
                    continue

                base_block_id = block_ids[base_state_idx]
                src_block_id = block_ids[src_block_idx]
                dst_ssm_block_id = block_ids[dst_ssm_block_idx]
                layer_names = self.kv_cache_config.kv_cache_groups[
                    mamba_group_id
                ].layer_names
                for layer_name in layer_names:
                    attention = self.compilation_config.static_forward_context.get(
                        layer_name
                    )
                    kv_caches = getattr(attention, "kv_cache", None)
                    if not isinstance(kv_caches, (list, tuple)) or len(kv_caches) < 2:
                        continue

                    conv_state, ssm_state = kv_caches[0], kv_caches[1]
                    conv_width = int(getattr(attention, "conv_kernel_size", 1)) - 1
                    if conv_width <= 0 or conv_state.ndim < 3:
                        continue

                    conv_dim = getattr(attention, "conv_dim", None)
                    # SD layout: [block, state_len, dim]. DS layout:
                    # [block, dim, state_len]. Default Qwen3.6/vLLM uses SD,
                    # but keep this robust for future SSM layout flips.
                    if conv_dim is not None and int(conv_state.shape[-1]) == int(conv_dim):
                        state_len = int(conv_state.shape[1])
                        if dst_compact + conv_width > state_len:
                            continue
                        conv_state[
                            base_block_id,
                            dst_compact : dst_compact + conv_width,
                        ].copy_(
                            conv_state[src_block_id, :conv_width].clone()
                        )
                    else:
                        state_len = int(conv_state.shape[-1])
                        if dst_compact + conv_width > state_len:
                            continue
                        conv_state[
                            base_block_id,
                            :,
                            dst_compact : dst_compact + conv_width,
                        ].copy_(
                            conv_state[src_block_id, :, :conv_width].clone()
                        )

                    if (
                        ssm_state.ndim >= 2
                        and src_block_id != dst_ssm_block_id
                        and src_block_id < ssm_state.shape[0]
                        and dst_ssm_block_id < ssm_state.shape[0]
                    ):
                        ssm_state[dst_ssm_block_id].copy_(
                            ssm_state[src_block_id].clone()
                        )

        if os.environ.get("DDTREE_LOG_STATE_COMPACT", "0") == "1" and not getattr(
            self, "_ddtree_m8r_logged", False
        ):
            logger.warning(
                "DDTree M8R recurrent state compaction enabled for %s request(s)",
                len(runtime_req_ids),
            )
            self._ddtree_m8r_logged = True

""",
        """    def _compact_ddtree_recurrent_states(
        self,
        num_reqs: int,
        spec_decode_common_attn_metadata: CommonAttentionMetadata | None,
    ) -> None:
        # aeon_dflash_ddtree_m8s
        # M8R used align-mode mamba_state_idx, but Qwen3.6 Spark runs commonly
        # use non-align mode. Build the exact GDN state block table from the
        # same common metadata the attention builders consume, then flatten the
        # chosen DDTree branch into vLLM's flat speculative recurrent layout.
        if os.environ.get("DDTREE_USE_RUNTIME_SAMPLER", "0") != "1":
            return
        if os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "1") != "1":
            return
        if spec_decode_common_attn_metadata is None:
            return
        runtime_req_ids = getattr(self, "_last_ddtree_runtime_req_ids", None)
        bonus_parent_by_req = getattr(
            self, "_last_ddtree_bonus_parent_compact_indices", None
        )
        if not runtime_req_ids or not bonus_parent_by_req:
            return

        try:
            mamba_group_ids, _ = mamba_utils.get_mamba_groups(self.kv_cache_config)
        except Exception:
            return

        accepted_counts = self.num_accepted_tokens.gpu[:num_reqs].detach().cpu()
        req_to_batch_index = {
            req_id: idx for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs])
        }
        compacted = 0

        for mamba_group_id in mamba_group_ids:
            kv_group = self.kv_cache_config.kv_cache_groups[mamba_group_id]
            kv_cache_spec = kv_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = kv_cache_spec.kv_cache_specs[kv_group.layer_names[0]]
            if not isinstance(kv_cache_spec, MambaSpec):
                continue
            block_table = mamba_get_block_table_tensor(
                spec_decode_common_attn_metadata.block_table_tensor,
                spec_decode_common_attn_metadata.seq_lens,
                kv_cache_spec,
                self.cache_config.mamba_cache_mode,
            )[:num_reqs].detach().cpu()

            for req_id, bonus_parent in zip(
                runtime_req_ids, bonus_parent_by_req, strict=False
            ):
                req_i = req_to_batch_index.get(req_id)
                if req_i is None:
                    continue
                output_count = int(accepted_counts[req_i].item())
                if output_count <= 0:
                    continue
                src_compact = int(bonus_parent)
                dst_compact = output_count - 1
                if (
                    src_compact == dst_compact
                    or src_compact < 0
                    or dst_compact < 0
                    or src_compact >= block_table.shape[1]
                    or dst_compact >= block_table.shape[1]
                ):
                    continue

                base_block_id = int(block_table[req_i, 0].item())
                src_block_id = int(block_table[req_i, src_compact].item())
                dst_ssm_block_id = int(block_table[req_i, dst_compact].item())
                if base_block_id <= 0 or src_block_id <= 0 or dst_ssm_block_id <= 0:
                    continue

                for layer_name in kv_group.layer_names:
                    attention = self.compilation_config.static_forward_context.get(
                        layer_name
                    )
                    kv_caches = getattr(attention, "kv_cache", None)
                    if not isinstance(kv_caches, (list, tuple)) or len(kv_caches) < 2:
                        continue

                    conv_state, ssm_state = kv_caches[0], kv_caches[1]
                    conv_width = int(getattr(attention, "conv_kernel_size", 1)) - 1
                    if conv_width <= 0 or conv_state.ndim < 3:
                        continue

                    conv_dim = getattr(attention, "conv_dim", None)
                    if conv_dim is not None and int(conv_state.shape[-1]) == int(conv_dim):
                        state_len = int(conv_state.shape[1])
                        if dst_compact + conv_width > state_len:
                            continue
                        conv_state[
                            base_block_id,
                            dst_compact : dst_compact + conv_width,
                        ].copy_(conv_state[src_block_id, :conv_width].clone())
                    else:
                        state_len = int(conv_state.shape[-1])
                        if dst_compact + conv_width > state_len:
                            continue
                        conv_state[
                            base_block_id,
                            :,
                            dst_compact : dst_compact + conv_width,
                        ].copy_(conv_state[src_block_id, :, :conv_width].clone())

                    if (
                        ssm_state.ndim >= 2
                        and src_block_id != dst_ssm_block_id
                        and src_block_id < ssm_state.shape[0]
                        and dst_ssm_block_id < ssm_state.shape[0]
                    ):
                        ssm_state[dst_ssm_block_id].copy_(
                            ssm_state[src_block_id].clone()
                        )
                    compacted += 1

        if os.environ.get("DDTREE_LOG_STATE_COMPACT", "0") == "1" and not getattr(
            self, "_ddtree_m8s_logged", False
        ):
            logger.warning(
                "DDTree M8S recurrent state compaction block-table path compacted=%s",
                compacted,
            )
            self._ddtree_m8s_logged = True

""",
    )
    replace_exact(
        path,
        """        self._compact_ddtree_recurrent_states(num_reqs)
""",
        """        self._compact_ddtree_recurrent_states(
            num_reqs, spec_decode_common_attn_metadata
        )
""",
    )


def verify_static(pkg_root: Path) -> None:
    runner = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m8s",
        "mamba_get_block_table_tensor",
        "spec_decode_common_attn_metadata: CommonAttentionMetadata | None",
        "DDTree M8S recurrent state compaction block-table path",
        "self._compact_ddtree_recurrent_states(",
    ):
        if needle not in runner:
            raise RuntimeError(f"Static M8S verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_imports(pkg_root)
    patch_recurrent_compaction(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree recurrent compaction via GDN block table verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
