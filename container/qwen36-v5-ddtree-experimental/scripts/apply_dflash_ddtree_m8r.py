#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8r"


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


RECURRENT_COMPACT_FN = r'''    def _compact_ddtree_recurrent_states(
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

'''


def patch_gpu_model_runner(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """    def _ddtree_state_token_counts(
        self,
        num_reqs: int,
    ) -> torch.Tensor | None:
        # For hybrid recurrent state, vLLM's flat speculative path uses
        # output_count - 1 as the accepted draft state index. DDTree must instead
        # use the compact parent of the bonus token: the last accepted tree node.
        if os.environ.get("DDTREE_USE_RUNTIME_SAMPLER", "0") != "1":
            return None
        runtime_req_ids = getattr(self, "_last_ddtree_runtime_req_ids", None)
        bonus_parent_by_req = getattr(
            self, "_last_ddtree_bonus_parent_compact_indices", None
        )
        if not runtime_req_ids or not bonus_parent_by_req:
            return None
        state_counts = self.num_accepted_tokens.gpu[:num_reqs].clone()
        req_to_batch_index = {
            req_id: idx for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs])
        }
        for req_id, bonus_parent in zip(
            runtime_req_ids, bonus_parent_by_req, strict=False
        ):
            req_i = req_to_batch_index.get(req_id)
            if req_i is not None:
                state_counts[req_i] = int(bonus_parent) + 1
        return state_counts

""",
        RECURRENT_COMPACT_FN
        + """    def _ddtree_state_token_counts(
        self,
        num_reqs: int,
    ) -> torch.Tensor | None:
        # aeon_dflash_ddtree_m8r
        # M8R compacts the chosen branch state back into vLLM's flat recurrent
        # locations, so postprocess_mamba should receive the true committed
        # output count. Keep the old compact-index count available behind an
        # escape hatch for diagnostics.
        if os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "1") == "1":
            return None
        if os.environ.get("DDTREE_USE_RUNTIME_SAMPLER", "0") != "1":
            return None
        runtime_req_ids = getattr(self, "_last_ddtree_runtime_req_ids", None)
        bonus_parent_by_req = getattr(
            self, "_last_ddtree_bonus_parent_compact_indices", None
        )
        if not runtime_req_ids or not bonus_parent_by_req:
            return None
        state_counts = self.num_accepted_tokens.gpu[:num_reqs].clone()
        req_to_batch_index = {
            req_id: idx for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs])
        }
        for req_id, bonus_parent in zip(
            runtime_req_ids, bonus_parent_by_req, strict=False
        ):
            req_i = req_to_batch_index.get(req_id)
            if req_i is not None:
                state_counts[req_i] = int(bonus_parent) + 1
        return state_counts

""",
    )
    replace_exact(
        path,
        """        self._compact_ddtree_full_attention_kv(
            slot_mappings, spec_decode_common_attn_metadata
        )
        state_token_counts = self._ddtree_state_token_counts(num_reqs)
""",
        """        self._compact_ddtree_full_attention_kv(
            slot_mappings, spec_decode_common_attn_metadata
        )
        self._compact_ddtree_recurrent_states(num_reqs)
        state_token_counts = self._ddtree_state_token_counts(num_reqs)
""",
    )


def patch_fused_gdn_root_reload(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/fla/ops/fused_sigmoid_gating.py"
    replace_exact(
        path,
        """        if IS_DDTREE:
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
""",
        """        if IS_DDTREE:
            parent_t = tl.load(
                ddtree_parent_ids
                + i_n * stride_parent_ids_seq
                + i_t * stride_parent_ids_tok
            ).to(tl.int64)
            # aeon_dflash_ddtree_m8r
            # A root child at compact row >0 must reload the same rolling root
            # state used by row 0, not its own not-yet-written compact state.
            # Branch children still reload from their compact parent row.
            root_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            root_t = tl.maximum(root_t, 0)
            reload_t = tl.where(parent_t < 0, root_t, parent_t)
            should_reload = (i_t > 0) & (parent_t != i_t - 1)
            reload_state_idx = tl.load(
                ssm_state_indices
                + i_n * stride_indices_seq
                + reload_t * stride_indices_tok
            ).to(tl.int64)
""",
    )


def verify_static(pkg_root: Path) -> None:
    runner = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    fused = (
        pkg_root / "model_executor/layers/fla/ops/fused_sigmoid_gating.py"
    ).read_text()
    checks = {
        "gpu_model_runner.py": (
            "aeon_dflash_ddtree_m8r",
            "_compact_ddtree_recurrent_states",
            "DDTREE_COMPACT_RECURRENT_STATE",
            "conv_state[src_block_id, :conv_width].clone()",
            "ssm_state[dst_ssm_block_id].copy_",
            "self._compact_ddtree_recurrent_states(num_reqs)",
        ),
        "fused_sigmoid_gating.py": (
            "aeon_dflash_ddtree_m8r",
            "root_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1",
            "reload_t = tl.where(parent_t < 0, root_t, parent_t)",
        ),
    }
    texts = {"gpu_model_runner.py": runner, "fused_sigmoid_gating.py": fused}
    for rel, needles in checks.items():
        for needle in needles:
            if needle not in texts[rel]:
                raise RuntimeError(f"Static M8R verification failed: {rel} missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_gpu_model_runner(pkg_root)
    patch_fused_gdn_root_reload(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree flat recurrent state compaction verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
