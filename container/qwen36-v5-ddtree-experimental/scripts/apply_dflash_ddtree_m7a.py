#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m7a"


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
        """                tree_sample = greedy_sample_ddtree(runtime_metadata, logits)
                self._last_ddtree_accepted_compact_indices = (
                    tree_sample.accepted_compact_indices
                )
                self._last_ddtree_bonus_parent_compact_indices = (
                    tree_sample.bonus_parent_compact_indices
                )
                return SamplerOutput(sampled_token_ids=tree_sample.output_token_ids)
""",
        """                tree_sample = greedy_sample_ddtree(runtime_metadata, logits)
                # aeon_dflash_ddtree_m7a
                # Keep request ids paired with sampler branch metadata so the
                # state rollback code can map compact accepted nodes back to the
                # active batch order.
                self._last_ddtree_runtime_req_ids = [
                    request.req_id for request in runtime_metadata.requests
                ]
                self._last_ddtree_accepted_compact_indices = (
                    tree_sample.accepted_compact_indices
                )
                self._last_ddtree_bonus_parent_compact_indices = (
                    tree_sample.bonus_parent_compact_indices
                )
                return SamplerOutput(sampled_token_ids=tree_sample.output_token_ids)
""",
    )
    replace_exact(
        path,
        """        self._last_ddtree_metadata_payload = None
        self._last_ddtree_parent_ids_gpu = None
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
""",
        """        self._last_ddtree_metadata_payload = None
        self._last_ddtree_parent_ids_gpu = None
        self._last_ddtree_runtime_req_ids = None
        self._last_ddtree_accepted_compact_indices = None
        self._last_ddtree_bonus_parent_compact_indices = None
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
""",
    )
    replace_exact(
        path,
        """    def _update_states_after_model_execute(
        self, output_token_ids: torch.Tensor, scheduler_output: "SchedulerOutput"
    ) -> None:
""",
        """    def _compact_ddtree_full_attention_kv(
        self,
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
        spec_decode_common_attn_metadata: CommonAttentionMetadata | None,
    ) -> None:
        # aeon_dflash_ddtree_m7a
        # The target verifier executes root+tree nodes in compact DFS/tree
        # order, but the scheduler commits accepted tokens as a contiguous
        # autoregressive spine. Copy accepted branch KV states into the first
        # committed draft slots so the next decode step sees a normal sequence.
        if os.environ.get("DDTREE_USE_RUNTIME_SAMPLER", "0") != "1":
            return
        if spec_decode_common_attn_metadata is None or not isinstance(
            slot_mappings, dict
        ):
            return
        runtime_req_ids = getattr(self, "_last_ddtree_runtime_req_ids", None)
        accepted_by_req = getattr(self, "_last_ddtree_accepted_compact_indices", None)
        if not runtime_req_ids or not accepted_by_req:
            return

        query_start_loc = spec_decode_common_attn_metadata.query_start_loc
        req_to_batch_index = {
            req_id: idx for idx, req_id in enumerate(self.input_batch.req_ids)
        }
        for req_id, accepted_compact in zip(
            runtime_req_ids, accepted_by_req, strict=False
        ):
            req_i = req_to_batch_index.get(req_id)
            if req_i is None or not accepted_compact:
                continue
            q_start = int(query_start_loc[req_i].item())
            q_end = int(query_start_loc[req_i + 1].item())
            q_len = q_end - q_start
            if q_len <= 1:
                continue

            for layer_name, layer_slot_mapping in slot_mappings.items():
                attention = self.compilation_config.static_forward_context.get(
                    layer_name
                )
                kv_cache = getattr(attention, "kv_cache", None)
                if (
                    not isinstance(kv_cache, torch.Tensor)
                    or kv_cache.ndim < 5
                    or kv_cache.shape[0] != 2
                ):
                    continue
                block_size = kv_cache.shape[2]
                for dest_compact, src_compact in enumerate(accepted_compact, start=1):
                    if src_compact <= 0 or src_compact >= q_len:
                        continue
                    src_slot = int(layer_slot_mapping[q_start + src_compact].item())
                    dst_slot = int(layer_slot_mapping[q_start + dest_compact].item())
                    if src_slot < 0 or dst_slot < 0 or src_slot == dst_slot:
                        continue
                    src_block = src_slot // block_size
                    src_offset = src_slot % block_size
                    dst_block = dst_slot // block_size
                    dst_offset = dst_slot % block_size
                    kv_cache[:, dst_block, dst_offset].copy_(
                        kv_cache[:, src_block, src_offset].clone()
                    )

    def _ddtree_state_token_counts(
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

    def _update_states_after_model_execute(
        self,
        output_token_ids: torch.Tensor,
        scheduler_output: "SchedulerOutput",
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
        spec_decode_common_attn_metadata: CommonAttentionMetadata | None = None,
    ) -> None:
""",
    )
    replace_exact(
        path,
        """        # Count the number of accepted tokens for each sequence.
        # Valid tokens are contiguous from position 0, so counting non-(-1)
        # tokens gives us the first -1 position (i.e., number of accepted).
        num_reqs = output_token_ids.size(0)
        self.num_accepted_tokens.gpu[:num_reqs] = (output_token_ids != -1).sum(dim=1)

        if self.cache_config.mamba_cache_mode == "align":
            for i, num_tokens in enumerate(
                self.num_accepted_tokens.gpu[:num_reqs].cpu().numpy()
            ):
                self.input_batch.num_accepted_tokens_cpu[i] = num_tokens
            mamba_utils.postprocess_mamba(
""",
        """        # Count the number of accepted tokens for each sequence.
        # Valid tokens are contiguous from position 0, so counting non-(-1)
        # tokens gives us the first -1 position (i.e., number of accepted).
        num_reqs = output_token_ids.size(0)
        self.num_accepted_tokens.gpu[:num_reqs] = (output_token_ids != -1).sum(dim=1)
        self._compact_ddtree_full_attention_kv(
            slot_mappings, spec_decode_common_attn_metadata
        )
        state_token_counts = self._ddtree_state_token_counts(num_reqs)
        if state_token_counts is None:
            state_token_counts = self.num_accepted_tokens.gpu[:num_reqs]

        if self.cache_config.mamba_cache_mode == "align":
            for i, num_tokens in enumerate(state_token_counts.cpu().numpy()):
                self.input_batch.num_accepted_tokens_cpu[i] = num_tokens
            mamba_utils.postprocess_mamba(
""",
    )
    replace_exact(
        path,
        """        else:
            self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
                self.num_accepted_tokens.gpu[:num_reqs], non_blocking=True
            )
""",
        """        else:
            self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
                state_token_counts, non_blocking=True
            )
""",
    )
    replace_exact(
        path,
        """        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids, scheduler_output
        )
""",
        """        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids,
            scheduler_output,
            slot_mappings,
            spec_decode_common_attn_metadata,
        )
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m7a",
        "_compact_ddtree_full_attention_kv",
        "_ddtree_state_token_counts",
        "kv_cache[:, dst_block, dst_offset].copy_",
        "_last_ddtree_runtime_req_ids",
        "state_token_counts",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M7A verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_gpu_model_runner(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree accepted-branch KV/state compaction verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
