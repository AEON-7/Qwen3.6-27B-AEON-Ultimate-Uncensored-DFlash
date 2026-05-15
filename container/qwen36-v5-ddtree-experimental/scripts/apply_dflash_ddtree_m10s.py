#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m10s"


def replace_exact(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Could not find expected text in {path}:\n{old}")
    path.write_text(text.replace(old, new, 1))


def clear_python_caches(pkg_root: Path) -> None:
    for pyc in pkg_root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    for pycache in pkg_root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


DDTREE_DRAFTER_CONTEXT_COMPACT_FN = r'''    def _compact_ddtree_drafter_context(
        self,
        hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_common_attn_metadata: CommonAttentionMetadata | None,
    ) -> None:
        # aeon_dflash_ddtree_m10s
        # DFlash uses target hidden states as the context feature stream for
        # the next draft pass. Full-attention KV and GDN recurrent caches are
        # compacted from DFS/tree order into the committed flat spine after a
        # non-flat DDTree branch is accepted; the hidden-state feature stream
        # needs the same treatment. This mirrors Lucebox's target_feat
        # compaction: accepted DFS slots are copied into slots 1..k so the next
        # drafter conditions on the branch that was actually committed.
        if os.environ.get("DDTREE_USE_RUNTIME_SAMPLER", "0") != "1":
            return
        if os.environ.get("DDTREE_COMPACT_DRAFTER_CONTEXT", "1") != "1":
            return
        if spec_decode_common_attn_metadata is None:
            return
        runtime_req_ids = getattr(self, "_last_ddtree_runtime_req_ids", None)
        accepted_by_req = getattr(self, "_last_ddtree_accepted_compact_indices", None)
        if not runtime_req_ids or accepted_by_req is None:
            return

        query_start_loc = spec_decode_common_attn_metadata.query_start_loc
        req_to_batch_index = {
            req_id: idx for idx, req_id in enumerate(self.input_batch.req_ids)
        }
        tensors: list[torch.Tensor] = [hidden_states]
        if aux_hidden_states is not None:
            tensors.extend(aux_hidden_states)

        copied = 0
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

            for dest_compact, src_compact in enumerate(accepted_compact, start=1):
                src_compact = int(src_compact)
                if (
                    src_compact <= 0
                    or src_compact >= q_len
                    or dest_compact >= q_len
                    or src_compact == dest_compact
                ):
                    continue

                src_i = q_start + src_compact
                dst_i = q_start + dest_compact
                for tensor in tensors:
                    if src_i < tensor.shape[0] and dst_i < tensor.shape[0]:
                        tensor[dst_i].copy_(tensor[src_i].clone())

                # Keep token ids aligned with the compacted context stream used
                # by DFlash's copy_and_expand_dflash_inputs kernel.
                if (
                    src_i < self.input_ids.gpu.shape[0]
                    and dst_i < self.input_ids.gpu.shape[0]
                ):
                    self.input_ids.gpu[dst_i].copy_(self.input_ids.gpu[src_i])
                copied += 1

        if os.environ.get("DDTREE_LOG_CONTEXT_COMPACT", "0") == "1":
            log_count = getattr(self, "_ddtree_m10s_context_log_count", 0)
            log_limit = int(os.environ.get("DDTREE_LOG_COMPACT_LIMIT", "16"))
            if log_count < log_limit:
                logger.warning(
                    "DDTree M10S drafter context compaction copied=%s "
                    "runtime_req_ids=%s accepted=%s",
                    copied,
                    runtime_req_ids,
                    accepted_by_req,
                )
                self._ddtree_m10s_context_log_count = log_count + 1

'''


def patch_gpu_model_runner(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """    def _update_states_after_model_execute(
        self,
        output_token_ids: torch.Tensor,
        scheduler_output: "SchedulerOutput",
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
        spec_decode_common_attn_metadata: CommonAttentionMetadata | None = None,
    ) -> None:
""",
        DDTREE_DRAFTER_CONTEXT_COMPACT_FN
        + """    def _update_states_after_model_execute(
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
        """        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids,
            scheduler_output,
            slot_mappings,
            spec_decode_common_attn_metadata,
        )
""",
        """        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids,
            scheduler_output,
            slot_mappings,
            spec_decode_common_attn_metadata,
        )
        self._compact_ddtree_drafter_context(
            hidden_states,
            aux_hidden_states,
            spec_decode_common_attn_metadata,
        )
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m10s",
        "DDTREE_COMPACT_DRAFTER_CONTEXT",
        "DDTree M10S drafter context compaction",
        "self._compact_ddtree_drafter_context(",
        "self.input_ids.gpu[dst_i].copy_",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M10S verification failed: missing {needle}")


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
    print(f"[{MARKER}] branch drafter-context compaction installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
