#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m11a"


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


def patch_outputs(pkg_root: Path) -> None:
    path = pkg_root / "v1/outputs.py"
    replace_exact(
        path,
        """    # information related to cudagraph execution
    cudagraph_stats: CUDAGraphStat | None = None
""",
        """    # information related to cudagraph execution
    cudagraph_stats: CUDAGraphStat | None = None

    # aeon_dflash_ddtree_m11a
    # Optional request-id -> accepted draft token count. Stock vLLM infers this
    # as len(generated_tokens) - 1 because flat speculative decoding always
    # emits an uncomputed target bonus. DDTree full-branch commit can emit only
    # computed accepted branch tokens, so the scheduler needs an explicit count.
    spec_decode_num_accepted: dict[str, int] | None = None
""",
    )


def patch_scheduler(pkg_root: Path) -> None:
    path = pkg_root / "v1/core/sched/scheduler.py"
    replace_exact(
        path,
        """            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
""",
        """            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                explicit_num_accepted = getattr(
                    model_runner_output, "spec_decode_num_accepted", None
                )
                if explicit_num_accepted and req_id in explicit_num_accepted:
                    # aeon_dflash_ddtree_m11a
                    # DDTree branch commit can return accepted branch tokens
                    # without a target bonus. Preserve vLLM's flat default when
                    # no explicit count is supplied.
                    num_accepted = int(explicit_num_accepted[req_id])
                    num_accepted = max(0, min(num_accepted, num_draft_tokens))
                else:
                    num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
""",
    )


def patch_gpu_model_runner(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """        ddtree_payload = getattr(self, "_last_ddtree_metadata_payload", None)
""",
        """        # aeon_dflash_ddtree_m11a
        # Avoid carrying a previous DDTree accepted-count map into a non-DDTree
        # step. A new map is installed only when the runtime tree sampler runs.
        self._last_ddtree_num_accepted_tokens = None
        ddtree_payload = getattr(self, "_last_ddtree_metadata_payload", None)
""",
    )
    replace_exact(
        path,
        """                self._last_ddtree_bonus_parent_compact_indices = (
                    tree_sample.bonus_parent_compact_indices
                )
                return SamplerOutput(
""",
        """                self._last_ddtree_bonus_parent_compact_indices = (
                    tree_sample.bonus_parent_compact_indices
                )
                self._last_ddtree_num_accepted_tokens = {
                    request.req_id: len(accepted_compact)
                    for request, accepted_compact in zip(
                        runtime_metadata.requests,
                        tree_sample.accepted_compact_indices,
                        strict=False,
                    )
                }
                return SamplerOutput(
""",
    )
    replace_exact(
        path,
        """            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output
                if self.supports_mm_inputs
                else None,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
                routed_experts_dict=routed_experts_dict,
            )
""",
        """            ddtree_num_accepted = getattr(
                self, "_last_ddtree_num_accepted_tokens", None
            )
            if ddtree_num_accepted:
                ddtree_num_accepted = {
                    req_id: int(ddtree_num_accepted[req_id])
                    for req_id in req_ids_output_copy
                    if req_id in ddtree_num_accepted
                } or None
            self._last_ddtree_num_accepted_tokens = None

            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output
                if self.supports_mm_inputs
                else None,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
                routed_experts_dict=routed_experts_dict,
                spec_decode_num_accepted=ddtree_num_accepted,
            )
""",
    )


def verify_static(pkg_root: Path) -> None:
    checks = {
        "v1/outputs.py": (
            "spec_decode_num_accepted: dict[str, int] | None = None",
            "aeon_dflash_ddtree_m11a",
        ),
        "v1/core/sched/scheduler.py": (
            "explicit_num_accepted",
            "spec_decode_num_accepted",
            "num_accepted = int(explicit_num_accepted[req_id])",
        ),
        "v1/worker/gpu_model_runner.py": (
            "_last_ddtree_num_accepted_tokens",
            "spec_decode_num_accepted=ddtree_num_accepted",
        ),
        "v1/spec_decode/ddtree_runtime_sampler.py": (
            "DDTREE_FULL_BRANCH_COMMIT",
            "computed accepted tokens",
        ),
    }
    for rel, needles in checks.items():
        text = (pkg_root / rel).read_text()
        for needle in needles:
            if needle not in text:
                raise RuntimeError(f"M11A static verification failed: {rel} missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_outputs(pkg_root)
    patch_scheduler(pkg_root)
    patch_gpu_model_runner(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] explicit DDTree accepted-count channel installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
