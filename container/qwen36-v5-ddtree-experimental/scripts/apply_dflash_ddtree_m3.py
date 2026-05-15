#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m3"


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


def patch_outputs(pkg_root: Path) -> None:
    path = pkg_root / "v1/outputs.py"
    replace_exact(
        path,
        """@dataclass
class DraftTokenIds:
    # [num_reqs]
    req_ids: list[str]
    # num_reqs x num_draft_tokens
    draft_token_ids: list[list[int]]
""",
        """@dataclass
class DraftTokenIds:
    # [num_reqs]
    req_ids: list[str]
    # num_reqs x num_draft_tokens
    draft_token_ids: list[list[int]]
    # aeon_dflash_ddtree_m3
    # Optional request_id -> tree verifier payload. Flat spec decode leaves this
    # unset; DDTree-capable proposers can attach flattened tree metadata here
    # without changing the existing draft-token ABI.
    draft_trees: dict[str, object] | None = None
""",
    )


def patch_request(pkg_root: Path) -> None:
    path = pkg_root / "v1/request.py"
    replace_exact(
        path,
        """        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
""",
        """        self.spec_token_ids: list[int] = []
        # aeon_dflash_ddtree_m3
        # Optional DDTree verifier payload paired with spec_token_ids. It is
        # consumed and cleared by the scheduler in the same step as the flat
        # draft tokens.
        self.spec_tree: Any | None = None
        self.num_computed_tokens = 0
""",
    )


def patch_scheduler_output(pkg_root: Path) -> None:
    path = pkg_root / "v1/core/sched/output.py"
    replace_exact(
        path,
        """    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> encoder input indices that need processing.
""",
        """    scheduled_spec_decode_tokens: dict[str, list[int]]
    # aeon_dflash_ddtree_m3
    # req_id -> optional DDTree verifier payload for requests whose scheduled
    # spec tokens represent flattened tree nodes.
    scheduled_spec_decode_trees: dict[str, object]
    # req_id -> encoder input indices that need processing.
""",
    )
    replace_exact(
        path,
        """            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
""",
        """            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_spec_decode_trees={},
            scheduled_encoder_inputs={},
""",
    )


def patch_scheduler(pkg_root: Path) -> None:
    path = pkg_root / "v1/core/sched/scheduler.py"
    replace_exact(
        path,
        """        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
""",
        """        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        # aeon_dflash_ddtree_m3: optional tree metadata paired with scheduled
        # flattened spec tokens.
        scheduled_spec_decode_trees: dict[str, object] = {}

        # For logging.
""",
    )
    replace_exact(
        path,
        """                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
""",
        """                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            scheduled_spec_decode_trees.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
""",
    )
    replace_exact(
        path,
        """                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []
""",
        """                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
                    if request.spec_tree is not None:
                        tree_payload = request.spec_tree
                        if isinstance(tree_payload, dict):
                            tree_payload = dict(tree_payload)
                            tree_payload["scheduled_token_ids"] = list(spec_token_ids)
                            tree_payload["num_scheduled_tokens"] = len(spec_token_ids)
                        scheduled_spec_decode_trees[request.request_id] = tree_payload

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []
                request.spec_tree = None
""",
    )
    replace_exact(
        path,
        """            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
""",
        """            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_spec_decode_trees=scheduled_spec_decode_trees,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
""",
    )
    replace_exact(
        path,
        """    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
""",
        """    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        draft_trees = draft_token_ids.draft_trees or {}
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
""",
    )
    replace_exact(
        path,
        """            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue
""",
        """            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                request.spec_tree = None
                continue
""",
    )
    replace_exact(
        path,
        """            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids
""",
        """            # Add newly generated spec token ids to the request.
            request.spec_tree = draft_trees.get(req_id)
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
                # Grammar validation may trim or rewrite a flat draft. Drop the
                # tree payload when that happens so the verifier never sees a
                # topology that no longer matches the scheduled token list.
                request.spec_tree = None
            request.spec_token_ids = spec_token_ids
""",
    )
    replace_exact(
        path,
        """        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
""",
        """        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        sched_spec_trees = scheduler_output.scheduled_spec_decode_trees
        for req_id, spec_token_ids in zip(
""",
    )
    replace_exact(
        path,
        """            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids
""",
        """            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens
                sched_spec_trees.pop(req_id, None)
            elif req_id in sched_spec_trees:
                tree_payload = sched_spec_trees[req_id]
                if isinstance(tree_payload, dict):
                    tree_payload = dict(tree_payload)
                    tree_payload["scheduled_token_ids"] = list(spec_token_ids)
                    tree_payload["num_scheduled_tokens"] = len(spec_token_ids)
                    sched_spec_trees[req_id] = tree_payload

            sched_spec_tokens[req_id] = spec_token_ids
""",
    )


def patch_gpu_model_runner(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """            spec_decode_tokens_copy = (
                scheduler_output.scheduled_spec_decode_tokens.copy()
            )
            scheduler_output = replace(
                scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens_copy,
                scheduled_spec_decode_tokens=spec_decode_tokens_copy,
            )
""",
        """            spec_decode_tokens_copy = (
                scheduler_output.scheduled_spec_decode_tokens.copy()
            )
            spec_decode_trees_copy = scheduler_output.scheduled_spec_decode_trees.copy()
            scheduler_output = replace(
                scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens_copy,
                scheduled_spec_decode_tokens=spec_decode_tokens_copy,
                scheduled_spec_decode_trees=spec_decode_trees_copy,
            )
""",
    )
    replace_exact(
        path,
        """        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
""",
        """        # aeon_dflash_ddtree_m3: bridge-only hook. Later milestones consume
        # this payload to build tree masks/logit gathers; flat DFlash keeps the
        # existing SpecDecodeMetadata path below.
        self._last_ddtree_metadata_payload = None
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
""",
    )
    replace_exact(
        path,
        """            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens
            )
            logits_indices = spec_decode_metadata.logits_indices
""",
        """            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens
            )
            self._last_ddtree_metadata_payload = (
                scheduler_output.scheduled_spec_decode_trees or None
            )
            logits_indices = spec_decode_metadata.logits_indices
""",
    )


def verify_static(pkg_root: Path) -> None:
    expected = {
        "v1/outputs.py": ("draft_trees",),
        "v1/request.py": ("self.spec_tree",),
        "v1/core/sched/output.py": ("scheduled_spec_decode_trees",),
        "v1/core/sched/scheduler.py": (
            "draft_trees = draft_token_ids.draft_trees or {}",
            "scheduled_spec_decode_trees",
        ),
        "v1/worker/gpu_model_runner.py": ("_last_ddtree_metadata_payload",),
    }
    for rel, needles in expected.items():
        text = (pkg_root / rel).read_text()
        for needle in needles:
            if needle not in text:
                raise RuntimeError(f"Static M3 verification failed: {rel} missing {needle}")


def verify_imports() -> None:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import DraftTokenIds

    tree_payload = {
        "tree_token_ids": [11, 21, 31],
        "parent_indices": [-1, 0, 1],
    }
    draft = DraftTokenIds(
        req_ids=["req-1"],
        draft_token_ids=[[11, 21, 31]],
        draft_trees={"req-1": tree_payload},
    )
    if draft.draft_trees != {"req-1": tree_payload}:
        raise RuntimeError("DraftTokenIds did not preserve draft_trees")

    empty = SchedulerOutput.make_empty()
    if empty.scheduled_spec_decode_trees != {}:
        raise RuntimeError("SchedulerOutput.make_empty did not initialize tree payloads")


def main() -> int:
    pkg_root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if pkg_root_override:
        pkg_root = Path(pkg_root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_outputs(pkg_root)
    patch_request(pkg_root)
    patch_scheduler_output(pkg_root)
    patch_scheduler(pkg_root)
    patch_gpu_model_runner(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    if not pkg_root_override:
        verify_imports()
    print(f"[{MARKER}] DDTree scheduler/model-runner bridge verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
