#!/usr/bin/env python3
from __future__ import annotations

import inspect
from dataclasses import fields


def test_draft_token_ids_carries_tree_payload() -> None:
    from vllm.v1.outputs import DraftTokenIds

    field_names = {field.name for field in fields(DraftTokenIds)}
    assert "draft_trees" in field_names

    payload = {"tree_token_ids": [11, 21], "parent_indices": [-1, 0]}
    draft = DraftTokenIds(
        req_ids=["req-a"],
        draft_token_ids=[[11, 21]],
        draft_trees={"req-a": payload},
    )
    assert draft.draft_trees == {"req-a": payload}

    flat = DraftTokenIds(req_ids=["req-b"], draft_token_ids=[[31, 41]])
    assert flat.draft_trees is None


def test_scheduler_output_has_empty_tree_payload_map() -> None:
    from vllm.v1.core.sched.output import SchedulerOutput

    field_names = {field.name for field in fields(SchedulerOutput)}
    assert "scheduled_spec_decode_trees" in field_names
    assert SchedulerOutput.make_empty().scheduled_spec_decode_trees == {}


def test_request_and_scheduler_sources_keep_tree_payload_in_step() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request

    request_init = inspect.getsource(Request.__init__)
    assert "self.spec_tree" in request_init

    update_drafts = inspect.getsource(Scheduler.update_draft_token_ids)
    assert "draft_trees = draft_token_ids.draft_trees or {}" in update_drafts
    assert "request.spec_tree = draft_trees.get(req_id)" in update_drafts

    schedule = inspect.getsource(Scheduler.schedule)
    assert "scheduled_spec_decode_trees" in schedule
    assert "request.spec_tree = None" in schedule


def test_gpu_model_runner_sources_receive_tree_payload() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    execute = inspect.getsource(GPUModelRunner.execute_model)
    assert "scheduled_spec_decode_trees=spec_decode_trees_copy" in execute

    prepare = inspect.getsource(GPUModelRunner._prepare_inputs)
    assert "_last_ddtree_metadata_payload" in prepare
    assert "scheduler_output.scheduled_spec_decode_trees or None" in prepare


def main() -> int:
    test_draft_token_ids_carries_tree_payload()
    test_scheduler_output_has_empty_tree_payload_map()
    test_request_and_scheduler_sources_keep_tree_payload_in_step()
    test_gpu_model_runner_sources_receive_tree_payload()
    print("dflash_ddtree M3 bridge tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
