#!/usr/bin/env python3
"""M11O: keep Qwen3.6 GDN running state authoritative after branch commit.

M11L mirrors the accepted non-flat branch state into vLLM's running
``mamba_state_idx`` block. In align-mode, vLLM's stock ``postprocess_mamba``
then advances/copies that running block using the flat speculative-token
cursor. For arbitrary DDTree branches that cursor can point at a stale flat
offset even though the running block already contains the branch-final state.

This patch adds an explicit diagnostic/deploy knob:

``DDTREE_FULL_BRANCH_FREEZE_MAMBA_POSTPROCESS=1``

When a non-flat branch is committed, report a postprocess state count of 1 for
that request. That preserves scheduler-visible accepted-token accounting via
``spec_decode_num_accepted`` while making the next GDN verifier step read the
branch-final state mirrored into the running block at offset 0.
"""

from __future__ import annotations

from pathlib import Path

import vllm


ROOT = Path(vllm.__file__).resolve().parent
RUNNER = ROOT / "v1" / "worker" / "gpu_model_runner.py"


def replace_exact(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        raise SystemExit(f"pattern not found in {path}: {old[:160]!r}")
    path.write_text(text.replace(old, new, 1))


replace_exact(
    RUNNER,
    """            if accepted_by_req is None:
                return None
            state_counts = self.num_accepted_tokens.gpu[:num_reqs].clone()
            req_to_batch_index = {
""",
    """            if accepted_by_req is None:
                return None
            if os.environ.get("DDTREE_FULL_BRANCH_FREEZE_MAMBA_POSTPROCESS", "0") == "1":
                state_counts = self.num_accepted_tokens.gpu[:num_reqs].clone()
                req_to_batch_index = {
                    req_id: idx
                    for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs])
                }
                for req_id, accepted_compact in zip(
                    runtime_req_ids, accepted_by_req, strict=False
                ):
                    req_i = req_to_batch_index.get(req_id)
                    if req_i is None or not accepted_compact:
                        continue
                    nonflat = any(
                        int(compact) != index + 1
                        for index, compact in enumerate(accepted_compact)
                    )
                    if nonflat:
                        state_counts[req_i] = 1
                return state_counts
            state_counts = self.num_accepted_tokens.gpu[:num_reqs].clone()
            req_to_batch_index = {
""",
)

text = RUNNER.read_text()
if "DDTREE_FULL_BRANCH_FREEZE_MAMBA_POSTPROCESS" not in text:
    raise SystemExit("M11O verification failed")

print("Applied AEON DFlash DDTree M11O Mamba postprocess freeze knob")
