#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m10l"


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


def patch_state_token_counts(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """        if os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "0") == "1":
            return None
        if os.environ.get("DDTREE_USE_RUNTIME_SAMPLER", "0") != "1":
            return None
""",
        """        if os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "0") == "1":
            if os.environ.get("DDTREE_ALLOW_BRANCH_STATE_COMPACTION", "0") != "1":
                return None
        if os.environ.get("DDTREE_USE_RUNTIME_SAMPLER", "0") != "1":
            return None
""",
    )
    replace_exact(
        path,
        """        # aeon_dflash_ddtree_m8w
        # M7A used bonus_parent+1 as the recurrent state cursor because branch
        # rollback had not yet compacted recurrent states into the committed
        # autoregressive prefix. M8T now performs that compaction explicitly.
        # Once compaction is enabled, keep vLLM's normal output-count cursor so
        # the next verifier step reloads the compacted state, not the stale
        # branch row in the previous tree window.
        if os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "0") == "1":
            return None

        state_counts = self.num_accepted_tokens.gpu[:num_reqs].clone()
""",
        """        # aeon_dflash_ddtree_m10l
        # Mamba postprocess uses state_count - 1 as the number of computed
        # verifier states to keep. In full branch-compaction mode M10J emits no
        # uncomputed target bonus for non-flat branches, so output_count is one
        # smaller than the computed branch-state cursor. Supply a state count
        # based on the number of accepted compact tree nodes while preserving
        # normal output-count behavior for safe/no-compaction mode.
        if os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "1") == "1":
            if os.environ.get("DDTREE_ALLOW_BRANCH_STATE_COMPACTION", "0") != "1":
                return None
            accepted_by_req = getattr(
                self, "_last_ddtree_accepted_compact_indices", None
            )
            if accepted_by_req is None:
                return None
            state_counts = self.num_accepted_tokens.gpu[:num_reqs].clone()
            req_to_batch_index = {
                req_id: idx
                for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs])
            }
            for req_id, accepted_compact in zip(
                runtime_req_ids, accepted_by_req, strict=False
            ):
                req_i = req_to_batch_index.get(req_id)
                if req_i is not None and accepted_compact:
                    state_counts[req_i] = len(accepted_compact) + 1
            return state_counts

        state_counts = self.num_accepted_tokens.gpu[:num_reqs].clone()
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m10l",
        "output_count is one",
        "DDTREE_ALLOW_BRANCH_STATE_COMPACTION",
        "state_counts[req_i] = len(accepted_compact) + 1",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M10L verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_state_token_counts(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] branch compaction Mamba state-count override installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
