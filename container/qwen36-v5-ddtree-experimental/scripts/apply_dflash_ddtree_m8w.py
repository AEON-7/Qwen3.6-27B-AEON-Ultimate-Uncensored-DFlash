#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8w"


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
        """        state_counts = self.num_accepted_tokens.gpu[:num_reqs].clone()
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
        """        # aeon_dflash_ddtree_m8w
        # M7A used bonus_parent+1 as the recurrent state cursor because branch
        # rollback had not yet compacted recurrent states into the committed
        # autoregressive prefix. M8T now performs that compaction explicitly.
        # Once compaction is enabled, keep vLLM's normal output-count cursor so
        # the next verifier step reloads the compacted state, not the stale
        # branch row in the previous tree window.
        if os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "1") == "1":
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


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m8w",
        "Once compaction is enabled",
        "DDTREE_COMPACT_RECURRENT_STATE",
        "return None",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M8W verification failed: missing {needle}")


def main() -> int:
    import os

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
    print(f"[{MARKER}] DDTree compacted recurrent-state cursor verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
