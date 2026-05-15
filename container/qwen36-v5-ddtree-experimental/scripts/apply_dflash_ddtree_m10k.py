#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m10k"


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


def patch_recurrent_destination(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """        bonus_parent_by_req = getattr(
            self, "_last_ddtree_bonus_parent_compact_indices", None
        )
        if not runtime_req_ids or not bonus_parent_by_req:
            return

        accepted_counts = self.num_accepted_tokens.gpu[:num_reqs].detach().cpu()
""",
        """        bonus_parent_by_req = getattr(
            self, "_last_ddtree_bonus_parent_compact_indices", None
        )
        accepted_by_req = getattr(
            self, "_last_ddtree_accepted_compact_indices", None
        )
        if not runtime_req_ids or not bonus_parent_by_req or accepted_by_req is None:
            return

        accepted_counts = self.num_accepted_tokens.gpu[:num_reqs].detach().cpu()
""",
    )
    replace_exact(
        path,
        """            for req_id, bonus_parent in zip(
                runtime_req_ids, bonus_parent_by_req, strict=False
            ):
                req_i = req_to_batch_index.get(req_id)
                if req_i is None or req_i not in row_by_req_index:
                    continue
                output_count = int(accepted_counts[req_i].item())
                if output_count <= 0:
                    continue
                src_compact = int(bonus_parent)
                dst_compact = output_count - 1
                row = state_indices[row_by_req_index[req_i]]
""",
        """            for req_id, bonus_parent, accepted_compact in zip(
                runtime_req_ids, bonus_parent_by_req, accepted_by_req, strict=False
            ):
                req_i = req_to_batch_index.get(req_id)
                if req_i is None or req_i not in row_by_req_index:
                    continue
                output_count = int(accepted_counts[req_i].item())
                if output_count <= 0:
                    continue
                src_compact = int(bonus_parent)
                # aeon_dflash_ddtree_m10k
                # Full-attention KV compaction writes accepted tree states into
                # compact destination slots 1..len(accepted_compact). Recurrent
                # state must use the same destination. The old output_count-1
                # formula only worked when a target bonus token was emitted;
                # branch-compaction mode intentionally emits no uncomputed
                # bonus, so output_count-1 pointed one slot too early.
                dst_compact = len(accepted_compact)
                row = state_indices[row_by_req_index[req_i]]
""",
    )
    replace_exact(
        path,
        """                    "runtime_req_ids=%s bonus_parent=%s accepted_counts=%s",
                    compacted,
                    len(entries),
                    runtime_req_ids,
                    bonus_parent_by_req,
                    accepted_counts.tolist(),
                )
""",
        """                    "runtime_req_ids=%s bonus_parent=%s accepted=%s "
                    "accepted_counts=%s",
                    compacted,
                    len(entries),
                    runtime_req_ids,
                    bonus_parent_by_req,
                    accepted_by_req,
                    accepted_counts.tolist(),
                )
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m10k",
        "accepted_by_req = getattr(",
        "dst_compact = len(accepted_compact)",
        "branch-compaction mode intentionally emits no uncomputed",
        "accepted=%s",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M10K verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_recurrent_destination(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] recurrent compaction destination matches KV compaction")
    return 0


if __name__ == "__main__":
    sys.exit(main())
