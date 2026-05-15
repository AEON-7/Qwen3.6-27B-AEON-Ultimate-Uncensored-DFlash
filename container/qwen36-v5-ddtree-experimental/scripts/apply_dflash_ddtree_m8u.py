#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8u"


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
                # aeon_dflash_ddtree_m7a
                # Keep request ids paired with sampler branch metadata so the
                # state rollback code can map compact accepted nodes back to the
                # active batch order.
                self._last_ddtree_runtime_req_ids = [
                    request.req_id for request in runtime_metadata.requests
                ]
""",
        """                tree_sample = greedy_sample_ddtree(runtime_metadata, logits)
                # aeon_dflash_ddtree_m8u
                # Diagnostic visibility for quality bring-up: log the compact
                # branch actually accepted by the target verifier. A non-flat
                # accepted_compact path must be paired with recurrent-state
                # compaction after sampling, otherwise the next step reads the
                # wrong GDN state.
                if os.environ.get("DDTREE_LOG_SAMPLE", "0") == "1":
                    log_count = getattr(self, "_ddtree_m8u_sample_log_count", 0)
                    log_limit = int(os.environ.get("DDTREE_LOG_SAMPLE_LIMIT", "16"))
                    if log_count < log_limit:
                        first_request = runtime_metadata.requests[0]
                        logger.warning(
                            "DDTree M8U sample accepted=%s bonus_parent=%s "
                            "output=%s first_tokens=%s first_parents=%s",
                            tree_sample.accepted_compact_indices,
                            tree_sample.bonus_parent_compact_indices,
                            tree_sample.output_token_ids.detach().cpu().tolist(),
                            list(first_request.tree_token_ids)[:16],
                            list(first_request.parent_indices)[:16],
                        )
                        self._ddtree_m8u_sample_log_count = log_count + 1
                # aeon_dflash_ddtree_m7a
                # Keep request ids paired with sampler branch metadata so the
                # state rollback code can map compact accepted nodes back to the
                # active batch order.
                self._last_ddtree_runtime_req_ids = [
                    request.req_id for request in runtime_metadata.requests
                ]
""",
    )
    replace_exact(
        path,
        """        if os.environ.get("DDTREE_LOG_STATE_COMPACT", "0") == "1" and not getattr(
            self, "_ddtree_m8t_logged", False
        ):
            logger.warning(
                "DDTree M8T recurrent state compaction from GDN metadata compacted=%s",
                compacted,
            )
            self._ddtree_m8t_logged = True
""",
        """        if os.environ.get("DDTREE_LOG_STATE_COMPACT", "0") == "1":
            # aeon_dflash_ddtree_m8u
            log_count = getattr(self, "_ddtree_m8u_compact_log_count", 0)
            log_limit = int(os.environ.get("DDTREE_LOG_COMPACT_LIMIT", "16"))
            if log_count < log_limit:
                logger.warning(
                    "DDTree M8U recurrent compaction compacted=%s entries=%s "
                    "runtime_req_ids=%s bonus_parent=%s accepted_counts=%s",
                    compacted,
                    len(entries),
                    runtime_req_ids,
                    bonus_parent_by_req,
                    accepted_counts.tolist(),
                )
                self._ddtree_m8u_compact_log_count = log_count + 1
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m8u",
        "DDTree M8U sample accepted",
        "DDTree M8U recurrent compaction",
        "_ddtree_m8u_sample_log_count",
        "_ddtree_m8u_compact_log_count",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M8U verification failed: missing {needle}")


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
    print(f"[{MARKER}] DDTree sampler/compaction diagnostics verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
