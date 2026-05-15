#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8h"


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


def patch_pending_payload_rehydrate(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """            self._last_ddtree_metadata_payload = (
                scheduler_output.scheduled_spec_decode_trees or None
            )
            # aeon_dflash_ddtree_m8c
""",
        """            self._last_ddtree_metadata_payload = (
                scheduler_output.scheduled_spec_decode_trees or None
            )
            # aeon_dflash_ddtree_m8h
            # Some DFlash paths return tree payloads from the proposer before
            # the scheduler bridge has draft_trees populated. Preserve that
            # real proposer payload across the next target step rather than
            # letting this per-step reset erase it before attention metadata is
            # built.
            if self._last_ddtree_metadata_payload is None:
                pending_payload = getattr(
                    self, "_pending_ddtree_payload_by_req_id", None
                )
                if pending_payload:
                    self._last_ddtree_metadata_payload = pending_payload
                    if not getattr(
                        self, "_ddtree_pending_payload_rehydrate_logged", False
                    ):
                        logger.warning(
                            "DDTree rehydrated pending proposer payloads=%s "
                            "for attention metadata",
                            len(pending_payload),
                        )
                        self._ddtree_pending_payload_rehydrate_logged = True
            # aeon_dflash_ddtree_m8c
""",
    )
    replace_exact(
        path,
        """                        self._last_ddtree_metadata_payload = payload_by_req_id
                        self._last_ddtree_parent_ids_gpu = (
                            live_parent_metadata.parent_ids
                        )
""",
        """                        self._pending_ddtree_payload_by_req_id = payload_by_req_id
                        self._last_ddtree_metadata_payload = payload_by_req_id
                        self._last_ddtree_parent_ids_gpu = (
                            live_parent_metadata.parent_ids
                        )
""",
    )


def patch_take_draft_diagnostics(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if not self.num_spec_tokens or not self._draft_token_req_ids:
            return None
        draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
""",
        """    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if self.num_spec_tokens and not self._draft_token_req_ids:
            fallback_req_ids = getattr(self, "_draft_tree_req_ids_cpu", None)
            if fallback_req_ids:
                self._draft_token_req_ids = fallback_req_ids
        if not self.num_spec_tokens or not self._draft_token_req_ids:
            if not getattr(self, "_ddtree_take_draft_early_log_once", False):
                logger.warning(
                    "DDTree take_draft_token_ids early none num_spec=%s "
                    "req_ids=%s payloads=%s draft_ids_type=%s",
                    self.num_spec_tokens,
                    self._draft_token_req_ids,
                    len(getattr(self, "_draft_tree_payloads_cpu", None) or ()),
                    type(getattr(self, "_draft_token_ids", None)).__name__,
                )
                self._ddtree_take_draft_early_log_once = True
            return None
        draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
""",
    )


def patch_core_post_step_trace(pkg_root: Path) -> None:
    path = pkg_root / "v1/engine/core.py"
    replace_exact(
        path,
        """        if not self.async_scheduling and self.use_spec_decode and model_executed:
            # Take the draft token ids.
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)
""",
        """        if not self.async_scheduling and self.use_spec_decode and model_executed:
            if not getattr(self, "_ddtree_core_post_step_log_once", False):
                logger.warning(
                    "DDTree core post_step taking drafts async=%s model_executed=%s",
                    self.async_scheduling,
                    model_executed,
                )
                self._ddtree_core_post_step_log_once = True
            # Take the draft token ids.
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                if not getattr(self, "_ddtree_core_draft_taken_log_once", False):
                    logger.warning(
                        "DDTree core received draft ids trees=%s",
                        (
                            len(draft_token_ids.draft_trees)
                            if draft_token_ids.draft_trees
                            else None
                        ),
                    )
                    self._ddtree_core_draft_taken_log_once = True
                self.scheduler.update_draft_token_ids(draft_token_ids)
""",
    )


def verify_static(pkg_root: Path) -> None:
    checks = {
        "v1/worker/gpu_model_runner.py": (
            "DDTree rehydrated pending proposer payloads",
            "_pending_ddtree_payload_by_req_id",
            "DDTree take_draft_token_ids early none",
        ),
        "v1/engine/core.py": (
            "DDTree core post_step taking drafts",
            "DDTree core received draft ids",
        ),
    }
    for rel, needles in checks.items():
        text = (pkg_root / rel).read_text()
        for needle in needles:
            if needle not in text:
                raise RuntimeError(f"Static M8H verification failed: {rel} missing {needle}")


def main() -> int:
    pkg_root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if pkg_root_override:
        pkg_root = Path(pkg_root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_pending_payload_rehydrate(pkg_root)
    patch_take_draft_diagnostics(pkg_root)
    patch_core_post_step_trace(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] pending DDTree payload rehydrate verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
