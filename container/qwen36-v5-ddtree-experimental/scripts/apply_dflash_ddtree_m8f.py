#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8f"


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


def patch_live_parent_metadata(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """                if not getattr(self, "_ddtree_pop_payloads_log_once", False):
                    logger.warning(
                        "DDTree runner popped payloads=%s active_req_ids=%s",
                        (
                            len(self._draft_tree_payloads_cpu)
                            if self._draft_tree_payloads_cpu is not None
                            else None
                        ),
                        len(self._draft_tree_req_ids_cpu),
                    )
                    self._ddtree_pop_payloads_log_once = True
            else:
""",
        """                if not getattr(self, "_ddtree_pop_payloads_log_once", False):
                    logger.warning(
                        "DDTree runner popped payloads=%s active_req_ids=%s",
                        (
                            len(self._draft_tree_payloads_cpu)
                            if self._draft_tree_payloads_cpu is not None
                            else None
                        ),
                        len(self._draft_tree_req_ids_cpu),
                    )
                    self._ddtree_pop_payloads_log_once = True
                # aeon_dflash_ddtree_m8f
                # The verifier GDN window can execute before the next scheduler
                # round exposes scheduled_spec_decode_trees. Install parent ids
                # immediately from the real DFlash DDTree proposer payload so
                # Triton parent-state replay sees true branch metadata without
                # using the M8D synthetic inline fallback.
                if self._draft_tree_payloads_cpu and self._draft_tree_req_ids_cpu:
                    payload_by_req_id = dict(
                        zip(
                            self._draft_tree_req_ids_cpu,
                            self._draft_tree_payloads_cpu,
                            strict=False,
                        )
                    )
                    live_parent_metadata = build_padded_parent_ids(
                        self.input_batch.req_ids,
                        payload_by_req_id,
                        device=self.device,
                    )
                    if live_parent_metadata is not None:
                        self._last_ddtree_metadata_payload = payload_by_req_id
                        self._last_ddtree_parent_ids_gpu = (
                            live_parent_metadata.parent_ids
                        )
                        self._last_ddtree_parent_metadata = live_parent_metadata
                        if not getattr(
                            self, "_ddtree_live_parent_payload_log_once", False
                        ):
                            logger.warning(
                                "DDTree runner installed live parent metadata "
                                "payloads=%s parent_shape=%s",
                                len(payload_by_req_id),
                                tuple(live_parent_metadata.parent_ids.shape),
                            )
                            self._ddtree_live_parent_payload_log_once = True
            else:
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m8f",
        "DDTree runner installed live parent metadata",
        "live_parent_metadata = build_padded_parent_ids",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M8F verification failed: missing {needle}")


def main() -> int:
    pkg_root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if pkg_root_override:
        pkg_root = Path(pkg_root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_live_parent_metadata(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] live DDTree parent metadata install verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
