#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8e"


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


def patch_dflash_payload_log(pkg_root: Path) -> None:
    path = pkg_root / "v1/spec_decode/dflash.py"
    replace_exact(
        path,
        """        self._last_ddtree_payloads = payloads
        return draft_token_ids
""",
        """        self._last_ddtree_payloads = payloads
        if not getattr(self, "_ddtree_payload_log_once", False):
            first_nodes = (
                len(payloads[0].get("tree_token_ids", ())) if payloads else 0
            )
            logger.warning(
                "DDTree proposer built payloads=%s first_nodes=%s",
                len(payloads),
                first_nodes,
            )
            self._ddtree_payload_log_once = True
        return draft_token_ids
""",
    )


def patch_gpu_runner_payload_logs(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """            if spec_config.method == "dflash_ddtree":
                pop_payloads = getattr(self.drafter, "pop_last_ddtree_payloads", None)
                self._draft_tree_payloads_cpu = (
                    pop_payloads() if pop_payloads is not None else None
                )
                self._draft_tree_req_ids_cpu = self.input_batch.req_ids.copy()
            else:
""",
        """            if spec_config.method == "dflash_ddtree":
                pop_payloads = getattr(self.drafter, "pop_last_ddtree_payloads", None)
                self._draft_tree_payloads_cpu = (
                    pop_payloads() if pop_payloads is not None else None
                )
                self._draft_tree_req_ids_cpu = self.input_batch.req_ids.copy()
                if not getattr(self, "_ddtree_pop_payloads_log_once", False):
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
    )
    replace_exact(
        path,
        """        return DraftTokenIds(req_ids, draft_token_ids, draft_trees=draft_trees)
""",
        """        if not getattr(self, "_ddtree_take_draft_log_once", False):
            logger.warning(
                "DDTree take_draft_token_ids req_ids=%s payloads=%s draft_trees=%s",
                len(req_ids),
                len(payloads) if payloads else None,
                len(draft_trees) if draft_trees else None,
            )
            self._ddtree_take_draft_log_once = True
        return DraftTokenIds(req_ids, draft_token_ids, draft_trees=draft_trees)
""",
    )


def verify_static(pkg_root: Path) -> None:
    checks = {
        "v1/spec_decode/dflash.py": ("DDTree proposer built payloads",),
        "v1/worker/gpu_model_runner.py": (
            "DDTree runner popped payloads",
            "DDTree take_draft_token_ids",
        ),
    }
    for rel, needles in checks.items():
        text = (pkg_root / rel).read_text()
        for needle in needles:
            if needle not in text:
                raise RuntimeError(f"Static M8E verification failed: {rel} missing {needle}")


def main() -> int:
    pkg_root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if pkg_root_override:
        pkg_root = Path(pkg_root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_dflash_payload_log(pkg_root)
    patch_gpu_runner_payload_logs(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree payload handoff tracing verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
