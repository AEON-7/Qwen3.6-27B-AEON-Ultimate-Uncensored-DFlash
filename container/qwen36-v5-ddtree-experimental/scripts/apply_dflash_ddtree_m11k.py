#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m11k"


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


def patch_skip_drafter_after_nonflat(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """        copied = 0
        for req_id, accepted_compact in zip(
""",
        """        copied = 0
        # aeon_dflash_ddtree_m11k
        # Diagnostic isolation switch: when a non-flat branch is committed,
        # optionally suppress exactly one following DFlash proposal. If target
        # output quality recovers, corruption is downstream of target-state
        # compaction; if it persists, the committed KV/GDN state is wrong.
        nonflat_committed = False
        for req_id, accepted_compact in zip(
""",
    )
    replace_exact(
        path,
        """                if (
                    src_i < self.input_ids.gpu.shape[0]
                    and dst_i < self.input_ids.gpu.shape[0]
                ):
                    self.input_ids.gpu[dst_i].copy_(self.input_ids.gpu[src_i])
                copied += 1
""",
        """                if (
                    src_i < self.input_ids.gpu.shape[0]
                    and dst_i < self.input_ids.gpu.shape[0]
                ):
                    self.input_ids.gpu[dst_i].copy_(self.input_ids.gpu[src_i])
                copied += 1
                nonflat_committed = True

        if (
            nonflat_committed
            and os.environ.get("DDTREE_SKIP_DRAFTER_AFTER_NONFLAT", "0") == "1"
        ):
            self._ddtree_skip_drafter_once_after_nonflat = True
""",
    )
    replace_exact(
        path,
        """        spec_config = self.speculative_config
        propose_drafts_after_bookkeeping = False
        if spec_config is not None:
""",
        """        # aeon_dflash_ddtree_m11k
        # The flag is set during DDTree context compaction above and consumed
        # immediately here so it suppresses only the next drafter call.
        skip_ddtree_drafter_once = (
            os.environ.get("DDTREE_SKIP_DRAFTER_AFTER_NONFLAT", "0") == "1"
            and bool(getattr(self, "_ddtree_skip_drafter_once_after_nonflat", False))
        )
        self._ddtree_skip_drafter_once_after_nonflat = False

        spec_config = self.speculative_config
        propose_drafts_after_bookkeeping = False
        if spec_config is not None:
""",
    )
    replace_exact(
        path,
        """            input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (
                spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
                <= self.effective_drafter_max_model_len
            )
""",
        """            input_fits_in_drafter = (
                not skip_ddtree_drafter_once
                and spec_decode_common_attn_metadata is not None
                and (
                    spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
                    <= self.effective_drafter_max_model_len
                )
            )
            if (
                skip_ddtree_drafter_once
                and os.environ.get("DDTREE_LOG_CONTEXT_COMPACT", "0") == "1"
            ):
                logger.warning(
                    "DDTree M11K skipped one DFlash proposal after non-flat commit"
                )
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    checks = (
        MARKER,
        "DDTREE_SKIP_DRAFTER_AFTER_NONFLAT",
        "nonflat_committed = False",
        "skip_ddtree_drafter_once",
        "DDTree M11K skipped one DFlash proposal after non-flat commit",
    )
    for needle in checks:
        if needle not in text:
            raise RuntimeError(f"M11K verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_skip_drafter_after_nonflat(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree non-flat commit can suppress the next DFlash proposal for isolation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
