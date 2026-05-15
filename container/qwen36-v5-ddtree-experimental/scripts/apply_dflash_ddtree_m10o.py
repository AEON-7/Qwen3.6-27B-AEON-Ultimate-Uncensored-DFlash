#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m10o"


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


def patch_gpu_model_runner(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """        state_token_counts = self._ddtree_state_token_counts(num_reqs)
        if state_token_counts is None:
            state_token_counts = self.num_accepted_tokens.gpu[:num_reqs]

        if self.cache_config.mamba_cache_mode == "align":
""",
        """        state_token_counts = self._ddtree_state_token_counts(num_reqs)
        if state_token_counts is None:
            state_token_counts = self.num_accepted_tokens.gpu[:num_reqs]
        elif state_token_counts.data_ptr() != self.num_accepted_tokens.gpu[:num_reqs].data_ptr():
            # aeon_dflash_ddtree_m10o
            # Branch-compaction mode may emit no uncomputed target bonus, so
            # output_token_ids contains fewer tokens than the number of computed
            # recurrent states available to the next verifier step. The GDN
            # metadata builder reads self.num_accepted_tokens.gpu on the next
            # step to choose the root rolling-state offset; keep that GPU tensor
            # synchronized with the state cursor, not just the emitted token
            # count.
            self.num_accepted_tokens.gpu[:num_reqs].copy_(state_token_counts)

        if self.cache_config.mamba_cache_mode == "align":
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m10o",
        "root rolling-state offset",
        "self.num_accepted_tokens.gpu[:num_reqs].copy_(state_token_counts)",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M10O verification failed: missing {needle}")


def main() -> int:
    root_override = __import__("os").environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_gpu_model_runner(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] branch state cursor synced to next-step GPU metadata")
    return 0


if __name__ == "__main__":
    sys.exit(main())
