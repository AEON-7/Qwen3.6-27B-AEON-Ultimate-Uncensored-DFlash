#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m9e"


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
        """                flat_parent_row = torch.arange(
                    self.num_spec_tokens + 1,
                    device=self.device,
                    dtype=torch.int64,
                ) - 1
""",
        """                flat_parent_row = torch.arange(
                    self.num_spec_tokens + 1,
                    device=self.device,
                    dtype=torch.int32,
                ) - 1
""",
    )
    replace_exact(
        path,
        """            extra_attn_metadata_args = {}
""",
        """            # aeon_dflash_ddtree_m9e
            # The graph-safe Triton verifier reads parent ids inside CUDA graph
            # capture. Normalize once while building metadata so the attention
            # path never has to allocate/copy during capture.
            if ddtree_parent_ids_for_attn is not None and (
                ddtree_parent_ids_for_attn.device != self.device
                or ddtree_parent_ids_for_attn.dtype != torch.int32
            ):
                ddtree_parent_ids_for_attn = ddtree_parent_ids_for_attn.to(
                    device=self.device,
                    dtype=torch.int32,
                )

            extra_attn_metadata_args = {}
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m9e",
        "dtype=torch.int32",
        "ddtree_parent_ids_for_attn.to(",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M9E verification failed: missing {needle}")


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
    print(f"[{MARKER}] DDTree parent-id dtype normalization installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
