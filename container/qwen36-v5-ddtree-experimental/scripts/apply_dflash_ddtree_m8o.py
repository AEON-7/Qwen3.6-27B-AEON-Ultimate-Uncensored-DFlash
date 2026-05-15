#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8o"


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


def patch_slow_reference_num_accepted(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py"
    replace_exact(
        path,
        """        ddtree_parent_ids = attn_metadata.ddtree_parent_ids
        assert spec_query_start_loc is not None
""",
        """        ddtree_parent_ids = attn_metadata.ddtree_parent_ids
        # aeon_dflash_ddtree_m8o
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        assert spec_query_start_loc is not None
""",
    )
    replace_exact(
        path,
        """        assert ddtree_parent_ids is not None
        assert attn_metadata.spec_sequence_masks is not None
""",
        """        assert ddtree_parent_ids is not None
        assert num_accepted_tokens is not None
        assert attn_metadata.spec_sequence_masks is not None
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m8o",
        "num_accepted_tokens = attn_metadata.num_accepted_tokens",
        "assert num_accepted_tokens is not None",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M8O verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_slow_reference_num_accepted(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] slow DDTree GDN num_accepted_tokens reference fix verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
