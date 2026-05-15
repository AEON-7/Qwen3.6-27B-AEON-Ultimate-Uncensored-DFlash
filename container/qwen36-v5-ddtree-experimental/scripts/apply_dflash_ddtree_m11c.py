#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m11c"


def clear_python_caches(pkg_root: Path) -> None:
    for pyc in pkg_root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    for pycache in pkg_root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


def patch_root_sibling_offset(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py"
    text = path.read_text()
    if MARKER in text:
        return

    old = """            root_offset = int(num_accepted_tokens[req_i].item()) - 1
            root_offset = max(0, root_offset)
"""
    new = """            root_offset = int(num_accepted_tokens[req_i].item()) - 1
            root_offset = max(0, root_offset)
            # aeon_dflash_ddtree_m11c
            # Lucebox tree replay treats root siblings as children of the
            # pre-tree recurrent state. vLLM's hybrid state cursor can point
            # at the most recent accepted/bonus slot instead, which is useful
            # for flat-chain replay but poisons root-sibling branch states.
            # Keep the old cursor behavior as the default while exposing a
            # zero-offset research switch for true root-sibling validation.
            if os.environ.get(
                "DDTREE_ROOT_SIBLING_STATE_OFFSET", "cursor"
            ).lower() in ("0", "zero", "root", "pre", "pre_tree"):
                root_offset = 0
"""
    count = text.count(old)
    if count < 2:
        raise RuntimeError(
            f"Expected at least two root_offset blocks in {path}, found {count}"
        )
    path.write_text(text.replace(old, new))


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py").read_text()
    for needle in (
        MARKER,
        "DDTREE_ROOT_SIBLING_STATE_OFFSET",
        "pre_tree",
    ):
        if needle not in text:
            raise RuntimeError(f"M11C static verification failed: missing {needle}")


def main() -> int:
    import os

    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_root_sibling_offset(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree root-sibling state offset switch installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
