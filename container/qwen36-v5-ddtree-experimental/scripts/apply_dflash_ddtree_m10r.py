#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m10r"


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


def patch_parent_metadata(pkg_root: Path) -> None:
    path = pkg_root / "v1/spec_decode/ddtree_parent_metadata.py"
    replace_exact(
        path,
        """    - `parent_indices[i] == -1` means parent is root compact node 0
    - otherwise `parent_indices[i]` is a non-root node index and maps to
      compact parent `parent_indices[i] + 1`
""",
        """    - `parent_indices[i] == -1` means read the pre-tree cached state
    - otherwise `parent_indices[i]` is a non-root node index and maps to
      compact parent `parent_indices[i] + 1`

    This is intentionally different from sampler traversal, where root
    children are looked up under compact cursor 0. Attention/GDN replay needs
    the real state parent, and row 0 is the root-logits row, not an accepted
    draft node.
""",
    )
    replace_exact(
        path,
        """        parents.append(0 if parent_int < 0 else parent_int + 1)
""",
        """        parents.append(ROOT_PARENT if parent_int < 0 else parent_int + 1)
""",
    )


def verify_runtime(pkg_root: Path) -> None:
    text = (pkg_root / "v1/spec_decode/ddtree_parent_metadata.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m10r",
        "pre-tree cached state",
        "parents.append(ROOT_PARENT if parent_int < 0 else parent_int + 1)",
    ):
        if needle == "aeon_dflash_ddtree_m10r":
            continue
        if needle not in text:
            raise RuntimeError(f"Static M10R verification failed: missing {needle}")

    import importlib.util

    module_path = pkg_root / "v1/spec_decode/ddtree_parent_metadata.py"
    spec = importlib.util.spec_from_file_location("ddtree_parent_metadata_m10r", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    parents = module.full_parent_ids_from_payload(
        {"tree_token_ids": [10, 20, 21, 30], "parent_indices": [-1, 0, 0, 2]}
    )
    if parents != [-1, -1, 1, 1, 3]:
        raise RuntimeError(f"M10R parent semantics failed: {parents}")


def main() -> int:
    root_override = __import__("os").environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_parent_metadata(pkg_root)
    marker_path = pkg_root / "v1/spec_decode/ddtree_parent_metadata.py"
    if MARKER not in marker_path.read_text():
        marker_path.write_text(marker_path.read_text() + f"\n# {MARKER}\n")
    clear_python_caches(pkg_root)
    verify_runtime(pkg_root)
    print(f"[{MARKER}] model replay parent ids keep root children on pre-tree state")
    return 0


if __name__ == "__main__":
    sys.exit(main())
