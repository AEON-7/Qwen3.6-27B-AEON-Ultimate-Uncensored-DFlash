#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m2"


def clear_python_caches(pkg_root: Path) -> None:
    for pyc in pkg_root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    for pycache in pkg_root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pkg_root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if pkg_root_override:
        pkg_root = Path(pkg_root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    spec_decode_root = pkg_root / "v1/spec_decode"
    if not spec_decode_root.exists():
        raise RuntimeError(f"Missing vLLM spec_decode package: {spec_decode_root}")

    tree_source = root / "prototypes/ddtree_tree.py"
    metadata_source = root / "prototypes/ddtree_vllm_metadata.py"
    tree_target = spec_decode_root / "ddtree_tree.py"
    metadata_target = spec_decode_root / "ddtree_metadata.py"

    tree_target.write_text(tree_source.read_text())
    metadata_text = metadata_source.read_text().replace(
        "from ddtree_tree import DDTree",
        "from vllm.v1.spec_decode.ddtree_tree import DDTree",
    )
    metadata_target.write_text(metadata_text)
    clear_python_caches(pkg_root)

    if not pkg_root_override:
        from vllm.v1.spec_decode.ddtree_metadata import TreeVerifierMetadata
        from vllm.v1.spec_decode.ddtree_tree import build_ddtree

        tree = build_ddtree(
            [[(11, -0.1), (12, -0.2)], [(21, -0.1), (22, -0.2)]],
            budget=3,
            top_k=2,
        )
        metadata = TreeVerifierMetadata.from_tree(prompt_len=5, tree=tree)
        if metadata.compact_logits_indices[0] != 4:
            raise RuntimeError("DDTree metadata import smoke test failed")

    print(f"[{MARKER}] vLLM DDTree metadata module installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
