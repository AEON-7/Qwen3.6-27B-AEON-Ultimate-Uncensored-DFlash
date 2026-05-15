#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m10t"


def clear_python_caches(pkg_root: Path) -> None:
    for pyc in pkg_root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    for pycache in pkg_root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


def install_runtime_sampler(root: Path, pkg_root: Path) -> None:
    source = root / "prototypes/ddtree_runtime_sampler.py"
    target = pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py"
    text = source.read_text()
    if MARKER not in text:
        text += f"\n# {MARKER}\n"
    target.write_text(text)


def verify_runtime(pkg_root: Path) -> None:
    import importlib.util

    import torch

    module_path = pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py"
    spec = importlib.util.spec_from_file_location("ddtree_runtime_sampler_m10t", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    metadata = module.DDTreeRuntimeMetadata.from_payloads(
        ["req-a"],
        {
            "req-a": {
                "tree_token_ids": [10, 20, 21, 30],
                "parent_indices": [-1, 0, -1, 2],
            }
        },
    )
    logits = torch.zeros((5, 64), dtype=torch.float32)
    logits[0, 21] = 1.0
    logits[3, 30] = 1.0
    logits[4, 42] = 1.0
    sample = module.greedy_sample_ddtree(metadata, logits)
    expected = [[21, -1, -1, -1, -1]]
    if sample.output_token_ids.tolist() != expected:
        raise RuntimeError(
            "M10T sibling-as-bonus output failed: "
            f"{sample.output_token_ids.tolist()}"
        )
    if sample.accepted_compact_indices != [[]]:
        raise RuntimeError(
            "M10T sibling-as-bonus accepted path failed: "
            f"{sample.accepted_compact_indices}"
        )
    if sample.bonus_parent_compact_indices != [0]:
        raise RuntimeError(
            "M10T sibling-as-bonus parent failed: "
            f"{sample.bonus_parent_compact_indices}"
        )

    metadata = module.DDTreeRuntimeMetadata.from_payloads(
        ["req-a"],
        {
            "req-a": {
                "tree_token_ids": [10, 20, 21, 30],
                "parent_indices": [-1, 0, 0, 2],
            }
        },
    )
    logits = torch.zeros((5, 64), dtype=torch.float32)
    logits[0, 10] = 1.0
    logits[1, 21] = 1.0
    logits[3, 30] = 1.0
    sample = module.greedy_sample_ddtree(metadata, logits)
    expected = [[10, 21, -1, -1, -1]]
    if sample.output_token_ids.tolist() != expected:
        raise RuntimeError(
            "M10T flat-prefix sibling bonus output failed: "
            f"{sample.output_token_ids.tolist()}"
        )
    if sample.accepted_compact_indices != [[1]]:
        raise RuntimeError(
            "M10T flat-prefix accepted path failed: "
            f"{sample.accepted_compact_indices}"
        )
    if sample.bonus_parent_compact_indices != [1]:
        raise RuntimeError(
            "M10T flat-prefix bonus parent failed: "
            f"{sample.bonus_parent_compact_indices}"
        )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    install_runtime_sampler(root, pkg_root)
    clear_python_caches(pkg_root)
    verify_runtime(pkg_root)
    print(f"[{MARKER}] non-flat DDTree walks adapt to vLLM bonus semantics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
