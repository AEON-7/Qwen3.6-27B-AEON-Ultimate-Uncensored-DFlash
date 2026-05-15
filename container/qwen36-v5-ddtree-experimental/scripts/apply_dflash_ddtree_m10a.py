#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m10a"


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


def patch_runtime_sampler(pkg_root: Path) -> None:
    path = pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py"
    replace_exact(
        path,
        """        emitted = accepted_tokens + [bonus_token]
        if emitted:
            output_token_ids[req_index, : len(emitted)] = torch.tensor(
                emitted,
                dtype=torch.int32,
                device=compact_logits.device,
            )
""",
        """        # aeon_dflash_ddtree_m10a
        # A bonus token is safe only when the accepted path is the normal flat
        # DFlash chain. For root/branch alternatives, the next-token row depends
        # on branch recurrent state; until that state path is fully proven,
        # emit only the accepted branch token(s) and let the next decode step
        # recompute target state from the committed sequence.
        flat_chain = accepted_compact == list(range(1, len(accepted_compact) + 1))
        if flat_chain or not accepted_tokens:
            emitted = accepted_tokens + [bonus_token]
        else:
            emitted = accepted_tokens
        if emitted:
            output_token_ids[req_index, : len(emitted)] = torch.tensor(
                emitted,
                dtype=torch.int32,
                device=compact_logits.device,
            )
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m10a",
        "flat_chain = accepted_compact == list(range(1, len(accepted_compact) + 1))",
        "emit only the accepted branch token",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M10A verification failed: missing {needle}")


def verify_runtime(pkg_root: Path) -> None:
    import importlib.util
    import sys as py_sys
    import torch

    module_path = pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py"
    spec = importlib.util.spec_from_file_location("ddtree_runtime_sampler_m10a", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import patched DDTree runtime sampler")
    module = importlib.util.module_from_spec(spec)
    py_sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    metadata = module.DDTreeRuntimeMetadata.from_payloads(
        ["req-a"],
        {
            "req-a": {
                "tree_token_ids": [10, 20],
                "parent_indices": [-1, -1],
            }
        },
    )
    logits = torch.zeros((3, 64), dtype=torch.float32)
    logits[0, 20] = 1.0  # accept alternate root node compact 2
    logits[2, 42] = 1.0  # unsafe branch bonus must not be emitted
    sample = module.greedy_sample_ddtree(metadata, logits)
    if sample.output_token_ids.tolist() != [[20, -1, -1]]:
        raise RuntimeError(f"M10A branch-bonus guard failed: {sample.output_token_ids.tolist()}")

    logits.zero_()
    logits[0, 10] = 1.0  # accept flat-chain root
    logits[1, 7] = 1.0   # flat-chain bonus is safe
    sample = module.greedy_sample_ddtree(metadata, logits)
    if sample.output_token_ids.tolist() != [[10, 7, -1]]:
        raise RuntimeError(f"M10A flat bonus guard failed: {sample.output_token_ids.tolist()}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    runtime_sampler = pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py"
    if "_adapt_tree_walk_to_vllm_contract" in runtime_sampler.read_text():
        print(f"[{MARKER}] M10T sampler already present; skipping superseded M10A patch")
        return 0
    patch_runtime_sampler(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    verify_runtime(pkg_root)
    print(f"[{MARKER}] DDTree branch bonus safety guard installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
