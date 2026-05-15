#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m10b"


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
        accepted_compact_by_req.append(accepted_compact)
        bonus_parent_by_req.append(bonus_parent)
""",
        """        # aeon_dflash_ddtree_m10b
        # Only the normal flat DFlash chain has proven KV/recurrent-state
        # compaction semantics. If the verified path jumps to a branch/root
        # alternative, emit the first non-flat token as a target bonus after the
        # longest safe flat prefix. That preserves output quality because vLLM
        # does not compact/reuse the unproven branch state.
        flat_prefix_len = 0
        for expected_compact, compact_index in enumerate(accepted_compact, start=1):
            if compact_index != expected_compact:
                break
            flat_prefix_len += 1
        has_nonflat_accept = flat_prefix_len < len(accepted_compact)
        if has_nonflat_accept:
            safe_accepted_tokens = accepted_tokens[:flat_prefix_len]
            safe_accepted_compact = accepted_compact[:flat_prefix_len]
            branch_bonus_token = accepted_tokens[flat_prefix_len]
            emitted = safe_accepted_tokens + [branch_bonus_token]
            reported_bonus_parent = flat_prefix_len
        else:
            safe_accepted_compact = accepted_compact
            emitted = accepted_tokens + [bonus_token]
            reported_bonus_parent = bonus_parent
        if emitted:
            output_token_ids[req_index, : len(emitted)] = torch.tensor(
                emitted,
                dtype=torch.int32,
                device=compact_logits.device,
            )
        accepted_compact_by_req.append(safe_accepted_compact)
        bonus_parent_by_req.append(reported_bonus_parent)
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m10b",
        "flat_prefix_len = 0",
        "reported_bonus_parent = flat_prefix_len",
        "accepted_compact_by_req.append(safe_accepted_compact)",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M10B verification failed: missing {needle}")


def verify_runtime(pkg_root: Path) -> None:
    import importlib.util
    import sys as py_sys

    import torch

    module_path = pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py"
    spec = importlib.util.spec_from_file_location("ddtree_runtime_sampler_m10b", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import patched DDTree runtime sampler")
    module = importlib.util.module_from_spec(spec)
    py_sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Alternate root token: emit it as a root bonus, but report zero accepted
    # compact nodes so branch KV/state is not reused.
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
    logits[0, 20] = 1.0
    logits[2, 42] = 1.0
    sample = module.greedy_sample_ddtree(metadata, logits)
    if sample.output_token_ids.tolist() != [[20, -1, -1]]:
        raise RuntimeError(f"M10B root-branch bonus output failed: {sample.output_token_ids.tolist()}")
    if sample.accepted_compact_indices != [[]] or sample.bonus_parent_compact_indices != [0]:
        raise RuntimeError(
            "M10B root-branch state report failed: "
            f"{sample.accepted_compact_indices}, {sample.bonus_parent_compact_indices}"
        )

    # Flat chain: preserve normal DFlash behavior and compact all accepted rows.
    metadata = module.DDTreeRuntimeMetadata.from_payloads(
        ["req-b"],
        {
            "req-b": {
                "tree_token_ids": [10, 20],
                "parent_indices": [-1, 0],
            }
        },
    )
    logits = torch.zeros((3, 64), dtype=torch.float32)
    logits[0, 10] = 1.0
    logits[1, 20] = 1.0
    logits[2, 7] = 1.0
    sample = module.greedy_sample_ddtree(metadata, logits)
    if sample.output_token_ids.tolist() != [[10, 20, 7]]:
        raise RuntimeError(f"M10B flat-chain output failed: {sample.output_token_ids.tolist()}")
    if sample.accepted_compact_indices != [[1, 2]] or sample.bonus_parent_compact_indices != [2]:
        raise RuntimeError(
            "M10B flat-chain state report failed: "
            f"{sample.accepted_compact_indices}, {sample.bonus_parent_compact_indices}"
        )

    # Flat prefix then branch: compact only the safe prefix and emit the branch
    # token as that prefix's bonus.
    metadata = module.DDTreeRuntimeMetadata.from_payloads(
        ["req-c"],
        {
            "req-c": {
                "tree_token_ids": [10, 20, 30],
                "parent_indices": [-1, 0, 0],
            }
        },
    )
    logits = torch.zeros((4, 64), dtype=torch.float32)
    logits[0, 10] = 1.0
    logits[1, 30] = 1.0
    logits[3, 42] = 1.0
    sample = module.greedy_sample_ddtree(metadata, logits)
    if sample.output_token_ids.tolist() != [[10, 30, -1, -1]]:
        raise RuntimeError(f"M10B prefix-branch output failed: {sample.output_token_ids.tolist()}")
    if sample.accepted_compact_indices != [[1]] or sample.bonus_parent_compact_indices != [1]:
        raise RuntimeError(
            "M10B prefix-branch state report failed: "
            f"{sample.accepted_compact_indices}, {sample.bonus_parent_compact_indices}"
        )


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
        print(f"[{MARKER}] M10T sampler already present; skipping superseded M10B patch")
        return 0
    patch_runtime_sampler(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    verify_runtime(pkg_root)
    print(f"[{MARKER}] DDTree safe branch-bonus state reporting installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
