#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m10j"


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


def patch_runtime_sampler(pkg_root: Path) -> None:
    path = pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py"
    replace_exact(
        path,
        """        if allow_branch_state_compaction:
            safe_accepted_compact = accepted_compact
            emitted = accepted_tokens + [bonus_token]
            reported_bonus_parent = bonus_parent
        else:
            flat_prefix_len = 0
            for expected_compact, compact_index in enumerate(
                accepted_compact, start=1
            ):
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
""",
        """        flat_prefix_len = 0
        for expected_compact, compact_index in enumerate(
            accepted_compact, start=1
        ):
            if compact_index != expected_compact:
                break
            flat_prefix_len += 1
        has_nonflat_accept = flat_prefix_len < len(accepted_compact)

        if allow_branch_state_compaction and has_nonflat_accept:
            # aeon_dflash_ddtree_m10j
            # Full branch compaction can safely reuse recurrent state only for
            # verifier rows that were actually computed. The target bonus row is
            # a logit sample after the accepted branch, but the model has not
            # computed KV/GDN state after that bonus token yet. Emitting it here
            # advances vLLM's cursor past available recurrent state and causes
            # the repeated-token collapse seen in prose. For non-flat branches,
            # commit the verified branch path and compact the state of its last
            # accepted node. Flat-chain paths keep the normal speculative bonus.
            safe_accepted_compact = accepted_compact
            emitted = accepted_tokens
            reported_bonus_parent = accepted_compact[-1] if accepted_compact else 0
        elif allow_branch_state_compaction:
            safe_accepted_compact = accepted_compact
            emitted = accepted_tokens + [bonus_token]
            reported_bonus_parent = bonus_parent
        elif has_nonflat_accept:
            safe_accepted_tokens = accepted_tokens[:flat_prefix_len]
            safe_accepted_compact = accepted_compact[:flat_prefix_len]
            branch_bonus_token = accepted_tokens[flat_prefix_len]
            emitted = safe_accepted_tokens + [branch_bonus_token]
            reported_bonus_parent = flat_prefix_len
        else:
            safe_accepted_compact = accepted_compact
            emitted = accepted_tokens + [bonus_token]
            reported_bonus_parent = bonus_parent
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m10j",
        "Full branch compaction can safely reuse recurrent state",
        "emitted = accepted_tokens",
        "reported_bonus_parent = accepted_compact[-1] if accepted_compact else 0",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M10J verification failed: missing {needle}")


def verify_runtime(pkg_root: Path) -> None:
    import importlib.util
    import sys as py_sys

    import torch

    module_path = pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py"
    spec = importlib.util.spec_from_file_location("ddtree_runtime_sampler_m10j", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import patched DDTree runtime sampler")
    module = importlib.util.module_from_spec(spec)
    py_sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    old = os.environ.get("DDTREE_ALLOW_BRANCH_STATE_COMPACTION")
    os.environ["DDTREE_ALLOW_BRANCH_STATE_COMPACTION"] = "1"
    try:
        # Alternate root: emit only the verified branch token and report that
        # branch row as the recurrent-state source.
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
            raise RuntimeError(f"M10J root branch output failed: {sample.output_token_ids.tolist()}")
        if sample.accepted_compact_indices != [[2]] or sample.bonus_parent_compact_indices != [2]:
            raise RuntimeError(
                "M10J root branch state report failed: "
                f"{sample.accepted_compact_indices}, {sample.bonus_parent_compact_indices}"
            )

        # Flat chain still emits the target bonus and reports the bonus parent.
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
            raise RuntimeError(f"M10J flat-chain output failed: {sample.output_token_ids.tolist()}")
        if sample.accepted_compact_indices != [[1, 2]] or sample.bonus_parent_compact_indices != [2]:
            raise RuntimeError(
                "M10J flat-chain state report failed: "
                f"{sample.accepted_compact_indices}, {sample.bonus_parent_compact_indices}"
            )
    finally:
        if old is None:
            os.environ.pop("DDTREE_ALLOW_BRANCH_STATE_COMPACTION", None)
        else:
            os.environ["DDTREE_ALLOW_BRANCH_STATE_COMPACTION"] = old


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
        print(f"[{MARKER}] M10T sampler already present; skipping superseded M10J patch")
        return 0
    patch_runtime_sampler(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    verify_runtime(pkg_root)
    print(f"[{MARKER}] branch compaction emits only computed branch state")
    return 0


if __name__ == "__main__":
    sys.exit(main())
