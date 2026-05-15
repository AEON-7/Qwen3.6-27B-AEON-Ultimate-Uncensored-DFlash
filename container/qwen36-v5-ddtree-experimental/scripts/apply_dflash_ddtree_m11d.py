#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m11d"


OLD = """    if (
        os.environ.get("DDTREE_FULL_BRANCH_COMMIT", "0") == "1"
        and os.environ.get("DDTREE_ALLOW_BRANCH_STATE_COMPACTION", "0") == "1"
    ):
        # aeon_dflash_ddtree_m11a
        # Full branch commit is only safe when the caller also patches vLLM's
        # speculative scheduler with an explicit accepted-token count. In that
        # mode, a non-flat DDTree branch is emitted as computed accepted tokens,
        # not as accepted-prefix plus an uncomputed target bonus.
        bonus_parent = accepted_compact[-1] if accepted_compact else 0
        return accepted_tokens, accepted_compact, bonus_parent
"""


NEW = """    if (
        os.environ.get("DDTREE_FULL_BRANCH_COMMIT", "0") == "1"
        and os.environ.get("DDTREE_ALLOW_BRANCH_STATE_COMPACTION", "0") == "1"
        and os.environ.get("DDTREE_UNSAFE_FULL_BRANCH_RESEARCH", "0") == "1"
    ):
        # aeon_dflash_ddtree_m11a
        # Full branch commit is only safe when the caller also patches vLLM's
        # speculative scheduler with an explicit accepted-token count. In that
        # mode, a non-flat DDTree branch is emitted as computed accepted tokens,
        # not as accepted-prefix plus an uncomputed target bonus.
        # aeon_dflash_ddtree_m11d
        # Qwen3.6's hybrid GDN recurrent state is not yet safe for arbitrary
        # branch-state commit in vLLM. Keep this path behind an explicit
        # research-only flag so deployable DDTree always falls back to the
        # quality-preserving branch-as-bonus contract.
        bonus_parent = accepted_compact[-1] if accepted_compact else 0
        return accepted_tokens, accepted_compact, bonus_parent
"""


def patch_file(path: Path) -> None:
    text = path.read_text()
    if "aeon_dflash_ddtree_m11g" in text:
        return
    if NEW in text:
        return
    if OLD not in text:
        raise RuntimeError(f"Could not find M11A full-branch guard in {path}")
    path.write_text(text.replace(OLD, NEW, 1))


def replace_exact(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Could not find expected text in {path}:\n{old}")
    path.write_text(text.replace(old, new, 1))


def patch_gpu_model_runner(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """        if os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "0") != "1":
            return
        entries = getattr(self, "_last_ddtree_gdn_state_metadata", None)
""",
        """        if os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "0") != "1":
            return
        if (
            os.environ.get("DDTREE_ALLOW_BRANCH_STATE_COMPACTION", "0") == "1"
            and os.environ.get("DDTREE_UNSAFE_FULL_BRANCH_RESEARCH", "0") != "1"
        ):
            # aeon_dflash_ddtree_m11d
            # Full branch recurrent compaction is still research-only for
            # Qwen3.6 hybrid GDN layers. If stale full-branch env vars are
            # present without the explicit research flag, keep deployable
            # quality by leaving vLLM on the safe branch-as-bonus contract.
            return
        entries = getattr(self, "_last_ddtree_gdn_state_metadata", None)
""",
    )
    replace_exact(
        path,
        """        if os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "1") == "1":
            if os.environ.get("DDTREE_ALLOW_BRANCH_STATE_COMPACTION", "0") != "1":
                return None
            accepted_by_req = getattr(
""",
        """        if os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "1") == "1":
            if os.environ.get("DDTREE_ALLOW_BRANCH_STATE_COMPACTION", "0") != "1":
                return None
            if os.environ.get("DDTREE_UNSAFE_FULL_BRANCH_RESEARCH", "0") != "1":
                # aeon_dflash_ddtree_m11d
                # Do not override Mamba state counts for full branch commit
                # unless the explicit research flag is set. The safe DDTree
                # path emits a normal vLLM bonus token and should use stock
                # output-token counts.
                return None
            accepted_by_req = getattr(
""",
    )


def clear_python_caches(pkg_root: Path) -> None:
    for pyc in pkg_root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    for pycache in pkg_root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


def verify_imports() -> None:
    from vllm.v1.spec_decode.ddtree_runtime_sampler import (
        DDTreeRequestRuntime,
        _adapt_tree_walk_to_vllm_contract,
        _walk_one_tree,
    )

    import torch

    request = DDTreeRequestRuntime(
        req_id="req-a",
        tree_token_ids=(10, 20, 30),
        parent_indices=(-1, 0, -1),
    )
    # Root chooses token 30, which is compact node 3: a non-flat root sibling.
    # Row 3 then chooses 42, so full branch mode would emit [30] without bonus.
    rows = torch.tensor([30, 99, 99, 42], dtype=torch.int64)
    accepted_tokens, accepted_compact, bonus_token, bonus_parent = _walk_one_tree(
        request,
        rows,
    )

    os.environ["DDTREE_FULL_BRANCH_COMMIT"] = "1"
    os.environ["DDTREE_ALLOW_BRANCH_STATE_COMPACTION"] = "1"
    os.environ.pop("DDTREE_UNSAFE_FULL_BRANCH_RESEARCH", None)
    safe = _adapt_tree_walk_to_vllm_contract(
        accepted_tokens,
        accepted_compact,
        bonus_token,
        bonus_parent,
    )
    if safe != ([30], [], 0):
        raise RuntimeError(f"M11D safe guard failed, got {safe!r}")

    os.environ["DDTREE_UNSAFE_FULL_BRANCH_RESEARCH"] = "1"
    unsafe = _adapt_tree_walk_to_vllm_contract(
        accepted_tokens,
        accepted_compact,
        bonus_token,
        bonus_parent,
    )
    if unsafe not in (([30], [3], 3), ([30, 42], [3], 3)):
        raise RuntimeError(f"M11D research escape hatch failed, got {unsafe!r}")


def main() -> int:
    pkg_root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if pkg_root_override:
        pkg_root = Path(pkg_root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_file(pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py")
    patch_gpu_model_runner(pkg_root)
    clear_python_caches(pkg_root)

    if not pkg_root_override:
        verify_imports()
    print(f"[{MARKER}] unsafe full-branch commit requires explicit research flag")
    return 0


if __name__ == "__main__":
    sys.exit(main())
