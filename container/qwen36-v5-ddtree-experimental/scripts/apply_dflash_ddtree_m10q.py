#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m10q"


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
        """        if allow_branch_state_compaction and has_nonflat_accept:
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
""",
        """        if allow_branch_state_compaction and has_nonflat_accept:
            # aeon_dflash_ddtree_m10q
            # vLLM's speculative scheduler expects sampler rows to be shaped as
            # accepted draft tokens plus one uncomputed target bonus. M10J tried
            # to emit only the computed non-flat branch path, but then vLLM
            # treated that computed branch token as the uncomputed bonus and
            # corrected num_computed_tokens/state cursors incorrectly. Keep the
            # standard accepted+bonus contract, while M10K/M10N/M10P compact the
            # computed branch state into the flat accepted-token slot.
            safe_accepted_compact = accepted_compact
            emitted = accepted_tokens + [bonus_token]
            reported_bonus_parent = accepted_compact[-1] if accepted_compact else 0
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m10q",
        "accepted draft tokens plus one uncomputed target bonus",
        "emitted = accepted_tokens + [bonus_token]",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M10Q verification failed: missing {needle}")


def main() -> int:
    root_override = __import__("os").environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    runtime_sampler = pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py"
    if "_adapt_tree_walk_to_vllm_contract" in runtime_sampler.read_text():
        print(f"[{MARKER}] M10T sampler already present; skipping superseded M10Q patch")
        return 0
    patch_runtime_sampler(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] non-flat branch sampler preserves accepted+bonus contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
