#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m10n"


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


def patch_gpu_model_runner(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """                    if src_block_id != dst_ssm_block_id:
                        ssm_state[dst_ssm_block_id].copy_(
                            ssm_state[src_block_id].clone()
                        )
                    compacted += 1
""",
        """                    # aeon_dflash_ddtree_m10n
                    # vLLM's align-mode Mamba postprocess copies the running
                    # recurrent state from mamba_state_idx[req_id], which is
                    # the base/root block (row[0]), not the compact destination
                    # block. Earlier DDTree compaction moved branch conv state
                    # into the base block but left SSM state only in the compact
                    # destination, so postprocess copied a stale flat-chain SSM
                    # state after accepting a non-flat branch. Keep the branch
                    # SSM in the base running-state block and mirror it to the
                    # compact destination for diagnostics/consistency.
                    src_ssm_state = ssm_state[src_block_id].clone()
                    if src_block_id != base_block_id:
                        ssm_state[base_block_id].copy_(src_ssm_state)
                    if (
                        src_block_id != dst_ssm_block_id
                        and dst_ssm_block_id != base_block_id
                    ):
                        ssm_state[dst_ssm_block_id].copy_(src_ssm_state)
                    compacted += 1
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m10n",
        "base running-state block",
        "src_ssm_state = ssm_state[src_block_id].clone()",
        "ssm_state[base_block_id].copy_(src_ssm_state)",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M10N verification failed: missing {needle}")


def main() -> int:
    root_override = __import__("os").environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_gpu_model_runner(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] branch SSM compaction targets base running state")
    return 0


if __name__ == "__main__":
    sys.exit(main())
