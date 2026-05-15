#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m10p"


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
        """                    conv_width = int(getattr(attention, "conv_kernel_size", 1)) - 1
                    if conv_width <= 0 or conv_state.ndim < 3:
                        continue

                    conv_dim = getattr(attention, "conv_dim", None)
                    if conv_dim is not None and int(conv_state.shape[-1]) == int(conv_dim):
                        state_len = int(conv_state.shape[1])
                        if dst_compact + conv_width > state_len:
                            continue
                        conv_state[
                            base_block_id,
                            dst_compact : dst_compact + conv_width,
                        ].copy_(conv_state[src_block_id, :conv_width].clone())
                    else:
                        state_len = int(conv_state.shape[-1])
                        if dst_compact + conv_width > state_len:
                            continue
                        conv_state[
                            base_block_id,
                            :,
                            dst_compact : dst_compact + conv_width,
                        ].copy_(conv_state[src_block_id, :, :conv_width].clone())

                    # aeon_dflash_ddtree_m10n
""",
        """                    conv_width = int(getattr(attention, "conv_kernel_size", 1)) - 1
                    if conv_width <= 0 or conv_state.ndim < 3:
                        continue

                    # aeon_dflash_ddtree_m10p
                    # Align with vLLM's hybrid/Mamba postprocess convention.
                    # If the running-state source and destination block are the
                    # same, postprocess_mamba resets num_accepted_tokens_cpu to
                    # 1 and the next verifier reads root offset 0. Branch
                    # compaction had only stored the accepted branch conv state
                    # at dst_compact, so the next step could read stale root
                    # state from offset 0. Mirror the branch rolling conv state
                    # into both offset 0 and dst_compact; whichever cursor
                    # convention vLLM selects on the next step reads the same
                    # accepted branch state.
                    conv_dim = getattr(attention, "conv_dim", None)
                    if conv_dim is not None and int(conv_state.shape[-1]) == int(conv_dim):
                        state_len = int(conv_state.shape[1])
                        if dst_compact + conv_width > state_len:
                            continue
                        branch_conv_state = conv_state[src_block_id, :conv_width].clone()
                        conv_state[
                            base_block_id,
                            :conv_width,
                        ].copy_(branch_conv_state)
                        conv_state[
                            base_block_id,
                            dst_compact : dst_compact + conv_width,
                        ].copy_(branch_conv_state)
                    else:
                        state_len = int(conv_state.shape[-1])
                        if dst_compact + conv_width > state_len:
                            continue
                        branch_conv_state = conv_state[src_block_id, :, :conv_width].clone()
                        conv_state[
                            base_block_id,
                            :,
                            :conv_width,
                        ].copy_(branch_conv_state)
                        conv_state[
                            base_block_id,
                            :,
                            dst_compact : dst_compact + conv_width,
                        ].copy_(branch_conv_state)

                    # aeon_dflash_ddtree_m10n
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m10p",
        "Mirror the branch rolling conv state",
        "branch_conv_state = conv_state[src_block_id",
        "base_block_id,",
        ":conv_width",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M10P verification failed: missing {needle}")


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
    print(f"[{MARKER}] branch conv compaction mirrors root and cursor offsets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
