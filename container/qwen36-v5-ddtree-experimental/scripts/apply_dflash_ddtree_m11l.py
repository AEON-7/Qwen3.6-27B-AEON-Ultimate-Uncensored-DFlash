#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m11l"


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


def patch_running_state_mirror(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """        compacted = 0

        for layer_names, spec_req_indices, state_indices in entries:
""",
        """        compacted = 0
        # aeon_dflash_ddtree_m11l
        # Keep a tiny diagnostic trail of which recurrent blocks are touched.
        # row[0] is usually the running-state block, but vLLM also tracks the
        # authoritative running block in self.mamba_state_idx. Mirror to both
        # when requested so branch commits cannot resume from stale GDN state.
        compact_diag: list[tuple[str, int, int, int, int, int, int]] = []

        for layer_names, spec_req_indices, state_indices in entries:
""",
    )
    replace_exact(
        path,
        """                base_block_id = int(row[0].item())
                src_block_id = int(row[src_compact].item())
                dst_ssm_block_id = int(row[dst_compact].item())
                if base_block_id <= 0 or src_block_id <= 0 or dst_ssm_block_id <= 0:
                    continue

                for layer_name in layer_names:
""",
        """                base_block_id = int(row[0].item())
                src_block_id = int(row[src_compact].item())
                dst_ssm_block_id = int(row[dst_compact].item())
                running_block_id = int(self.mamba_state_idx.get(req_id, base_block_id))
                if (
                    base_block_id <= 0
                    or src_block_id <= 0
                    or dst_ssm_block_id <= 0
                    or running_block_id <= 0
                ):
                    continue

                compact_diag.append(
                    (
                        str(req_id),
                        int(src_compact),
                        int(dst_compact),
                        int(base_block_id),
                        int(src_block_id),
                        int(dst_ssm_block_id),
                        int(running_block_id),
                    )
                )

                for layer_name in layer_names:
""",
    )
    replace_exact(
        path,
        """                    conv_dim = getattr(attention, "conv_dim", None)
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
""",
        """                    conv_dim = getattr(attention, "conv_dim", None)
                    mirror_running = (
                        os.environ.get("DDTREE_MIRROR_BRANCH_TO_MAMBA_RUNNING", "1")
                        == "1"
                    )
                    target_conv_blocks = [base_block_id]
                    if mirror_running and running_block_id != base_block_id:
                        target_conv_blocks.append(running_block_id)
                    if conv_dim is not None and int(conv_state.shape[-1]) == int(conv_dim):
                        state_len = int(conv_state.shape[1])
                        if dst_compact + conv_width > state_len:
                            continue
                        branch_conv_state = conv_state[src_block_id, :conv_width].clone()
                        for target_block_id in target_conv_blocks:
                            if target_block_id >= conv_state.shape[0]:
                                continue
                            conv_state[
                                target_block_id,
                                :conv_width,
                            ].copy_(branch_conv_state)
                            conv_state[
                                target_block_id,
                                dst_compact : dst_compact + conv_width,
                            ].copy_(branch_conv_state)
                    else:
                        state_len = int(conv_state.shape[-1])
                        if dst_compact + conv_width > state_len:
                            continue
                        branch_conv_state = conv_state[src_block_id, :, :conv_width].clone()
                        for target_block_id in target_conv_blocks:
                            if target_block_id >= conv_state.shape[0]:
                                continue
                            conv_state[
                                target_block_id,
                                :,
                                :conv_width,
                            ].copy_(branch_conv_state)
                            conv_state[
                                target_block_id,
                                :,
                                dst_compact : dst_compact + conv_width,
                            ].copy_(branch_conv_state)
""",
    )
    replace_exact(
        path,
        """                    src_ssm_state = ssm_state[src_block_id].clone()
                    if src_block_id != base_block_id:
                        ssm_state[base_block_id].copy_(src_ssm_state)
                    if (
                        src_block_id != dst_ssm_block_id
                        and dst_ssm_block_id != base_block_id
                    ):
                        ssm_state[dst_ssm_block_id].copy_(src_ssm_state)
                    compacted += 1
""",
        """                    src_ssm_state = ssm_state[src_block_id].clone()
                    target_ssm_blocks = [base_block_id]
                    if (
                        os.environ.get("DDTREE_MIRROR_BRANCH_TO_MAMBA_RUNNING", "1")
                        == "1"
                        and running_block_id != base_block_id
                    ):
                        target_ssm_blocks.append(running_block_id)
                    if (
                        dst_ssm_block_id != base_block_id
                        and dst_ssm_block_id != running_block_id
                    ):
                        target_ssm_blocks.append(dst_ssm_block_id)
                    for target_block_id in dict.fromkeys(target_ssm_blocks):
                        if (
                            target_block_id < ssm_state.shape[0]
                            and src_block_id != target_block_id
                        ):
                            ssm_state[target_block_id].copy_(src_ssm_state)
                    compacted += 1
""",
    )
    replace_exact(
        path,
        """                    "accepted_counts=%s",
                    compacted,
                    len(entries),
                    runtime_req_ids,
                    bonus_parent_by_req,
                    accepted_by_req,
                    accepted_counts.tolist(),
                )
""",
        """                    "accepted_counts=%s compact_diag=%s",
                    compacted,
                    len(entries),
                    runtime_req_ids,
                    bonus_parent_by_req,
                    accepted_by_req,
                    accepted_counts.tolist(),
                    compact_diag[:8],
                )
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        MARKER,
        "DDTREE_MIRROR_BRANCH_TO_MAMBA_RUNNING",
        "running_block_id = int(self.mamba_state_idx.get(req_id, base_block_id))",
        "compact_diag[:8]",
    ):
        if needle not in text:
            raise RuntimeError(f"M11L verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_running_state_mirror(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree mirrors branch GDN state into vLLM's running mamba_state_idx block")
    return 0


if __name__ == "__main__":
    sys.exit(main())
