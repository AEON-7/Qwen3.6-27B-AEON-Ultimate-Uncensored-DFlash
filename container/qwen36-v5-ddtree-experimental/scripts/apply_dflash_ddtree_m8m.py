#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8m"


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


def patch_tree_depth_positions(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """        self.input_batch.block_table.compute_slot_mapping(
            num_reqs,
            self.query_start_loc.gpu[: num_reqs + 1],
            self.positions[:total_num_scheduled_tokens],
        )

        # Copy the tensors to the GPU.
        self._prepare_input_ids(
""",
        """        self.input_batch.block_table.compute_slot_mapping(
            num_reqs,
            self.query_start_loc.gpu[: num_reqs + 1],
            self.positions[:total_num_scheduled_tokens],
        )

        # aeon_dflash_ddtree_m8m
        # DDTree verifier nodes need two coordinate systems:
        # - sequential KV slots, already computed above from the ordinary flat
        #   positions, so every compact node has a unique cache slot;
        # - semantic model/RoPE positions based on tree depth, so sibling nodes
        #   at the same depth receive the same positional phase.
        #
        # Apply the tree-depth positions only after slot mapping has consumed
        # the sequential positions.
        ddtree_scheduled_payload = (
            scheduler_output.scheduled_spec_decode_trees or None
        )
        if ddtree_scheduled_payload:
            token_offset = 0
            for req_idx, req_id in enumerate(self.input_batch.req_ids):
                num_sched = int(num_scheduled_tokens[req_idx])
                payload = ddtree_scheduled_payload.get(req_id)
                if isinstance(payload, dict) and num_sched > 0:
                    node_depths = payload.get("node_depths", ())
                    if isinstance(node_depths, list):
                        compact_depths = [0] + [int(depth) for depth in node_depths]
                        if len(compact_depths) >= num_sched:
                            base_pos = int(
                                self.input_batch.num_computed_tokens_cpu[req_idx]
                            )
                            depth_positions_np = np.asarray(
                                [
                                    base_pos + depth
                                    for depth in compact_depths[:num_sched]
                                ],
                                dtype=np.int64,
                            )
                            depth_positions_cpu = torch.from_numpy(depth_positions_np)
                            self.positions[
                                token_offset : token_offset + num_sched
                            ].copy_(
                                depth_positions_cpu.to(
                                    device=self.device, non_blocking=True
                                )
                            )
                            if self.uses_mrope:
                                self.mrope_positions.cpu[
                                    :, token_offset : token_offset + num_sched
                                ].copy_(
                                    depth_positions_cpu.view(1, -1).expand(
                                        self.mrope_positions.cpu.shape[0], -1
                                    )
                                )
                            if self.uses_xdrope_dim > 0:
                                self.xdrope_positions.cpu[
                                    :, token_offset : token_offset + num_sched
                                ].copy_(
                                    depth_positions_cpu.view(1, -1).expand(
                                        self.xdrope_positions.cpu.shape[0], -1
                                    )
                                )
                            if not getattr(
                                self, "_ddtree_depth_positions_logged", False
                            ):
                                logger.warning(
                                    "DDTree applied depth positions req=%s "
                                    "positions=%s",
                                    req_id,
                                    depth_positions_np[:16].tolist(),
                                )
                                self._ddtree_depth_positions_logged = True
                token_offset += num_sched

        # Copy the tensors to the GPU.
        self._prepare_input_ids(
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m8m",
        "DDTree applied depth positions",
        "compact_depths = [0]",
        "self.mrope_positions.cpu",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M8M verification failed: missing {needle}")


def main() -> int:
    pkg_root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if pkg_root_override:
        pkg_root = Path(pkg_root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_tree_depth_positions(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree depth-position override verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
