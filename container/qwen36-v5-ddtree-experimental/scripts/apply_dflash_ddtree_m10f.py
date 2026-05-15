#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m10f"


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


def patch_slow_gdn_root_snapshots(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py"

    replace_exact(
        path,
        """        starts_cpu = (
            spec_query_start_loc[: attn_metadata.num_spec_decodes + 1]
            .detach()
            .cpu()
        )

        for req_i in range(attn_metadata.num_spec_decodes):
""",
        """        starts_cpu = (
            spec_query_start_loc[: attn_metadata.num_spec_decodes + 1]
            .detach()
            .cpu()
        )

        # aeon_dflash_ddtree_m10f
        # DDTree root siblings must read the pre-tree recurrent state. vLLM's
        # in-place speculative state layout lets row 0 overwrite that same cache
        # slot before later root siblings execute, which corrupts branch logits.
        # Snapshot the root conv/SSM state once, before any verifier row mutates
        # cache, matching Lucebox's separate curr_state + intermediate-state
        # design.
        root_offsets: list[int] = []
        root_conv_snapshots: list[torch.Tensor | None] = []
        root_ssm_snapshots: list[torch.Tensor | None] = []
        for req_i in range(attn_metadata.num_spec_decodes):
            root_offset = int(num_accepted_tokens[req_i].item()) - 1
            root_offset = max(0, root_offset)
            root_offsets.append(root_offset)
            conv_root_idx = int(state_indices_cpu[req_i, 0].item())
            ssm_root_idx = int(
                state_indices_cpu[
                    req_i, min(root_offset, state_indices_cpu.shape[1] - 1)
                ].item()
            )
            root_conv_snapshots.append(
                conv_state[conv_root_idx].clone() if conv_root_idx > 0 else None
            )
            root_ssm_snapshots.append(
                ssm_state[ssm_root_idx].clone() if ssm_root_idx > 0 else None
            )
        if os.environ.get("DDTREE_LOG_SLOW_GDN", "0") == "1" and not getattr(
            self, "_ddtree_m10f_slow_gdn_logged", False
        ):
            logger.warning(
                "DDTree M10F slow GDN root snapshots active reqs=%s "
                "root_offsets=%s parent_shape=%s",
                attn_metadata.num_spec_decodes,
                root_offsets,
                tuple(ddtree_parent_ids.shape),
            )
            self._ddtree_m10f_slow_gdn_logged = True

        for req_i in range(attn_metadata.num_spec_decodes):
""",
    )

    replace_exact(
        path,
        """            root_offset = int(num_accepted_tokens[req_i].item()) - 1
            root_offset = max(0, root_offset)
            width_minus_one = conv_weights.shape[-1] - 1
            for compact_i in range(end - start):
                token_row = start + compact_i
                dst_state_idx = int(state_indices_cpu[req_i, compact_i].item())
                if dst_state_idx <= 0:
                    continue
                parent_compact = int(parent_ids_cpu[req_i, compact_i].item())
                if parent_compact < 0:
                    src_state_idx = int(state_indices_cpu[req_i, 0].item())
                    src_offset = root_offset
                else:
                    src_state_idx = int(state_indices_cpu[req_i, parent_compact].item())
                    src_offset = 0
                if src_state_idx <= 0:
                    continue

                parent_conv_full = conv_state[src_state_idx].clone()
                parent_conv = parent_conv_full[
                    :, src_offset : src_offset + width_minus_one
                ]
""",
        """            root_offset = root_offsets[req_i]
            width_minus_one = conv_weights.shape[-1] - 1
            for compact_i in range(end - start):
                token_row = start + compact_i
                dst_state_idx = int(state_indices_cpu[req_i, compact_i].item())
                if dst_state_idx <= 0:
                    continue
                parent_compact = int(parent_ids_cpu[req_i, compact_i].item())
                if parent_compact < 0:
                    parent_conv_full = root_conv_snapshots[req_i]
                    src_offset = root_offset
                else:
                    src_state_idx = int(state_indices_cpu[req_i, parent_compact].item())
                    if src_state_idx <= 0:
                        continue
                    parent_conv_full = conv_state[src_state_idx].clone()
                    src_offset = 0
                if parent_conv_full is None:
                    continue

                parent_conv = parent_conv_full[
                    :, src_offset : src_offset + width_minus_one
                ]
""",
    )

    replace_exact(
        path,
        """            root_offset = int(num_accepted_tokens[req_i].item()) - 1
            root_offset = max(0, root_offset)
            for compact_i in range(end - start):
                token_row = start + compact_i
                dst_state_idx = int(state_indices_cpu[req_i, compact_i].item())
                if dst_state_idx <= 0:
                    continue
                parent_compact = int(parent_ids_cpu[req_i, compact_i].item())
                if parent_compact < 0:
                    src_state_idx = int(state_indices_cpu[req_i, root_offset].item())
                else:
                    src_state_idx = int(state_indices_cpu[req_i, parent_compact].item())
                if src_state_idx <= 0:
                    continue

                h = ssm_state[src_state_idx].to(torch.float32).clone()
""",
        """            root_offset = root_offsets[req_i]
            for compact_i in range(end - start):
                token_row = start + compact_i
                dst_state_idx = int(state_indices_cpu[req_i, compact_i].item())
                if dst_state_idx <= 0:
                    continue
                parent_compact = int(parent_ids_cpu[req_i, compact_i].item())
                if parent_compact < 0:
                    root_h = root_ssm_snapshots[req_i]
                    if root_h is None:
                        continue
                    h = root_h.to(torch.float32).clone()
                else:
                    src_state_idx = int(state_indices_cpu[req_i, parent_compact].item())
                    if src_state_idx <= 0:
                        continue
                    h = ssm_state[src_state_idx].to(torch.float32).clone()
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m10f",
        "root_conv_snapshots",
        "root_ssm_snapshots",
        "DDTree M10F slow GDN root snapshots active",
        "parent_conv_full = root_conv_snapshots[req_i]",
        "root_h = root_ssm_snapshots[req_i]",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M10F verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_slow_gdn_root_snapshots(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] slow DDTree GDN pre-root state snapshot verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
