#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m9a"


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


def patch_flash_attention_gpu_branch_mask(pkg_root: Path) -> None:
    path = pkg_root / "v1/attention/backends/flash_attn.py"
    replace_exact(
        path,
        """            parent_row = None
            if req_i < parent_ids.shape[0] and q_len <= parent_ids.shape[1]:
                candidate_parent_row = parent_ids[req_i, :q_len].detach().cpu().tolist()
                if candidate_parent_row and int(candidate_parent_row[0]) < 0:
                    parent_row = candidate_parent_row

            if branch_only:
                if parent_row is None:
                    continue
                manual_rows = [
                    row_i
                    for row_i, parent_i in enumerate(parent_row)
                    if not (
                        (row_i == 0 and int(parent_i) < 0)
                        or int(parent_i) == row_i - 1
                    )
                ]
                if not manual_rows:
                    continue
                row_indices = torch.tensor(
                    manual_rows, device=query.device, dtype=torch.long
                )
            else:
                row_indices = torch.arange(q_len, device=query.device, dtype=torch.long)
""",
        """            # aeon_dflash_ddtree_m9a
            # Keep branch-mask construction on GPU. The M8V debug path converted
            # parent ids and row indices to Python lists in every attention layer,
            # which made a one-row branch verifier pay hundreds of CPU syncs per
            # response. We still use a tiny eager correction for non-flat rows,
            # but all branch-row selection and ancestor masks remain tensors.
            parent_row_tensor = None
            if req_i < parent_ids.shape[0] and q_len <= parent_ids.shape[1]:
                candidate_parent_row = parent_ids[req_i, :q_len].to(
                    device=query.device, dtype=torch.long
                )
                if candidate_parent_row.numel() and int(candidate_parent_row[0].item()) < 0:
                    parent_row_tensor = candidate_parent_row

            if branch_only:
                if parent_row_tensor is None:
                    continue
                local_rows = torch.arange(q_len, device=query.device, dtype=torch.long)
                flat_chain_rows = (
                    ((local_rows == 0) & (parent_row_tensor < 0))
                    | (parent_row_tensor == (local_rows - 1))
                )
                row_indices = torch.nonzero(
                    ~flat_chain_rows, as_tuple=False
                ).flatten()
                if row_indices.numel() == 0:
                    continue
            else:
                row_indices = torch.arange(q_len, device=query.device, dtype=torch.long)
""",
    )
    replace_exact(
        path,
        """            if parent_row is None:
                local_kv = torch.arange(q_len, device=query.device)
                visible[:, context_len : context_len + q_len] = (
                    local_kv.unsqueeze(0) <= row_indices.unsqueeze(1)
                )
            else:
                ancestor = torch.zeros(
                    (row_indices.numel(), q_len), device=query.device, dtype=torch.bool
                )
                for out_i, q_local_tensor in enumerate(row_indices.detach().cpu()):
                    cur = int(q_local_tensor.item())
                    while cur >= 0:
                        ancestor[out_i, cur] = True
                        cur = int(parent_row[cur])
                visible[:, context_len : context_len + q_len] = ancestor
""",
        """            if parent_row_tensor is None:
                local_kv = torch.arange(q_len, device=query.device)
                visible[:, context_len : context_len + q_len] = (
                    local_kv.unsqueeze(0) <= row_indices.unsqueeze(1)
                )
            else:
                ancestor = torch.zeros(
                    (row_indices.numel(), q_len), device=query.device, dtype=torch.bool
                )
                if row_indices.numel() > 0:
                    out_rows = torch.arange(
                        row_indices.numel(), device=query.device, dtype=torch.long
                    )
                    cur = row_indices.clone()
                    for _ in range(q_len):
                        valid = cur >= 0
                        safe_cur = cur.clamp(min=0, max=q_len - 1)
                        ancestor[out_rows, safe_cur] = (
                            ancestor[out_rows, safe_cur] | valid
                        )
                        next_cur = parent_row_tensor[safe_cur]
                        cur = torch.where(valid, next_cur, cur)
                visible[:, context_len : context_len + q_len] = ancestor
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/attention/backends/flash_attn.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m9a",
        "parent_row_tensor",
        "flat_chain_rows",
        "ancestor[out_rows, safe_cur]",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M9A verification failed: missing {needle}")
    if "candidate_parent_row = parent_ids[req_i, :q_len].detach().cpu().tolist()" in text:
        raise RuntimeError("Static M9A verification failed: CPU parent-row list remains")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_flash_attention_gpu_branch_mask(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree FlashAttention GPU branch-mask verifier verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
