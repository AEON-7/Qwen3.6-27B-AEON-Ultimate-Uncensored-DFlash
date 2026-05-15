#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m6a"


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


def install_parent_metadata(root: Path, pkg_root: Path) -> None:
    source = root / "prototypes/ddtree_parent_metadata.py"
    target = pkg_root / "v1/spec_decode/ddtree_parent_metadata.py"
    target.write_text(source.read_text())


def patch_gpu_model_runner(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """from vllm.v1.spec_decode.ddtree_runtime_sampler import (
    DDTreeRuntimeMetadata,
    greedy_sample_ddtree,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
""",
        """from vllm.v1.spec_decode.ddtree_parent_metadata import build_padded_parent_ids
from vllm.v1.spec_decode.ddtree_runtime_sampler import (
    DDTreeRuntimeMetadata,
    greedy_sample_ddtree,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
""",
    )
    replace_exact(
        path,
        """                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                        :num_reqs_padded
                    ],
                )
""",
        """                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                        :num_reqs_padded
                    ],
                    # aeon_dflash_ddtree_m6a
                    # Parent ids are root+tree compact coordinates, padded in
                    # active request order. GDN filters them down to spec rows.
                    ddtree_parent_ids=getattr(
                        self, "_last_ddtree_parent_ids_gpu", None
                    ),
                )
""",
    )
    replace_exact(
        path,
        """        self._last_ddtree_metadata_payload = None
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
""",
        """        self._last_ddtree_metadata_payload = None
        self._last_ddtree_parent_ids_gpu = None
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
""",
    )
    replace_exact(
        path,
        """            self._last_ddtree_metadata_payload = (
                scheduler_output.scheduled_spec_decode_trees or None
            )
            logits_indices = spec_decode_metadata.logits_indices
""",
        """            self._last_ddtree_metadata_payload = (
                scheduler_output.scheduled_spec_decode_trees or None
            )
            parent_metadata = build_padded_parent_ids(
                self.input_batch.req_ids,
                self._last_ddtree_metadata_payload,
                device=self.device,
            )
            self._last_ddtree_parent_ids_gpu = (
                parent_metadata.parent_ids if parent_metadata is not None else None
            )
            self._last_ddtree_parent_metadata = parent_metadata
            logits_indices = spec_decode_metadata.logits_indices
""",
    )


def patch_gdn_attention_metadata(pkg_root: Path) -> None:
    path = pkg_root / "v1/attention/backends/gdn_attn.py"
    replace_exact(
        path,
        """    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
""",
        """    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]
    # aeon_dflash_ddtree_m6a
    # Optional root+tree parent ids in compact verifier coordinates. Shape is
    # [num_spec_decodes, max_tree_tokens + 1] after builder filtering.
    ddtree_parent_ids: torch.Tensor | None = None

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
""",
    )
    replace_exact(
        path,
        """        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
""",
        """        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        ddtree_parent_ids: torch.Tensor | None = None,
        fast_build: bool = False,
""",
    )
    replace_exact(
        path,
        """            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
""",
        """            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
            ddtree_parent_ids = None
""",
    )
    replace_exact(
        path,
        """            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

        chunk_indices: torch.Tensor | None = None
""",
        """            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]
            if ddtree_parent_ids is not None:
                ddtree_parent_ids = ddtree_parent_ids[spec_sequence_masks_cpu]

        chunk_indices: torch.Tensor | None = None
""",
    )
    replace_exact(
        path,
        """            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
""",
        """            num_accepted_tokens=num_accepted_tokens,
            ddtree_parent_ids=ddtree_parent_ids,
            nums_dict=nums_dict,
""",
    )


def verify_static(pkg_root: Path) -> None:
    expected = {
        "v1/spec_decode/ddtree_parent_metadata.py": (
            "build_padded_parent_ids",
            "full_parent_ids_from_payload",
        ),
        "v1/worker/gpu_model_runner.py": (
            "build_padded_parent_ids",
            "_last_ddtree_parent_ids_gpu",
            "ddtree_parent_ids=getattr",
        ),
        "v1/attention/backends/gdn_attn.py": (
            "ddtree_parent_ids",
            "ddtree_parent_ids[spec_sequence_masks_cpu]",
        ),
    }
    for rel, needles in expected.items():
        text = (pkg_root / rel).read_text()
        for needle in needles:
            if needle not in text:
                raise RuntimeError(f"Static M6A verification failed: {rel} missing {needle}")


def verify_imports() -> None:
    from vllm.v1.spec_decode.ddtree_parent_metadata import build_padded_parent_ids

    import torch

    metadata = build_padded_parent_ids(
        ["req-a"],
        {
            "req-a": {
                "tree_token_ids": [11, 21, 22],
                "parent_indices": [-1, 0, 0],
            }
        },
        device=torch.device("cpu"),
    )
    if metadata is None or metadata.parent_ids.tolist() != [[-1, -1, 1, 1]]:
        raise RuntimeError("M6A parent metadata import smoke test failed")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pkg_root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if pkg_root_override:
        pkg_root = Path(pkg_root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    install_parent_metadata(root, pkg_root)
    patch_gpu_model_runner(pkg_root)
    patch_gdn_attention_metadata(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    if not pkg_root_override:
        verify_imports()
    print(f"[{MARKER}] DDTree parent metadata plumbing verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
