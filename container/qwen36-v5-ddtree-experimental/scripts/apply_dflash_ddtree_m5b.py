#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m5b"


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


def install_runtime_sampler(root: Path, pkg_root: Path) -> None:
    source = root / "prototypes/ddtree_runtime_sampler.py"
    target = pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py"
    target.write_text(source.read_text())


def patch_scheduler_tree_gate(pkg_root: Path) -> None:
    path = pkg_root / "v1/core/sched/scheduler.py"
    replace_exact(
        path,
        """import itertools
import time
""",
        """import itertools
import os
import time
""",
    )
    replace_exact(
        path,
        """            # Add newly generated spec token ids to the request.
            request.spec_tree = draft_trees.get(req_id)
            if self.structured_output_manager.should_advance(request):
""",
        """            # Add newly generated spec token ids to the request.
            request.spec_tree = draft_trees.get(req_id)
            # aeon_dflash_ddtree_m5b
            # Off by default. When enabled, schedule flattened tree nodes for
            # target verification instead of the flat top-1 fallback. The
            # target-side tree sampler/state rollback path is still guarded by
            # DDTREE_USE_RUNTIME_SAMPLER so production DFlash remains unchanged.
            if (
                os.environ.get("DDTREE_TARGET_VERIFY", "0") == "1"
                and isinstance(request.spec_tree, dict)
            ):
                tree_token_ids = request.spec_tree.get("tree_token_ids")
                if isinstance(tree_token_ids, list) and tree_token_ids:
                    spec_token_ids = [int(token_id) for token_id in tree_token_ids]
            if self.structured_output_manager.should_advance(request):
""",
    )


def patch_gpu_model_runner(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """import itertools
import threading
""",
        """import itertools
import os
import threading
""",
    )
    replace_exact(
        path,
        """from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer_gpu import (
""",
        """from vllm.v1.spec_decode.ddtree_runtime_sampler import (
    DDTreeRuntimeMetadata,
    greedy_sample_ddtree,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer_gpu import (
""",
    )
    replace_exact(
        path,
        """        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        return sampler_output
""",
        """        # aeon_dflash_ddtree_m5b
        # Quality-first tree sampler hook. This only activates when the
        # scheduler has actually fed root+tree-node logits to the target and
        # the request is greedy; otherwise we keep vLLM's flat rejection path.
        ddtree_payload = getattr(self, "_last_ddtree_metadata_payload", None)
        if (
            os.environ.get("DDTREE_USE_RUNTIME_SAMPLER", "0") == "1"
            and ddtree_payload
            and sampling_metadata.all_greedy
        ):
            runtime_metadata = DDTreeRuntimeMetadata.from_payloads(
                self.input_batch.req_ids,
                ddtree_payload,
            )
            expected_rows = sum(
                1 + request.num_nodes for request in runtime_metadata.requests
            )
            if expected_rows == logits.shape[0] and runtime_metadata.num_requests:
                tree_sample = greedy_sample_ddtree(runtime_metadata, logits)
                self._last_ddtree_accepted_compact_indices = (
                    tree_sample.accepted_compact_indices
                )
                self._last_ddtree_bonus_parent_compact_indices = (
                    tree_sample.bonus_parent_compact_indices
                )
                return SamplerOutput(sampled_token_ids=tree_sample.output_token_ids)

        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        return sampler_output
""",
    )


def verify_static(pkg_root: Path) -> None:
    expected = {
        "v1/spec_decode/ddtree_runtime_sampler.py": (
            "greedy_sample_ddtree",
            "accepted_compact_indices",
        ),
        "v1/core/sched/scheduler.py": (
            "DDTREE_TARGET_VERIFY",
            "tree_token_ids",
        ),
        "v1/worker/gpu_model_runner.py": (
            "DDTREE_USE_RUNTIME_SAMPLER",
            "greedy_sample_ddtree",
            "_last_ddtree_accepted_compact_indices",
        ),
    }
    for rel, needles in expected.items():
        text = (pkg_root / rel).read_text()
        for needle in needles:
            if needle not in text:
                raise RuntimeError(f"Static M5B verification failed: {rel} missing {needle}")


def verify_imports() -> None:
    from vllm.v1.spec_decode.ddtree_runtime_sampler import (
        DDTreeRuntimeMetadata,
        greedy_sample_ddtree,
    )

    import torch

    metadata = DDTreeRuntimeMetadata.from_payloads(
        ["req-a"],
        {"req-a": {"tree_token_ids": [11], "parent_indices": [-1]}},
    )
    logits = torch.zeros((2, 32), dtype=torch.float32)
    logits[0, 11] = 1
    logits[1, 7] = 1
    sample = greedy_sample_ddtree(metadata, logits)
    if sample.output_token_ids.tolist() != [[11, 7]]:
        raise RuntimeError("M5B runtime sampler import smoke test failed")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pkg_root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if pkg_root_override:
        pkg_root = Path(pkg_root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    install_runtime_sampler(root, pkg_root)
    patch_scheduler_tree_gate(pkg_root)
    patch_gpu_model_runner(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    if not pkg_root_override:
        verify_imports()
    print(f"[{MARKER}] DDTree runtime greedy sampler hook verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
