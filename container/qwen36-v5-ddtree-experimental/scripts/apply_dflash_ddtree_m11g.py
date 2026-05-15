#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m11g"


OLD = """        bonus_parent = accepted_compact[-1] if accepted_compact else 0
        return accepted_tokens, accepted_compact, bonus_parent
"""


NEW = """        # aeon_dflash_ddtree_m11g
        # Preserve the normal vLLM speculative shape even for full non-flat
        # branch commit: computed accepted branch tokens plus one target bonus.
        # M11A's accepted-count side channel tells the scheduler how many
        # draft nodes were accepted, while the emitted bonus keeps
        # postprocess_mamba on the stock accepted+bonus cursor convention.
        bonus_parent = accepted_compact[-1] if accepted_compact else 0
        if os.environ.get("DDTREE_FULL_BRANCH_SUPPRESS_BONUS", "0") == "1":
            return accepted_tokens, accepted_compact, bonus_parent
        return accepted_tokens + [bonus_token], accepted_compact, bonus_parent
"""


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
    text = path.read_text()
    if MARKER in text:
        return
    replace_exact(path, OLD, NEW)


def patch_state_count_default(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """                        state_bias = int(
                            os.environ.get("DDTREE_FULL_BRANCH_STATE_COUNT_BIAS", "0")
                        )
""",
        """                        state_bias = int(
                            os.environ.get("DDTREE_FULL_BRANCH_STATE_COUNT_BIAS", "1")
                        )
""",
    )


def verify_runtime(pkg_root: Path) -> None:
    import importlib.util

    import torch

    module_path = pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py"
    spec = importlib.util.spec_from_file_location("ddtree_runtime_sampler_m11g", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    old_full = os.environ.get("DDTREE_FULL_BRANCH_COMMIT")
    old_allow = os.environ.get("DDTREE_ALLOW_BRANCH_STATE_COMPACTION")
    old_unsafe = os.environ.get("DDTREE_UNSAFE_FULL_BRANCH_RESEARCH")
    old_suppress = os.environ.get("DDTREE_FULL_BRANCH_SUPPRESS_BONUS")
    try:
        os.environ["DDTREE_FULL_BRANCH_COMMIT"] = "1"
        os.environ["DDTREE_ALLOW_BRANCH_STATE_COMPACTION"] = "1"
        os.environ["DDTREE_UNSAFE_FULL_BRANCH_RESEARCH"] = "1"
        os.environ.pop("DDTREE_FULL_BRANCH_SUPPRESS_BONUS", None)

        metadata = module.DDTreeRuntimeMetadata.from_payloads(
            ["req-a"],
            {
                "req-a": {
                    "tree_token_ids": [10, 20, 21, 30],
                    "parent_indices": [-1, 0, 0, 2],
                }
            },
        )
        logits = torch.zeros((5, 64), dtype=torch.float32)
        logits[0, 10] = 1.0
        logits[1, 21] = 1.0
        logits[3, 30] = 1.0
        logits[4, 42] = 1.0
        sample = module.greedy_sample_ddtree(metadata, logits)
        if sample.output_token_ids.tolist() != [[10, 21, 30, 42, -1]]:
            raise RuntimeError(f"M11G full-branch bonus output failed: {sample.output_token_ids.tolist()}")
        if sample.accepted_compact_indices != [[1, 3, 4]]:
            raise RuntimeError(f"M11G accepted path failed: {sample.accepted_compact_indices}")
        if sample.bonus_parent_compact_indices != [4]:
            raise RuntimeError(f"M11G bonus parent failed: {sample.bonus_parent_compact_indices}")

        os.environ["DDTREE_FULL_BRANCH_SUPPRESS_BONUS"] = "1"
        sample = module.greedy_sample_ddtree(metadata, logits)
        if sample.output_token_ids.tolist() != [[10, 21, 30, -1, -1]]:
            raise RuntimeError(f"M11G suppress-bonus output failed: {sample.output_token_ids.tolist()}")
    finally:
        for key, value in (
            ("DDTREE_FULL_BRANCH_COMMIT", old_full),
            ("DDTREE_ALLOW_BRANCH_STATE_COMPACTION", old_allow),
            ("DDTREE_UNSAFE_FULL_BRANCH_RESEARCH", old_unsafe),
            ("DDTREE_FULL_BRANCH_SUPPRESS_BONUS", old_suppress),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def verify_static(pkg_root: Path) -> None:
    sampler = (pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py").read_text()
    runner = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        MARKER,
        "DDTREE_FULL_BRANCH_SUPPRESS_BONUS",
        "return accepted_tokens + [bonus_token], accepted_compact, bonus_parent",
    ):
        if needle not in sampler:
            raise RuntimeError(f"M11G sampler verification failed: missing {needle}")
    if 'DDTREE_FULL_BRANCH_STATE_COUNT_BIAS", "1"' not in runner:
        raise RuntimeError("M11G state-count default verification failed")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_runtime_sampler(pkg_root)
    patch_state_count_default(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    if not root_override:
        verify_runtime(pkg_root)
    print(f"[{MARKER}] full-branch commit preserves accepted+bonus contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
