#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m10g"


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


def patch_slow_gdn_guard(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py"
    replace_exact(
        path,
        """            and spec_sequence_masks is not None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes == 0
        ):
            return self._forward_core_ddtree_slow(
""",
        """            and spec_sequence_masks is not None
            and attn_metadata.num_prefills == 0
        ):
            # aeon_dflash_ddtree_m10g
            # Current vLLM labels pure speculative verifier windows as decode
            # windows in some hybrid paths. The tree-parent correctness path is
            # still valid as long as there is no prefill mixed into the batch.
            if os.environ.get("DDTREE_LOG_GDN_META", "0") == "1" and not getattr(
                self, "_ddtree_m10g_slow_guard_logged", False
            ):
                logger.warning(
                    "DDTree M10G slow GDN guard passed num_prefills=%s "
                    "num_decodes=%s num_spec_decodes=%s num_actual_tokens=%s "
                    "spec_mask_shape=%s parent_shape=%s",
                    attn_metadata.num_prefills,
                    attn_metadata.num_decodes,
                    attn_metadata.num_spec_decodes,
                    attn_metadata.num_actual_tokens,
                    (
                        tuple(spec_sequence_masks.shape)
                        if hasattr(spec_sequence_masks, "shape")
                        else None
                    ),
                    tuple(attn_metadata.ddtree_parent_ids.shape),
                )
                self._ddtree_m10g_slow_guard_logged = True
            return self._forward_core_ddtree_slow(
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m10g",
        "DDTree M10G slow GDN guard passed",
        "and attn_metadata.num_prefills == 0",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M10G verification failed: missing {needle}")
    if "and attn_metadata.num_decodes == 0\n        ):\n            # aeon_dflash_ddtree_m10g" in text:
        raise RuntimeError("Static M10G verification failed: old num_decodes guard remains")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_slow_gdn_guard(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] slow DDTree GDN decode-window guard verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
