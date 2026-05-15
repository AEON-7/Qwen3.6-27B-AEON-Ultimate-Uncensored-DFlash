#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m10h"


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


def patch_gdn_metadata_diagnostics(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py"
    replace_exact(
        path,
        """        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        # aeon_dflash_ddtree_m6b
""",
        """        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        # aeon_dflash_ddtree_m10h
        if os.environ.get("DDTREE_LOG_GDN_META", "0") == "1" and not getattr(
            self, "_ddtree_m10h_meta_logged", False
        ):
            parent_ids = getattr(attn_metadata, "ddtree_parent_ids", None)
            logger.warning(
                "DDTree M10H GDN meta prefix=%s env_slow=%s env_triton=%s "
                "parent_present=%s parent_shape=%s spec_mask_present=%s "
                "spec_mask_shape=%s num_prefills=%s num_decodes=%s "
                "num_spec_decodes=%s num_actual_tokens=%s",
                self.prefix,
                os.environ.get("DDTREE_SLOW_TREE_GDN"),
                os.environ.get("DDTREE_TRITON_TREE_GDN"),
                parent_ids is not None,
                tuple(parent_ids.shape) if parent_ids is not None else None,
                spec_sequence_masks is not None,
                (
                    tuple(spec_sequence_masks.shape)
                    if hasattr(spec_sequence_masks, "shape")
                    else None
                ),
                attn_metadata.num_prefills,
                attn_metadata.num_decodes,
                attn_metadata.num_spec_decodes,
                num_actual_tokens,
            )
            self._ddtree_m10h_meta_logged = True

        # aeon_dflash_ddtree_m6b
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m10h",
        "DDTree M10H GDN meta",
        "parent_present=%s",
        "self._ddtree_m10h_meta_logged",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M10H verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_gdn_metadata_diagnostics(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] GDN metadata diagnostics installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
