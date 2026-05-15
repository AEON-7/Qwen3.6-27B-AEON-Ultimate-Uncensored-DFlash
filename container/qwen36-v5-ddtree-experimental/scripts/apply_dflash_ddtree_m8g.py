#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8g"


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


def patch_cached_metadata_refresh(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """                if builder.supports_update_block_table:
                    cached_attn_metadata[cache_key] = attn_metadata_i

            if ubid is None:
""",
        """                if builder.supports_update_block_table:
                    cached_attn_metadata[cache_key] = attn_metadata_i

            # aeon_dflash_ddtree_m8g
            # Cached attention metadata can be reused via update_block_table()
            # after it was first built without DDTree parent ids. Refresh the
            # mutable metadata object directly so GDN/Flash layers see the live
            # parent tensor installed by M8F.
            if (
                ddtree_parent_ids_for_attn is not None
                and hasattr(attn_metadata_i, "ddtree_parent_ids")
            ):
                ddtree_parent_ids_for_metadata = ddtree_parent_ids_for_attn
                spec_sequence_masks = getattr(
                    attn_metadata_i, "spec_sequence_masks", None
                )
                if spec_sequence_masks is not None:
                    mask = spec_sequence_masks[: ddtree_parent_ids_for_attn.shape[0]]
                    if (
                        getattr(mask, "dtype", None) == torch.bool
                        and bool(mask.any().item())
                    ):
                        ddtree_parent_ids_for_metadata = ddtree_parent_ids_for_attn[
                            mask
                        ]
                attn_metadata_i.ddtree_parent_ids = ddtree_parent_ids_for_metadata
                if not getattr(self, "_ddtree_refresh_cached_metadata_logged", False):
                    logger.warning(
                        "DDTree refreshed cached attention metadata builder=%s "
                        "parent_shape=%s",
                        type(builder).__name__,
                        tuple(ddtree_parent_ids_for_metadata.shape),
                    )
                    self._ddtree_refresh_cached_metadata_logged = True

            if ubid is None:
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m8g",
        "DDTree refreshed cached attention metadata",
        "attn_metadata_i.ddtree_parent_ids = ddtree_parent_ids_for_metadata",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M8G verification failed: missing {needle}")


def main() -> int:
    pkg_root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if pkg_root_override:
        pkg_root = Path(pkg_root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_cached_metadata_refresh(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] cached DDTree attention metadata refresh verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
