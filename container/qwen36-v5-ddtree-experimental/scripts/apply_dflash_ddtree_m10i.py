#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m10i"


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


def patch_gdn_parent_present_diagnostics(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py"
    replace_exact(
        path,
        """        # aeon_dflash_ddtree_m10h
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
        """        # aeon_dflash_ddtree_m10i
        if os.environ.get("DDTREE_LOG_GDN_META", "0") == "1":
            parent_ids = getattr(attn_metadata, "ddtree_parent_ids", None)
            parent_present = parent_ids is not None
            log_prefill = os.environ.get("DDTREE_LOG_GDN_META_PREFILL", "0") == "1"
            should_log = False
            if parent_present and not getattr(
                self, "_ddtree_m10i_parent_meta_logged", False
            ):
                should_log = True
            elif (
                log_prefill
                and not parent_present
                and not getattr(self, "_ddtree_m10i_prefill_meta_logged", False)
            ):
                should_log = True
            if should_log:
                logger.warning(
                    "DDTree M10I GDN meta prefix=%s env_slow=%s env_triton=%s "
                    "parent_present=%s parent_shape=%s spec_mask_present=%s "
                    "spec_mask_shape=%s num_prefills=%s num_decodes=%s "
                    "num_spec_decodes=%s num_actual_tokens=%s parent_preview=%s",
                    self.prefix,
                    os.environ.get("DDTREE_SLOW_TREE_GDN"),
                    os.environ.get("DDTREE_TRITON_TREE_GDN"),
                    parent_present,
                    tuple(parent_ids.shape) if parent_present else None,
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
                    (
                        [int(v) for v in parent_ids[0, :16].detach().cpu().tolist()]
                        if parent_present and parent_ids.shape[0] > 0
                        else None
                    ),
                )
                if parent_present:
                    self._ddtree_m10i_parent_meta_logged = True
                else:
                    self._ddtree_m10i_prefill_meta_logged = True

        # aeon_dflash_ddtree_m6b
""",
    )


def patch_per_builder_refresh_logging(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """                if not getattr(self, "_ddtree_refresh_cached_metadata_logged", False):
                    logger.warning(
                        "DDTree refreshed cached attention metadata builder=%s "
                        "parent_shape=%s",
                        type(builder).__name__,
                        tuple(ddtree_parent_ids_for_metadata.shape),
                    )
                    self._ddtree_refresh_cached_metadata_logged = True
""",
        """                ddtree_refresh_logged_builders = getattr(
                    self, "_ddtree_refresh_cached_metadata_logged_builders", set()
                )
                ddtree_builder_name = type(builder).__name__
                if ddtree_builder_name not in ddtree_refresh_logged_builders:
                    logger.warning(
                        "DDTree refreshed cached attention metadata builder=%s "
                        "metadata=%s parent_shape=%s",
                        ddtree_builder_name,
                        type(attn_metadata_i).__name__,
                        tuple(ddtree_parent_ids_for_metadata.shape),
                    )
                    ddtree_refresh_logged_builders.add(ddtree_builder_name)
                    self._ddtree_refresh_cached_metadata_logged_builders = (
                        ddtree_refresh_logged_builders
                    )
""",
    )


def patch_flash_parent_diagnostics(pkg_root: Path) -> None:
    path = pkg_root / "v1/attention/backends/flash_attn.py"
    replace_exact(
        path,
        """        if (
            os.environ.get("DDTREE_EAGER_TREE_ATTN", "0") == "1"
            and attn_metadata.ddtree_parent_ids is not None
            and not is_quantized_kv_cache(self.kv_cache_dtype)
            and not attn_metadata.use_cascade
            and self.dcp_world_size == 1
        ):
""",
        """        if os.environ.get("DDTREE_LOG_FLASH_ATTN", "0") == "1":
            ddtree_flash_parent_ids = getattr(attn_metadata, "ddtree_parent_ids", None)
            if ddtree_flash_parent_ids is not None and not getattr(
                self, "_ddtree_m10i_flash_parent_logged", False
            ):
                logger.warning(
                    "DDTree M10I Flash meta parent_shape=%s parent_mode=%s "
                    "parent_preview=%s quantized_kv=%s use_cascade=%s "
                    "dcp_world_size=%s num_actual_tokens=%s query_shape=%s",
                    tuple(ddtree_flash_parent_ids.shape),
                    self._ddtree_parent_mode(attn_metadata),
                    self._ddtree_parent_preview(attn_metadata),
                    is_quantized_kv_cache(self.kv_cache_dtype),
                    attn_metadata.use_cascade,
                    self.dcp_world_size,
                    num_actual_tokens,
                    tuple(query.shape),
                )
                self._ddtree_m10i_flash_parent_logged = True

        if (
            os.environ.get("DDTREE_EAGER_TREE_ATTN", "0") == "1"
            and attn_metadata.ddtree_parent_ids is not None
            and not is_quantized_kv_cache(self.kv_cache_dtype)
            and not attn_metadata.use_cascade
            and self.dcp_world_size == 1
        ):
""",
    )


def verify_static(pkg_root: Path) -> None:
    checks = {
        "model_executor/layers/mamba/gdn_linear_attn.py": (
            "aeon_dflash_ddtree_m10i",
            "DDTree M10I GDN meta",
            "_ddtree_m10i_parent_meta_logged",
            "parent_preview=%s",
        ),
        "v1/worker/gpu_model_runner.py": (
            "_ddtree_refresh_cached_metadata_logged_builders",
            "metadata=%s parent_shape=%s",
        ),
        "v1/attention/backends/flash_attn.py": (
            "DDTree M10I Flash meta",
            "_ddtree_m10i_flash_parent_logged",
            "quantized_kv=%s use_cascade=%s",
        ),
    }
    for rel, needles in checks.items():
        text = (pkg_root / rel).read_text()
        for needle in needles:
            if needle not in text:
                raise RuntimeError(f"Static M10I verification failed: {rel} missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_gdn_parent_present_diagnostics(pkg_root)
    patch_per_builder_refresh_logging(pkg_root)
    patch_flash_parent_diagnostics(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] parent-present DDTree diagnostics installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
