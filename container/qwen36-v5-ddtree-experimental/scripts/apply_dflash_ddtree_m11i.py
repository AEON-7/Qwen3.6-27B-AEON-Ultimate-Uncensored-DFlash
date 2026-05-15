#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m11i"


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


def patch_gpu_model_runner(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """        if spec_decode_common_attn_metadata is None or not isinstance(
            slot_mappings, dict
        ):
            return
""",
        """        if spec_decode_common_attn_metadata is None:
            return
        # aeon_dflash_ddtree_m11i
        # vLLM may pass full-attention slot mappings as either a single
        # layer-name dict or a list of per-cache-group dicts. Full branch commit
        # needs KV replay for every full-attention layer; otherwise the hybrid
        # GDN state can advance down the accepted branch while attention history
        # still points at the old flat DFlash chain.
        if isinstance(slot_mappings, dict):
            slot_mapping_items = list(slot_mappings.items())
        elif isinstance(slot_mappings, list):
            slot_mapping_items = []
            for mapping in slot_mappings:
                if isinstance(mapping, dict):
                    slot_mapping_items.extend(mapping.items())
        else:
            return
        if not slot_mapping_items:
            return
""",
    )
    replace_exact(
        path,
        """        for req_id, accepted_compact in zip(
            runtime_req_ids, accepted_by_req, strict=False
        ):
""",
        """        copied_kv = 0
        for req_id, accepted_compact in zip(
            runtime_req_ids, accepted_by_req, strict=False
        ):
""",
    )
    replace_exact(
        path,
        """            for layer_name, layer_slot_mapping in slot_mappings.items():
""",
        """            for layer_name, layer_slot_mapping in slot_mapping_items:
""",
    )
    replace_exact(
        path,
        """                    kv_cache[:, dst_block, dst_offset].copy_(
                        kv_cache[:, src_block, src_offset].clone()
                    )

    def _compact_ddtree_recurrent_states(
""",
        """                    kv_cache[:, dst_block, dst_offset].copy_(
                        kv_cache[:, src_block, src_offset].clone()
                    )
                    copied_kv += 1

        if os.environ.get("DDTREE_LOG_KV_COMPACT", "0") == "1":
            log_count = getattr(self, "_ddtree_m11i_kv_log_count", 0)
            log_limit = int(os.environ.get("DDTREE_LOG_COMPACT_LIMIT", "16"))
            if log_count < log_limit:
                logger.warning(
                    "DDTree M11I full-attn KV compaction copied=%s "
                    "slot_mapping_groups=%s runtime_req_ids=%s accepted=%s",
                    copied_kv,
                    len(slot_mapping_items),
                    runtime_req_ids,
                    accepted_by_req,
                )
                self._ddtree_m11i_kv_log_count = log_count + 1

    def _compact_ddtree_recurrent_states(
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m11i",
        "slot_mapping_items = []",
        "DDTree M11I full-attn KV compaction",
        "copied_kv += 1",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M11I verification failed: missing {needle}")
    if "not isinstance(\n            slot_mappings, dict\n        )" in text:
        raise RuntimeError("Static M11I verification failed: dict-only guard remains")


def main() -> int:
    root_override = __import__("os").environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_gpu_model_runner(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree full-attention KV compaction handles list slot mappings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
