#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8c"


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


def patch_gpu_model_runner(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """            self._last_ddtree_metadata_payload = (
                scheduler_output.scheduled_spec_decode_trees or None
            )
            parent_metadata = build_padded_parent_ids(
""",
        """            self._last_ddtree_metadata_payload = (
                scheduler_output.scheduled_spec_decode_trees or None
            )
            # aeon_dflash_ddtree_m8c
            # Safety fallback for live sequential DFlash paths: if no true
            # branch payload made it through the scheduler, represent the
            # scheduled flat draft as a chain-shaped tree. This preserves exact
            # flat-DFlash behavior while still feeding parent metadata into
            # Qwen3.6 GDN layers so the parent-state path is exercised.
            spec_cfg = self.vllm_config.speculative_config
            if (
                spec_cfg is not None
                and getattr(spec_cfg, "method", None) == "dflash_ddtree"
            ):
                payloads = (
                    dict(self._last_ddtree_metadata_payload)
                    if isinstance(self._last_ddtree_metadata_payload, dict)
                    else {}
                )
                added_chain_payload = False
                for req_id, token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()
                ):
                    existing_payload = payloads.get(req_id)
                    if (
                        isinstance(existing_payload, dict)
                        and existing_payload.get("tree_token_ids")
                    ):
                        continue
                    clean_token_ids = [
                        int(token_id) for token_id in token_ids if int(token_id) >= 0
                    ]
                    if clean_token_ids:
                        payloads[req_id] = {
                            "method": "dflash_ddtree",
                            "version": 1,
                            "source": "flat_chain_fallback",
                            "tree_token_ids": clean_token_ids,
                            "parent_indices": [-1]
                            + list(range(max(0, len(clean_token_ids) - 1))),
                        }
                        added_chain_payload = True
                if added_chain_payload:
                    if not getattr(self, "_ddtree_chain_fallback_logged", False):
                        logger.warning(
                            "Using DDTree flat-chain metadata fallback; true "
                            "branch payload was not present in scheduler output"
                        )
                        self._ddtree_chain_fallback_logged = True
                    self._last_ddtree_metadata_payload = payloads
            parent_metadata = build_padded_parent_ids(
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m8c",
        "flat_chain_fallback",
        "Using DDTree flat-chain metadata fallback",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M8C verification failed: missing {needle}")


def main() -> int:
    pkg_root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if pkg_root_override:
        pkg_root = Path(pkg_root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_gpu_model_runner(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] flat-chain DDTree metadata fallback verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
