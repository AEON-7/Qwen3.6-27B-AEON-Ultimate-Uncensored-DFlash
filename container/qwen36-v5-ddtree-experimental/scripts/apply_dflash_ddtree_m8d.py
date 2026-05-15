#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8d"


def replace_exact(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Could not find expected text in {path}:\n{old}")
    path.write_text(text.replace(old, new, 1))
    return True


def replace_all_exact(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if old not in text:
        if new in text:
            return False
        raise RuntimeError(f"Could not find expected text in {path}:\n{old}")
    path.write_text(text.replace(old, new))
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
        """            extra_attn_metadata_args = {}
            if use_spec_decode and isinstance(
""",
        """            ddtree_parent_ids_for_attn = getattr(
                self, "_last_ddtree_parent_ids_gpu", None
            )
            # aeon_dflash_ddtree_m8d
            # Last-line safety net: if the scheduler/proposer did not attach a
            # branch payload, represent the active draft as a flat chain right
            # where attention metadata is built. This preserves flat behavior
            # but guarantees GDN/Flash metadata receives parent ids.
            spec_cfg = self.vllm_config.speculative_config
            if (
                ddtree_parent_ids_for_attn is None
                and use_spec_decode
                and spec_cfg is not None
                and getattr(spec_cfg, "method", None) == "dflash_ddtree"
                and os.environ.get("DDTREE_INLINE_PARENT_FALLBACK", "0") == "1"
            ):
                flat_parent_row = torch.arange(
                    self.num_spec_tokens + 1,
                    device=self.device,
                    dtype=torch.int64,
                ) - 1
                ddtree_parent_ids_for_attn = flat_parent_row.unsqueeze(0).expand(
                    num_reqs_padded,
                    -1,
                )
                if not getattr(self, "_ddtree_inline_parent_fallback_logged", False):
                    logger.warning(
                        "Using opt-in inline DDTree flat-chain parent ids for attention "
                        "metadata; true branch parent tensor was not available"
                    )
                    self._ddtree_inline_parent_fallback_logged = True

            extra_attn_metadata_args = {}
            if use_spec_decode and isinstance(
""",
    )
    replace_all_exact(
        path,
        """                    ddtree_parent_ids=getattr(
                        self, "_last_ddtree_parent_ids_gpu", None
                    ),
""",
        """                    ddtree_parent_ids=ddtree_parent_ids_for_attn,
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m8d",
        "ddtree_parent_ids_for_attn",
        "DDTREE_INLINE_PARENT_FALLBACK",
        "Using opt-in inline DDTree flat-chain parent ids",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M8D verification failed: missing {needle}")


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
    print(f"[{MARKER}] inline flat-chain attention parent fallback verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
