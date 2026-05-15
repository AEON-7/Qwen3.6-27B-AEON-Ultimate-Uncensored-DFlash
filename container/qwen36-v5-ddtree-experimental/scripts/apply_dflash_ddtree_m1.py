#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
import os
from dataclasses import fields
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m1"


def replace_exact(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Could not find expected text in {path}:\n{old}")
    path.write_text(text.replace(old, new, 1))
    return True


def replace_all(path: Path, old: str, new: str) -> int:
    text = path.read_text()
    if old not in text:
        if new in text:
            return 0
        raise RuntimeError(f"Could not find expected text in {path}:\n{old}")
    count = text.count(old)
    path.write_text(text.replace(old, new))
    return count


def clear_python_caches(pkg_root: Path) -> None:
    for pyc in pkg_root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    for pycache in pkg_root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


def patch_speculative_config(pkg_root: Path) -> None:
    path = pkg_root / "config/speculative.py"

    replace_exact(
        path,
        'DFlashModelTypes = Literal["dflash"]',
        'DFlashModelTypes = Literal["dflash", "dflash_ddtree"]',
    )

    replace_exact(
        path,
        '''    num_speculative_tokens: int = Field(default=None, gt=0)  # type: ignore[assignment]
    """The number of speculative tokens, if provided. It will default to the
    number in the draft model config if present, otherwise, it is required."""
    model: str | None = None
''',
        '''    num_speculative_tokens: int = Field(default=None, gt=0)  # type: ignore[assignment]
    """The number of speculative tokens, if provided. It will default to the
    number in the draft model config if present, otherwise, it is required."""

    # aeon_dflash_ddtree_m1
    # Accepted for the experimental DDTree method. M1 routes through the proven
    # flat DFlash verifier; later milestones consume these fields in the tree
    # builder and verifier.
    ddtree_budget: int | None = Field(default=None, ge=1)
    """Maximum non-root DDTree verifier nodes per request."""
    ddtree_top_k: int = Field(default=8, ge=1)
    """Number of candidate siblings to keep per DFlash draft position."""
    ddtree_temperature: float = Field(default=1.0, gt=0)
    """Temperature used when scoring DFlash distributions for tree expansion."""
    ddtree_chain_seed: bool = True
    """Seed the tree with the top-1 chain before adding sibling branches."""

    model: str | None = None
''',
    )

    replace_exact(
        path,
        '''        uses_aux_hidden_states = self.method in (
            "eagle3",
            "extract_hidden_states",
            "dflash",
        )
''',
        '''        uses_aux_hidden_states = self.method in (
            "eagle3",
            "extract_hidden_states",
            "dflash",
            "dflash_ddtree",
        )
''',
    )

    replace_all(
        path,
        'if self.method in ("eagle", "eagle3", "dflash"):',
        'if self.method in ("eagle", "eagle3", "dflash", "dflash_ddtree"):',
    )

    replace_exact(
        path,
        '''                if self.method == "dflash":
                    self.parallel_drafting = True
''',
        '''                if self.method in ("dflash", "dflash_ddtree"):
                    self.parallel_drafting = True
''',
    )

    replace_exact(
        path,
        'self.method in ("eagle3", "extract_hidden_states", "dflash")',
        'self.method in ("eagle3", "extract_hidden_states", "dflash", "dflash_ddtree")',
    )

    replace_exact(
        path,
        'return self.method in ("eagle", "eagle3", "mtp", "dflash")',
        'return self.method in ("eagle", "eagle3", "mtp", "dflash", "dflash_ddtree")',
    )

    replace_exact(
        path,
        'return self.method == "dflash"',
        'return self.method in ("dflash", "dflash_ddtree")',
    )


def patch_eagle_config(pkg_root: Path) -> None:
    path = pkg_root / "transformers_utils/configs/eagle.py"
    replace_exact(
        path,
        '        elif method == "dflash":\n',
        '        elif method in ("dflash", "dflash_ddtree"):\n',
    )
    replace_exact(
        path,
        '                "eagle, eagle3, and dflash."\n',
        '                "eagle, eagle3, dflash, and dflash_ddtree."\n',
    )


def patch_base_proposer(pkg_root: Path) -> None:
    path = pkg_root / "v1/spec_decode/llm_base_proposer.py"
    replace_exact(
        path,
        '1 if (self.pass_hidden_states_to_model and self.method != "dflash") else 0',
        '1 if (self.pass_hidden_states_to_model and self.method not in ("dflash", "dflash_ddtree")) else 0',
    )
    replace_exact(
        path,
        'if self.method in ("eagle3", "dflash"):',
        'if self.method in ("eagle3", "dflash", "dflash_ddtree"):',
    )
    replace_exact(
        path,
        'return self.method not in ("mtp", "draft_model", "dflash")',
        'return self.method not in ("mtp", "draft_model", "dflash", "dflash_ddtree")',
    )


def patch_dflash_proposer(pkg_root: Path) -> None:
    path = pkg_root / "v1/spec_decode/dflash.py"
    replace_exact(
        path,
        '        assert vllm_config.speculative_config.method == "dflash"\n',
        '        assert vllm_config.speculative_config.method in ("dflash", "dflash_ddtree")\n',
    )


def main() -> int:
    pkg_root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if pkg_root_override:
        pkg_root = Path(pkg_root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent
    print(f"[{MARKER}] vLLM package root: {pkg_root}")

    patch_speculative_config(pkg_root)
    patch_eagle_config(pkg_root)
    patch_base_proposer(pkg_root)
    patch_dflash_proposer(pkg_root)
    clear_python_caches(pkg_root)

    if pkg_root_override:
        speculative_text = (pkg_root / "config/speculative.py").read_text()
        for token in (
            '"dflash_ddtree"',
            "ddtree_budget",
            "ddtree_top_k",
            "ddtree_temperature",
            "ddtree_chain_seed",
        ):
            if token not in speculative_text:
                raise RuntimeError(f"Static verification failed: missing {token}")
    else:
        from vllm.config.speculative import SpeculativeConfig
        from vllm.transformers_utils.configs.eagle import EAGLEConfig
        from vllm.v1.spec_decode.dflash import DFlashProposer

        assert DFlashProposer is not None
        assert EAGLEConfig is not None
        config_fields = {field.name for field in fields(SpeculativeConfig)}
        for field in (
            "ddtree_budget",
            "ddtree_top_k",
            "ddtree_temperature",
            "ddtree_chain_seed",
        ):
            if field not in config_fields:
                raise RuntimeError(f"SpeculativeConfig missing {field}")

    print(f"[{MARKER}] dflash_ddtree M1 overlay verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
