#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8b"


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


def patch_llm_base_proposer(pkg_root: Path) -> None:
    path = pkg_root / "v1/spec_decode/llm_base_proposer.py"
    replace_exact(
        path,
        """        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
""",
        """        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]
        # aeon_dflash_ddtree_m8b
        # Some DFlash configs take the sequential proposer path rather than
        # the parallel-drafting early exit hooked in M4B. Preserve tree payload
        # generation there by retaining each draft step's hidden state and
        # building the DDTree payload after the sequential chain is complete.
        ddtree_hidden_states_list = (
            [sample_hidden_states] if self.method == "dflash_ddtree" else None
        )

        cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
""",
    )
    replace_exact(
        path,
        """            hidden_states = hidden_states[:batch_size]
            draft_token_ids = self._greedy_sample(last_hidden_states[:batch_size])
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids
""",
        """            hidden_states = hidden_states[:batch_size]
            step_sample_hidden_states = last_hidden_states[:batch_size]
            if ddtree_hidden_states_list is not None:
                ddtree_hidden_states_list.append(step_sample_hidden_states)
            draft_token_ids = self._greedy_sample(step_sample_hidden_states)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if self.method == "dflash_ddtree" and ddtree_hidden_states_list is not None:
            build_payloads = getattr(self, "_build_ddtree_payloads_from_logits", None)
            if build_payloads is not None:
                ddtree_hidden_states = torch.stack(
                    ddtree_hidden_states_list,
                    dim=1,
                ).reshape(-1, ddtree_hidden_states_list[0].shape[-1])
                ddtree_logits = self.model.compute_logits(ddtree_hidden_states)
                return build_payloads(ddtree_logits, batch_size).view(
                    batch_size, self.num_speculative_tokens
                )
        return draft_token_ids
""",
    )


def verify_static(pkg_root: Path) -> None:
    text = (pkg_root / "v1/spec_decode/llm_base_proposer.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m8b",
        "ddtree_hidden_states_list",
        "_build_ddtree_payloads_from_logits",
        "ddtree_logits",
    ):
        if needle not in text:
            raise RuntimeError(f"Static M8B verification failed: missing {needle}")


def main() -> int:
    pkg_root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if pkg_root_override:
        pkg_root = Path(pkg_root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_llm_base_proposer(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] sequential DFlash DDTree payload path verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
