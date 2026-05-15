#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m4b"


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


def patch_dflash_proposer(pkg_root: Path) -> None:
    path = pkg_root / "v1/spec_decode/dflash.py"
    replace_exact(
        path,
        """from typing import Any

import torch
""",
        """from typing import Any
import os

import torch
""",
    )
    replace_exact(
        path,
        """from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import copy_and_expand_dflash_inputs_kernel
""",
        """from vllm.v1.spec_decode.ddtree_tree import DraftCandidate, build_ddtree
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import copy_and_expand_dflash_inputs_kernel
""",
    )
    replace_exact(
        path,
        """        # DFlash embeds mask tokens directly.
        self.parallel_drafting_hidden_state_tensor = None
""",
        """        # DFlash embeds mask tokens directly.
        self.parallel_drafting_hidden_state_tensor = None

        # aeon_dflash_ddtree_m4b
        # Experimental DDTree payload builder. This leaves flat DFlash behavior
        # intact: the returned tensor is still the top-1 chain, while the
        # best-first tree rides beside it through the M3 scheduler bridge.
        self.ddtree_budget = int(os.environ.get("DDTREE_BUDGET", "22"))
        self.ddtree_top_k = int(os.environ.get("DDTREE_TOP_K", "4"))
        self._last_ddtree_payloads: list[dict[str, object]] | None = None
""",
    )
    replace_exact(
        path,
        """    @override
    @torch.inference_mode()
    def dummy_run(
""",
        """    def _build_ddtree_payloads_from_logits(
        self,
        logits: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if self.ddtree_budget < 1:
            raise ValueError("DDTREE_BUDGET must be >= 1")
        if self.ddtree_top_k < 1:
            raise ValueError("DDTREE_TOP_K must be >= 1")

        top_k = min(self.ddtree_top_k, logits.shape[-1])
        top_values, top_ids = torch.topk(logits, k=top_k, dim=-1)
        draft_token_ids = top_ids[:, 0].view(batch_size, self.num_speculative_tokens)

        top_ids_cpu = (
            top_ids.view(batch_size, self.num_speculative_tokens, top_k)
            .detach()
            .cpu()
            .tolist()
        )
        top_scores_cpu = (
            top_values.view(batch_size, self.num_speculative_tokens, top_k)
            .detach()
            .float()
            .cpu()
            .tolist()
        )

        payloads: list[dict[str, object]] = []
        for req_index in range(batch_size):
            candidates_by_depth: list[list[DraftCandidate]] = []
            for depth in range(self.num_speculative_tokens):
                candidates_by_depth.append(
                    [
                        DraftCandidate(token_id=int(token_id), logprob=float(score))
                        for token_id, score in zip(
                            top_ids_cpu[req_index][depth],
                            top_scores_cpu[req_index][depth],
                            strict=True,
                        )
                    ]
                )
            tree = build_ddtree(
                candidates_by_depth,
                budget=min(self.ddtree_budget, self.num_speculative_tokens * top_k),
                top_k=top_k,
                chain_seed=True,
            )
            payloads.append(
                {
                    "method": "dflash_ddtree",
                    "version": 1,
                    "budget": self.ddtree_budget,
                    "effective_budget": len(tree.non_root_nodes),
                    "top_k": top_k,
                    "num_speculative_tokens": self.num_speculative_tokens,
                    "score_type": "draft_logits",
                    "tree_token_ids": list(tree.token_ids_for_verifier()),
                    "parent_indices": list(tree.parent_indices_for_verifier()),
                    "node_depths": [node.depth for node in tree.non_root_nodes],
                    "node_scores": [float(node.score) for node in tree.non_root_nodes],
                    "flat_fallback_token_ids": draft_token_ids[req_index].tolist(),
                }
            )

        self._last_ddtree_payloads = payloads
        return draft_token_ids

    def _ddtree_greedy_sample(
        self,
        hidden_states: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        logits = self.model.compute_logits(hidden_states)
        return self._build_ddtree_payloads_from_logits(logits, batch_size)

    def pop_last_ddtree_payloads(self) -> list[dict[str, object]] | None:
        payloads = self._last_ddtree_payloads
        self._last_ddtree_payloads = None
        return payloads

    @override
    @torch.inference_mode()
    def dummy_run(
""",
    )


def patch_base_proposer(pkg_root: Path) -> None:
    path = pkg_root / "v1/spec_decode/llm_base_proposer.py"
    replace_exact(
        path,
        """        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            draft_token_ids = self._greedy_sample(sample_hidden_states)
            return draft_token_ids.view(-1, self.num_speculative_tokens)
""",
        """        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            if self.method == "dflash_ddtree":
                draft_token_ids = self._ddtree_greedy_sample(  # type: ignore[attr-defined]
                    sample_hidden_states, batch_size
                )
            else:
                draft_token_ids = self._greedy_sample(sample_hidden_states)
            return draft_token_ids.view(-1, self.num_speculative_tokens)
""",
    )


def patch_gpu_model_runner(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """        draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
        return DraftTokenIds(req_ids, draft_token_ids)
""",
        """        draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
        draft_trees = None
        payloads = getattr(self, "_draft_tree_payloads_cpu", None)
        payload_req_ids = getattr(self, "_draft_tree_req_ids_cpu", None)
        if payloads and payload_req_ids:
            payload_by_req_id = dict(zip(payload_req_ids, payloads, strict=False))
            draft_trees = {
                req_id: payload_by_req_id[req_id]
                for req_id in req_ids
                if req_id in payload_by_req_id
            }
            if not draft_trees:
                draft_trees = None
        return DraftTokenIds(req_ids, draft_token_ids, draft_trees=draft_trees)
""",
    )
    replace_exact(
        path,
        """            draft_token_ids = self.drafter.propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                sampling_metadata=sampling_metadata,
                common_attn_metadata=common_attn_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                slot_mappings=slot_mappings,
            )
""",
        """            draft_token_ids = self.drafter.propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                sampling_metadata=sampling_metadata,
                common_attn_metadata=common_attn_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                slot_mappings=slot_mappings,
            )
            if spec_config.method == "dflash_ddtree":
                pop_payloads = getattr(self.drafter, "pop_last_ddtree_payloads", None)
                self._draft_tree_payloads_cpu = (
                    pop_payloads() if pop_payloads is not None else None
                )
                self._draft_tree_req_ids_cpu = self.input_batch.req_ids.copy()
            else:
                self._draft_tree_payloads_cpu = None
                self._draft_tree_req_ids_cpu = None
""",
    )


def verify_static(pkg_root: Path) -> None:
    expected = {
        "v1/spec_decode/dflash.py": (
            "_build_ddtree_payloads_from_logits",
            "pop_last_ddtree_payloads",
            "flat_fallback_token_ids",
        ),
        "v1/spec_decode/llm_base_proposer.py": (
            'self.method == "dflash_ddtree"',
            "_ddtree_greedy_sample",
        ),
        "v1/worker/gpu_model_runner.py": (
            "_draft_tree_payloads_cpu",
            "DraftTokenIds(req_ids, draft_token_ids, draft_trees=draft_trees)",
        ),
    }
    for rel, needles in expected.items():
        text = (pkg_root / rel).read_text()
        for needle in needles:
            if needle not in text:
                raise RuntimeError(f"Static M4B verification failed: {rel} missing {needle}")


def main() -> int:
    pkg_root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if pkg_root_override:
        pkg_root = Path(pkg_root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_dflash_proposer(pkg_root)
    patch_base_proposer(pkg_root)
    patch_gpu_model_runner(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DFlash top-k DDTree payload builder verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
