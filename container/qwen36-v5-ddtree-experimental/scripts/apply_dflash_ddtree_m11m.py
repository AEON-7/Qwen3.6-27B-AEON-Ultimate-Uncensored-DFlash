#!/usr/bin/env python3
"""M11M: native stochastic sampling for DDTree verifier rows.

M10M forced DDTree verifier batches through greedy tree sampling because the
stock flat rejection sampler treats root+tree rows as a linear speculative
chain. That preserved DDTree shape but also changed the model's natural
sampling policy. M11M keeps DDTree active while sampling each verifier row with
vLLM's normal sampler, then walks the tree using those sampled row tokens.
"""

from __future__ import annotations

from pathlib import Path

import vllm


ROOT = Path(vllm.__file__).resolve().parent
RUNNER = ROOT / "v1" / "worker" / "gpu_model_runner.py"
SAMPLER = ROOT / "v1" / "spec_decode" / "ddtree_runtime_sampler.py"


def replace_exact(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        raise SystemExit(f"pattern not found in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new))


runtime_helper = r'''
def sample_ddtree_from_row_tokens(
    metadata: DDTreeRuntimeMetadata,
    row_token_ids: torch.Tensor,
) -> DDTreeGreedySample:
    """DDTree walk using one already-sampled token per compact verifier row.

    ``row_token_ids`` must be laid out exactly like compact verifier logits:
    root row followed by one row per non-root tree node for each request.
    The caller decides how those row tokens are sampled. This lets GPU runner
    reuse vLLM's native temperature/top-p sampler while preserving DDTree's
    tree-aware accept/bonus contract.
    """

    if row_token_ids.ndim != 1:
        row_token_ids = row_token_ids.reshape(-1)
    expected_rows = sum(1 + request.num_nodes for request in metadata.requests)
    if row_token_ids.shape[0] != expected_rows:
        raise ValueError(
            f"row_token_ids row mismatch: expected {expected_rows}, "
            f"got {row_token_ids.shape[0]}"
        )

    max_output_len = metadata.max_num_nodes + 1
    output_token_ids = torch.full(
        (metadata.num_requests, max_output_len),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=row_token_ids.device,
    )

    accepted_compact_by_req: list[list[int]] = []
    bonus_parent_by_req: list[int] = []
    offset = 0
    for req_index, request in enumerate(metadata.requests):
        rows = row_token_ids[offset : offset + request.num_nodes + 1]
        accepted_tokens, accepted_compact, bonus_token, bonus_parent = _walk_one_tree(
            request, rows
        )
        emitted, reported_accepted, reported_bonus_parent = (
            _adapt_tree_walk_to_vllm_contract(
                accepted_tokens,
                accepted_compact,
                bonus_token,
                bonus_parent,
            )
        )
        if emitted:
            output_token_ids[req_index, : len(emitted)] = torch.tensor(
                emitted,
                dtype=torch.int32,
                device=row_token_ids.device,
            )
        accepted_compact_by_req.append(reported_accepted)
        bonus_parent_by_req.append(reported_bonus_parent)
        offset += request.num_nodes + 1

    return DDTreeGreedySample(
        output_token_ids=output_token_ids,
        accepted_compact_indices=accepted_compact_by_req,
        bonus_parent_compact_indices=bonus_parent_by_req,
    )


'''

replace_exact(
    SAMPLER,
    "\n\ndef greedy_sample_ddtree(\n",
    "\n\n" + runtime_helper + "def greedy_sample_ddtree(\n",
)

replace_exact(
    RUNNER,
    "from vllm.v1.spec_decode.ddtree_runtime_sampler import (\n"
    "    DDTreeRuntimeMetadata,\n"
    "    greedy_sample_ddtree,\n"
    ")\n",
    "from vllm.v1.spec_decode.ddtree_runtime_sampler import (\n"
    "    DDTreeRuntimeMetadata,\n"
    "    greedy_sample_ddtree,\n"
    "    sample_ddtree_from_row_tokens,\n"
    ")\n",
)

expand_method = r'''
    def _ddtree_expand_sampling_metadata(
        self,
        sampling_metadata,
        runtime_metadata: DDTreeRuntimeMetadata,
    ):
        """Repeat per-request sampling metadata for DDTree verifier rows.

        vLLM's sampler expects one metadata row per logits row. DDTree verifier
        logits are laid out as root+tree rows per request, so native stochastic
        tree sampling needs the request's temperature/top-p/penalties repeated
        across its verifier rows before calling the normal Sampler.
        """

        req_to_batch_index = {
            req_id: idx for idx, req_id in enumerate(self.input_batch.req_ids)
        }
        row_to_req_index: list[int] = []
        for request in runtime_metadata.requests:
            req_index = req_to_batch_index.get(request.req_id)
            if req_index is None:
                return sampling_metadata
            row_to_req_index.extend([req_index] * (request.num_nodes + 1))
        if not row_to_req_index:
            return sampling_metadata

        max_req_index = max(row_to_req_index)

        def expand_tensor(value):
            if value is None or not isinstance(value, torch.Tensor):
                return value
            if value.ndim == 0 or value.shape[0] <= max_req_index:
                return value
            row_index = torch.tensor(
                row_to_req_index,
                dtype=torch.long,
                device=value.device,
            )
            return value.index_select(0, row_index)

        def expand_list(value):
            if value is None:
                return None
            if len(value) <= max_req_index:
                return value
            expanded = []
            for req_index in row_to_req_index:
                item = value[req_index]
                expanded.append(copy(item) if isinstance(item, list) else item)
            return expanded

        expanded_generators = {}
        for row_index, req_index in enumerate(row_to_req_index):
            generator = sampling_metadata.generators.get(req_index)
            if generator is not None:
                expanded_generators[row_index] = generator

        expanded_bad_words = {}
        for row_index, req_index in enumerate(row_to_req_index):
            bad_words = sampling_metadata.bad_words_token_ids.get(req_index)
            if bad_words is not None:
                expanded_bad_words[row_index] = bad_words

        return replace(
            sampling_metadata,
            temperature=expand_tensor(sampling_metadata.temperature),
            top_p=expand_tensor(sampling_metadata.top_p),
            top_k=expand_tensor(sampling_metadata.top_k),
            prompt_token_ids=expand_tensor(sampling_metadata.prompt_token_ids),
            frequency_penalties=expand_tensor(
                sampling_metadata.frequency_penalties
            ),
            presence_penalties=expand_tensor(sampling_metadata.presence_penalties),
            repetition_penalties=expand_tensor(
                sampling_metadata.repetition_penalties
            ),
            allowed_token_ids_mask=expand_tensor(
                sampling_metadata.allowed_token_ids_mask
            ),
            output_token_ids=expand_list(sampling_metadata.output_token_ids),
            spec_token_ids=expand_list(sampling_metadata.spec_token_ids),
            generators=expanded_generators,
            bad_words_token_ids=expanded_bad_words,
            max_num_logprobs=None,
            logprob_token_ids=None,
            thinking_budget_state_holder=None,
        )


'''

replace_exact(
    RUNNER,
    "    def _sample(\n",
    expand_method + "    def _sample(\n",
)

replace_exact(
    RUNNER,
    "                sampling_metadata.all_greedy\n"
    "                or os.environ.get(\"DDTREE_FORCE_GREEDY_TREE_SAMPLER\", \"1\") == \"1\"\n"
    "            )\n",
    "                sampling_metadata.all_greedy\n"
    "                or os.environ.get(\"DDTREE_FORCE_GREEDY_TREE_SAMPLER\", \"1\") == \"1\"\n"
    "                or os.environ.get(\"DDTREE_NATIVE_TREE_SAMPLER\", \"1\") == \"1\"\n"
    "            )\n",
)

replace_exact(
    RUNNER,
    "            if expected_rows == logits.shape[0] and runtime_metadata.num_requests:\n"
    "                tree_sample = greedy_sample_ddtree(runtime_metadata, logits)\n",
    "            if expected_rows == logits.shape[0] and runtime_metadata.num_requests:\n"
    "                use_native_tree_sampling = (\n"
    "                    not sampling_metadata.all_greedy\n"
    "                    and os.environ.get(\"DDTREE_FORCE_GREEDY_TREE_SAMPLER\", \"1\") != \"1\"\n"
    "                    and os.environ.get(\"DDTREE_NATIVE_TREE_SAMPLER\", \"1\") == \"1\"\n"
    "                )\n"
    "                if use_native_tree_sampling:\n"
    "                    expanded_sampling_metadata = self._ddtree_expand_sampling_metadata(\n"
    "                        sampling_metadata,\n"
    "                        runtime_metadata,\n"
    "                    )\n"
    "                    row_sample = self.sampler(\n"
    "                        logits=logits,\n"
    "                        sampling_metadata=expanded_sampling_metadata,\n"
    "                    )\n"
    "                    sampled_rows = row_sample.sampled_token_ids.squeeze(-1).to(\n"
    "                        torch.int64\n"
    "                    )\n"
    "                    tree_sample = sample_ddtree_from_row_tokens(\n"
    "                        runtime_metadata,\n"
    "                        sampled_rows,\n"
    "                    )\n"
    "                    if os.environ.get(\"DDTREE_LOG_SAMPLE\", \"0\") == \"1\":\n"
    "                        native_count = getattr(\n"
    "                            self, \"_ddtree_m11m_native_log_count\", 0\n"
    "                        )\n"
    "                        native_limit = int(\n"
    "                            os.environ.get(\"DDTREE_LOG_SAMPLE_LIMIT\", \"16\")\n"
    "                        )\n"
    "                        if native_count < native_limit:\n"
    "                            logger.warning(\n"
    "                                \"DDTree M11M native row sampler rows=%s sample=%s\",\n"
    "                                int(sampled_rows.shape[0]),\n"
    "                                sampled_rows[:16].detach().cpu().tolist(),\n"
    "                            )\n"
    "                            self._ddtree_m11m_native_log_count = native_count + 1\n"
    "                else:\n"
    "                    tree_sample = greedy_sample_ddtree(runtime_metadata, logits)\n",
)

print("Applied AEON DFlash DDTree M11M native stochastic tree sampler patch")
