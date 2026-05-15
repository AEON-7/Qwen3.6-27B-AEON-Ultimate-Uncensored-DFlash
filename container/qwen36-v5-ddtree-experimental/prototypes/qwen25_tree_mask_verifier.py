#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototypes"))

from ddtree_tree import DDTree  # noqa: E402
from ddtree_vllm_metadata import (  # noqa: E402
    TreeVerifierMetadata,
    greedy_sample_from_compact_logits,
    make_position_ids,
    make_prefill_tree_attention_mask,
)
from qwen25_tree_oracle import (  # noqa: E402
    build_self_draft_candidates,
    encode_prompt,
    greedy_tokens,
    load_model_and_tokenizer,
    next_logits,
    summarize_tree,
)
from ddtree_tree import build_ddtree  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen2.5 one-pass DDTree ancestor-mask verifier prototype."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="Write one vivid sentence about sunrise over the ocean.")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


@torch.inference_mode()
def tree_verify_logits(model, prompt_ids: torch.Tensor, tree: DDTree) -> torch.Tensor:
    tree_ids = torch.tensor(
        [tree.token_ids_for_verifier()],
        device=prompt_ids.device,
        dtype=prompt_ids.dtype,
    )
    input_ids = torch.cat([prompt_ids, tree_ids], dim=-1)
    position_ids = make_position_ids(prompt_ids.shape[-1], tree, prompt_ids.device)
    mask_dtype = next(model.parameters()).dtype
    attention_mask = make_prefill_tree_attention_mask(
        prompt_len=prompt_ids.shape[-1],
        tree=tree,
        device=prompt_ids.device,
        dtype=mask_dtype,
    )
    mask_mapping = {
        "full_attention": attention_mask,
        "sliding_attention": attention_mask,
    }

    outputs = model.model(
        input_ids=input_ids,
        attention_mask=mask_mapping,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = model.lm_head(outputs.last_hidden_state).float()
    return logits[:, prompt_ids.shape[-1] :, :].squeeze(0)


@torch.inference_mode()
def replay_logits_for_tree(model, prompt_ids: torch.Tensor, tree: DDTree) -> dict[tuple[int, ...], torch.Tensor]:
    logits_by_path: dict[tuple[int, ...], torch.Tensor] = {}
    for node in tree.non_root_nodes:
        path = tree.path_token_ids(node.index)
        path_ids = torch.tensor([path], device=prompt_ids.device, dtype=prompt_ids.dtype)
        input_ids = torch.cat([prompt_ids, path_ids], dim=-1)
        logits_by_path[path] = next_logits(model, input_ids)
    return logits_by_path


def main() -> int:
    args = parse_args()
    # Force eager attention so the prototype's 4D ancestor mask is honored by
    # the reference Transformers model instead of an optimized causal kernel.
    args.extra_model_kwargs = {"attn_implementation": "eager"}
    started = time.perf_counter()

    tokenizer, model, _ = load_model_and_tokenizer(args)
    if getattr(model.config, "_attn_implementation", None) != "eager":
        model.config._attn_implementation = "eager"

    prompt_ids = encode_prompt(tokenizer, args.prompt).to(next(model.parameters()).device)
    candidates_by_depth, chain_tokens = build_self_draft_candidates(
        model,
        prompt_ids,
        depth=args.depth,
        top_k=args.top_k,
    )
    tree = build_ddtree(
        candidates_by_depth,
        budget=args.budget,
        top_k=args.top_k,
        chain_seed=True,
        root_token_id=int(prompt_ids[0, -1].item()),
    )
    verifier_metadata = TreeVerifierMetadata.from_tree(
        prompt_len=int(prompt_ids.shape[-1]),
        tree=tree,
    )

    batched_tree_logits = tree_verify_logits(model, prompt_ids, tree)
    replay_logits = replay_logits_for_tree(model, prompt_ids, tree)

    comparisons: list[dict[str, object]] = []
    max_abs_diff = 0.0
    top1_matches = 0
    root_logits = next_logits(model, prompt_ids)

    for offset, node in enumerate(tree.non_root_nodes):
        path = tree.path_token_ids(node.index)
        batched_logits = batched_tree_logits[offset]
        replay = replay_logits[path]
        diff = torch.max(torch.abs(batched_logits - replay)).item()
        batched_top1 = int(torch.argmax(batched_logits).item())
        replay_top1 = int(torch.argmax(replay).item())
        if batched_top1 == replay_top1:
            top1_matches += 1
        max_abs_diff = max(max_abs_diff, float(diff))
        comparisons.append(
            {
                "node_index": node.index,
                "path": path,
                "batched_top1": batched_top1,
                "replay_top1": replay_top1,
                "top1_match": batched_top1 == replay_top1,
                "max_abs_logit_diff": round(float(diff), 6),
            }
        )

    compact_logits = torch.vstack([root_logits.unsqueeze(0), batched_tree_logits])
    walk = greedy_sample_from_compact_logits(tree=tree, compact_logits=compact_logits)
    baseline = greedy_tokens(model, prompt_ids, len(walk.output_token_ids))

    result = {
        "model": args.model,
        "prompt": args.prompt,
        "prompt_tokens": int(prompt_ids.shape[-1]),
        "budget": args.budget,
        "depth": args.depth,
        "top_k": args.top_k,
        "tree_nodes_including_root": len(tree.nodes),
        "compact_logits_indices": verifier_metadata.compact_logits_indices,
        "edge_parent_compact_indices": verifier_metadata.edge_parent_compact_indices,
        "node_compact_indices": verifier_metadata.node_compact_indices,
        "self_draft_chain_tokens": chain_tokens,
        "top1_matches": top1_matches,
        "verifier_nodes": len(tree.non_root_nodes),
        "all_tree_logits_top1_match_replay": top1_matches == len(tree.non_root_nodes),
        "max_abs_logit_diff": round(max_abs_diff, 6),
        "accepted_token_ids": walk.accepted_token_ids,
        "bonus_token_id": walk.bonus_token_id,
        "tree_output_token_ids": walk.output_token_ids,
        "baseline_greedy_token_ids": baseline,
        "matches_baseline_greedy": tuple(baseline) == walk.output_token_ids,
        "tree_output_text": tokenizer.decode(walk.output_token_ids, skip_special_tokens=False),
        "baseline_greedy_text": tokenizer.decode(baseline, skip_special_tokens=False),
        "comparisons": comparisons,
        "tree": summarize_tree(tree),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    output = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n")
    print(output)
    ok = result["matches_baseline_greedy"] and result["all_tree_logits_top1_match_replay"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
