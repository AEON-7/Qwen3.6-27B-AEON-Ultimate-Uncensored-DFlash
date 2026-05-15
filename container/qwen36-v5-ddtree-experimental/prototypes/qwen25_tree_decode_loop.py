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

from ddtree_tree import build_ddtree  # noqa: E402
from ddtree_vllm_metadata import greedy_sample_from_compact_logits  # noqa: E402
from qwen25_tree_mask_verifier import tree_verify_logits  # noqa: E402
from qwen25_tree_oracle import (  # noqa: E402
    build_self_draft_candidates,
    encode_prompt,
    greedy_tokens,
    load_model_and_tokenizer,
    next_logits,
    summarize_tree,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen2.5 multi-step DDTree decode-loop correctness prototype. "
            "This uses the target as its own drafter so any mismatch is a "
            "tree verifier/sampler bug, not a draft-model quality issue."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="Write one vivid sentence about sunrise over the ocean.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


@torch.inference_mode()
def run_tree_decode_loop(
    *,
    model,
    tokenizer,
    prompt: str,
    model_name: str,
    max_new_tokens: int,
    budget: int,
    depth: int,
    top_k: int,
    category: str | None = None,
    started: float | None = None,
) -> dict[str, object]:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if started is None:
        started = time.perf_counter()

    device = next(model.parameters()).device
    prompt_ids = encode_prompt(tokenizer, prompt).to(device)
    context_ids = prompt_ids
    generated_ids: list[int] = []
    steps: list[dict[str, object]] = []

    while len(generated_ids) < max_new_tokens:
        remaining = max_new_tokens - len(generated_ids)
        step_depth = max(1, min(depth, remaining))
        step_budget = max(1, min(budget, remaining))
        step_started = time.perf_counter()

        candidates_by_depth, chain_tokens = build_self_draft_candidates(
            model,
            context_ids,
            depth=step_depth,
            top_k=top_k,
        )
        tree = build_ddtree(
            candidates_by_depth,
            budget=step_budget,
            top_k=top_k,
            chain_seed=True,
            root_token_id=int(context_ids[0, -1].item()),
        )

        root_logits = next_logits(model, context_ids)
        tree_logits = tree_verify_logits(model, context_ids, tree)
        compact_logits = torch.vstack([root_logits.unsqueeze(0), tree_logits])
        walk = greedy_sample_from_compact_logits(tree=tree, compact_logits=compact_logits)

        emitted = list(walk.output_token_ids)[:remaining]
        if not emitted:
            raise RuntimeError("DDTree decode loop emitted no tokens")

        generated_ids.extend(emitted)
        emitted_tensor = torch.tensor([emitted], device=context_ids.device, dtype=context_ids.dtype)
        context_ids = torch.cat([context_ids, emitted_tensor], dim=-1)

        steps.append(
            {
                "step_index": len(steps),
                "remaining_before_step": remaining,
                "context_tokens_before_step": int(context_ids.shape[-1] - len(emitted)),
                "tree_nodes_including_root": len(tree.nodes),
                "verifier_nodes": len(tree.non_root_nodes),
                "self_draft_chain_tokens": chain_tokens,
                "accepted_token_ids": list(walk.accepted_token_ids),
                "bonus_token_id": int(walk.bonus_token_id),
                "emitted_token_ids": emitted,
                "accepted_count": len(walk.accepted_token_ids),
                "emitted_count": len(emitted),
                "tree": summarize_tree(tree),
                "elapsed_seconds": round(time.perf_counter() - step_started, 3),
            }
        )

    baseline_ids = greedy_tokens(model, prompt_ids, max_new_tokens)
    matches = tuple(baseline_ids) == tuple(generated_ids)
    first_mismatch: dict[str, object] | None = None
    if not matches:
        for offset, (expected, actual) in enumerate(zip(baseline_ids, generated_ids, strict=False)):
            if expected != actual:
                first_mismatch = {
                    "offset": offset,
                    "baseline_token_id": int(expected),
                    "tree_token_id": int(actual),
                    "baseline_text": tokenizer.decode([expected], skip_special_tokens=False),
                    "tree_text": tokenizer.decode([actual], skip_special_tokens=False),
                }
                break
        if first_mismatch is None:
            first_mismatch = {
                "offset": min(len(baseline_ids), len(generated_ids)),
                "baseline_length": len(baseline_ids),
                "tree_length": len(generated_ids),
            }

    accepted_counts = [int(step["accepted_count"]) for step in steps]
    emitted_counts = [int(step["emitted_count"]) for step in steps]
    mean_accepted = sum(accepted_counts) / len(accepted_counts)
    mean_emitted = sum(emitted_counts) / len(emitted_counts)

    return {
        "model": model_name,
        "category": category,
        "prompt": prompt,
        "prompt_tokens": int(prompt_ids.shape[-1]),
        "max_new_tokens": max_new_tokens,
        "budget": budget,
        "depth": depth,
        "top_k": top_k,
        "decode_steps": len(steps),
        "mean_accepted_tokens_per_step": round(mean_accepted, 3),
        "mean_emitted_tokens_per_step": round(mean_emitted, 3),
        "total_tree_tokens": len(generated_ids),
        "matches_baseline_greedy": matches,
        "first_mismatch": first_mismatch,
        "tree_output_token_ids": generated_ids,
        "baseline_greedy_token_ids": baseline_ids,
        "tree_output_text": tokenizer.decode(generated_ids, skip_special_tokens=False),
        "baseline_greedy_text": tokenizer.decode(baseline_ids, skip_special_tokens=False),
        "steps": steps,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> int:
    args = parse_args()
    args.extra_model_kwargs = {"attn_implementation": "eager"}
    started = time.perf_counter()
    tokenizer, model, _ = load_model_and_tokenizer(args)
    if getattr(model.config, "_attn_implementation", None) != "eager":
        model.config._attn_implementation = "eager"

    result = run_tree_decode_loop(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        budget=args.budget,
        depth=args.depth,
        top_k=args.top_k,
        started=started,
    )
    output = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n")
    print(output)
    return 0 if result["matches_baseline_greedy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
