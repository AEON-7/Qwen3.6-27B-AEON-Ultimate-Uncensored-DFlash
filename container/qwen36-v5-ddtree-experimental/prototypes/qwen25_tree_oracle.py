#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototypes"))

from ddtree_tree import DDTree, DraftCandidate, build_ddtree, greedy_tree_walk  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen2.5 DDTree greedy-oracle correctness prototype."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="Write one vivid sentence about sunrise over the ocean.")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--output", default=None, help="Optional path to write JSON output.")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def dtype_from_name(name: str):
    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def encode_prompt(tokenizer, prompt: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(encoded, torch.Tensor):
        return encoded
    if isinstance(encoded, dict):
        return encoded["input_ids"]
    if hasattr(encoded, "input_ids"):
        return encoded.input_ids
    raise TypeError(f"Unsupported chat template output: {type(encoded)!r}")


@torch.inference_mode()
def next_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    output = model(input_ids=input_ids)
    return output.logits[:, -1, :].float().squeeze(0)


def topk_candidates(logits: torch.Tensor, top_k: int) -> list[DraftCandidate]:
    log_probs = torch.log_softmax(logits, dim=-1)
    values, indices = torch.topk(log_probs, k=top_k)
    return [
        DraftCandidate(token_id=int(token_id), logprob=float(logprob))
        for token_id, logprob in zip(indices.tolist(), values.tolist(), strict=True)
    ]


@torch.inference_mode()
def greedy_tokens(model, prefix_ids: torch.Tensor, steps: int) -> list[int]:
    tokens: list[int] = []
    input_ids = prefix_ids
    for _ in range(steps):
        token_id = int(torch.argmax(next_logits(model, input_ids)).item())
        tokens.append(token_id)
        next_id = torch.tensor([[token_id]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, next_id], dim=-1)
    return tokens


def build_self_draft_candidates(
    model,
    prompt_ids: torch.Tensor,
    *,
    depth: int,
    top_k: int,
) -> tuple[list[list[DraftCandidate]], list[int]]:
    candidates_by_depth: list[list[DraftCandidate]] = []
    chain_tokens: list[int] = []
    input_ids = prompt_ids

    for _ in range(depth):
        candidates = topk_candidates(next_logits(model, input_ids), top_k)
        candidates_by_depth.append(candidates)
        top1 = candidates[0].token_id
        chain_tokens.append(top1)
        next_id = torch.tensor([[top1]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, next_id], dim=-1)

    return candidates_by_depth, chain_tokens


def summarize_tree(tree: DDTree) -> list[dict[str, object]]:
    return [
        {
            "index": node.index,
            "parent_index": node.parent_index,
            "token_id": node.token_id,
            "depth": node.depth,
            "score": round(node.score, 6),
            "path": tree.path_token_ids(node.index),
        }
        for node in tree.nodes
    ]


def load_model_and_tokenizer(args: argparse.Namespace):
    started = time.perf_counter()
    dtype = dtype_from_name(args.dtype)
    extra_model_kwargs = getattr(args, "extra_model_kwargs", {})

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        **extra_model_kwargs,
    )
    model.to(args.device)
    model.eval()
    return tokenizer, model, started


def run_oracle(
    *,
    model,
    tokenizer,
    prompt: str,
    model_name: str,
    budget: int,
    depth: int,
    top_k: int,
    started: float | None = None,
    category: str | None = None,
) -> dict[str, object]:
    if started is None:
        started = time.perf_counter()
    device = next(model.parameters()).device

    prompt_ids = encode_prompt(tokenizer, prompt).to(device)
    candidates_by_depth, chain_tokens = build_self_draft_candidates(
        model,
        prompt_ids,
        depth=depth,
        top_k=top_k,
    )
    tree = build_ddtree(
        candidates_by_depth,
        budget=budget,
        top_k=top_k,
        chain_seed=True,
        root_token_id=int(prompt_ids[0, -1].item()),
    )

    def next_token_for_path(path: tuple[int, ...]) -> int:
        if path:
            path_ids = torch.tensor([path], device=prompt_ids.device, dtype=prompt_ids.dtype)
            input_ids = torch.cat([prompt_ids, path_ids], dim=-1)
        else:
            input_ids = prompt_ids
        return int(torch.argmax(next_logits(model, input_ids)).item())

    walk = greedy_tree_walk(tree, next_token_for_path)
    baseline = greedy_tokens(model, prompt_ids, len(walk.output_token_ids))

    return {
        "model": model_name,
        "category": category,
        "prompt": prompt,
        "prompt_tokens": int(prompt_ids.shape[-1]),
        "budget": budget,
        "depth": depth,
        "top_k": top_k,
        "tree_nodes_including_root": len(tree.nodes),
        "verifier_token_ids": tree.token_ids_for_verifier(),
        "verifier_parent_indices": tree.parent_indices_for_verifier(),
        "self_draft_chain_tokens": chain_tokens,
        "accepted_token_ids": walk.accepted_token_ids,
        "bonus_token_id": walk.bonus_token_id,
        "tree_output_token_ids": walk.output_token_ids,
        "baseline_greedy_token_ids": baseline,
        "matches_baseline_greedy": tuple(baseline) == walk.output_token_ids,
        "accepted_text": tokenizer.decode(walk.accepted_token_ids, skip_special_tokens=False),
        "bonus_text": tokenizer.decode([walk.bonus_token_id], skip_special_tokens=False),
        "tree_output_text": tokenizer.decode(walk.output_token_ids, skip_special_tokens=False),
        "baseline_greedy_text": tokenizer.decode(baseline, skip_special_tokens=False),
        "tree": summarize_tree(tree),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> int:
    args = parse_args()
    tokenizer, model, started = load_model_and_tokenizer(args)
    result = run_oracle(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        model_name=args.model,
        budget=args.budget,
        depth=args.depth,
        top_k=args.top_k,
        started=started,
    )
    output = json.dumps(result, indent=2, default=lambda value: asdict(value))
    if args.output:
        Path(args.output).write_text(output + "\n")
    print(output)
    return 0 if result["matches_baseline_greedy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
