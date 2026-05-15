#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototypes"))

from qwen25_tree_decode_loop import run_tree_decode_loop  # noqa: E402
from qwen25_tree_oracle import load_model_and_tokenizer  # noqa: E402
from qwen25_tree_oracle_suite import PROMPTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multi-step Qwen2.5 DDTree decode loop across six prompt categories."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--output", default="/tmp/qwen25_tree_decode_suite.json")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.extra_model_kwargs = {"attn_implementation": "eager"}
    suite_started = time.perf_counter()
    tokenizer, model, _ = load_model_and_tokenizer(args)
    if getattr(model.config, "_attn_implementation", None) != "eager":
        model.config._attn_implementation = "eager"

    results: list[dict[str, object]] = []
    for category, prompt in PROMPTS:
        started = time.perf_counter()
        result = run_tree_decode_loop(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            budget=args.budget,
            depth=args.depth,
            top_k=args.top_k,
            category=category,
            started=started,
        )
        results.append(result)

    summary = {
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "budget": args.budget,
        "depth": args.depth,
        "top_k": args.top_k,
        "prompt_count": len(results),
        "all_match_baseline_greedy": all(
            bool(result["matches_baseline_greedy"]) for result in results
        ),
        "mean_decode_steps": round(
            sum(int(result["decode_steps"]) for result in results) / len(results),
            3,
        ),
        "mean_accepted_tokens_per_step": round(
            sum(float(result["mean_accepted_tokens_per_step"]) for result in results) / len(results),
            3,
        ),
        "mean_emitted_tokens_per_step": round(
            sum(float(result["mean_emitted_tokens_per_step"]) for result in results) / len(results),
            3,
        ),
        "elapsed_seconds": round(time.perf_counter() - suite_started, 3),
        "results": results,
    }

    output = json.dumps(summary, indent=2)
    Path(args.output).write_text(output + "\n")
    print(output)
    return 0 if summary["all_match_baseline_greedy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
