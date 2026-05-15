#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototypes"))

from qwen25_tree_oracle import load_model_and_tokenizer, run_oracle  # noqa: E402


PROMPTS: tuple[tuple[str, str], ...] = (
    (
        "coding",
        "Write a compact Python function that returns the first duplicate item in a list.",
    ),
    (
        "math",
        "A shop discounts a $48 item by 25%, then adds 8% tax. What is the final price?",
    ),
    (
        "reasoning",
        "Mira is older than Sol, and Sol is older than Ren. Who is the youngest?",
    ),
    (
        "prose",
        "Write one vivid sentence about sunrise over the ocean.",
    ),
    (
        "natural",
        "Explain why rainbows appear after storms in plain language.",
    ),
    (
        "extraction",
        "Extract JSON with name and city from: 'Ada Lovelace moved to London.'",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Qwen2.5 DDTree oracle prompt suite.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--output", default="/tmp/qwen25_tree_oracle_suite.json")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite_started = time.perf_counter()
    tokenizer, model, _ = load_model_and_tokenizer(args)

    results: list[dict[str, object]] = []
    for category, prompt in PROMPTS:
        started = time.perf_counter()
        result = run_oracle(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            model_name=args.model,
            budget=args.budget,
            depth=args.depth,
            top_k=args.top_k,
            started=started,
            category=category,
        )
        results.append(result)

    summary = {
        "model": args.model,
        "budget": args.budget,
        "depth": args.depth,
        "top_k": args.top_k,
        "prompt_count": len(results),
        "all_match_baseline_greedy": all(
            bool(result["matches_baseline_greedy"]) for result in results
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
