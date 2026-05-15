#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


PROMPTS: dict[str, list[str]] = {
    "coding": [
        "Write a Python function that parses nginx access logs and returns the top 10 IP addresses by request count. Include edge-case handling.",
        "Implement a small TypeScript LRU cache class with get, set, delete, and clear methods. Explain the complexity of each operation.",
        "Review this design: a background worker pulls jobs from Redis and writes status to Postgres. What failure modes should I handle?",
        "Write a Rust function that validates an IPv4 CIDR string and returns the network address and broadcast address.",
    ],
    "math": [
        "Compute 173 multiplied by 487 and show the arithmetic clearly.",
        "A rectangle has its length increased by 25 percent and width decreased by 20 percent. What happens to the area?",
        "Solve for x: 3x^2 - 14x + 8 = 0. Show each step.",
        "Explain why the harmonic series diverges using a grouping argument.",
    ],
    "reasoning": [
        "A farmer needs to cross a river with a wolf, a goat, and a cabbage. The boat carries the farmer plus one item. How can all cross safely?",
        "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. What does the ball cost, and why is the common answer wrong?",
        "You have 12 identical-looking coins; one is counterfeit and may be heavier or lighter. How do you identify it in three weighings?",
        "If every glorp is a wug and no wug is a dax, can any glorp be a dax? Explain the logic.",
    ],
    "prose": [
        "Write a vivid 250-word scene about a lighthouse keeper hearing an impossible song during a storm.",
        "Compose a warm letter from an old astronaut to a child who wants to visit the stars.",
        "Write the opening of a noir mystery set in a rain-soaked railway station at midnight.",
        "Describe a hidden library beneath a desert city, focusing on sensory details and atmosphere.",
    ],
    "natural_language": [
        "Explain to a non-technical homeowner why their Wi-Fi may be slow even though their internet plan is fast.",
        "Help me decide whether to repair an old laptop or buy a new one. Ask clarifying questions and give practical criteria.",
        "Summarize the tradeoffs between working remotely and working in an office for a small engineering team.",
        "Give a thoughtful answer to someone who feels overwhelmed by too many unfinished projects.",
    ],
    "extraction_json": [
        'Extract the people, dates, dollar amounts, and action items from this note as JSON: "Maya met Jordan on April 12. They approved $4,700 for the lab upgrade and asked Priya to order sensors by Friday."',
        'Return valid JSON with keys title, priority, blockers, and next_steps from this ticket: "Deploy failed after the auth migration. High priority. Blocked by missing staging secrets. Next: rotate token and rerun CI."',
        'Convert this inventory sentence into JSON array items with sku, count, and location: "A12 has 14 units in rack 7, B44 has 3 units in cold storage, and C09 has 28 units in receiving."',
        'Read this meeting note and emit compact JSON: "On May 3, Alex will draft the API plan, Sam will review security, and Lena will schedule the rollout for June 1."',
    ],
}


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    ordered = sorted(xs)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def trimmed(xs: list[float], trim_fraction: float) -> list[float]:
    if not xs:
        return []
    if trim_fraction <= 0 or len(xs) < 3:
        return sorted(xs)
    ordered = sorted(xs)
    drop = int(len(ordered) * trim_fraction)
    if drop == 0 or drop * 2 >= len(ordered):
        return ordered
    return ordered[drop:-drop]


def mean_or_none(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


async def stream_one(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    request_id: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{prompt}\n\nRequest marker: {request_id}"}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    started = time.perf_counter()
    ttft = None
    prompt_tokens = 0
    completion_tokens = 0
    chunks = 0
    try:
        async with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=None,
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                return {
                    "ok": False,
                    "error": f"http_{response.status_code}: {body[:240].decode(errors='ignore')}",
                }
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = event.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens) or prompt_tokens
                    completion_tokens = usage.get("completion_tokens", completion_tokens) or completion_tokens
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    text = (
                        delta.get("content")
                        or delta.get("reasoning")
                        or delta.get("reasoning_content")
                        or ""
                    )
                    if text:
                        chunks += 1
                        if ttft is None:
                            ttft = time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__}

    total = time.perf_counter() - started
    decode_s = max(0.0, total - (ttft or 0.0))
    tpot_ms = (decode_s / completion_tokens * 1000.0) if completion_tokens and decode_s else None
    decode_tps = (completion_tokens / decode_s) if completion_tokens and decode_s else None
    pp_tps = (prompt_tokens / ttft) if prompt_tokens and ttft else None
    return {
        "ok": True,
        "ttft_ms": (ttft * 1000.0) if ttft is not None else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_s": total,
        "decode_s": decode_s,
        "tpot_ms": tpot_ms,
        "decode_tps": decode_tps,
        "pp_tps": pp_tps,
        "chunks": chunks,
    }


async def run_batch(
    base_url: str,
    model: str,
    category: str,
    prompts: list[str],
    concurrency: int,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    run_index: int,
) -> dict[str, Any]:
    selected = (prompts * ((concurrency // len(prompts)) + 1))[:concurrency]
    timeout = httpx.Timeout(None, connect=30.0)
    limits = httpx.Limits(max_connections=max(16, concurrency + 8), max_keepalive_connections=max(16, concurrency))
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        started = time.perf_counter()
        results = await asyncio.gather(
            *[
                stream_one(
                    client,
                    base_url,
                    model,
                    prompt,
                    max_tokens,
                    temperature,
                    enable_thinking,
                    f"{category}-{concurrency}-run{run_index}-{i}",
                )
                for i, prompt in enumerate(selected)
            ]
        )
        wall_s = time.perf_counter() - started
    ok = [r for r in results if r.get("ok")]
    errors = [r for r in results if not r.get("ok")]
    total_tokens = sum(int(r.get("completion_tokens") or 0) for r in ok)
    total_prompt_tokens = sum(int(r.get("prompt_tokens") or 0) for r in ok)
    ttfts = [float(r["ttft_ms"]) for r in ok if r.get("ttft_ms") is not None]
    tpots = [float(r["tpot_ms"]) for r in ok if r.get("tpot_ms") is not None]
    decode_tps = [float(r["decode_tps"]) for r in ok if r.get("decode_tps") is not None]
    pp_tps = [float(r["pp_tps"]) for r in ok if r.get("pp_tps") is not None]
    first_token_window_s = (max(ttfts) / 1000.0) if ttfts else None
    return {
        "category": category,
        "concurrency": concurrency,
        "run_index": run_index,
        "requests": concurrency,
        "ok": len(ok),
        "errors": len(errors),
        "first_error": errors[0].get("error") if errors else None,
        "wall_s": wall_s,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_tokens,
        "aggregate_tok_s": total_tokens / wall_s if wall_s > 0 else None,
        "aggregate_prompt_tok_s": (
            total_prompt_tokens / first_token_window_s
            if total_prompt_tokens and first_token_window_s else None
        ),
        "aggregate_prompt_tok_s_wall": total_prompt_tokens / wall_s if wall_s > 0 else None,
        "first_token_window_ms": max(ttfts) if ttfts else None,
        "per_request_tok_s_p50": percentile(decode_tps, 0.50),
        "per_request_tok_s_peak": max(decode_tps) if decode_tps else None,
        "pp_tok_s_p50": percentile(pp_tps, 0.50),
        "pp_tok_s_peak": max(pp_tps) if pp_tps else None,
        "ttft_ms_p50": percentile(ttfts, 0.50),
        "ttft_ms_p95": percentile(ttfts, 0.95),
        "tpot_ms_p50": percentile(tpots, 0.50),
        "tpot_ms_p95": percentile(tpots, 0.95),
        "samples": results,
    }


def summarize_runs(runs: list[dict[str, Any]], trim_fraction: float) -> dict[str, Any]:
    if not runs:
        raise ValueError("No runs to summarize")

    def values(key: str) -> list[float]:
        return [float(r[key]) for r in runs if r.get(key) is not None]

    def trimmed_mean(key: str) -> float | None:
        return mean_or_none(trimmed(values(key), trim_fraction))

    first = runs[0]
    errors = sum(int(r.get("errors") or 0) for r in runs)
    ok = sum(int(r.get("ok") or 0) for r in runs)
    prompt_tokens = sum(int(r.get("prompt_tokens") or 0) for r in runs)
    completion_tokens = sum(int(r.get("completion_tokens") or 0) for r in runs)
    first_error = next((r.get("first_error") for r in runs if r.get("first_error")), None)

    return {
        "category": first["category"],
        "concurrency": first["concurrency"],
        "runs": len(runs),
        "requests": sum(int(r.get("requests") or 0) for r in runs),
        "ok": ok,
        "errors": errors,
        "first_error": first_error,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "aggregate_tok_s_mean": trimmed_mean("aggregate_tok_s"),
        "aggregate_tok_s_peak": max(values("aggregate_tok_s") or [0.0]),
        "aggregate_prompt_tok_s_mean": trimmed_mean("aggregate_prompt_tok_s"),
        "aggregate_prompt_tok_s_peak": max(values("aggregate_prompt_tok_s") or [0.0]),
        "per_request_tok_s_p50_mean": trimmed_mean("per_request_tok_s_p50"),
        "per_request_tok_s_peak": max(values("per_request_tok_s_peak") or [0.0]),
        "pp_tok_s_p50_mean": trimmed_mean("pp_tok_s_p50"),
        "pp_tok_s_peak": max(values("pp_tok_s_peak") or [0.0]),
        "ttft_ms_p50_mean": trimmed_mean("ttft_ms_p50"),
        "ttft_ms_p95_peak": max(values("ttft_ms_p95") or [0.0]),
        "tpot_ms_p50_mean": trimmed_mean("tpot_ms_p50"),
        "tpot_ms_p95_peak": max(values("tpot_ms_p95") or [0.0]),
        "raw_runs": runs,
        "trim_fraction": trim_fraction,
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="aeon-fast")
    parser.add_argument("--levels", default="1,4,8,16,32,64,128,256")
    parser.add_argument("--categories", default=",".join(PROMPTS))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runs-per-point", type=int, default=1)
    parser.add_argument("--min-samples-per-point", type=int, default=0)
    parser.add_argument("--trim-fraction", type=float, default=0.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    categories = [x.strip() for x in args.categories.split(",") if x.strip()]
    missing = [c for c in categories if c not in PROMPTS]
    if missing:
        raise SystemExit(f"Unknown categories: {missing}")

    print(
        f"Qwen3.6 category benchmark: model={args.model} max_tokens={args.max_tokens} "
        f"temperature={args.temperature} enable_thinking={args.enable_thinking} "
        f"runs_per_point={args.runs_per_point} min_samples_per_point={args.min_samples_per_point} "
        f"trim_fraction={args.trim_fraction}"
    )
    print(
        f"{'category':<18} {'c':>4} {'runs':>4} {'ok':>4} {'err':>4} "
        f"{'agg tok/s':>10} {'peak req':>9} {'agg PP':>9} {'PP p50':>9} "
        f"{'TTFT p50':>10} {'TPOT p50':>10}"
    )
    print("-" * 122)

    rows = []
    raw_runs = []
    for category in categories:
        for concurrency in levels:
            runs = []
            target_runs = max(
                args.runs_per_point,
                math.ceil(args.min_samples_per_point / concurrency)
                if args.min_samples_per_point > 0 else 1,
            )
            for run_index in range(target_runs):
                run = await run_batch(
                    args.base_url,
                    args.model,
                    category,
                    PROMPTS[category],
                    concurrency,
                    args.max_tokens,
                    args.temperature,
                    args.enable_thinking,
                    run_index,
                )
                runs.append(run)
                raw_runs.append(run)
            row = summarize_runs(runs, args.trim_fraction)
            rows.append(row)
            print(
                f"{category:<18} {concurrency:>4} {row['runs']:>4} "
                f"{row['ok']:>4} {row['errors']:>4} "
                f"{(row['aggregate_tok_s_mean'] or 0):>10.2f} "
                f"{(row['per_request_tok_s_peak'] or 0):>9.2f} "
                f"{(row['aggregate_prompt_tok_s_mean'] or 0):>9.0f} "
                f"{(row['pp_tok_s_p50_mean'] or 0):>9.0f} "
                f"{(row['ttft_ms_p50_mean'] or 0):>9.0f}ms "
                f"{(row['tpot_ms_p50_mean'] or 0):>9.2f}ms"
            )
            if row["first_error"]:
                print(f"  first_error: {row['first_error']}")

    result = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "levels": levels,
        "categories": categories,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "enable_thinking": args.enable_thinking,
        "runs_per_point": args.runs_per_point,
        "min_samples_per_point": args.min_samples_per_point,
        "trim_fraction": args.trim_fraction,
        "rows": rows,
        "raw_runs": raw_runs,
    }
    output = args.output or f"qwen36-category-bench-{int(time.time())}.json"
    Path(output).write_text(json.dumps(result, indent=2))
    print(f"\nSaved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
