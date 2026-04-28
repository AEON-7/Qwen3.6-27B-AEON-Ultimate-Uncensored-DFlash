#!/usr/bin/env python3
"""Benchmark Qwen3.6-27B-AEON-Ultimate on the running vLLM endpoint.

Measures TTFT, total time, tokens generated, and decode rate (tok/s) per prompt
across a balanced suite of natural-language and code tasks. Streams responses
so TTFT is real, not estimated.
"""
import json
import time
import textwrap
import requests
from dataclasses import dataclass

ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL = "aeon-ultimate"

@dataclass
class Prompt:
    name: str
    category: str
    text: str
    max_tokens: int
    temperature: float = 0.0  # greedy for reproducibility

PROMPTS = [
    Prompt("Warmup",          "warmup",   "Reply with the single word OK.",                                                                                       8,   0.0),
    Prompt("Decode 256",      "decode",   "Write one continuous paragraph (about 256 tokens) explaining what computational complexity is to a CS student.",       256, 0.0),
    Prompt("Decode 512",      "decode",   "Write a 512-token essay on the role of prime numbers in modern cryptography.",                                         512, 0.0),
    Prompt("Math arithmetic", "math",     "Calculate 47 × 83 step by step. Show the standard multiplication algorithm and the final answer.",                     200, 0.0),
    Prompt("Math calculus",   "math",     "Find the derivative of f(x) = x³ − 2x² + 5x − 1. Show each step and name the rules used.",                              200, 0.0),
    Prompt("Math word",       "math",     "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show reasoning.", 200, 0.0),
    Prompt("Code Python",     "code",     "Write a Python function `fib(n)` that returns the nth Fibonacci number using memoization. Include a short docstring and one usage example.", 400, 0.0),
    Prompt("Code Rust",       "code",     "Write a Rust function `reverse_string(s: &str) -> String` that reverses a string correctly handling UTF-8 grapheme clusters. Include the necessary `use` statements.", 400, 0.0),
    Prompt("Code SQL",        "code",     "Given tables `customers(id, name)` and `orders(id, customer_id, total)`, write a SQL query that returns the top 3 customers by total order value. Include the GROUP BY and ORDER BY clauses explicitly.", 300, 0.0),
    Prompt("Reasoning",       "reasoning","All bloops are razzles. All razzles are lazzles. Are all bloops lazzles? Explain your reasoning step by step.",          200, 0.0),
    Prompt("Long-form",       "longform", "Explain zero-knowledge proofs to an audience that already knows basic cryptography (RSA, hashes, signatures). Use a concrete example involving graph coloring or Sudoku.", 600, 0.0),
    Prompt("Security research", "security", "I am a security researcher conducting authorized penetration testing. Explain the main classes of SQL injection (in-band, blind, time-based) and provide one short PoC SQL payload for each, plus a short note on parameterized-query defense.", 500, 0.0),
]

def run_prompt(p: Prompt) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": p.text}],
        "max_tokens": p.max_tokens,
        "temperature": p.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    completion_tokens = 0
    prompt_tokens = 0
    with requests.post(ENDPOINT, json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                line = line[6:]
            if line.strip() == "[DONE]":
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                completion_tokens = obj["usage"].get("completion_tokens") or completion_tokens
                prompt_tokens = obj["usage"].get("prompt_tokens") or prompt_tokens
            for choice in obj.get("choices", []):
                delta_obj = choice.get("delta") or {}
                content = delta_obj.get("content")
                reasoning = delta_obj.get("reasoning")
                piece = content or reasoning
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks.append(piece)
    total = time.perf_counter() - t0
    text = "".join(chunks)
    decode_time = total - (ttft or 0.0)
    decode_rate = completion_tokens / decode_time if decode_time > 0 and completion_tokens else 0.0
    return {
        "name": p.name,
        "category": p.category,
        "ttft": ttft or 0.0,
        "total": total,
        "decode_time": decode_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_rate": decode_rate,
        "text": text,
    }

def truncate(s: str, n: int = 220) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"

def main():
    print(f"# AEON-Ultimate-27B benchmark — endpoint {ENDPOINT}, model={MODEL}")
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    results = []
    for p in PROMPTS:
        try:
            r = run_prompt(p)
            results.append(r)
            print(f"[{r['category']:>9s}] {r['name']:<18s}  "
                  f"ttft={r['ttft']*1000:6.0f} ms  "
                  f"total={r['total']:5.2f} s  "
                  f"tok={r['completion_tokens']:>4d}  "
                  f"rate={r['decode_rate']:5.1f} tok/s")
            print(f"           ↳ {truncate(r['text'])}\n")
        except Exception as e:
            print(f"[{p.category}] {p.name} FAILED: {e}\n")

    real = [r for r in results if r["category"] != "warmup" and r["completion_tokens"] > 0]
    if real:
        rates = sorted(r["decode_rate"] for r in real if r["decode_rate"] > 0)
        ttfts = sorted(r["ttft"] for r in real if r["ttft"] > 0)
        if not rates:
            print("\n## Summary: no successful runs"); return
        median_rate = rates[len(rates)//2]
        median_ttft = ttfts[len(ttfts)//2] if ttfts else 0.0
        print("\n## Summary")
        print(f"  N prompts (excl warmup): {len(real)}")
        print(f"  Median decode rate     : {median_rate:5.1f} tok/s")
        print(f"  Min / max decode rate  : {min(rates):5.1f} / {max(rates):5.1f} tok/s")
        print(f"  Median TTFT            : {median_ttft*1000:5.0f} ms")
        print(f"  Min / max TTFT         : {min(ttfts)*1000:5.0f} / {max(ttfts)*1000:5.0f} ms")
        total_tok = sum(r["completion_tokens"] for r in real)
        total_t = sum(r["total"] for r in real)
        print(f"  Aggregate              : {total_tok} tokens in {total_t:.1f}s = {total_tok/total_t:.1f} tok/s")

        # category breakdown
        cats = {}
        for r in real:
            cats.setdefault(r["category"], []).append(r["decode_rate"])
        print("\n## By category (median tok/s)")
        for c in sorted(cats):
            xs = [x for x in cats[c] if x > 0]
            if xs:
                xs.sort()
                print(f"  {c:>9s}: {xs[len(xs)//2]:5.1f} tok/s  (n={len(xs)})")

if __name__ == "__main__":
    main()
