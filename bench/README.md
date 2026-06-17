# Benchmark scripts

The two scripts in this folder are exactly what we use to measure the throughput numbers cited in the top-level [README's Performance section](../README.md#performance--dgx-spark-dflash-vs-raw-baseline). Run them yourself to verify the numbers on your hardware, or to debug a configuration that isn't hitting expected throughput.

## Scripts

| Script | Thinking mode | Use case |
|---|---|---|
| [`bench_aeon.py`](bench_aeon.py) | **disabled** (`chat_template_kwargs.enable_thinking=false`) | Clean decode-rate measurement. The numbers in our headline performance table come from this script. Best for like-for-like comparisons against other speculative-decoding setups. |
| [`bench_aeon_thinking.py`](bench_aeon_thinking.py) | **enabled** (default user-facing path) | Real-world tok/s as users see it. Counts both reasoning tokens and content tokens as output. Slightly lower median (~13 % gap on our DGX Spark) but represents the default chat experience. |

Both scripts:

- Hit `http://localhost:8000/v1/chat/completions`
- Use **streaming** to measure TTFT correctly
- Set `temperature=0` (greedy) for deterministic measurement
- Run **single-stream sequentially** — one prompt at a time, no concurrent load
- Cover 11 prompts spanning math, code, reasoning, long-form, security, and pure-decode workloads
- Compute decode rate as `completion_tokens / decode_time` (excluding TTFT prefill)
- Report both per-prompt and median/min/max + by-category aggregates

## Single-stream is intentional

The numbers we publish are **single-stream greedy** by design. Under concurrent serving, per-stream throughput drops roughly linearly with the number of concurrent requests — a single GPU can only do so much work in parallel. Two concurrent streams ≈ half per-stream tok/s; eight concurrent streams ≈ one-eighth.

If your production workload is multi-user / batched, the relevant number is **aggregate** throughput, not per-stream. To estimate aggregate at your concurrency level, you can scale up to your `--max-num-seqs` setting and measure with a load generator like `locust` or `wrk`.

## Running

The scripts assume vLLM is up at `http://localhost:8000` and listening as `aeon-ultimate`. Override the model name and endpoint via env vars or by editing the constants at the top of the script.

```bash
# Thinking off (matches our headline numbers)
python3 bench_aeon.py

# Thinking on (real user-facing default)
python3 bench_aeon_thinking.py
```

Each run takes ~2.5 minutes on a healthy DGX Spark. Both scripts require only `requests` from the Python standard ecosystem — no vLLM client lib, no HF lib.

## Expected numbers (DGX Spark / GB10 / sm_121a, single-stream)

From reference passes on an earlier per-model image (the superseded `qwen36-v2.1` lineage); these have **not** yet been re-run on the unified production image `ghcr.io/aeon-7/aeon-vllm-ultimate:latest`:

| Metric | Thinking OFF (headline) | Thinking ON (user-default) |
|---|---|---|
| Median decode | **32.5 tok/s** | **28.3 tok/s** |
| Peak decode | 56.7 tok/s (Code Python) | 46.1 tok/s (Reasoning) |
| Min decode | 14.7 tok/s (Decode 256) | 19.1 tok/s (Long-form) |
| Median TTFT | 325 ms | 329 ms (effectively the same) |

If your numbers are within ±10 %, the stack is healthy. If you're substantially below, something specific to your environment is in play — see the [AGENTS.md diagnostics ladder](../AGENTS.md#diagnostics--confirming-the-stack-is-healthy).

## Heads-up on `bench_aeon_thinking.py`

When thinking is enabled, the model spends a substantial fraction of its output budget reasoning before the final answer block begins. With our default `max_tokens` budgets (200-600 depending on prompt), **most prompts are truncated mid-`<think>`** — they never reach the final answer. The output of `bench_aeon_thinking.py` will report `(TRUNCATED IN <think>)` for these cases. The decode rate is still meaningful (it's the speed of generating reasoning tokens, which matches the speed of generating final-answer tokens within rounding), but if you want to *see* the answer, bump `max_tokens` substantially or pass `chat_template_kwargs.enable_thinking=false` per-request.
