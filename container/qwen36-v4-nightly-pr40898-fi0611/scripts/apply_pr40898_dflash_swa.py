#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

import vllm


PR40898_SHA = "998deadea74b3df21c6c829f85b62583337a774a"

RUNTIME_FILES = [
    "vllm/config/speculative.py",
    "vllm/model_executor/models/qwen3_dflash.py",
    "vllm/transformers_utils/configs/speculators/algos.py",
    "vllm/v1/core/kv_cache_utils.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/spec_decode/dflash.py",
    "vllm/v1/spec_decode/llm_base_proposer.py",
    "vllm/v1/worker/gpu_model_runner.py",
]


def patch_gdn_rearrange_capture_safety(pkg_root: Path) -> None:
    target = pkg_root / "model_executor/layers/mamba/gdn_linear_attn.py"
    text = target.read_text()
    marker = "# aeon_disable_rearrange_mixed_qkv_compile_for_cudagraph_capture"
    if marker in text:
        print("[aeon] gdn rearrange_mixed_qkv compile patch already present")
        return

    old = "    @torch.compile(fullgraph=True)\n    def rearrange_mixed_qkv(self, mixed_qkv):\n"
    new = (
        f"    {marker}\n"
        "    # TorchInductor autotunes this tiny cat/split helper on first call.\n"
        "    # In vLLM nightly + torch 2.11, the first call can happen inside CUDA\n"
        "    # graph memory profiling, where autotune's cuda synchronize is illegal.\n"
        "    # Keep CUDA graphs for the model and run this reshape helper eagerly.\n"
        "    def rearrange_mixed_qkv(self, mixed_qkv):\n"
    )
    if old not in text:
        raise RuntimeError(f"Could not find rearrange_mixed_qkv compile decorator in {target}")
    target.write_text(text.replace(old, new, 1))
    print("[aeon] disabled torch.compile on gdn_linear_attn.rearrange_mixed_qkv")


def fetch_raw(rel_path: str) -> str:
    url = f"https://raw.githubusercontent.com/vllm-project/vllm/{PR40898_SHA}/{rel_path}"
    req = urllib.request.Request(url, headers={"User-Agent": "AEON-7-build"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read().decode("utf-8")


def clear_python_caches(pkg_root: Path) -> None:
    for pyc in pkg_root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    for pycache in pkg_root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


def main() -> int:
    pkg_root = Path(vllm.__file__).resolve().parent
    site_root = pkg_root.parent

    print(f"[pr40898] vLLM package root: {pkg_root}")
    print(f"[pr40898] overlay commit: {PR40898_SHA}")

    for rel in RUNTIME_FILES:
        dst = site_root / rel
        if not dst.exists():
            raise FileNotFoundError(f"Cannot overlay missing installed file: {dst}")
        text = fetch_raw(rel)
        dst.write_text(text)
        print(f"[pr40898] overlaid {rel}")

    patch_gdn_rearrange_capture_safety(pkg_root)
    clear_python_caches(pkg_root)

    from vllm.model_executor.models.qwen3_dflash import DFlashAttention
    from vllm.v1.spec_decode.dflash import DFlashProposer

    qwen_file = pkg_root / "model_executor/models/qwen3_dflash.py"
    dflash_file = pkg_root / "v1/spec_decode/dflash.py"
    qwen_text = qwen_file.read_text()
    dflash_text = dflash_file.read_text()
    required = [
        ("DFlashAttention", DFlashAttention),
        ("DFlashProposer", DFlashProposer),
        ("sliding_attention_layer_names", qwen_text),
        ("build_for_drafting", dflash_text),
        ("causal_layers", dflash_text),
    ]
    missing = [name for name, haystack in required if name not in str(haystack)]
    if missing:
        raise RuntimeError(f"PR40898 overlay verification failed: missing {missing}")

    print("[pr40898] DFlash SWA overlay verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
