FROM vllm/vllm-openai:nightly

WORKDIR /opt/qwen36-aeon-dflash-spark-vllm

COPY docker ./docker
COPY scripts ./scripts
COPY prototypes ./prototypes
COPY tests ./tests
COPY README.md ./

ARG FLASHINFER_VERSION=0.6.11
ARG FLASHINFER_CUDA_SUFFIX=cu130

RUN chmod +x docker/*.sh scripts/*.sh scripts/*.py \
 && python3 -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache \
 && python3 -m pip install --no-cache-dir --no-deps \
      "flashinfer-python==${FLASHINFER_VERSION}" \
      "flashinfer-cubin==${FLASHINFER_VERSION}" \
 && python3 -m pip install --no-cache-dir --no-deps \
      "flashinfer-jit-cache==${FLASHINFER_VERSION}+${FLASHINFER_CUDA_SUFFIX}" \
      --index-url "https://flashinfer.ai/whl/${FLASHINFER_CUDA_SUFFIX}" \
 && python3 scripts/apply_pr40898_dflash_swa.py \
 && python3 scripts/apply_dflash_ddtree_m1.py \
 && python3 scripts/apply_dflash_ddtree_m2.py \
 && python3 scripts/apply_dflash_ddtree_m3.py \
 && python3 scripts/apply_dflash_ddtree_m4b.py \
 && python3 scripts/apply_dflash_ddtree_m5b.py \
 && python3 scripts/apply_dflash_ddtree_m6a.py \
 && python3 scripts/apply_dflash_ddtree_m6b.py \
 && python3 scripts/apply_dflash_ddtree_m6c.py \
 && python3 scripts/apply_dflash_ddtree_m6d.py \
 && python3 scripts/apply_dflash_ddtree_m6e.py \
 && python3 scripts/apply_dflash_ddtree_m6f.py \
 && python3 scripts/apply_dflash_ddtree_m6g.py \
 && python3 scripts/apply_dflash_ddtree_m7a.py \
 && python3 scripts/apply_dflash_ddtree_m8a.py \
 && python3 scripts/apply_dflash_ddtree_m8b.py \
 && python3 scripts/apply_dflash_ddtree_m8c.py \
 && python3 scripts/apply_dflash_ddtree_m8d.py \
 && python3 scripts/apply_dflash_ddtree_m8e.py \
 && python3 scripts/apply_dflash_ddtree_m8f.py \
 && python3 scripts/apply_dflash_ddtree_m8g.py \
 && python3 scripts/apply_dflash_ddtree_m8h.py \
 && python3 scripts/apply_dflash_ddtree_m8i.py \
 && python3 scripts/apply_dflash_ddtree_m8j.py \
 && python3 scripts/apply_dflash_ddtree_m8k.py \
 && python3 scripts/apply_dflash_ddtree_m8l.py \
 && python3 scripts/apply_dflash_ddtree_m8m.py \
 && python3 scripts/apply_dflash_ddtree_m8n.py \
 && python3 scripts/apply_dflash_ddtree_m8o.py \
 && python3 scripts/apply_dflash_ddtree_m8p.py \
 && python3 scripts/apply_dflash_ddtree_m8q.py \
 && python3 scripts/apply_dflash_ddtree_m8r.py \
 && python3 scripts/apply_dflash_ddtree_m8s.py \
 && python3 scripts/apply_dflash_ddtree_m8t.py \
 && python3 scripts/apply_dflash_ddtree_m8u.py \
 && python3 scripts/apply_dflash_ddtree_m8v.py \
 && python3 scripts/apply_dflash_ddtree_m8w.py \
 && python3 scripts/apply_dflash_ddtree_m8x.py \
 && python3 scripts/apply_dflash_ddtree_m8y.py \
 && python3 scripts/apply_dflash_ddtree_m8z.py \
 && python3 scripts/apply_dflash_ddtree_m9a.py \
 && python3 scripts/apply_dflash_ddtree_m9b.py \
 && python3 scripts/apply_dflash_ddtree_m9c.py \
 && python3 scripts/apply_dflash_ddtree_m9d.py \
 && python3 scripts/apply_dflash_ddtree_m9e.py \
 && python3 scripts/apply_dflash_ddtree_m10a.py \
 && python3 scripts/apply_dflash_ddtree_m10b.py \
 && python3 scripts/apply_dflash_ddtree_m10c.py \
 && python3 scripts/apply_dflash_ddtree_m10d.py \
 && python3 scripts/apply_dflash_ddtree_m10e.py \
 && python3 scripts/apply_dflash_ddtree_m10f.py \
 && python3 scripts/apply_dflash_ddtree_m10g.py \
 && python3 scripts/apply_dflash_ddtree_m10h.py \
 && python3 scripts/apply_dflash_ddtree_m10i.py \
 && python3 scripts/apply_dflash_ddtree_m10j.py \
 && python3 scripts/apply_dflash_ddtree_m10k.py \
 && python3 scripts/apply_dflash_ddtree_m10l.py \
 && python3 scripts/apply_dflash_ddtree_m10m.py \
 && python3 scripts/apply_dflash_ddtree_m10n.py \
 && python3 scripts/apply_dflash_ddtree_m10o.py \
 && python3 scripts/apply_dflash_ddtree_m10p.py \
 && python3 scripts/apply_dflash_ddtree_m10q.py \
 && python3 scripts/apply_dflash_ddtree_m10r.py \
 && python3 scripts/apply_dflash_ddtree_m10s.py \
 && python3 scripts/apply_dflash_ddtree_m10t.py \
 && python3 scripts/apply_dflash_ddtree_m11a.py \
 && python3 scripts/apply_dflash_ddtree_m11b.py \
 && python3 scripts/apply_dflash_ddtree_m11c.py \
 && python3 scripts/apply_dflash_ddtree_m11d.py \
 && python3 scripts/apply_dflash_ddtree_m11g.py \
 && python3 scripts/apply_dflash_ddtree_m11h.py \
 && python3 scripts/apply_dflash_ddtree_m11i.py \
 && python3 scripts/apply_dflash_ddtree_m11j.py \
 && python3 scripts/apply_dflash_ddtree_m11k.py \
 && python3 scripts/apply_dflash_ddtree_m11l.py \
 && python3 scripts/apply_dflash_ddtree_m11m.py \
 && python3 scripts/apply_dflash_ddtree_m11n.py \
 && python3 scripts/apply_dflash_ddtree_m11o.py

RUN python3 - <<'PY'
import py_compile
from pathlib import Path
import vllm

pkg = Path(vllm.__file__).resolve().parent
for rel in (
    "model_executor/layers/mamba/ops/causal_conv1d.py",
    "model_executor/layers/fla/ops/fused_sigmoid_gating.py",
    "model_executor/layers/mamba/gdn_linear_attn.py",
):
    py_compile.compile(str(pkg / rel), doraise=True)
PY

RUN python3 -m py_compile \
      prototypes/ddtree_parent_metadata.py \
      prototypes/ddtree_runtime_sampler.py \
      prototypes/ddtree_gdn_reference.py \
      prototypes/qwen25_tree_decode_loop.py \
      prototypes/qwen25_tree_decode_suite.py \
 && python3 tests/test_ddtree_tree.py \
 && python3 tests/test_ddtree_vllm_metadata.py \
 && python3 tests/test_dflash_ddtree_m3_bridge.py \
 && python3 tests/test_dflash_ddtree_m4b_payload.py \
 && python3 tests/test_ddtree_runtime_sampler.py \
 && python3 tests/test_ddtree_parent_metadata.py \
 && python3 tests/test_ddtree_gdn_reference.py

RUN python3 - <<'PY'
import importlib.metadata as metadata
import torch

for package in ("vllm", "flashinfer-python", "flashinfer-cubin",
                "flashinfer-jit-cache", "torch", "triton"):
    print(f"{package}={metadata.version(package)}")
print(f"torch_cuda={torch.version.cuda}")
PY

ENTRYPOINT ["/opt/qwen36-aeon-dflash-spark-vllm/docker/entrypoint.sh"]
CMD ["dflash"]

LABEL org.opencontainers.image.title="vllm-aeon-ultimate-dflash" \
      org.opencontainers.image.description="Qwen3.6 AEON Ultimate DFlash/DDTree experimental image for DGX Spark. Based on official vLLM nightly with vLLM PR #40898 DFlash sliding-window-attention overlay, FlashInfer 0.6.11 cu130 packages, AEON dflash_ddtree M1/M2/M3 bridge overlays, the M4A Qwen2.5 multi-step correctness oracle, M4B DFlash top-k tree payload generation, the M5A tree-aware GDN state reference, the M5B gated runtime tree sampler hook, M6A parent-id metadata plumbing for Qwen3.6 GDN layers, M6B guarded slow DDTree GDN parent-state replay, M6C FlexAttention ancestor-only tree masking, M6D FlexAttention non-contiguous hybrid KV cache support, M6E power-of-two Flex correctness blocks for hybrid page alignment, M6F uncompiled Flex correctness mode for dynamic DDTree ancestor masks, M6G a guarded FlashAttention eager DDTree verifier path, M7A accepted-branch KV/state compaction, M7B safe DDTree smoke defaults, M8A opt-in Triton DDTree conv/GDN fast path, M8B sequential DFlash DDTree payload generation, M8C flat-chain metadata fallback, M8D opt-in inline attention parent-id fallback, M8E DDTree payload handoff tracing, M8F live parent metadata install from real proposer payloads, M8G cached attention metadata parent refresh, M8H pending proposer payload rehydration, M8I current vLLM SamplerOutput signature compatibility, M8J flat-chain attention bypass plus branch diagnostics, M8K log-prob tree scoring plus speculative-buffer guards, M8L forced-root-branch validation support, M8M tree-depth RoPE/model position override, M8N GDN rolling-state offset replay, M8O slow-reference num_accepted fix, M8P slow-conv/fast-SSM isolation, M8Q root-leaf exact DDTree topology, M8R accepted-branch recurrent-state flattening, M8S non-align GDN block-table recurrent compaction, M8T GDN metadata-backed recurrent compaction, M8V branch-only attention overlay, M8W compacted recurrent-state cursor alignment, M8X recurrent conv compaction for flat accepted paths, M8Y compiled Flex tree-mask restoration, M8Z compiled Flex CUDA graph step markers, M9A GPU-side FlashAttention branch-mask correction, M9B opt-in Triton branch-row attention correction, M9C graph-safe Triton verifier capture path, M9D paged-KV graph-safe Triton verifier support, M9E CUDA-int32 parent metadata normalization, M10A branch bonus safety guard, M10B safe branch-bonus state reporting, M10C opt-in recurrent compaction default, M10D same-compact compaction guard, M10E branch-state compaction research switch, M10F slow-path pre-root GDN state snapshots, M10G slow GDN decode-window guard, M10H GDN metadata diagnostics, M10I parent-present DDTree diagnostics, M10J branch compaction without uncomputed bonus emission, M10K recurrent/KV destination alignment, M10L branch-compaction Mamba state-count override, M10M non-greedy flat-sampler guard, M10N branch SSM compaction into the base running-state block, M10O next-step branch state cursor synchronization, M10P branch conv compaction mirrored into root/cursor offsets, M10Q non-flat branch accepted+bonus scheduler contract restoration, M10R model-replay root children on pre-tree cached state, M10S DFlash drafter-context hidden-state compaction, M10T vLLM-safe non-flat DDTree walk adaptation, M11A explicit accepted-count plumbing for full branch commit experiments, M11B full-branch state-count bias testing, M11C root-sibling recurrent-state offset testing, M11D research-only guard for unsafe full-branch commit, M11E deployable safe DDTree entrypoint defaults, M11F tightened smoke compile range defaults, M11G full-branch accepted+bonus contract restoration, M11H width-guarded Triton GDN parent replay, M11I full-attention KV compaction for list slot mappings, M11J root-sibling GDN replay from the pre-tree root state, M11K one-step DFlash suppression after non-flat commits for target-vs-drafter isolation, M11L branch GDN state mirroring into vLLM's running mamba_state_idx block, M11M native stochastic row sampling for tree-aware DDTree without forced greedy decoding, M11N internal-branch topology controls that block extra root children, and M11O optional Mamba postprocess freeze after non-flat branch commits." \
      org.opencontainers.image.source="https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash" \
      org.opencontainers.image.version="qwen36-v5-ddtree-experimental-m11o"
