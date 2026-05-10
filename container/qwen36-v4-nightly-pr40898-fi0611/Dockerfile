FROM vllm/vllm-openai:nightly

WORKDIR /opt/qwen36-aeon-dflash-spark-vllm

COPY docker ./docker
COPY scripts ./scripts
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
 && python3 scripts/apply_pr40898_dflash_swa.py

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
      org.opencontainers.image.description="Qwen3.6 AEON Ultimate DFlash image for DGX Spark. Based on official vLLM nightly with vLLM PR #40898 DFlash sliding-window-attention overlay and FlashInfer 0.6.11 cu130 packages." \
      org.opencontainers.image.source="https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash" \
      org.opencontainers.image.version="qwen36-v4-nightly-pr40898-fi0611"
