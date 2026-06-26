# Production embers platform image — runs `embers up` on a real GPU.
# Pinned to a CUDA-12.4-compatible stack;
# bump on a newer-driver host. Needs: GPU access (--gpus all) and, for
# scale-to-zero park/unpark, the CAP_SYS_ADMIN/checkpoint privileges or a
# bare-metal host (RunPod containers free the GPU via cuda-checkpoint only).
# For a no-GPU local test image, use ./Dockerfile.local instead.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv curl && \
    rm -rf /var/lib/apt/lists/*

# Known-good stack for this CUDA base (matches the validated benchmark runs).
RUN pip3 install --no-cache-dir \
        vllm==0.8.5 transformers==4.51.3 \
        fastapi uvicorn pydantic httpx pyyaml

# cuda-checkpoint (snapshot/restore) — driver R550+ required at runtime.
RUN curl -fsSL -o /usr/local/bin/cuda-checkpoint \
        https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint && \
    chmod +x /usr/local/bin/cuda-checkpoint

COPY embers /app/embers
WORKDIR /app
ENV PYTHONPATH=/app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD curl -fsS http://localhost:8080/v1/models || exit 1

# Mount your platform.yaml at /etc/embers/platform.yaml (or bake it in).
# ENTRYPOINT = base command, CMD = default args (k8s `args:` overrides CMD).
ENTRYPOINT ["python3", "-m", "embers.cli"]
CMD ["up", "--config", "/etc/embers/platform.yaml"]
