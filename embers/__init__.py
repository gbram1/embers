"""embers — fast model cold-start + scale-to-zero serving for vLLM.

A serverless-GPU serving platform: gateway → autoscaler → loader → scheduler →
serving unit, with NVIDIA cuda-checkpoint snapshot/restore making scale-to-zero
fast (cold start ~minutes → restore ~seconds). Scales across GPUs (data + tensor
parallel), serves many LoRA adapters off one base, and spans multiple nodes.

Two ways to use it:

  * As a service (the common path) — run it and point any OpenAI client at it:
        $ pip install 'embers[serve,gpu]'
        $ embers init && embers up --config platform.yaml
        >>> from openai import OpenAI
        >>> OpenAI(base_url="http://localhost:8080/v1", api_key="x").chat...

  * Embedded in your app (programmatic) — the public API is lazily exported here,
    so `import embers` is cheap and the heavy bits load only when you touch them:
        >>> import embers
        >>> p = embers.Platform(embers.load_config("platform.yaml"))
        >>> p.serve()                      # needs the [serve] extra

The snapshot/restore is done via NVIDIA's cuda-checkpoint (see gpu_backend.py),
not a bundled native extension — embers is pure Python.
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Platform", "PlatformConfig", "ModelConfig", "NodeConfig", "load_config",
    "build_node", "run_node",
    "ModelUnit",
    "ControlPlane", "NodeAgent",
    "ControlPlaneStore",
]

# Lazy attribute access (PEP 562): keep `import embers` dependency-free; pull in
# the serve/GPU machinery only when a public symbol is actually used.
_LAZY = {
    "Platform": "embers.platform",
    "PlatformConfig": "embers.platform",
    "ModelConfig": "embers.platform",
    "NodeConfig": "embers.platform",
    "load_config": "embers.platform",
    "build_node": "embers.platform",
    "run_node": "embers.platform",
    "ModelUnit": "embers.server",
    "ControlPlane": "embers.controlplane",
    "NodeAgent": "embers.controlplane",
    "ControlPlaneStore": "embers.store",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'embers' has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(__all__)
