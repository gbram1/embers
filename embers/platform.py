"""The runnable platform — assembles every organ from a config file and serves.

`embers up --config platform.yaml` builds Scheduler + GpuLauncher + Autoscaler
+ gateway, registers the declared models, starts a background control loop
(tick → scale-to-zero / reap on an interval), and serves the OpenAI-compatible
API with /metrics and /stats. This is the entry point that turns the library
into a system a user actually runs.

Config (YAML):

    host: 0.0.0.0
    port: 8080
    api_keys: []          # bearer keys; empty/omitted = open (dev)
    tick_interval: 15     # control-loop period, seconds
    gpus: auto            # or a list of {id, vram_mb}
    models:
      - name: Qwen/Qwen2.5-3B
        vram_mb: 6000
        min_replicas: 0
        max_replicas: 3
        idle_ttl: 300
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import yaml

from embers.autoscaler import Autoscaler
from embers.controlplane import ControlPlane, NodeAgent
from embers.gateway import LocalBackend, Router, create_gateway_app
from embers.gpu_backend import GpuLauncher
from embers.metrics import Registry, platform_snapshot
from embers.scheduler import GPU, Scheduler
from embers.server import ModelUnit


@dataclass
class ModelConfig:
    name: str
    vram_mb: int                       # per-GPU footprint (one tensor-parallel shard)
    min_replicas: int = 0
    max_replicas: int = 3
    idle_ttl: float = 300.0
    tensor_parallel_size: int = 1      # GPUs to shard ONE unit across (>1 = TP)
    adapters: dict[str, str] = field(default_factory=dict)  # LoRA name -> path
    gpu_memory_utilization: float | None = None  # fraction of the GPU to grab (packing)
    max_model_len: int | None = None   # cap context (smaller → less KV-cache memory)


@dataclass
class NodeConfig:
    """One node in the fleet. Either LOCAL (in-process, give `gpus`) or REMOTE
    (give `url` — a NodeServer reachable over the network, run via `embers node`).
    Multiple nodes → the control plane spreads models across them."""
    id: str
    gpus: list[GPU] | None = None       # local node: its GPUs
    url: str | None = None              # remote node: its control URL


@dataclass
class PlatformConfig:
    models: list[ModelConfig]
    gpus: list[GPU] | str = "auto"      # single-box: "auto" → detect via nvidia-smi
    nodes: list[NodeConfig] | None = None  # multi-node: explicit boxes (overrides gpus)
    host: str = "0.0.0.0"
    port: int = 8080
    api_keys: list[str] = field(default_factory=list)
    tick_interval: float = 15.0
    # serving units bind 127.0.0.1:serve_port_base+N — keep it HIGH: RunPod and
    # other hosts proxy low localhost ports (8000/8001/8888).
    serve_port_base: int = 19000
    state_db: str | None = None         # path → durable control-plane registry (SQLite)
    overcommit: bool = False            # pack more models than fit; evict idle on demand
    eviction_policy: str = "lru"        # demand-eviction victim choice: lru | cost_aware


def detect_gpus(query: Callable[[], str] | None = None) -> list[GPU]:
    """Enumerate local GPUs via nvidia-smi (id + total VRAM). Injectable."""
    if query is None:
        import subprocess

        def query() -> str:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True)
            return out.stdout
    gpus = []
    for line in query().strip().splitlines():
        idx, mem = (p.strip() for p in line.split(","))
        gpus.append(GPU(f"gpu{idx}", int(mem)))
    return gpus


def _resolve_gpus(gpus_cfg, mock: bool) -> list[GPU]:
    """Resolve the `gpus` config to a GPU list. An explicit list passes through.
    "auto" detects via nvidia-smi — EXCEPT in mock mode (no real GPU, so use a
    fake one) or when nvidia-smi is missing (a clear error, not a raw traceback)."""
    if gpus_cfg != "auto":
        return list(gpus_cfg)
    if mock:                                    # mock never touches a real GPU
        return [GPU("gpu0", 80000)]
    try:
        return detect_gpus()
    except Exception as e:                       # noqa: BLE001 — any detect failure
        raise RuntimeError(
            "gpus: auto needs nvidia-smi to detect GPUs, but it failed/was not "
            "found. Run with --mock (no GPU needed), or set gpus explicitly in "
            "the config, e.g. gpus: [{id: gpu0, vram_mb: 24000}].") from e


def load_config(path: str) -> PlatformConfig:
    raw = yaml.safe_load(open(path)) or {}
    models = [ModelConfig(**m) for m in raw.get("models", [])]
    gpus_raw = raw.get("gpus", "auto")
    gpus = gpus_raw if gpus_raw == "auto" else [
        GPU(g["id"], g["vram_mb"]) for g in gpus_raw]
    nodes = None
    if raw.get("nodes"):                 # multi-node: each {id, gpus:[...]} or {id, url}
        nodes = []
        for n in raw["nodes"]:
            gpus_n = ([GPU(g["id"], g["vram_mb"]) for g in n["gpus"]]
                      if n.get("gpus") else None)
            nodes.append(NodeConfig(n["id"], gpus=gpus_n, url=n.get("url")))
    return PlatformConfig(
        models=models, gpus=gpus, nodes=nodes, host=raw.get("host", "0.0.0.0"),
        port=raw.get("port", 8080), api_keys=raw.get("api_keys", []) or [],
        tick_interval=raw.get("tick_interval", 15.0),
        serve_port_base=raw.get("serve_port_base", 19000),
        state_db=raw.get("state_db"), overcommit=raw.get("overcommit", False),
        eviction_policy=raw.get("eviction_policy", "lru"))


from embers.starter import STARTER_CONFIG     # re-exported for back-compat


def _engine_by_model(models) -> dict[str, dict]:
    """Per-model serving-engine knobs (gpu_memory_utilization, max_model_len),
    omitting unset ones — passed to each serving unit so models can be packed."""
    out: dict[str, dict] = {}
    for m in models:
        eng = {}
        if m.gpu_memory_utilization is not None:
            eng["gpu_memory_utilization"] = m.gpu_memory_utilization
        if m.max_model_len is not None:
            eng["max_model_len"] = m.max_model_len
        if eng:
            out[m.name] = eng
    return out


def build_node(config: PlatformConfig, *, node_id: str = "node0",
               mock: bool = False, serve_host: str = "127.0.0.1",
               clock: Callable[[], float] = time.monotonic):
    """Build a single LOCAL NodeAgent for THIS box (its GPUs + launcher). The
    data plane for one node — driven by a (possibly remote) control plane.
    `serve_host` is where serving units bind (0.0.0.0 to be reachable off-box)."""
    gpus = _resolve_gpus(config.gpus, mock)
    if not gpus:
        raise RuntimeError("no GPUs configured/detected")
    sched = Scheduler(gpus)
    router = Router()
    adapters = {m.name: dict(m.adapters) for m in config.models}
    launcher = None if mock else GpuLauncher(
        port_base=config.serve_port_base, serve_host=serve_host,
        adapters_by_model=adapters, engine_by_model=_engine_by_model(config.models))
    if mock:
        def launch(model, gpu_ids):
            u = ModelUnit(model, mock=True, adapters=adapters.get(model) or None)
            u.load()
            return LocalBackend(u)
        on_deactivate = on_evict = None
    else:
        launch, on_deactivate = launcher.launch, launcher.deactivate
        on_evict = launcher.discard            # demand eviction stops (no park)
    executor = None if mock else ThreadPoolExecutor(
        max_workers=4, thread_name_prefix=f"embers-{node_id}")
    auto = Autoscaler(sched, router, launch, clock=clock,
                      on_deactivate=on_deactivate, on_evict=on_evict,
                      eviction_policy=config.eviction_policy, executor=executor)
    return NodeAgent(node_id, auto, launcher=launcher)


def run_node(config: PlatformConfig, *, node_id: str, host: str, port: int,
             advertise_host: str, serve_host: str = "127.0.0.1",
             mock: bool = False) -> None:
    """`embers node` — serve one box's control API. The control plane drives it
    over RPC (register/begin/tick); serving units bind `serve_host` on this box
    and are reachable at `advertise_host` (use 0.0.0.0 + the LAN IP for off-box)."""
    import uvicorn

    from embers.noderpc import create_node_app
    node = build_node(config, node_id=node_id, mock=mock, serve_host=serve_host)
    app = create_node_app(node, advertise_host=advertise_host)
    print(f"[node {node_id}] control API on http://{host}:{port} "
          f"(advertise serving at {advertise_host}); driven by a control plane")
    uvicorn.run(app, host=host, port=port, log_level="info")


class Platform:
    """Assembled, runnable platform. Holds the wired organs + the gateway app,
    and runs a background control loop driving scale-to-zero and reaping."""

    def __init__(self, config: PlatformConfig, *, mock: bool = False,
                 clock: Callable[[], float] = time.monotonic,
                 remote_client=None):
        self.config = config
        self.mock = mock
        self.registry = Registry()
        # LoRA adapters per base model — the launch hooks load them on the unit.
        self._adapters = {m.name: dict(m.adapters) for m in config.models}
        # how to reach a remote node's control API (injectable for tests)
        self._remote_client = remote_client or self._default_remote_client

        # Build a node agent per node: LOCAL (in-process organs) or REMOTE (an
        # RPC client to a `embers node` server). Single-box (no `nodes:`) → one
        # local node. `_bundles` holds only the LOCAL nodes' organs (for shutdown).
        self._bundles: list[dict] = []
        self.nodes: list[NodeAgent] = []
        local_i = 0
        for nc in self._resolve_nodes(config):
            if nc.url:                          # REMOTE node — connect over RPC
                from embers.noderpc import RemoteNodeAgent
                self.nodes.append(RemoteNodeAgent(nc.url, http=self._remote_client(nc.url)))
                continue
            sched = Scheduler(list(nc.gpus))
            router = Router()
            # offset each local node's serve ports so co-located nodes don't collide
            launcher = None if mock else GpuLauncher(
                port_base=config.serve_port_base + local_i * 1000,
                adapters_by_model=self._adapters,
                engine_by_model=_engine_by_model(config.models))
            launch, on_deactivate, on_evict = self._launch_hooks(launcher)
            executor = None if mock else ThreadPoolExecutor(
                max_workers=4, thread_name_prefix=f"embers-{nc.id}")
            auto = Autoscaler(sched, router, launch, clock=clock,
                              on_deactivate=on_deactivate, on_evict=on_evict,
                              eviction_policy=config.eviction_policy,
                              executor=executor)
            self.nodes.append(NodeAgent(nc.id, auto, launcher=launcher))
            self._bundles.append({"scheduler": sched, "autoscaler": auto,
                                  "launcher": launcher})
            local_i += 1

        # the control plane spans all nodes: global placement + request routing.
        # A `state_db` makes placement durable: on restart we re-apply persisted
        # assignments, then place any NEW config models.
        store = None
        if config.state_db:
            from embers.store import ControlPlaneStore
            store = ControlPlaneStore(config.state_db)
        self.controlplane = ControlPlane(self.nodes, clock=clock, store=store,
                                         overcommit=config.overcommit)
        restored = self.controlplane.restore()
        if restored:
            print(f"[platform] restored {restored} placement(s) from {config.state_db}",
                  flush=True)
        for m in config.models:
            if m.name in self.controlplane._owner:   # already placed (restored)
                continue
            self.controlplane.assign(
                m.name, m.vram_mb, min_replicas=m.min_replicas,
                max_replicas=m.max_replicas, idle_ttl=m.idle_ttl,
                tensor_parallel_size=m.tensor_parallel_size,
                adapters=m.adapters)

        # single-node aliases (back-compat: most code/tests use one local box)
        if self._bundles:
            self.scheduler = self._bundles[0]["scheduler"]
            self.autoscaler = self._bundles[0]["autoscaler"]
            self.launcher = self._bundles[0]["launcher"]
        else:
            self.scheduler = self.autoscaler = self.launcher = None

        self.app = create_gateway_app(
            Router(), api_keys=set(config.api_keys) or None,
            registry=self.registry, autoscaler=self.controlplane,
            snapshot_fn=self.snapshot)
        self._stop = threading.Event()
        self._loop_thread: threading.Thread | None = None

    def _resolve_nodes(self, config: PlatformConfig) -> list[NodeConfig]:
        """The node list — explicit `nodes:` if given, else one local node from `gpus`."""
        if config.nodes:
            return config.nodes
        gpus = _resolve_gpus(config.gpus, self.mock)
        if not gpus:
            raise RuntimeError("no GPUs configured/detected")
        return [NodeConfig("node0", gpus=gpus)]

    @staticmethod
    def _default_remote_client(url: str):
        import httpx
        # generous: a /begin RPC blocks while the node cold-starts (~100s)
        return httpx.Client(base_url=url.rstrip("/"), timeout=600.0)

    def _launch_hooks(self, launcher):
        if self.mock:                       # in-process mock units, no GPU
            def launch(model, gpu_ids):     # gpu_ids: the unit's GPU group
                u = ModelUnit(model, mock=True,
                              adapters=self._adapters.get(model) or None)
                u.load()
                return LocalBackend(u)
            return launch, None, None
        return launcher.launch, launcher.deactivate, launcher.discard

    def snapshot(self) -> dict:
        """Aggregate every node's view (local organs or remote RPC) into one
        cluster snapshot — uniform across local and remote nodes."""
        snap: dict = {"gpus": [], "replicas": {},
                      "scaling": {"cold_starts": 0, "scale_ups": 0,
                                  "scale_downs": 0, "scaled_to_zero": 0}}
        embers = {"cold_loads": 0, "restores": 0, "invalidations": 0,
                  "unpark_failures": 0}
        has_loader = False
        for node in self.nodes:
            s = node.snapshot()             # local: from organs; remote: over RPC
            snap["gpus"] += s.get("gpus", [])
            snap["replicas"].update(s.get("replicas", {}))
            for k in snap["scaling"]:
                snap["scaling"][k] += s.get("scaling", {}).get(k, 0)
            if "embers" in s:
                has_loader = True
                for k in embers:
                    embers[k] += s["embers"].get(k, 0)
        tot = sum(g["total_mb"] for g in snap["gpus"])
        used = sum(g["used_mb"] for g in snap["gpus"])
        snap["cluster_util_pct"] = (100 * used // tot) if tot else 0
        if has_loader:
            t = embers["restores"] + embers["cold_loads"]
            embers["snapshot_hit_rate"] = round(embers["restores"] / t, 3) if t else 0.0
            snap["embers"] = embers
        if len(self.nodes) > 1:
            snap["nodes"] = self.controlplane._owner
            snap["health"] = self.controlplane.health
        return snap

    def tick(self) -> None:
        self.controlplane.tick()

    def start_control_loop(self) -> None:
        def loop():
            while not self._stop.wait(self.config.tick_interval):
                try:
                    self.controlplane.check_health()   # heartbeat (proactive)
                    self.controlplane.tick()           # scale-to-zero + reap
                except Exception as e:      # noqa: BLE001 — never kill the loop
                    print(f"[platform] control-loop tick error: {e}", flush=True)
        self._loop_thread = threading.Thread(target=loop, daemon=True)
        self._loop_thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)
        for b in self._bundles:                 # tear down every node
            b["autoscaler"].shutdown()
            if b["launcher"] is not None:
                b["launcher"].shutdown()

    def serve(self) -> None:
        import uvicorn

        self.start_control_loop()
        n_local_gpu = sum(len(b["scheduler"].gpus) for b in self._bundles)
        n_node = len(self.nodes)
        n_remote = n_node - len(self._bundles)
        models = ", ".join(m.name for m in self.config.models)
        print(f"[platform] {len(self.config.models)} model(s) [{models}] across "
              f"{n_node} node(s) ({len(self._bundles)} local / {n_remote} remote, "
              f"{n_local_gpu} local GPU(s)) on "
              f"http://{self.config.host}:{self.config.port}")
        print("[platform] endpoints: /v1/completions /v1/chat/completions "
              "/v1/models /metrics /stats")
        try:
            uvicorn.run(self.app, host=self.config.host, port=self.config.port,
                        log_level="info")
        finally:
            self.shutdown()
