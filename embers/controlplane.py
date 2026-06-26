"""Phase B — control plane / data plane split (multi-node).

Today's Autoscaler+Scheduler+Router are one *node's* brain (one GPU box). To span
many boxes we add two layers:

  * NodeAgent  — the data plane for ONE node: thin wrapper over that box's
    Autoscaler (+ its Scheduler/Router/launcher). Exposes capacity + the
    request/lifecycle interface the control plane drives.
  * ControlPlane — the brain across nodes: assigns each model to a node (global
    placement), then routes requests to the owning node. It exposes the SAME
    autoscaler-facing interface the gateway already uses
    (begin_request/end_request/served_models), so the gateway doesn't change —
    it just talks to a ControlPlane instead of a single Autoscaler.

Single node = a ControlPlane with one NodeAgent (today's behaviour, unchanged).
This is the seam that later lets node agents live on remote boxes behind an RPC.
"""
from __future__ import annotations

import time
from collections.abc import Callable

from embers.scheduler import NoCapacity


class NodeAgent:
    """The data plane for one GPU box — wraps that node's Autoscaler."""

    def __init__(self, node_id: str, autoscaler, launcher=None):
        self.node_id = node_id
        self.autoscaler = autoscaler
        self.launcher = launcher        # GpuLauncher (real) — for snapshot counters

    @property
    def scheduler(self):
        return self.autoscaler.scheduler

    def capacity_mb(self) -> int:
        return sum(g.total_mb for g in self.scheduler.gpus)

    def fits_statically(self, vram_mb: int, n_gpus: int) -> bool:
        """Could this node EVER host one unit (n_gpus distinct cards each ≥ vram_mb),
        ignoring current load? A hard physical-feasibility check."""
        return sum(1 for g in self.scheduler.gpus if g.total_mb >= vram_mb) >= n_gpus

    # --- lifecycle / request interface the control plane drives -----------

    def register(self, name: str, vram_mb: int, **kw) -> None:
        # teach this node's launcher the model's adapters (control plane sends
        # them at register time — for remote nodes the launcher was built empty).
        if self.launcher is not None and kw.get("adapters"):
            self.launcher._adapters_by_model[name] = dict(kw["adapters"])
        self.autoscaler.register_model(name, vram_mb, **kw)

    def begin_request(self, model: str):
        return self.autoscaler.begin_request(model)

    def end_request(self, model: str) -> None:
        self.autoscaler.end_request(model)

    def inflight(self, model: str) -> int:
        return self.autoscaler.inflight(model)

    def served_models(self) -> list[str]:
        return self.autoscaler.served_models()

    def tick(self) -> None:
        self.autoscaler.tick()

    def status(self) -> dict:
        return {"node": self.node_id,
                "capacity_mb": self.capacity_mb(),
                "free_mb": self.scheduler.total_free_mb(),
                "replicas": self.autoscaler.state()}

    def snapshot(self) -> dict:
        """This node's full organ view (GPUs + scaling + cold-start counters),
        tagged with the node id. Remote nodes return the same shape over RPC."""
        from embers.metrics import platform_snapshot
        snap = platform_snapshot(self.scheduler, self.autoscaler, self.launcher)
        snap["node"] = self.node_id
        return snap

    def healthy(self) -> bool:
        return True             # in-process node is up as long as we are


class ControlPlane:
    """Global brain: assigns models to nodes and routes requests to the owner.
    Presents the autoscaler-facing interface the gateway expects."""

    def __init__(self, nodes: list[NodeAgent], *,
                 clock: Callable[[], float] | None = None, store=None,
                 overcommit: bool = False):
        if not nodes:
            raise ValueError("control plane needs at least one node")
        self.nodes = {n.node_id: n for n in nodes}
        # over-commit: assign more model footprint to a node than fits at once;
        # the node's autoscaler evicts idle models on demand (density play).
        self.overcommit = overcommit
        self._owner: dict[str, str] = {}        # model / adapter name -> node_id
        self._committed: dict[str, int] = {}    # node_id -> anticipated VRAM footprint
        self._clock = clock or time.monotonic
        self.store = store                      # durable registry (ControlPlaneStore)
        # heartbeat state: node_id -> {healthy, last_ok, last_seen, fails}
        self.health: dict[str, dict] = {
            nid: {"healthy": True, "last_ok": None, "fails": 0} for nid in self.nodes}

    # --- global placement (which node owns a model) -----------------------

    def assign(self, name: str, vram_mb: int, *, tensor_parallel_size: int = 1,
               adapters: dict[str, str] | None = None, **kw) -> str:
        """Place a model on the node with the most headroom that can host it.
        Returns the chosen node_id. Raises NoCapacity if none fits.

        Headroom = node capacity − already-committed footprint (anticipated), so
        successive models SPREAD across nodes instead of piling on one (a model
        at 0 replicas consumes no live VRAM, so we reserve against intent)."""
        need = vram_mb * tensor_parallel_size

        def head(n):
            return n.capacity_mb() - self._committed.get(n.node_id, 0)

        # candidates: nodes that can EVER host one unit (physical feasibility)
        feasible = [n for n in self.nodes.values()
                    if n.fits_statically(vram_mb, tensor_parallel_size)]
        if not feasible:
            raise NoCapacity(
                f"no node can host {name} ({tensor_parallel_size}x {vram_mb}MB)")
        if self.overcommit:
            best = min(feasible, key=lambda n: self._committed.get(n.node_id, 0))
        else:                                   # footprints must fit simultaneously
            withroom = [n for n in feasible if head(n) >= need]
            if not withroom:
                raise NoCapacity(
                    f"no node has room for {name} ({need}MB); set overcommit: true "
                    f"to pack more than fits (evicts idle models on demand)")
            best = max(withroom, key=head)
        best.register(name, vram_mb, tensor_parallel_size=tensor_parallel_size,
                      adapters=adapters, **kw)
        self._committed[best.node_id] = self._committed.get(best.node_id, 0) + need
        self._owner[name] = best.node_id
        for adapter in (adapters or {}):        # adapters live with their base
            self._owner[adapter] = best.node_id
        if self.store is not None:              # persist the placement (durable)
            spec = {"vram_mb": vram_mb, "tensor_parallel_size": tensor_parallel_size,
                    "adapters": adapters or {}, **kw}
            self.store.save_assignment(name, best.node_id, spec)
        return best.node_id

    def restore(self) -> int:
        """Re-apply persisted assignments after a control-plane restart: rebuild
        the owner map + committed footprint, and re-register each model on its
        node only if that node lost it (e.g. the node restarted too). Returns the
        number of assignments restored. Idempotent."""
        if self.store is None:
            return 0
        n = 0
        for row in self.store.load_assignments():
            model, node_id, spec = row["model"], row["node_id"], row["spec"]
            node = self.nodes.get(node_id)
            if node is None:                    # the assigned node is gone
                continue
            tp = spec.get("tensor_parallel_size", 1)
            self._owner[model] = node_id
            self._committed[node_id] = self._committed.get(node_id, 0) + spec["vram_mb"] * tp
            for adapter in (spec.get("adapters") or {}):
                self._owner[adapter] = node_id
            if model not in node.served_models():   # node lost it → re-register
                rest = {k: v for k, v in spec.items() if k != "vram_mb"}
                node.register(model, spec["vram_mb"], **rest)
            n += 1
        return n

    def node_of(self, model: str) -> NodeAgent:
        nid = self._owner.get(model)
        if nid is None:
            raise KeyError(model)
        return self.nodes[nid]

    # --- autoscaler-facing interface (the gateway calls these) ------------

    def begin_request(self, model: str):
        return self.node_of(model).begin_request(model)

    def end_request(self, model: str) -> None:
        self.node_of(model).end_request(model)

    def inflight(self, model: str) -> int:
        return self.node_of(model).inflight(model)

    def served_models(self) -> list[str]:
        out: list[str] = []
        for n in self.nodes.values():
            out.extend(n.served_models())
        return out

    def tick(self) -> None:
        for n in self.nodes.values():
            try:
                n.tick()
            except Exception as e:      # noqa: BLE001 — one dead node mustn't
                print(f"[controlplane] node {n.node_id} tick failed: {e}",
                      flush=True)        # stall the others

    def check_health(self) -> dict[str, dict]:
        """Probe every node's liveness (the heartbeat). Records healthy/last_ok/
        consecutive-fails per node; called periodically by the control loop."""
        now = self._clock()
        for nid, node in self.nodes.items():
            h = self.health[nid]
            try:
                ok = node.healthy()
            except Exception:           # noqa: BLE001 — any error = unhealthy
                ok = False
            h["healthy"] = ok
            if ok:
                h["last_ok"] = now
                h["fails"] = 0
            else:
                h["fails"] += 1
        return self.health

    def healthy_nodes(self) -> list[str]:
        return [nid for nid, h in self.health.items() if h["healthy"]]

    def status(self) -> dict:
        return {"nodes": [n.status() for n in self.nodes.values()],
                "owners": dict(self._owner),
                "health": self.health}

