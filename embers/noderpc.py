"""Remote node agents — the RPC seam that lets node agents run on separate boxes.

A NodeServer (FastAPI) runs on each GPU box, wrapping that box's local NodeAgent
and exposing its control interface over HTTP. A RemoteNodeAgent is a client that
implements the SAME NodeAgent interface by calling a NodeServer — so the
ControlPlane drives local and remote nodes identically (it can't tell them apart).

Two channels:
  * control (this RPC): register / begin / end / inflight / tick / status
  * data: the serving-unit URL the node returns from /begin, which the gateway's
    HttpBackend forwards requests to directly (reachable at the node's advertised
    host). The control plane never proxies tokens — only placement decisions.
"""
from __future__ import annotations

from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException

from embers.gateway import HttpBackend, NoReadyBackend


def _advertise(base_url: str | None, host: str) -> str | None:
    """Rewrite a node-local serving URL (http://127.0.0.1:PORT) to the node's
    externally-reachable host so the gateway on another box can forward to it."""
    if not base_url:
        return None
    p = urlparse(base_url)
    return f"{p.scheme}://{host}:{p.port}"


def create_node_app(node, advertise_host: str = "127.0.0.1") -> FastAPI:
    """The control-plane-facing HTTP app for one node (wraps a local NodeAgent)."""
    app = FastAPI(title=f"embers node {node.node_id}")

    @app.get("/health")
    def health():                       # cheap liveness probe for the heartbeat
        return {"ok": True, "node": node.node_id}

    @app.get("/node/info")
    def info():
        return {"node_id": node.node_id,
                "gpus": [g.total_mb for g in node.scheduler.gpus],
                "capacity_mb": node.capacity_mb()}

    @app.get("/node/served")
    def served():
        return {"models": node.served_models()}

    @app.post("/node/register")
    def register(body: dict):
        name = body.pop("name")
        vram = body.pop("vram_mb")
        node.register(name, vram, **body)
        return {"ok": True}

    @app.post("/node/begin")
    def begin(body: dict):
        try:
            backend = node.begin_request(body["model"])
        except KeyError:
            raise HTTPException(404, "unknown model")
        except NoReadyBackend:
            raise HTTPException(503, "no ready backend")
        return {"url": _advertise(getattr(backend, "base_url", None), advertise_host)}

    @app.post("/node/end")
    def end(body: dict):
        node.end_request(body["model"])
        return {"ok": True}

    @app.get("/node/inflight")
    def inflight(model: str):
        return {"inflight": node.inflight(model)}

    @app.post("/node/tick")
    def tick():
        node.tick()
        return {"ok": True}

    @app.get("/node/status")
    def status():
        return node.status()

    @app.get("/node/snapshot")
    def snapshot():
        return node.snapshot()

    return app


class RemoteNodeAgent:
    """Drives a remote NodeServer over HTTP, presenting the NodeAgent interface.
    Pass a custom `http` client (e.g. an ASGI transport in tests, or a real one
    pointed at the node's control URL in production)."""

    def __init__(self, base_url: str, http: httpx.Client | None = None,
                 timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self._http = http or httpx.Client(base_url=self.base_url, timeout=timeout)
        info = self._get("/node/info")
        self.node_id = info["node_id"]
        self._gpus = info["gpus"]            # GPU total_mb inventory (cached)
        self._capacity = info["capacity_mb"]

    def _get(self, path: str, **params):
        r = self._http.get(path, params=params)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict | None = None):
        try:
            r = self._http.post(path, json=body or {})
        except httpx.HTTPError as e:          # node unreachable → treat as down
            raise NoReadyBackend(f"node {self.node_id} unreachable: {e}")
        if r.status_code == 404:
            raise KeyError((body or {}).get("model"))
        if r.status_code == 503:
            raise NoReadyBackend((body or {}).get("model"))
        r.raise_for_status()
        return r.json()

    # --- the NodeAgent interface the ControlPlane drives ------------------

    def capacity_mb(self) -> int:
        return self._capacity

    def fits_statically(self, vram_mb: int, n_gpus: int) -> bool:
        return sum(1 for t in self._gpus if t >= vram_mb) >= n_gpus

    def register(self, name: str, vram_mb: int, **kw) -> None:
        self._post("/node/register", {"name": name, "vram_mb": vram_mb, **kw})

    def begin_request(self, model: str):
        url = self._post("/node/begin", {"model": model})["url"]
        return HttpBackend(model, url)       # gateway forwards data here directly

    def end_request(self, model: str) -> None:
        self._post("/node/end", {"model": model})

    def inflight(self, model: str) -> int:
        return self._get("/node/inflight", model=model)["inflight"]

    def served_models(self) -> list[str]:
        return self._get("/node/served")["models"]

    def tick(self) -> None:
        self._post("/node/tick")

    def status(self) -> dict:
        return self._get("/node/status")

    def snapshot(self) -> dict:
        try:
            return self._get("/node/snapshot")
        except httpx.HTTPError as e:          # node down → minimal snapshot
            return {"node": self.node_id, "gpus": [], "replicas": {},
                    "unreachable": str(e)}

    def healthy(self) -> bool:
        try:
            return self._http.get("/health").status_code == 200
        except httpx.HTTPError:
            return False
