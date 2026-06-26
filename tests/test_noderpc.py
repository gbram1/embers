"""Remote node agents over RPC — a RemoteNodeAgent driving a NodeServer that
wraps a local NodeAgent, wired in-process via httpx's ASGI transport (no real
network, but the full serialize → HTTP → deserialize path). The ControlPlane
drives a remote node identically to a local one.
"""

import pytest
from fastapi.testclient import TestClient

from embers.autoscaler import Autoscaler
from embers.controlplane import ControlPlane, NodeAgent
from embers.gateway import Router
from embers.noderpc import RemoteNodeAgent, create_node_app
from embers.scheduler import GPU, Scheduler


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


class FakeServingBackend:
    """A serving unit with a URL (no network probe) — stands in for a real
    HttpBackend so /begin has a base_url to advertise."""
    def __init__(self, model, port):
        self.name = model
        self.base_url = f"http://127.0.0.1:{port}"

    @property
    def ready(self):
        return True

    def chat(self, *a, **k):
        return "ok"

    def complete(self, *a, **k):
        return "ok"


def local_node(node_id="boxA", gpus=2, gpu_mb=24000):
    sched = Scheduler([GPU(f"{node_id}-g{i}", gpu_mb) for i in range(gpus)])
    router = Router()
    ports = {"n": 19000}

    def launch(model, gpu_ids):
        ports["n"] += 1
        return FakeServingBackend(model, ports["n"])

    auto = Autoscaler(sched, router, launch, clock=Clock())
    return NodeAgent(node_id, auto)


def remote(node, advertise_host="10.0.0.7"):
    # TestClient is a sync httpx client that runs the ASGI app in-process — the
    # stand-in for a real httpx.Client pointed at the node's control URL.
    app = create_node_app(node, advertise_host=advertise_host)
    return RemoteNodeAgent("http://node", http=TestClient(app))


# --- the RPC contract -----------------------------------------------------

def test_remote_reflects_node_identity_and_capacity():
    r = remote(local_node("boxA", gpus=2, gpu_mb=24000))
    assert r.node_id == "boxA"
    assert r.capacity_mb() == 48000
    assert r.fits_statically(8000, 2) is True
    assert r.fits_statically(8000, 3) is False        # only 2 GPUs


def test_remote_register_and_served_models():
    r = remote(local_node())
    r.register("m", 6000, adapters={"sql": "/a/sql"})
    assert sorted(r.served_models()) == ["m", "sql"]


def test_remote_begin_returns_advertised_serving_url():
    r = remote(local_node(), advertise_host="10.0.0.7")
    r.register("m", 6000)
    backend = r.begin_request("m")
    # the URL was rewritten to the node's externally-reachable host
    assert backend.name == "m"
    assert backend.base_url.startswith("http://10.0.0.7:")
    r.end_request("m")
    assert r.inflight("m") == 0


def test_remote_unknown_model_raises_keyerror():
    r = remote(local_node())
    with pytest.raises(KeyError):
        r.begin_request("ghost")


def test_remote_tick_and_status():
    node = local_node("boxA")
    r = remote(node)
    r.register("m", 6000)
    r.tick()                                          # no error over RPC
    st = r.status()
    assert st["node"] == "boxA" and "replicas" in st


# --- ControlPlane drives a remote node like a local one -------------------

def test_control_plane_over_a_remote_node():
    r = remote(local_node("boxA", gpus=2))
    cp = ControlPlane([r])
    cp.assign("m", 6000, adapters={"sql": "/a/sql"})
    assert cp._owner["m"] == "boxA" and cp._owner["sql"] == "boxA"
    assert sorted(cp.served_models()) == ["m", "sql"]
    backend = cp.begin_request("m")                   # placement + routing over RPC
    assert backend.name == "m" and backend.base_url.startswith("http://")
    cp.end_request("m")


def test_control_plane_spreads_across_local_and_remote():
    # a mixed fleet: one in-process node, one "remote" node — placement spans both
    cp = ControlPlane([local_node("local", gpus=1), remote(local_node("remote", gpus=1))])
    cp.assign("a", 6000)
    cp.assign("b", 6000)
    assert {cp._owner["a"], cp._owner["b"]} == {"local", "remote"}
