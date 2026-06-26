"""`embers up` fronting REMOTE node servers — a node in the config given by `url`
becomes a RemoteNodeAgent; the control plane registers models on it and routes
over RPC. Wired in-process via TestClient transports (no real network)."""

import httpx
import pytest
from fastapi.testclient import TestClient

from embers.noderpc import RemoteNodeAgent, create_node_app
from embers.platform import (
    ModelConfig, NodeConfig, Platform, PlatformConfig, build_node,
)
from embers.scheduler import GPU


def _node_server(node_id, gpus=1, gpu_mb=24000):
    """A running node server (in-process) for `node_id`, returned as a TestClient."""
    cfg = PlatformConfig(models=[], gpus=[GPU(f"{node_id}-g{i}", gpu_mb)
                                         for i in range(gpus)])
    node = build_node(cfg, node_id=node_id, mock=True)
    return TestClient(create_node_app(node, advertise_host=f"10.0.0.{gpus}"))


def test_url_node_becomes_a_remote_agent():
    server = _node_server("box-r")
    p = Platform(
        PlatformConfig(models=[ModelConfig("m", 6000)],
                       nodes=[NodeConfig("box-r", url="http://box-r")]),
        mock=True, remote_client=lambda url: server)
    assert len(p.nodes) == 1
    assert isinstance(p.nodes[0], RemoteNodeAgent)
    assert p.nodes[0].node_id == "box-r"
    assert p._bundles == []                       # no local organs for a remote node


def test_model_registered_and_listed_on_remote_node():
    server = _node_server("box-r")
    p = Platform(
        PlatformConfig(models=[ModelConfig("m", 6000)],
                       nodes=[NodeConfig("box-r", url="http://box-r")]),
        mock=True, remote_client=lambda url: server)
    c = TestClient(p.app)
    # the gateway → control plane → remote node (RPC) lists the model
    assert [m["id"] for m in c.get("/v1/models").json()["data"]] == ["m"]
    assert p.controlplane._owner["m"] == "box-r"


def test_snapshot_aggregates_remote_node_over_rpc():
    server = _node_server("box-r", gpus=2)
    p = Platform(
        PlatformConfig(models=[ModelConfig("m", 6000)],
                       nodes=[NodeConfig("box-r", url="http://box-r")]),
        mock=True, remote_client=lambda url: server)
    snap = p.snapshot()                            # pulls the remote node via RPC
    assert {g["id"] for g in snap["gpus"]} == {"box-r-g0", "box-r-g1"}


def test_mixed_local_and_remote_fleet_spreads_models():
    server = _node_server("remote")
    p = Platform(
        PlatformConfig(
            models=[ModelConfig("a", 6000), ModelConfig("b", 6000)],
            nodes=[NodeConfig("local", gpus=[GPU("lg0", 24000)]),
                   NodeConfig("remote", url="http://remote")]),
        mock=True, remote_client=lambda url: server)
    assert {p.controlplane._owner["a"], p.controlplane._owner["b"]} == {"local", "remote"}
    # the local node has organs; the remote one doesn't
    assert len(p._bundles) == 1


def test_adapter_registers_on_remote_node():
    server = _node_server("box-r")
    p = Platform(
        PlatformConfig(
            models=[ModelConfig("base", 6000, adapters={"sql": "/a/sql"})],
            nodes=[NodeConfig("box-r", url="http://box-r")]),
        mock=True, remote_client=lambda url: server)
    c = TestClient(p.app)
    assert sorted(m["id"] for m in c.get("/v1/models").json()["data"]) == ["base", "sql"]
    assert p.controlplane._owner["sql"] == "box-r"


# --- node-down resilience -------------------------------------------------

class _DeadClient:
    """An http client whose calls always fail — simulates an unreachable node."""
    def get(self, *a, **k):
        raise httpx.ConnectError("connection refused")

    def post(self, *a, **k):
        raise httpx.ConnectError("connection refused")


def test_unreachable_node_request_raises_no_ready_backend():
    # build a remote agent against a live server (so /info works), then kill the link
    from embers.gateway import NoReadyBackend
    server = _node_server("box-r")
    agent = RemoteNodeAgent("http://box-r", http=server)
    agent.register("m", 6000)
    agent._http = _DeadClient()                    # node goes down
    with pytest.raises(NoReadyBackend):
        agent.begin_request("m")                   # → gateway turns this into a 503


def test_dead_node_snapshot_is_marked_unreachable():
    server = _node_server("box-r")
    agent = RemoteNodeAgent("http://box-r", http=server)
    agent._http = _DeadClient()
    snap = agent.snapshot()
    assert snap["node"] == "box-r" and "unreachable" in snap
