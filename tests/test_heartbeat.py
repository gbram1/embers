"""Heartbeat — the control plane proactively probes node liveness, records
healthy/last_ok/consecutive-fails per node, and exposes it in /stats. A dead
node is detected by the heartbeat (not only reactively on a request)."""

import httpx
from fastapi.testclient import TestClient

from embers.autoscaler import Autoscaler
from embers.controlplane import ControlPlane, NodeAgent
from embers.gateway import Router
from embers.noderpc import RemoteNodeAgent, create_node_app
from embers.platform import ModelConfig, NodeConfig, Platform, PlatformConfig, build_node
from embers.scheduler import GPU, Scheduler


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def local_node(node_id="local"):
    sched = Scheduler([GPU(f"{node_id}-g0", 24000)])
    a = Autoscaler(sched, Router(), lambda m, g: None, clock=Clock())
    return NodeAgent(node_id, a)


def remote_node(node_id="remote"):
    cfg = PlatformConfig(models=[], gpus=[GPU(f"{node_id}-g0", 24000)])
    server = TestClient(create_node_app(build_node(cfg, node_id=node_id, mock=True)))
    return RemoteNodeAgent(f"http://{node_id}", http=server), server


class _DeadClient:
    def get(self, *a, **k):
        raise httpx.ConnectError("refused")

    def post(self, *a, **k):
        raise httpx.ConnectError("refused")


# --- health probing -------------------------------------------------------

def test_healthy_fleet_all_up():
    clock = Clock()
    r, _ = remote_node()
    cp = ControlPlane([local_node(), r], clock=clock)
    cp.check_health()
    assert set(cp.healthy_nodes()) == {"local", "remote"}
    assert cp.health["remote"]["last_ok"] == 100.0


def test_dead_remote_node_marked_unhealthy_with_fail_count():
    clock = Clock()
    r, _ = remote_node()
    cp = ControlPlane([local_node(), r], clock=clock)
    cp.check_health()                          # both up
    r._http = _DeadClient()                    # remote dies

    clock.t = 110.0
    cp.check_health()
    assert cp.health["remote"]["healthy"] is False
    assert cp.health["remote"]["fails"] == 1
    assert cp.health["remote"]["last_ok"] == 100.0   # frozen at last success
    assert cp.healthy_nodes() == ["local"]

    clock.t = 120.0
    cp.check_health()
    assert cp.health["remote"]["fails"] == 2          # consecutive fails accumulate


def test_node_recovers_resets_fail_count():
    r, server = remote_node()
    cp = ControlPlane([r])
    r._http = _DeadClient()
    cp.check_health()
    assert cp.health["remote"]["fails"] == 1
    r._http = server                            # link restored
    cp.check_health()
    assert cp.health["remote"]["healthy"] is True
    assert cp.health["remote"]["fails"] == 0


def test_local_node_always_healthy():
    cp = ControlPlane([local_node()])
    cp.check_health()
    assert cp.health["local"]["healthy"] is True


# --- exposed in the platform /stats ---------------------------------------

def test_platform_stats_includes_health():
    server = TestClient(create_node_app(
        build_node(PlatformConfig(models=[], gpus=[GPU("r-g0", 24000)]),
                   node_id="remote", mock=True)))
    p = Platform(
        PlatformConfig(models=[ModelConfig("m", 6000)],
                       nodes=[NodeConfig("local", gpus=[GPU("lg0", 24000)]),
                              NodeConfig("remote", url="http://remote")]),
        mock=True, remote_client=lambda url: server)
    p.controlplane.check_health()
    snap = p.snapshot()
    assert "health" in snap
    assert snap["health"]["local"]["healthy"] is True
    assert snap["health"]["remote"]["healthy"] is True
