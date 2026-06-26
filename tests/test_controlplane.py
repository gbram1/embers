"""Control plane / data plane split — global placement of models across multiple
node agents, and routing requests to the owning node. Single node = a control
plane with one node agent (today's behaviour)."""

import pytest

from embers.autoscaler import Autoscaler
from embers.controlplane import ControlPlane, NodeAgent
from embers.gateway import Router
from embers.scheduler import GPU, NoCapacity, Scheduler


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


class ReadyBackend:
    def __init__(self, model):
        self.name = model
        self._ready = True

    @property
    def ready(self):
        return self._ready

    def chat(self, *a, **k):
        return f"reply from {self.name}"

    def complete(self, *a, **k):
        return "ok"


def node(node_id, gpus, gpu_mb=24000):
    sched = Scheduler([GPU(f"{node_id}-g{i}", gpu_mb) for i in range(gpus)])
    router = Router()
    launched = []

    def launch(model, gpu_ids):
        launched.append((node_id, model))
        return ReadyBackend(model)

    a = Autoscaler(sched, router, launch, clock=Clock())
    agent = NodeAgent(node_id, a)
    agent._launched = launched          # for assertions
    return agent


# --- global placement spreads models across nodes -------------------------

def test_models_spread_across_nodes_by_headroom():
    cp = ControlPlane([node("n0", gpus=1), node("n1", gpus=1)])
    cp.assign("a", 6000)
    cp.assign("b", 6000)
    # two models, two equal nodes → one each (spread, not piled)
    assert {cp._owner["a"], cp._owner["b"]} == {"n0", "n1"}


def test_bigger_node_gets_the_model():
    cp = ControlPlane([node("small", gpus=1, gpu_mb=10000),
                       node("big", gpus=2, gpu_mb=24000)])
    # 'big' has far more headroom → wins the first placement
    assert cp.assign("m", 8000) == "big"


def test_request_routes_to_the_owning_node():
    n0, n1 = node("n0", gpus=1), node("n1", gpus=1)
    cp = ControlPlane([n0, n1])
    cp.assign("a", 6000)
    cp.assign("b", 6000)
    owner_a = cp._owner["a"]
    backend = cp.begin_request("a")
    cp.end_request("a")
    assert backend.name == "a"
    # the cold-start happened on the owning node only
    launched_on = {nid for nid, m in (n0._launched + n1._launched) if m == "a"}
    assert launched_on == {owner_a}


def test_served_models_aggregates_all_nodes_and_adapters():
    cp = ControlPlane([node("n0", gpus=1), node("n1", gpus=1)])
    cp.assign("base", 6000, adapters={"sql": "/a/sql"})
    cp.assign("other", 6000)
    assert sorted(cp.served_models()) == ["base", "other", "sql"]


def test_adapter_request_routes_to_its_base_node():
    n0, n1 = node("n0", gpus=1), node("n1", gpus=1)
    cp = ControlPlane([n0, n1])
    cp.assign("base", 6000, adapters={"sql": "/a/sql"})
    cp.assign("filler", 6000)             # pushes base/sql onto one node, filler the other
    base_node = cp._owner["base"]
    assert cp._owner["sql"] == base_node  # adapter co-located with its base
    cp.begin_request("sql")               # routes to the base's node
    cp.end_request("sql")


def test_no_node_can_host_raises():
    cp = ControlPlane([node("n0", gpus=1, gpu_mb=8000)])
    with pytest.raises(NoCapacity):
        cp.assign("huge", 20000)          # bigger than any single GPU


def test_tensor_parallel_needs_a_node_with_enough_gpus():
    cp = ControlPlane([node("single", gpus=1, gpu_mb=24000),
                       node("dual", gpus=2, gpu_mb=24000)])
    # tp=2 can only land on the 2-GPU node
    assert cp.assign("big", 8000, tensor_parallel_size=2) == "dual"


def test_committed_headroom_balances_across_equal_nodes():
    # two equal nodes; each assignment reserves headroom, so models alternate
    # rather than piling on whichever was chosen first.
    cp = ControlPlane([node("n0", gpus=4, gpu_mb=24000),
                       node("n1", gpus=4, gpu_mb=24000)])
    for i in range(6):
        cp.assign(f"m{i}", 8000)
    counts = {"n0": 0, "n1": 0}
    for i in range(6):
        counts[cp._owner[f"m{i}"]] += 1
    assert counts == {"n0": 3, "n1": 3}             # evenly balanced


def test_single_node_control_plane_behaves_like_today():
    n0 = node("only", gpus=2)
    cp = ControlPlane([n0])
    cp.assign("a", 6000)
    cp.assign("b", 6000)
    assert cp._owner == {"a": "only", "b": "only"}
    assert sorted(cp.served_models()) == ["a", "b"]
    assert cp.begin_request("a").name == "a"
    cp.end_request("a")
    cp.tick()                              # fans out to the one node, no error
