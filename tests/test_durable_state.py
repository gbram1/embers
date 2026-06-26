"""Durable control-plane state — a SQLite registry that survives a restart. The
control plane persists model→node placement on assign and re-applies it on
restart (rebuilding the owner map + re-registering models on their nodes)."""

from fastapi.testclient import TestClient

from embers.autoscaler import Autoscaler
from embers.controlplane import ControlPlane, NodeAgent
from embers.gateway import Router
from embers.platform import ModelConfig, NodeConfig, Platform, PlatformConfig
from embers.scheduler import GPU, Scheduler
from embers.store import ControlPlaneStore


class Clock:
    def __call__(self):
        return 0.0


def local_node(node_id):
    a = Autoscaler(Scheduler([GPU(f"{node_id}-g0", 24000)]), Router(),
                   lambda m, g: None, clock=Clock())
    return NodeAgent(node_id, a)


# --- the store ------------------------------------------------------------

def test_store_persists_assignments(tmp_path):
    db = str(tmp_path / "s.db")
    s = ControlPlaneStore(db)
    s.save_assignment("a", "n0", {"vram_mb": 6000, "adapters": {}})
    s.save_assignment("b", "n1", {"vram_mb": 8000, "tensor_parallel_size": 2})
    s.close()
    # reopen → data survives
    s2 = ControlPlaneStore(db)
    rows = {r["model"]: r for r in s2.load_assignments()}
    assert rows["a"]["node_id"] == "n0"
    assert rows["b"]["spec"]["tensor_parallel_size"] == 2


def test_store_snapshot_index_and_clear(tmp_path):
    s = ControlPlaneStore(str(tmp_path / "s.db"))
    s.set_snapshot("a", "n0")
    s.set_snapshot("a", "n1")            # upsert
    assert s.snapshot_index() == {"a": "n1"}
    s.clear()
    assert s.load_assignments() == [] and s.snapshot_index() == {}


# --- control-plane recovery -----------------------------------------------

def test_assign_persists_then_restore_rebuilds_placement(tmp_path):
    db = str(tmp_path / "s.db")
    cp1 = ControlPlane([local_node("n0"), local_node("n1")],
                       store=ControlPlaneStore(db))
    cp1.assign("a", 6000)
    cp1.assign("b", 6000, adapters={"sql": "/p"})
    owners = dict(cp1._owner)

    # restart: fresh nodes + a new control plane on the same db
    nodes2 = [local_node("n0"), local_node("n1")]
    cp2 = ControlPlane(nodes2, store=ControlPlaneStore(db))
    assert cp2._owner == {}              # starts empty
    n = cp2.restore()
    assert n == 2
    assert cp2._owner == owners          # same placement recovered (incl. adapter)
    # models were re-registered on their (fresh) nodes
    by_id = {x.node_id: x for x in nodes2}
    assert "a" in by_id[owners["a"]].served_models()
    assert "sql" in by_id[owners["sql"]].served_models()


def test_restore_skips_reregister_when_node_kept_the_model(tmp_path):
    db = str(tmp_path / "s.db")
    nodes = [local_node("n0")]
    cp1 = ControlPlane(nodes, store=ControlPlaneStore(db))
    cp1.assign("a", 6000)
    # same nodes still have 'a' → restore rebuilds owner map without double-register
    cp2 = ControlPlane(nodes, store=ControlPlaneStore(db))
    assert cp2.restore() == 1
    assert cp2._owner["a"] == "n0"
    assert nodes[0].served_models() == ["a"]     # not duplicated


# --- platform-level durability --------------------------------------------

def _cfg(db):
    return PlatformConfig(
        models=[ModelConfig("m", 6000)],
        nodes=[NodeConfig("n0", gpus=[GPU("g0", 24000)]),
               NodeConfig("n1", gpus=[GPU("g1", 24000)])],
        state_db=db)


def test_platform_recovers_placement_across_restart(tmp_path):
    db = str(tmp_path / "platform.db")
    p1 = Platform(_cfg(db), mock=True)
    owner = p1.controlplane._owner["m"]

    # "restart" the control plane on the same db
    p2 = Platform(_cfg(db), mock=True)
    assert p2.controlplane._owner["m"] == owner   # recovered, not re-decided
    c = TestClient(p2.app)
    assert [x["id"] for x in c.get("/v1/models").json()["data"]] == ["m"]
    # serving still works after recovery
    r = c.post("/v1/chat/completions",
               json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200


def test_new_model_added_after_restart_is_placed(tmp_path):
    db = str(tmp_path / "p.db")
    Platform(_cfg(db), mock=True)                 # places "m"
    # restart with an extra model in config
    cfg2 = _cfg(db)
    cfg2.models.append(ModelConfig("n", 6000))
    p2 = Platform(cfg2, mock=True)
    assert "m" in p2.controlplane._owner          # recovered
    assert "n" in p2.controlplane._owner          # newly placed
