"""Multi-node Platform — one `embers up` driving several node agents from config.
Models spread across nodes; requests route to the owning node; the snapshot
aggregates every node's GPUs. (In-process nodes; remote node agents come later.)"""

from fastapi.testclient import TestClient

from embers.platform import ModelConfig, NodeConfig, Platform, PlatformConfig, load_config
from embers.scheduler import GPU


def _two_node(models):
    cfg = PlatformConfig(
        models=models,
        nodes=[NodeConfig("n0", [GPU("n0-g0", 24000)]),
               NodeConfig("n1", [GPU("n1-g0", 24000)])])
    return Platform(cfg, mock=True)


def test_two_node_platform_builds_two_nodes():
    p = _two_node([ModelConfig("a", 6000), ModelConfig("b", 6000)])
    assert len(p.nodes) == 2
    assert {n.node_id for n in p.nodes} == {"n0", "n1"}


def test_models_spread_across_nodes():
    p = _two_node([ModelConfig("a", 6000), ModelConfig("b", 6000)])
    # global placement put the two models on different nodes
    assert {p.controlplane._owner["a"], p.controlplane._owner["b"]} == {"n0", "n1"}


def test_requests_route_to_owning_node_and_serve():
    p = _two_node([ModelConfig("a", 6000), ModelConfig("b", 6000)])
    c = TestClient(p.app)
    assert sorted(m["id"] for m in c.get("/v1/models").json()["data"]) == ["a", "b"]
    for model in ("a", "b"):
        r = c.post("/v1/chat/completions",
                   json={"model": model, "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
        assert r.json()["usage"]["total_tokens"] > 0
    # each model cold-started on its OWN node only
    a_node = p.controlplane._owner["a"]
    b_node = p.controlplane._owner["b"]
    by_id = {n.node_id: n for n in p.nodes}
    assert by_id[a_node].autoscaler.state().get("a") == 1
    assert by_id[b_node].autoscaler.state().get("b") == 1


def test_snapshot_aggregates_all_nodes_gpus():
    p = _two_node([ModelConfig("a", 6000), ModelConfig("b", 6000)])
    c = TestClient(p.app)
    c.post("/v1/chat/completions",
           json={"model": "a", "messages": [{"role": "user", "content": "hi"}]})
    snap = p.snapshot()
    # both nodes' GPUs show up in the unified view
    assert {g["id"] for g in snap["gpus"]} == {"n0-g0", "n1-g0"}
    assert snap["replicas"].get("a") == 1
    assert "nodes" in snap                        # owner map exposed for multi-node


def test_adapters_served_on_their_base_node():
    p = _two_node([ModelConfig("base", 6000, adapters={"sql": "/a/sql"}),
                   ModelConfig("other", 6000)])
    c = TestClient(p.app)
    assert sorted(m["id"] for m in c.get("/v1/models").json()["data"]) == \
        ["base", "other", "sql"]
    r = c.post("/v1/chat/completions",
               json={"model": "sql", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert "via sql" in r.json()["choices"][0]["message"]["content"]
    # adapter co-located with its base node
    assert p.controlplane._owner["sql"] == p.controlplane._owner["base"]


def test_load_config_parses_nodes(tmp_path):
    cfg = tmp_path / "p.yaml"
    cfg.write_text(
        "models:\n"
        "  - {name: m, vram_mb: 6000}\n"
        "nodes:\n"
        "  - id: box-a\n"
        "    gpus: [{id: a0, vram_mb: 24000}]\n"
        "  - id: box-b\n"
        "    gpus: [{id: b0, vram_mb: 24000}, {id: b1, vram_mb: 24000}]\n")
    parsed = load_config(str(cfg))
    assert [n.id for n in parsed.nodes] == ["box-a", "box-b"]
    assert len(parsed.nodes[1].gpus) == 2
    p = Platform(parsed, mock=True)
    assert len(p.nodes) == 2


def test_single_node_config_still_one_node():
    p = Platform(PlatformConfig(models=[ModelConfig("m", 6000)],
                                gpus=[GPU("g0", 24000)]), mock=True)
    assert len(p.nodes) == 1 and p.nodes[0].node_id == "node0"
