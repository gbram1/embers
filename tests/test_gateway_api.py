"""Integration tests for the gateway app — multi-model routing, load
distribution, auth, and OpenAI-compatible error codes (all mock, no GPU)."""

from fastapi.testclient import TestClient

from embers.gateway import LocalBackend, Router, create_gateway_app
from embers.server import ModelUnit
from tests.test_gateway import FakeBackend


def local(model: str, loaded: bool = True) -> LocalBackend:
    u = ModelUnit(model, mock=True)
    if loaded:
        u.load()
    return LocalBackend(u)


def gw(router: Router, **kw) -> TestClient:
    return TestClient(create_gateway_app(router, **kw))


def test_lists_all_registered_models():
    r = Router()
    r.register(local("a"))
    r.register(local("b"))
    ids = [m["id"] for m in gw(r).get("/v1/models").json()["data"]]
    assert ids == ["a", "b"]


def test_routes_to_correct_model():
    r = Router()
    r.register(local("alpha"))
    r.register(local("beta"))
    c = gw(r)
    body = c.post("/v1/completions",
                  json={"model": "beta", "prompt": "hi"}).json()
    assert body["model"] == "beta"
    assert "hi" in body["choices"][0]["text"]


def test_chat_routes_and_shapes():
    r = Router()
    r.register(local("m"))
    body = gw(r).post("/v1/chat/completions",
                      json={"model": "m",
                            "messages": [{"role": "user", "content": "yo"}]}).json()
    assert body["object"] == "chat.completion"
    assert "yo" in body["choices"][0]["message"]["content"]


def test_load_balances_across_replicas():
    r = Router()
    backs = [FakeBackend("m", i) for i in range(3)]
    for b in backs:
        r.register(b)
    c = gw(r)
    for _ in range(9):
        c.post("/v1/completions", json={"model": "m", "prompt": "x"})
    assert [b.calls for b in backs] == [3, 3, 3]  # evenly distributed


def test_unknown_model_404():
    r = gw(Router())
    assert r.post("/v1/completions",
                  json={"model": "ghost", "prompt": "x"}).status_code == 404


def test_no_ready_replica_503():
    r = Router()
    r.register(local("m", loaded=False))  # registered but not loaded
    assert gw(r).post("/v1/completions",
                      json={"model": "m", "prompt": "x"}).status_code == 503


def test_missing_model_field_400():
    r = Router()
    r.register(local("m"))
    assert gw(r).post("/v1/completions",
                      json={"prompt": "x"}).status_code == 400


def test_auth_required_when_keys_set():
    r = Router()
    r.register(local("m"))
    c = gw(r, api_keys={"secret"})
    # no token -> 401
    assert c.post("/v1/completions",
                  json={"model": "m", "prompt": "x"}).status_code == 401
    # wrong token -> 401
    assert c.post("/v1/completions", headers={"Authorization": "Bearer nope"},
                  json={"model": "m", "prompt": "x"}).status_code == 401
    # right token -> 200
    assert c.post("/v1/completions", headers={"Authorization": "Bearer secret"},
                  json={"model": "m", "prompt": "x"}).status_code == 200


def test_no_auth_when_keys_none():
    r = Router()
    r.register(local("m"))
    assert gw(r).post("/v1/completions",
                      json={"model": "m", "prompt": "x"}).status_code == 200
