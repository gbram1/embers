"""Phase 1 API tests — OpenAI-compatible endpoints in mock mode (no vLLM)."""

from fastapi.testclient import TestClient

from embers.server import ModelUnit, create_app


def client(loaded: bool = True) -> TestClient:
    unit = ModelUnit("test-model", mock=True)
    if loaded:
        unit.load()
    return TestClient(create_app(unit))


def test_health_is_liveness_even_before_load():
    c = client(loaded=False)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_503_until_loaded():
    assert client(loaded=False).get("/ready").status_code == 503
    assert client(loaded=True).get("/ready").status_code == 200


def test_models_lists_served_model():
    r = client().get("/v1/models")
    assert r.json()["data"][0]["id"] == "test-model"


def test_completions_openai_shape():
    r = client().post("/v1/completions",
                      json={"prompt": "The capital of France is"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert "France" in body["choices"][0]["text"]
    assert "usage" in body


def test_chat_completions_openai_shape():
    r = client().post("/v1/chat/completions",
                      json={"messages": [{"role": "user", "content": "hi there"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert "hi there" in body["choices"][0]["message"]["content"]


def test_completions_requires_prompt():
    r = client().post("/v1/completions", json={"max_tokens": 8})
    assert r.status_code == 422  # pydantic validation


def test_chat_requires_messages():
    r = client().post("/v1/chat/completions", json={})
    assert r.status_code == 422


def test_unknown_route_404():
    assert client().get("/nope").status_code == 404


def test_completion_respects_max_tokens_field():
    # mock ignores value but the field must be accepted without error
    r = client().post("/v1/completions",
                      json={"prompt": "x", "max_tokens": 1, "temperature": 0.7})
    assert r.status_code == 200


def test_generate_autoloads_when_not_eager():
    # unit starts unloaded; first completion must trigger load (cold-on-demand)
    from embers.server import ModelUnit, create_app
    unit = ModelUnit("m", mock=True)
    c = TestClient(create_app(unit))
    assert c.get("/ready").status_code == 503      # not loaded yet
    assert c.post("/v1/completions", json={"prompt": "x"}).status_code == 200
    assert c.get("/ready").status_code == 200       # loaded on demand
