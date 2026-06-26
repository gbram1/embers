"""LoRA multi-adapter serving — many fine-tuned adapters off ONE base model.

The serving unit resolves a request's `model` field to a base (None) or an adapter,
lists base+adapters in /v1/models, and (real mode) builds a vLLM LoRARequest. These
tests cover the unit + HTTP endpoints in mock mode (no GPU).
"""

import pytest
from fastapi.testclient import TestClient

from embers.server import ModelUnit, create_app


def unit(adapters=None):
    u = ModelUnit("base-7b", mock=True, adapters=adapters)
    u.load()
    return u


# --- adapter resolution ---------------------------------------------------

def test_lora_for_resolves_base_and_adapters():
    u = unit({"sql": "/a/sql", "chat": "/a/chat"})
    assert u.lora_for(None) is None            # default → base
    assert u.lora_for("base-7b") is None       # explicit base
    assert u.lora_for("sql") == "sql"          # an adapter
    assert u.lora_for("chat") == "chat"
    with pytest.raises(KeyError):
        u.lora_for("unknown")                  # not served here


def test_served_models_lists_base_and_adapters():
    u = unit({"sql": "/a/sql", "chat": "/a/chat"})
    assert u.served_models() == ["base-7b", "sql", "chat"]


def test_adapters_get_stable_distinct_lora_ids():
    u = unit({"sql": "/a/sql", "chat": "/a/chat"})
    assert u._lora_ids == {"sql": 1, "chat": 2}     # 1-based, unique


def test_no_adapters_is_plain_base_model():
    u = unit()
    assert u.served_models() == ["base-7b"]
    assert u.adapters == {} and u.lora_for(None) is None


def test_enable_lora_flag_set_only_when_adapters_present():
    assert ModelUnit("b", adapters={"x": "/p"}).engine_kwargs.get("enable_lora") is True
    assert "enable_lora" not in ModelUnit("b").engine_kwargs


# --- HTTP endpoints select the adapter ------------------------------------

def test_models_endpoint_lists_base_and_adapters():
    c = TestClient(create_app(unit({"sql": "/a/sql", "chat": "/a/chat"})))
    ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
    assert ids == ["base-7b", "sql", "chat"]


def test_chat_routes_to_the_requested_adapter():
    c = TestClient(create_app(unit({"sql": "/a/sql"})))
    base = c.post("/v1/chat/completions",
                  json={"model": "base-7b", "messages": [{"role": "user", "content": "hi"}]})
    sql = c.post("/v1/chat/completions",
                 json={"model": "sql", "messages": [{"role": "user", "content": "hi"}]})
    assert base.status_code == sql.status_code == 200
    base_txt = base.json()["choices"][0]["message"]["content"]
    sql_txt = sql.json()["choices"][0]["message"]["content"]
    assert "via sql" in sql_txt                 # adapter path taken
    assert "via" not in base_txt                # base path (no adapter)
    assert sql.json()["model"] == "sql"         # echoes the requested model


def test_completion_routes_to_adapter():
    c = TestClient(create_app(unit({"sql": "/a/sql"})))
    r = c.post("/v1/completions", json={"model": "sql", "prompt": "SELECT"})
    assert r.status_code == 200
    assert "via sql" in r.json()["choices"][0]["text"]


def test_unknown_model_is_404():
    c = TestClient(create_app(unit({"sql": "/a/sql"})))
    r = c.post("/v1/chat/completions",
               json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404


def test_streaming_through_an_adapter():
    c = TestClient(create_app(unit({"sql": "/a/sql"})))
    r = c.post("/v1/chat/completions",
               json={"model": "sql", "stream": True,
                     "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.text
    # mock streams word-by-word, so "via" and "sql" arrive as separate deltas
    assert '"via ' in body and '"sql ' in body    # adapter path taken
    assert "data: [DONE]" in body


# --- autoscaler: an adapter request resolves to (and cold-starts) the base ---

def _autoscaler_with_adapters():
    from embers.autoscaler import Autoscaler
    from embers.gateway import LocalBackend, Router
    from embers.scheduler import GPU, Scheduler

    class Clock:
        t = 0.0
        def __call__(self):
            return self.t

    sched = Scheduler([GPU("g0", 24000)])
    router = Router()

    def launch(model, gpu_ids):
        u = ModelUnit(model, mock=True, adapters={"sql": "/a/sql"})
        u.load()
        return LocalBackend(u)

    a = Autoscaler(sched, router, launch, clock=Clock())
    a.register_model("base-7b", 6000, adapters={"sql": "/a/sql"})
    return a


def test_adapter_request_cold_starts_the_base():
    a = _autoscaler_with_adapters()
    assert a.state()["base-7b"] == 0
    backend = a.handle_request("sql")          # request the ADAPTER
    assert backend is not None
    assert a.state()["base-7b"] == 1           # the BASE cold-started
    assert a.cold_starts == 1
    # the returned backend is the base unit, and it serves the adapter
    assert "via sql" in backend.chat([{"role": "user", "content": "x"}], 8, 0.0, "sql")


def test_served_models_includes_adapters_and_unknown_raises():
    a = _autoscaler_with_adapters()
    assert a.served_models() == ["base-7b", "sql"]
    with pytest.raises(KeyError):
        a.handle_request("ghost")


def test_inflight_tracked_against_base_for_adapter_requests():
    a = _autoscaler_with_adapters()
    a.begin_request("sql")
    assert a.inflight("sql") == 1 and a.inflight("base-7b") == 1   # same counter
    a.end_request("sql")
    assert a.inflight("base-7b") == 0


# --- platform mock: adapters end-to-end through `embers up` ----------------

def test_platform_routes_adapter_through_gateway_to_base_unit():
    from embers.platform import ModelConfig, Platform, PlatformConfig
    from embers.scheduler import GPU

    cfg = PlatformConfig(
        models=[ModelConfig("base-7b", 6000, adapters={"sql": "/a/sql", "chat": "/a/chat"})],
        gpus=[GPU("g0", 24000)])
    p = Platform(cfg, mock=True)
    c = TestClient(p.app)
    assert sorted(m["id"] for m in c.get("/v1/models").json()["data"]) == \
        ["base-7b", "chat", "sql"]
    r = c.post("/v1/chat/completions",
               json={"model": "sql", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert "via sql" in r.json()["choices"][0]["message"]["content"]
    assert p.autoscaler.state()["base-7b"] == 1     # one base unit serves both adapters


# --- launcher passes --lora to the serving process ------------------------

def test_launcher_builds_serve_process_with_lora_flags():
    from embers.gpu_backend import GpuLauncher
    gl = GpuLauncher(adapters_by_model={"base-7b": {"sql": "/a/sql"}})
    proc = gl._make("base-7b", 19000, ["gpu0"])
    assert proc.adapters == {"sql": "/a/sql"}
    spawned = {}

    def spawn(args):
        spawned["args"] = args

        class P:
            pid = 1
        return P()
    proc._spawn = spawn
    proc._ready = lambda: True
    proc._sleep = lambda s: None
    proc.start()
    a = spawned["args"]
    assert "--lora" in a and "sql=/a/sql" in a
