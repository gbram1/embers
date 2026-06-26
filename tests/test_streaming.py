"""Tests for OpenAI-compatible SSE streaming — format helpers, the serving
unit's stream endpoints, and the gateway streaming path (all mock, no GPU)."""

import json

from fastapi.testclient import TestClient

from embers.gateway import LocalBackend, Router, create_gateway_app
from embers.server import ModelUnit, create_app
from embers.streaming import (
    chat_sse,
    chunk_text,
    completion_sse,
    parse_sse_text,
)


# --- helpers ---------------------------------------------------------------

def test_chunk_text_preserves_content():
    assert "".join(chunk_text("Paris is nice")) == "Paris is nice"
    assert chunk_text("") == []


def test_completion_sse_shape():
    out = list(completion_sse(["Par", "is"], "m", cid="cmpl-1"))
    assert out[-1] == "data: [DONE]\n\n"
    first = json.loads(out[0][len("data: "):])
    assert first["object"] == "text_completion"
    assert first["choices"][0]["text"] == "Par"
    # a finish chunk precedes [DONE]
    finish = json.loads(out[-2][len("data: "):])
    assert finish["choices"][0]["finish_reason"] == "stop"


def test_chat_sse_has_role_then_content_then_done():
    out = list(chat_sse(["hi"], "m", cid="c1"))
    first = json.loads(out[0][len("data: "):])
    assert first["choices"][0]["delta"]["role"] == "assistant"
    content = json.loads(out[1][len("data: "):])
    assert content["choices"][0]["delta"]["content"] == "hi"
    assert out[-1] == "data: [DONE]\n\n"


def test_parse_sse_text_roundtrip_completion():
    sse = list(completion_sse(["Par", "is "], "m"))
    # SSE blocks are "data: {...}\n\n" — feed individual lines
    lines = "".join(sse).splitlines()
    assert "".join(parse_sse_text(lines, chat=False)) == "Paris "


def test_parse_sse_text_roundtrip_chat():
    sse = list(chat_sse(["he", "llo"], "m"))
    lines = "".join(sse).splitlines()
    assert "".join(parse_sse_text(lines, chat=True)) == "hello"


# --- serving unit stream endpoints -----------------------------------------

def unit_client():
    u = ModelUnit("m", mock=True)
    u.load()
    return TestClient(create_app(u))


def test_unit_streams_completion():
    r = unit_client().post("/v1/completions",
                           json={"prompt": "hi", "stream": True})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "data: [DONE]" in r.text
    # reconstruct the streamed text from the chunks
    text = "".join(parse_sse_text(r.text.splitlines(), chat=False))
    assert "hi" in text


def test_unit_streams_chat():
    r = unit_client().post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "yo"}],
                                 "stream": True})
    assert "chat.completion.chunk" in r.text
    text = "".join(parse_sse_text(r.text.splitlines(), chat=True))
    assert "yo" in text


def test_unit_non_stream_still_works():
    r = unit_client().post("/v1/completions", json={"prompt": "hi"})
    assert r.json()["object"] == "text_completion"   # default stream=false


# --- gateway streaming -----------------------------------------------------

def gw_client():
    r = Router()
    u = ModelUnit("m", mock=True)
    u.load()
    r.register(LocalBackend(u))
    return TestClient(create_gateway_app(r))


def test_gateway_streams_completion():
    r = gw_client().post("/v1/completions",
                         json={"model": "m", "prompt": "hello", "stream": True})
    assert "text/event-stream" in r.headers["content-type"]
    assert "data: [DONE]" in r.text
    text = "".join(parse_sse_text(r.text.splitlines(), chat=False))
    assert "hello" in text


def test_gateway_streams_chat():
    r = gw_client().post("/v1/chat/completions",
                         json={"model": "m", "stream": True,
                               "messages": [{"role": "user", "content": "ping"}]})
    text = "".join(parse_sse_text(r.text.splitlines(), chat=True))
    assert "ping" in text


def test_gateway_stream_through_autoscaler_releases_inflight():
    from embers.autoscaler import Autoscaler
    from embers.scheduler import GPU, Scheduler

    sched = Scheduler([GPU("g0", 24000)])
    router = Router()

    def launch(model, gpu_id):
        u = ModelUnit(model, mock=True)
        u.load()
        return LocalBackend(u)

    auto = Autoscaler(sched, router, launch)
    auto.register_model("m", 6000)
    c = TestClient(create_gateway_app(router, autoscaler=auto))
    r = c.post("/v1/completions",
               json={"model": "m", "prompt": "x", "stream": True})
    assert "data: [DONE]" in r.text          # full stream consumed by TestClient
    assert auto.inflight("m") == 0           # released after the stream closed
