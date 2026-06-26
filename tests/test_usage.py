"""Token-usage accounting — the serving unit reports real usage (not 0,0,0) and
the gateway meters tokens per model (for billing). Mock mode approximates token
counts by whitespace words; real mode uses vLLM token_ids.
"""

from fastapi.testclient import TestClient

from embers.gateway import LocalBackend, Router, create_gateway_app
from embers.metrics import Registry
from embers.server import ModelUnit, create_app


def _unit(adapters=None):
    u = ModelUnit("m", mock=True, adapters=adapters)
    u.load()
    return u


# --- serving unit reports real usage --------------------------------------

def test_completion_reports_nonzero_usage():
    c = TestClient(create_app(_unit()))
    r = c.post("/v1/completions", json={"model": "m", "prompt": "alpha beta gamma"})
    u = r.json()["usage"]
    assert u["prompt_tokens"] == 3                      # three words
    assert u["completion_tokens"] > 0
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_chat_reports_usage_summing_message_words():
    c = TestClient(create_app(_unit()))
    r = c.post("/v1/chat/completions", json={
        "model": "m",
        "messages": [{"role": "system", "content": "be terse"},
                     {"role": "user", "content": "hello there friend"}]})
    u = r.json()["usage"]
    assert u["prompt_tokens"] == 2 + 3                  # "be terse" + "hello there friend"
    assert u["completion_tokens"] > 0
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_usage_is_no_longer_stubbed_zero():
    c = TestClient(create_app(_unit()))
    r = c.post("/v1/completions", json={"model": "m", "prompt": "x y"})
    assert r.json()["usage"] != {"prompt_tokens": 0, "completion_tokens": 0,
                                 "total_tokens": 0}


# --- gateway meters tokens per model --------------------------------------

def _gateway():
    router = Router()
    router.register(LocalBackend(_unit()))
    reg = Registry()
    return TestClient(create_gateway_app(router, registry=reg)), reg


def test_gateway_returns_real_usage():
    c, _ = _gateway()
    r = c.post("/v1/chat/completions",
               json={"model": "m", "messages": [{"role": "user", "content": "a b c d"}]})
    assert r.status_code == 200
    assert r.json()["usage"]["prompt_tokens"] == 4
    assert r.json()["usage"]["total_tokens"] > 4


def test_gateway_meters_tokens_in_prometheus():
    c, reg = _gateway()
    c.post("/v1/chat/completions",
           json={"model": "m", "messages": [{"role": "user", "content": "one two three"}]})
    metrics = reg.render_prometheus()
    assert "embers_tokens_total" in metrics
    # the counter carries the model label and a positive value
    line = [ln for ln in metrics.splitlines()
            if ln.startswith("embers_tokens_total") and 'model="m"' in ln][0]
    assert float(line.rsplit(" ", 1)[1]) > 0


def test_gateway_tokens_accumulate_across_requests():
    c, reg = _gateway()
    for _ in range(3):
        c.post("/v1/completions", json={"model": "m", "prompt": "a b"})
    line = [ln for ln in reg.render_prometheus().splitlines()
            if ln.startswith("embers_tokens_total") and 'endpoint="completions"' in ln][0]
    # 3 requests each counted prompt(2)+completion(>0) → strictly more than one req
    assert float(line.rsplit(" ", 1)[1]) >= 3 * 3


# --- streaming usage (the include_usage final chunk + metering) -----------

import json


def test_streaming_emits_a_usage_chunk():
    c = TestClient(create_app(_unit()))
    r = c.post("/v1/chat/completions",
               json={"model": "m", "stream": True,
                     "messages": [{"role": "user", "content": "alpha beta gamma"}]})
    assert r.status_code == 200
    # find the final chunk carrying usage (empty choices + usage)
    chunks = [json.loads(ln[len("data:"):].strip())
              for ln in r.text.splitlines()
              if ln.startswith("data:") and ln.strip() != "data: [DONE]"]
    usage_chunks = [ch for ch in chunks if ch.get("usage")]
    assert len(usage_chunks) == 1
    u = usage_chunks[0]["usage"]
    assert u["prompt_tokens"] == 3 and u["completion_tokens"] > 0
    assert usage_chunks[0]["choices"] == []        # OpenAI: usage chunk has no choices


def test_gateway_meters_streaming_requests():
    c, reg = _gateway()
    # stream a request and fully consume it (TestClient reads the whole body)
    r = c.post("/v1/chat/completions",
               json={"model": "m", "stream": True,
                     "messages": [{"role": "user", "content": "one two three four"}]})
    assert r.status_code == 200
    # streaming requests are NOT free — tokens were metered
    line = [ln for ln in reg.render_prometheus().splitlines()
            if ln.startswith("embers_tokens_total") and 'endpoint="chat"' in ln][0]
    assert float(line.rsplit(" ", 1)[1]) >= 4      # at least the 4 prompt words


def test_parse_sse_text_fills_usage_from_chunk():
    from embers.streaming import parse_sse_text
    lines = [
        'data: {"choices":[{"delta":{"content":"hi"}}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}',
        'data: [DONE]',
    ]
    usage = {}
    pieces = list(parse_sse_text(lines, chat=True, usage=usage))
    assert pieces == ["hi"]
    assert usage == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
