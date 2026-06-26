"""Unit tests for the Phase 2 Router — round-robin, health-skipping, errors."""

import httpx
import pytest

from embers import gateway as gw_mod
from embers.gateway import (
    HttpBackend,
    NoReadyBackend,
    Router,
    router_from_backends,
)


class FakeBackend:
    """Records which backend served, with toggleable readiness."""

    def __init__(self, name: str, idx: int, ready: bool = True):
        self.name = name
        self.idx = idx
        self._ready = ready
        self.calls = 0

    @property
    def ready(self) -> bool:
        return self._ready

    def complete(self, *a):
        self.calls += 1
        return f"{self.name}#{self.idx}"

    def chat(self, *a):
        self.calls += 1
        return f"{self.name}#{self.idx}"


def test_unknown_model_raises_keyerror():
    with pytest.raises(KeyError):
        Router().pick("nope")


def test_single_backend_always_picked():
    r = Router()
    b = FakeBackend("m", 0)
    r.register(b)
    assert r.pick("m") is b
    assert r.pick("m") is b


def test_round_robin_distributes():
    r = Router()
    backs = [FakeBackend("m", i) for i in range(3)]
    for b in backs:
        r.register(b)
    picked = [r.pick("m").idx for _ in range(6)]
    assert picked == [0, 1, 2, 0, 1, 2]  # even rotation


def test_round_robin_skips_unready():
    r = Router()
    b0 = FakeBackend("m", 0, ready=True)
    b1 = FakeBackend("m", 1, ready=False)
    b2 = FakeBackend("m", 2, ready=True)
    for b in (b0, b1, b2):
        r.register(b)
    picked = [r.pick("m").idx for _ in range(4)]
    assert 1 not in picked          # the unready replica is never chosen
    assert set(picked) == {0, 2}


def test_no_ready_backend_raises():
    r = Router()
    r.register(FakeBackend("m", 0, ready=False))
    with pytest.raises(NoReadyBackend):
        r.pick("m")


def test_models_and_replicas():
    r = Router()
    r.register(FakeBackend("a", 0))
    r.register(FakeBackend("b", 0))
    r.register(FakeBackend("b", 1))
    assert r.models() == ["a", "b"]
    assert len(r.replicas("b")) == 2
    assert r.replicas("missing") == []


# --- HttpBackend (httpx mocked) -------------------------------------------

class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


def test_http_backend_ready_true(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(200))
    assert HttpBackend("m", "http://x:8000").ready is True


def test_http_backend_ready_false_on_503(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(503))
    assert HttpBackend("m", "http://x:8000").ready is False


def test_http_backend_ready_false_on_network_error(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx, "get", boom)
    assert HttpBackend("m", "http://x:8000").ready is False


def test_http_backend_complete_forwards(monkeypatch):
    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        captured["json"] = kw["json"]
        return _Resp(200, {"choices": [{"text": "remote-out"}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    out = HttpBackend("m", "http://x:8000/").complete("hi", 16, 0.5)
    assert out == "remote-out"
    assert captured["url"] == "http://x:8000/v1/completions"
    assert captured["json"]["prompt"] == "hi"
    assert captured["json"]["max_tokens"] == 16


def test_http_backend_chat_forwards(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda url, **k: _Resp(
        200, {"choices": [{"message": {"content": "reply"}}]}))
    out = HttpBackend("m", "http://x:8000").chat(
        [{"role": "user", "content": "q"}], 8, 0.0)
    assert out == "reply"


def test_router_from_backends_groups_replicas():
    r = router_from_backends([
        ("a", "http://h1:8000"),
        ("a", "http://h2:8000"),  # second replica of 'a'
        ("b", "http://h3:8000"),
    ])
    assert r.models() == ["a", "b"]
    assert len(r.replicas("a")) == 2
