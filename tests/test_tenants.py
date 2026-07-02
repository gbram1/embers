"""Per-tenant metering + quotas — attribute usage and rate-limit by tenant.
The multi-tenant business layer: who used what, and cut them off at their limit."""

import pytest
from fastapi.testclient import TestClient

from embers.gateway import (
    LocalBackend, QuotaLimiter, Router, create_gateway_app,
)
from embers.metrics import Registry
from embers.server import ModelUnit


def local(model="m"):
    u = ModelUnit(model, mock=True)
    u.load()
    return LocalBackend(u)


def gw(**kw):
    r = Router()
    r.register(local("m"))
    return TestClient(create_gateway_app(r, **kw))


def chat(c, key=None):
    h = {"Authorization": f"Bearer {key}"} if key else {}
    return c.post("/v1/chat/completions", headers=h,
                  json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})


# --- QuotaLimiter unit -----------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_quota_requests_per_min_then_429():
    clk = Clock()
    q = QuotaLimiter({"acme": {"requests_per_min": 2}}, clock=clk)
    q.check("acme")            # 1
    q.check("acme")            # 2
    with pytest.raises(Exception):
        q.check("acme")        # 3 → over


def test_quota_window_resets():
    clk = Clock()
    q = QuotaLimiter({"acme": {"requests_per_min": 1}}, clock=clk, window=60)
    q.check("acme")
    with pytest.raises(Exception):
        q.check("acme")
    clk.t = 61                 # new window
    q.check("acme")            # allowed again


def test_quota_tokens_per_min():
    clk = Clock()
    q = QuotaLimiter({"acme": {"tokens_per_min": 10}}, clock=clk)
    q.check("acme")            # under (0 tokens so far)
    q.add_tokens("acme", 12)   # now over budget
    with pytest.raises(Exception):
        q.check("acme")


def test_quota_untracked_tenant_unlimited():
    q = QuotaLimiter({"acme": {"requests_per_min": 1}})
    for _ in range(100):
        q.check("beta")        # no quota configured → never raises


# --- tenant resolution + metering (end to end) -----------------------------

def test_request_metered_under_its_tenant():
    reg = Registry()
    c = gw(tenants={"acme": "sk-acme"}, registry=reg)
    assert chat(c, "sk-acme").status_code == 200
    text = c.get("/metrics").text
    assert 'tenant="acme"' in text
    assert 'embers_tokens_total{endpoint="chat",model="m",tenant="acme"}' in text


def test_two_tenants_metered_separately():
    reg = Registry()
    c = gw(tenants={"acme": "sk-a", "beta": "sk-b"}, registry=reg)
    chat(c, "sk-a")
    chat(c, "sk-b")
    text = c.get("/metrics").text
    assert 'tenant="acme"' in text and 'tenant="beta"' in text


def test_tenant_key_is_also_a_valid_api_key():
    c = gw(tenants={"acme": "sk-acme"})
    assert chat(c, "sk-acme").status_code == 200      # tenant key authenticates
    assert chat(c, "wrong").status_code == 401        # unknown key rejected
    assert chat(c, None).status_code == 401           # missing token rejected


def test_no_auth_meters_anonymous():
    reg = Registry()
    c = gw(registry=reg)                               # no api_keys, no tenants
    assert chat(c).status_code == 200
    assert 'tenant="anonymous"' in c.get("/metrics").text


# --- quota enforcement through the gateway ---------------------------------

def test_gateway_429s_over_request_quota():
    clk = Clock()
    c = gw(tenants={"acme": "sk-acme"},
           quotas={"acme": {"requests_per_min": 2}}, clock=clk)
    assert chat(c, "sk-acme").status_code == 200
    assert chat(c, "sk-acme").status_code == 200
    assert chat(c, "sk-acme").status_code == 429      # third over the limit
    clk.t = 61
    assert chat(c, "sk-acme").status_code == 200       # window rolled


def test_quota_is_per_tenant_not_global():
    c = gw(tenants={"acme": "sk-a", "beta": "sk-b"},
           quotas={"acme": {"requests_per_min": 1}})
    assert chat(c, "sk-a").status_code == 200
    assert chat(c, "sk-a").status_code == 429          # acme capped
    assert chat(c, "sk-b").status_code == 200           # beta unaffected (no quota)
