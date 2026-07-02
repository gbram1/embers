"""Phase 2 gateway — one URL fronting many models/replicas.

Accepts OpenAI-compatible requests, authenticates, routes by the `model` field
to a registered backend, and round-robin load-balances across replicas of that
model. A backend is either in-process (LocalBackend, for single-node/tests) or a
remote serving unit (HttpBackend, the real multi-process/container case).

  POST /v1/completions       {model, prompt, ...}   -> routed to a replica
  POST /v1/chat/completions  {model, messages, ...}
  GET  /v1/models            -> every registered model

This is the layer the Phase-3 scheduler populates (which replica runs where) and
the Phase-4 autoscaler grows/shrinks.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from typing import Any, Protocol

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse

from embers.metrics import Registry
from embers.server import ChatRequest, CompletionRequest, ModelUnit
from embers.streaming import chat_sse, completion_sse


class NoReadyBackend(Exception):
    """No replica of a known model is ready to serve (-> 503)."""


class Backend(Protocol):
    name: str

    @property
    def ready(self) -> bool: ...
    def complete(self, prompt: str, max_tokens: int, temperature: float,
                 model: str | None = None) -> str: ...
    def chat(self, messages: list[dict], max_tokens: int, temperature: float,
             model: str | None = None) -> str: ...


class LocalBackend:
    """In-process backend wrapping a ModelUnit — single-node and tests."""

    def __init__(self, unit: ModelUnit):
        self.unit = unit

    @property
    def name(self) -> str:
        return self.unit.model

    @property
    def ready(self) -> bool:
        return self.unit.loaded

    def _lora(self, model):
        # `model` may name a LoRA adapter served off this unit's base.
        try:
            return self.unit.lora_for(model)
        except KeyError:
            return None

    def complete(self, prompt, max_tokens, temperature, model=None) -> str:
        return self.unit.generate(prompt, max_tokens, temperature, self._lora(model))

    def chat(self, messages, max_tokens, temperature, model=None) -> str:
        return self.unit.chat(messages, max_tokens, temperature, self._lora(model))

    def stream_complete(self, prompt, max_tokens, temperature, model=None, usage=None):
        return self.unit.stream(prompt, max_tokens, temperature,
                                self._lora(model), usage)

    def stream_chat(self, messages, max_tokens, temperature, model=None, usage=None):
        return self.unit.chat_stream(messages, max_tokens, temperature,
                                     self._lora(model), usage)

    def complete_usage(self, prompt, max_tokens, temperature, model=None):
        text = self.unit.generate(prompt, max_tokens, temperature, self._lora(model))
        return text, self.unit._usage(self.unit._mock_count(prompt),
                                      self.unit._mock_count(text))

    def chat_usage(self, messages, max_tokens, temperature, model=None):
        text = self.unit.chat(messages, max_tokens, temperature, self._lora(model))
        pt = sum(self.unit._mock_count(m.get("content", "")) for m in messages)
        return text, self.unit._usage(pt, self.unit._mock_count(text))


class HttpBackend:
    """Forwards to a remote serving unit's /v1 endpoints (real multi-node)."""

    def __init__(self, name: str, base_url: str, timeout: float = 60.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def ready(self) -> bool:
        try:
            return httpx.get(f"{self.base_url}/ready", timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    def complete(self, prompt, max_tokens, temperature, model=None) -> str:
        body = {"prompt": prompt, "max_tokens": max_tokens,
                "temperature": temperature}
        if model:                          # forward the adapter/model name
            body["model"] = model
        r = httpx.post(f"{self.base_url}/v1/completions", timeout=self.timeout,
                       json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["text"]

    def chat(self, messages, max_tokens, temperature, model=None) -> str:
        body = {"messages": messages, "max_tokens": max_tokens,
                "temperature": temperature}
        if model:
            body["model"] = model
        r = httpx.post(f"{self.base_url}/v1/chat/completions", timeout=self.timeout,
                       json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def complete_usage(self, prompt, max_tokens, temperature, model=None):
        body = {"prompt": prompt, "max_tokens": max_tokens,
                "temperature": temperature}
        if model:
            body["model"] = model
        r = httpx.post(f"{self.base_url}/v1/completions", timeout=self.timeout,
                       json=body)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["text"], j.get("usage", _ZERO_USAGE)

    def chat_usage(self, messages, max_tokens, temperature, model=None):
        body = {"messages": messages, "max_tokens": max_tokens,
                "temperature": temperature}
        if model:
            body["model"] = model
        r = httpx.post(f"{self.base_url}/v1/chat/completions", timeout=self.timeout,
                       json=body)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"], j.get("usage", _ZERO_USAGE)

    def _stream(self, path: str, body: dict, *, chat: bool, usage=None):
        from embers.streaming import parse_sse_text
        body = {**body, "stream": True}
        with httpx.stream("POST", f"{self.base_url}{path}", json=body,
                          timeout=self.timeout) as r:
            r.raise_for_status()
            yield from parse_sse_text(r.iter_lines(), chat=chat, usage=usage)

    def stream_complete(self, prompt, max_tokens, temperature, model=None, usage=None):
        body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        if model:
            body["model"] = model
        return self._stream("/v1/completions", body, chat=False, usage=usage)

    def stream_chat(self, messages, max_tokens, temperature, model=None, usage=None):
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if model:
            body["model"] = model
        return self._stream("/v1/chat/completions", body, chat=True, usage=usage)


class Router:
    """Registry of model -> replicas, with per-model round-robin that skips
    replicas that aren't ready."""

    def __init__(self):
        self._backends: dict[str, list[Backend]] = {}
        self._rr: dict[str, int] = {}

    def register(self, backend: Backend) -> None:
        self._backends.setdefault(backend.name, []).append(backend)

    def unregister(self, backend: Backend) -> bool:
        """Remove one replica (by identity). Drops the model entirely when its
        last replica goes. Returns True if something was removed."""
        lst = self._backends.get(backend.name)
        if not lst or backend not in lst:
            return False
        lst.remove(backend)
        if not lst:
            del self._backends[backend.name]
            self._rr.pop(backend.name, None)
        return True

    def models(self) -> list[str]:
        return sorted(self._backends)

    def replicas(self, model: str) -> list[Backend]:
        return list(self._backends.get(model, []))

    def pick(self, model: str) -> Backend:
        if model not in self._backends:
            raise KeyError(model)
        backends = self._backends[model]
        start = self._rr.get(model, 0)
        n = len(backends)
        for i in range(n):
            b = backends[(start + i) % n]
            try:
                ready = b.ready
            except Exception:        # noqa: BLE001 — a throwing probe = not ready
                ready = False
            if ready:
                self._rr[model] = (start + i + 1) % n
                return b
        raise NoReadyBackend(model)


_ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _complete_call(backend, prompt, max_tokens, temperature, model):
    """(text, usage) from a backend — usage-aware path, fallback to plain for
    backends that don't report usage."""
    if hasattr(backend, "complete_usage"):
        return backend.complete_usage(prompt, max_tokens, temperature, model)
    return backend.complete(prompt, max_tokens, temperature, model), dict(_ZERO_USAGE)


def _chat_call(backend, messages, max_tokens, temperature, model):
    if hasattr(backend, "chat_usage"):
        return backend.chat_usage(messages, max_tokens, temperature, model)
    return backend.chat(messages, max_tokens, temperature, model), dict(_ZERO_USAGE)


def _route(router: Router, model: str | None) -> Backend:
    if not model:
        raise HTTPException(400, "request must specify a 'model'")
    try:
        return router.pick(model)
    except KeyError:
        raise HTTPException(404, f"model '{model}' not registered")
    except NoReadyBackend:
        raise HTTPException(503, f"no ready replica for '{model}'")


class QuotaLimiter:
    """Per-tenant fixed-window rate limits (requests and/or tokens per window).
    Token usage is only known after a response, so it's checked at request start
    against the window's running total and added after — a tenant can slightly
    overshoot on the crossing request, then 429s until the window rolls. Clock is
    injectable for deterministic tests."""

    def __init__(self, quotas: dict[str, dict] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 window: float = 60.0):
        self._q = quotas or {}
        self._clock = clock
        self._window = window
        self._state: dict[str, dict] = {}      # tenant -> {start, reqs, tokens}
        self._lock = threading.Lock()

    def _win(self, tenant: str) -> dict:
        now = self._clock()
        st = self._state.get(tenant)
        if st is None or now - st["start"] >= self._window:
            st = {"start": now, "reqs": 0, "tokens": 0}
            self._state[tenant] = st
        return st

    def check(self, tenant: str) -> None:
        """Raise 429 if the tenant is already over a limit; else count the request."""
        limits = self._q.get(tenant)
        if not limits:
            return
        with self._lock:
            st = self._win(tenant)
            rpm, tpm = limits.get("requests_per_min"), limits.get("tokens_per_min")
            if rpm is not None and st["reqs"] >= rpm:
                raise HTTPException(429, f"tenant '{tenant}' over quota: "
                                    f"{rpm} requests/min")
            if tpm is not None and st["tokens"] >= tpm:
                raise HTTPException(429, f"tenant '{tenant}' over quota: "
                                    f"{tpm} tokens/min")
            st["reqs"] += 1

    def add_tokens(self, tenant: str, n: int) -> None:
        if not self._q.get(tenant) or not n:
            return
        with self._lock:
            self._win(tenant)["tokens"] += n


def create_gateway_app(router: Router,
                       api_keys: set[str] | None = None,
                       tenants: dict[str, str] | None = None,
                       quotas: dict[str, dict] | None = None,
                       registry: Registry | None = None,
                       snapshot_fn: Callable[[], dict] | None = None,
                       autoscaler: Any | None = None,
                       clock: Callable[[], float] = time.monotonic) -> FastAPI:
    """Build the gateway app. api_keys=None disables auth (dev); a non-empty set
    requires `Authorization: Bearer <key>`. `tenants` maps tenant name -> api key
    (multi-tenant: requests are attributed + metered per tenant, and each key
    doubles as a valid api key); `quotas` maps tenant -> {requests_per_min,
    tokens_per_min} enforced with 429. A registry enables `/metrics` (Prometheus);
    a snapshot_fn enables `/stats`. An autoscaler makes the gateway the real front
    door (begin_request/end_request cold-start + in-flight tracking)."""
    app = FastAPI(title="embers gateway")
    reg = registry or Registry()
    reqs = reg.counter("embers_requests_total", "requests routed by the gateway")
    latency = reg.histogram("embers_request_latency_seconds",
                            "end-to-end request latency at the gateway")
    tokens = reg.counter("embers_tokens_total",
                         "tokens served (prompt+completion) — for metering/billing")

    # tenant resolution: `tenants` (name->key) inverted to key->name; every key
    # in tenants is also a valid api key. Auth is on if api_keys OR tenants given.
    key2tenant = {key: name for name, key in (tenants or {}).items()}
    effective_keys = set(api_keys or set()) | set(key2tenant)
    auth_on = bool(effective_keys)
    quota = QuotaLimiter(quotas, clock=clock)

    def _record(endpoint: str, model: str, seconds: float,
                usage: dict | None = None, tenant: str = "anonymous") -> None:
        reqs.inc(model=model, endpoint=endpoint, tenant=tenant)
        latency.observe(seconds, model=model, endpoint=endpoint)
        if usage:
            n = usage.get("total_tokens", 0)
            tokens.inc(n, model=model, endpoint=endpoint, tenant=tenant)
            quota.add_tokens(tenant, n)

    def _acquire(model: str | None):
        """Get a backend to serve `model` + a release callback. Through the
        autoscaler when present (spins up cold models, marks in-flight); else
        straight router routing."""
        if not model:
            raise HTTPException(400, "request must specify a 'model'")
        if autoscaler is None:
            return _route(router, model), (lambda: None)
        try:
            backend = autoscaler.begin_request(model)
        except KeyError:
            raise HTTPException(404, f"model '{model}' not registered")
        except NoReadyBackend:
            raise HTTPException(503, f"no ready replica for '{model}'")
        return backend, (lambda: autoscaler.end_request(model))

    def auth(authorization: str | None = Header(None)) -> str:
        """Validate the bearer token and resolve it to a tenant id (used to meter
        + rate-limit). Returns 'anonymous' when auth is off."""
        if not auth_on:
            return "anonymous"
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing bearer token")
        key = authorization.split(" ", 1)[1]
        if key not in effective_keys:
            raise HTTPException(401, "invalid api key")
        return key2tenant.get(key, "default")

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        # list every servable model — including ones currently scaled to zero
        # (the autoscaler knows them even when the router has no live replica)
        # and LoRA adapters served off a base.
        names = (sorted(autoscaler.served_models()) if autoscaler is not None
                 else router.models())
        return {"object": "list",
                "data": [{"id": m, "object": "model"} for m in names]}

    def _streamer(endpoint, model, backend, release, sse_iter, usage=None,
                  tenant="anonymous"):
        """Wrap a backend stream as SSE, releasing the in-flight slot when the
        client finishes reading (the request spans the whole stream). `usage` is
        filled by the time the stream ends, so it's metered then."""
        def gen():
            t0 = time.perf_counter()
            try:
                yield from sse_iter
            finally:
                release()
                _record(endpoint, model, time.perf_counter() - t0, usage, tenant)
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/completions")
    def completions(req: CompletionRequest, tenant: str = Depends(auth)):
        quota.check(tenant)
        backend, release = _acquire(req.model)
        if req.stream:
            usage: dict = {}
            pieces = backend.stream_complete(req.prompt, req.max_tokens,
                                             req.temperature, req.model, usage)
            return _streamer("completions", req.model, backend, release,
                             completion_sse(pieces, req.model, usage=usage), usage,
                             tenant)
        t0 = time.perf_counter()
        try:
            text, usage = _complete_call(backend, req.prompt, req.max_tokens,
                                         req.temperature, req.model)
        finally:
            release()
        _record("completions", req.model, time.perf_counter() - t0, usage, tenant)
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion", "model": req.model,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": usage,
        }

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest, tenant: str = Depends(auth)):
        quota.check(tenant)
        backend, release = _acquire(req.model)
        msgs = [m.model_dump() for m in req.messages]
        if req.stream:
            usage: dict = {}
            pieces = backend.stream_chat(msgs, req.max_tokens, req.temperature,
                                         req.model, usage)
            return _streamer("chat", req.model, backend, release,
                             chat_sse(pieces, req.model, usage=usage), usage, tenant)
        t0 = time.perf_counter()
        try:
            text, usage = _chat_call(backend, msgs, req.max_tokens,
                                     req.temperature, req.model)
        finally:
            release()
        _record("chat", req.model, time.perf_counter() - t0, usage, tenant)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion", "model": req.model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": usage,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return reg.render_prometheus()

    if snapshot_fn is not None:
        @app.get("/stats")
        def stats() -> dict[str, Any]:
            return snapshot_fn()

    return app


def router_from_backends(backends: list[tuple[str, str]]) -> Router:
    """Build a Router from (model, url) pairs — each becomes an HttpBackend.
    Repeating a model registers it as another replica (load-balanced)."""
    router = Router()
    for model, url in backends:
        router.register(HttpBackend(model, url))
    return router


def serve_gateway(backends: list[tuple[str, str]], *, host: str = "0.0.0.0",
                  port: int = 8080, api_keys: set[str] | None = None) -> None:
    """Run the gateway in front of remote serving units (uvicorn, blocks)."""
    import uvicorn

    router = router_from_backends(backends)
    app = create_gateway_app(router, api_keys=api_keys)
    print(f"[gateway] fronting {len(backends)} backend(s) "
          f"for models {router.models()} on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
