"""GPU-free tests for the Phase 0 serving unit (mock mode — no vLLM)."""

from embers.server import ModelUnit


def test_unit_starts_unloaded_then_loads():
    u = ModelUnit("m", mock=True)
    assert u.loaded is False
    u.load()
    assert u.loaded is True
    assert u.load_seconds is not None


def test_generate_autoloads_and_returns_text():
    u = ModelUnit("m", mock=True)
    out = u.generate("hello")
    assert u.loaded is True
    assert "hello" in out  # mock echoes the prompt


def test_load_is_idempotent():
    u = ModelUnit("m", mock=True)
    u.load()
    first = u.load_seconds
    u.load()
    assert u.load_seconds == first  # second load() is a no-op


def test_chat_mock_echoes_last_message():
    u = ModelUnit("m", mock=True)
    out = u.chat([{"role": "user", "content": "ping"}])
    assert "ping" in out


def test_serve_wires_app_and_runs_uvicorn(monkeypatch):
    """serve() must build the app and hand it to uvicorn — without binding a
    socket or needing a GPU (mock=True). Catches wiring bugs in serve()."""
    import embers.server as srv

    captured = {}
    monkeypatch.setattr("uvicorn.run",
                        lambda app, **kw: captured.update(app=app, kw=kw))
    srv.serve("m", mock=True, port=1234, host="127.0.0.1")
    assert captured["app"] is not None
    assert captured["kw"]["port"] == 1234
    assert captured["kw"]["host"] == "127.0.0.1"


def test_serve_eager_load_false_defers(monkeypatch):
    import embers.server as srv

    monkeypatch.setattr("uvicorn.run", lambda app, **kw: None)
    # eager_load=False must not raise and must not pre-load (cold-on-demand)
    srv.serve("m", mock=True, eager_load=False)
