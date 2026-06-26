"""Tests for the async serving path — true-streaming ModelUnit methods and the
async SSE helpers (mock mode, no GPU). Drives the coroutines with asyncio."""

import asyncio
import json

import pytest

from embers.server import ModelUnit
from embers.streaming import achat_sse, acompletion_sse, parse_sse_text


def collect(agen):
    async def run():
        return [x async for x in agen]
    return asyncio.run(run())


def test_astream_yields_pieces_mock():
    u = ModelUnit("m", mock=True)
    pieces = collect(u.astream("The capital of France is"))
    assert "".join(pieces) == "[mock completion for 'The capital of France is']"
    assert len(pieces) > 1                         # streamed in multiple pieces


def test_acomplete_joins_stream():
    u = ModelUnit("m", mock=True)
    text = asyncio.run(u.acomplete("hi"))
    assert text == "[mock completion for 'hi']"


def test_achat_stream_and_achat():
    u = ModelUnit("m", mock=True)
    pieces = collect(u.achat_stream([{"role": "user", "content": "yo"}]))
    assert "yo" in "".join(pieces)
    assert asyncio.run(u.achat([{"role": "user", "content": "yo"}])) \
        == "[mock chat reply to 'yo']"


def test_real_sync_methods_refuse():
    # the sync path is mock-only; a real unit must serve via the async endpoints
    u = ModelUnit("m", mock=False)
    u._loaded = True                               # pretend loaded, no real engine
    with pytest.raises(RuntimeError):
        u.generate("x")
    with pytest.raises(RuntimeError):
        u.chat([{"role": "user", "content": "x"}])


def test_acompletion_sse_shape():
    async def src():
        for p in ["Par", "is"]:
            yield p
    out = collect(acompletion_sse(src(), "m", cid="cmpl-1"))
    assert out[-1] == "data: [DONE]\n\n"
    first = json.loads(out[0][len("data: "):])
    assert first["choices"][0]["text"] == "Par"
    # reconstructs to the full text
    assert "".join(parse_sse_text("".join(out).splitlines(), chat=False)) == "Paris"


def test_achat_sse_role_then_deltas():
    async def src():
        for p in ["he", "llo"]:
            yield p
    out = collect(achat_sse(src(), "m"))
    first = json.loads(out[0][len("data: "):])
    assert first["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(parse_sse_text("".join(out).splitlines(), chat=True)) == "hello"
