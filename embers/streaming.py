"""OpenAI-compatible SSE streaming — format helpers + text chunking.

Turns a stream of text pieces into the `text/event-stream` chunk protocol that
OpenAI clients (the `openai` SDK with stream=True, LangChain, chat UIs) expect:

    data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"Par"}}]}

    data: {...}

    data: [DONE]

NOTE: today the serving unit generates the full response (batch vLLM) and chunks
it out here — clients get the correct *protocol* but not lower time-to-first-token.
True incremental streaming needs vLLM's AsyncLLMEngine (a serving-unit change);
this is the compatibility layer that makes streaming clients work now.
"""
from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable, Iterator

DONE = "data: [DONE]\n\n"


def chunk_text(text: str) -> list[str]:
    """Split into streaming pieces, preserving whitespace (word + trailing ws)."""
    if not text:
        return []
    return re.findall(r"\S+\s*|\s+", text)


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _usage_chunk(cid: str, model: str, obj: str, usage: dict | None):
    """OpenAI `stream_options.include_usage` final chunk: empty choices + usage.
    Emitted only when `usage` is populated."""
    if usage:
        return _sse({"id": cid, "object": obj, "model": model,
                     "choices": [], "usage": usage})
    return None


def completion_sse(pieces: Iterable[str], model: str,
                   cid: str | None = None, usage: dict | None = None) -> Iterator[str]:
    cid = cid or f"cmpl-{uuid.uuid4().hex[:24]}"
    for p in pieces:
        yield _sse({"id": cid, "object": "text_completion", "model": model,
                    "choices": [{"index": 0, "text": p, "finish_reason": None}]})
    yield _sse({"id": cid, "object": "text_completion", "model": model,
                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]})
    chunk = _usage_chunk(cid, model, "text_completion", usage)
    if chunk:
        yield chunk
    yield DONE


def chat_sse(pieces: Iterable[str], model: str,
             cid: str | None = None, usage: dict | None = None) -> Iterator[str]:
    cid = cid or f"chatcmpl-{uuid.uuid4().hex[:24]}"
    # first chunk announces the assistant role (OpenAI convention)
    yield _sse({"id": cid, "object": "chat.completion.chunk", "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"},
                             "finish_reason": None}]})
    for p in pieces:
        yield _sse({"id": cid, "object": "chat.completion.chunk", "model": model,
                    "choices": [{"index": 0, "delta": {"content": p},
                                 "finish_reason": None}]})
    yield _sse({"id": cid, "object": "chat.completion.chunk", "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    chunk = _usage_chunk(cid, model, "chat.completion.chunk", usage)
    if chunk:
        yield chunk
    yield DONE


async def acompletion_sse(apieces, model: str, cid: str | None = None,
                          usage: dict | None = None):
    """Async version of completion_sse for a true token-incremental source.
    `usage` is read AFTER `apieces` is exhausted (the unit fills it at stream end)."""
    cid = cid or f"cmpl-{uuid.uuid4().hex[:24]}"
    async for p in apieces:
        yield _sse({"id": cid, "object": "text_completion", "model": model,
                    "choices": [{"index": 0, "text": p, "finish_reason": None}]})
    yield _sse({"id": cid, "object": "text_completion", "model": model,
                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]})
    chunk = _usage_chunk(cid, model, "text_completion", usage)
    if chunk:
        yield chunk
    yield DONE


async def achat_sse(apieces, model: str, cid: str | None = None,
                    usage: dict | None = None):
    cid = cid or f"chatcmpl-{uuid.uuid4().hex[:24]}"
    yield _sse({"id": cid, "object": "chat.completion.chunk", "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"},
                             "finish_reason": None}]})
    async for p in apieces:
        yield _sse({"id": cid, "object": "chat.completion.chunk", "model": model,
                    "choices": [{"index": 0, "delta": {"content": p},
                                 "finish_reason": None}]})
    yield _sse({"id": cid, "object": "chat.completion.chunk", "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    chunk = _usage_chunk(cid, model, "chat.completion.chunk", usage)
    if chunk:
        yield chunk
    yield DONE


def parse_sse_text(lines: Iterable[str], *, chat: bool,
                   usage: dict | None = None) -> Iterator[str]:
    """Pull text/content deltas out of an SSE stream (for proxying a remote unit's
    stream through the gateway). If `usage` is given, fill it from the usage chunk."""
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            return
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if usage is not None and obj.get("usage"):   # the include_usage final chunk
            usage.update(obj["usage"])
        try:
            choice = obj["choices"][0]
        except (KeyError, IndexError):
            continue
        piece = choice["delta"].get("content") if chat else choice.get("text")
        if piece:
            yield piece
