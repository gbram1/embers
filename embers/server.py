"""Phase 1 serving unit — one vLLM model behind an OpenAI-compatible HTTP API.

Grows the Phase-0 slice into the robust serving unit: OpenAI-compatible
`/v1/completions` + `/v1/chat/completions`, liveness/readiness probes, and config
knobs (dtype, max-model-len, gpu-memory-utilization). This is the unit the
gateway (Phase 2) fronts and the autoscaler (Phase 4) starts/stops; `load()` is
where snapshot/restore (Phase 5) plugs in.

  embers serve <model>                       # start the unit (uvicorn)
  POST /v1/completions       {model,prompt,...}
  POST /v1/chat/completions  {model,messages,...}
  GET  /health   (liveness)   GET /ready (model loaded)   GET /v1/models

--mock loads no model (no GPU/vLLM) so the API + loop are testable on macOS.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


class ModelUnit:
    """Holds the vLLM AsyncLLMEngine for one model — async so streaming is true
    token-by-token. Loading is explicit (`load()`) so start/stop and
    scale-to-zero are meaningful; the GPU-holding pid is what cuda-checkpoint
    parks (AsyncLLMEngine runs the GPU in a worker child).

    Two interfaces:
      * async (acomplete/astream/achat/achat_stream) — the real serving endpoints
      * sync (generate/chat/stream/chat_stream) — MOCK only, for the in-process
        LocalBackend path (tests / `embers up --mock`)."""

    def __init__(self, model: str, *, mock: bool = False,
                 adapters: dict[str, str] | None = None, **engine_kwargs):
        self.model = model
        self.mock = mock
        # LoRA adapters served off this ONE base model: name -> path. Each gets a
        # stable int id (vLLM's LoRARequest needs one). Many cheap adapters share
        # the base's GPU memory — the multi-tenant fine-tuning win.
        self.adapters = dict(adapters or {})
        self._lora_ids = {name: i + 1 for i, name in enumerate(self.adapters)}
        if self.adapters:
            engine_kwargs.setdefault("enable_lora", True)
            engine_kwargs.setdefault("max_loras", max(1, len(self.adapters)))
        self.engine_kwargs = engine_kwargs
        self._engine = None        # AsyncLLMEngine in real mode
        self._loaded = False
        self.load_seconds: float | None = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    def served_models(self) -> list[str]:
        """The base model + every adapter — all selectable via the `model` field."""
        return [self.model, *self.adapters]

    def lora_for(self, model_name: str | None) -> str | None:
        """Resolve a request's `model` to an adapter name (or None = base).
        Raises KeyError for a model this unit doesn't serve."""
        if not model_name or model_name == self.model:
            return None
        if model_name in self.adapters:
            return model_name
        raise KeyError(model_name)

    def load(self) -> None:
        if self._loaded:
            return
        t0 = time.perf_counter()
        if not self.mock:
            from vllm import AsyncEngineArgs, AsyncLLMEngine

            self._engine = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(model=self.model, disable_log_requests=True,
                                **self.engine_kwargs))
        self.load_seconds = time.perf_counter() - t0
        self._loaded = True

    def _lora_request(self, lora: str | None):
        """Build a vLLM LoRARequest for an adapter name, or None for the base."""
        if not lora:
            return None
        from vllm.lora.request import LoRARequest

        return LoRARequest(lora_name=lora, lora_int_id=self._lora_ids[lora],
                           lora_path=self.adapters[lora])

    def _sampling(self, max_tokens: int, temperature: float):
        from vllm import SamplingParams

        return SamplingParams(max_tokens=max_tokens, temperature=temperature)

    # --- token usage ------------------------------------------------------

    def _mock_completion_text(self, prompt: str, lora: str | None) -> str:
        via = f" via {lora}" if lora else ""
        return f"[mock completion{via} for {prompt!r}]"

    def _mock_chat_text(self, messages: list[dict], lora: str | None) -> str:
        last = messages[-1]["content"] if messages else ""
        via = f" via {lora}" if lora else ""
        return f"[mock chat reply{via} to {last!r}]"

    @staticmethod
    def _usage(prompt_tokens: int, completion_tokens: int) -> dict:
        return {"prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens}

    @staticmethod
    def _mock_count(text: str) -> int:
        return len(text.split())          # whitespace words ≈ tokens (mock only)

    @staticmethod
    def _usage_from_output(out) -> tuple[int, int]:
        """(prompt_tokens, completion_tokens) from a vLLM RequestOutput."""
        if out is None:
            return 0, 0
        return len(out.prompt_token_ids or []), len(out.outputs[0].token_ids or [])

    # --- async: true token-incremental (the real serving endpoints) -------

    async def astream(self, prompt: str, max_tokens: int = 32,
                      temperature: float = 0.0, lora: str | None = None,
                      usage: dict | None = None):
        """Yield text deltas AS they're generated (true streaming). `lora` selects
        a LoRA adapter (None = base model). If `usage` is given, it's filled with
        token counts once the stream ends (for metering streaming requests)."""
        if not self._loaded:
            self.load()
        if self.mock:
            from embers.streaming import chunk_text
            text = self._mock_completion_text(prompt, lora)
            for piece in chunk_text(text):
                yield piece
            if usage is not None:
                usage.update(self._usage(self._mock_count(prompt),
                                         self._mock_count(text)))
            return
        sp = self._sampling(max_tokens, temperature)
        prev, out = "", None
        async for out in self._engine.generate(
                prompt, sp, request_id=uuid.uuid4().hex,
                lora_request=self._lora_request(lora)):
            text = out.outputs[0].text
            delta = text[len(prev):]
            prev = text
            if delta:
                yield delta
        if usage is not None:
            pt, ct = self._usage_from_output(out)
            usage.update(self._usage(pt, ct))

    async def acomplete(self, prompt: str, max_tokens: int = 32,
                        temperature: float = 0.0, lora: str | None = None) -> str:
        return "".join([p async for p in
                        self.astream(prompt, max_tokens, temperature, lora)])

    async def _chat_prompt(self, messages: list[dict]) -> str:
        # apply the model's chat template (GPU-validate the exact call)
        tok = await self._engine.get_tokenizer()
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)

    async def achat_stream(self, messages: list[dict], max_tokens: int = 32,
                           temperature: float = 0.0, lora: str | None = None,
                           usage: dict | None = None):
        if not self._loaded:
            self.load()
        if self.mock:
            from embers.streaming import chunk_text
            text = self._mock_chat_text(messages, lora)
            for piece in chunk_text(text):
                yield piece
            if usage is not None:
                pt = sum(self._mock_count(m.get("content", "")) for m in messages)
                usage.update(self._usage(pt, self._mock_count(text)))
            return
        prompt = await self._chat_prompt(messages)
        async for piece in self.astream(prompt, max_tokens, temperature, lora, usage):
            yield piece

    async def achat(self, messages: list[dict], max_tokens: int = 32,
                   temperature: float = 0.0, lora: str | None = None) -> str:
        return "".join([p async for p in
                        self.achat_stream(messages, max_tokens, temperature, lora)])

    # --- non-streaming with real token usage (for metering/billing) -------

    async def acomplete_usage(self, prompt: str, max_tokens: int = 32,
                              temperature: float = 0.0, lora: str | None = None):
        """Return (text, usage). Real usage from vLLM token_ids; mock approximates
        by word count. This is the source of truth the gateway meters on."""
        if not self._loaded:
            self.load()
        if self.mock:
            text = self._mock_completion_text(prompt, lora)
            return text, self._usage(self._mock_count(prompt), self._mock_count(text))
        sp = self._sampling(max_tokens, temperature)
        final = None
        async for out in self._engine.generate(
                prompt, sp, request_id=uuid.uuid4().hex,
                lora_request=self._lora_request(lora)):
            final = out
        text = final.outputs[0].text if final else ""
        pt, ct = self._usage_from_output(final)
        return text, self._usage(pt, ct)

    async def achat_usage(self, messages: list[dict], max_tokens: int = 32,
                          temperature: float = 0.0, lora: str | None = None):
        if not self._loaded:
            self.load()
        if self.mock:
            text = self._mock_chat_text(messages, lora)
            pt = sum(self._mock_count(m.get("content", "")) for m in messages)
            return text, self._usage(pt, self._mock_count(text))
        prompt = await self._chat_prompt(messages)
        return await self.acomplete_usage(prompt, max_tokens, temperature, lora)

    # --- sync: MOCK only (in-process LocalBackend path) -------------------

    def generate(self, prompt: str, max_tokens: int = 32,
                 temperature: float = 0.0, lora: str | None = None) -> str:
        if not self._loaded:
            self.load()
        if not self.mock:
            raise RuntimeError("real ModelUnit serves via async endpoints "
                               "(acomplete); sync generate is mock-only")
        via = f" via {lora}" if lora else ""
        return f"[mock completion{via} for {prompt!r}]"

    def chat(self, messages: list[dict], max_tokens: int = 32,
             temperature: float = 0.0, lora: str | None = None) -> str:
        if not self._loaded:
            self.load()
        if not self.mock:
            raise RuntimeError("real ModelUnit serves via async endpoints (achat)")
        last = messages[-1]["content"] if messages else ""
        via = f" via {lora}" if lora else ""
        return f"[mock chat reply{via} to {last!r}]"

    def stream(self, prompt: str, max_tokens: int = 32, temperature: float = 0.0,
               lora: str | None = None, usage: dict | None = None):
        from embers.streaming import chunk_text
        text = self.generate(prompt, max_tokens, temperature, lora)
        yield from chunk_text(text)
        if usage is not None:
            usage.update(self._usage(self._mock_count(prompt),
                                     self._mock_count(text)))

    def chat_stream(self, messages: list[dict], max_tokens: int = 32,
                    temperature: float = 0.0, lora: str | None = None,
                    usage: dict | None = None):
        from embers.streaming import chunk_text
        text = self.chat(messages, max_tokens, temperature, lora)
        yield from chunk_text(text)
        if usage is not None:
            pt = sum(self._mock_count(m.get("content", "")) for m in messages)
            usage.update(self._usage(pt, self._mock_count(text)))


# --- OpenAI-compatible schemas (minimal subset) ---------------------------

class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = 32
    temperature: float = 0.0
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 32
    temperature: float = 0.0
    stream: bool = False


def create_app(unit: ModelUnit) -> FastAPI:
    """Build the FastAPI app for one serving unit. Separated from `serve()` so
    it's testable with fastapi.testclient without binding a socket."""
    app = FastAPI(title="embers serving unit")

    @app.get("/health")
    def health() -> dict[str, Any]:  # liveness — process is up
        return {"status": "ok", "model": unit.model}

    @app.get("/ready")
    def ready() -> dict[str, Any]:  # readiness — model is loaded & servable
        if not unit.loaded:
            raise HTTPException(503, "model not loaded")
        return {"status": "ready", "model": unit.model,
                "load_seconds": unit.load_seconds}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list",
                "data": [{"id": m, "object": "model"} for m in unit.served_models()]}

    def _resolve(model_name: str | None) -> str | None:
        try:
            return unit.lora_for(model_name)
        except KeyError:
            raise HTTPException(404, f"model {model_name!r} not served here")

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        lora = _resolve(req.model)
        if req.stream:    # TRUE token-incremental SSE
            from embers.streaming import acompletion_sse
            usage: dict = {}
            pieces = unit.astream(req.prompt, req.max_tokens, req.temperature,
                                  lora, usage)
            return StreamingResponse(
                acompletion_sse(pieces, req.model or unit.model, usage=usage),
                media_type="text/event-stream")
        text, usage = await unit.acomplete_usage(req.prompt, req.max_tokens,
                                                 req.temperature, lora)
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "model": req.model or unit.model,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": usage,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        lora = _resolve(req.model)
        msgs = [m.model_dump() for m in req.messages]
        if req.stream:
            from embers.streaming import achat_sse
            usage: dict = {}
            pieces = unit.achat_stream(msgs, req.max_tokens, req.temperature,
                                       lora, usage)
            return StreamingResponse(
                achat_sse(pieces, req.model or unit.model, usage=usage),
                media_type="text/event-stream")
        text, usage = await unit.achat_usage(msgs, req.max_tokens,
                                             req.temperature, lora)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "model": req.model or unit.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": usage,
        }

    return app


def serve(model: str, *, host: str = "0.0.0.0", port: int = 8000,
          mock: bool = False, eager_load: bool = True,
          adapters: dict[str, str] | None = None, **engine_kwargs) -> None:
    """Start the serving unit (uvicorn) and block. eager_load=True loads the
    model now; False defers to first request (cold-on-demand, the Phase-4
    autoscaler behaviour). `adapters` (name->path) serves LoRA adapters off this
    base model — selectable per-request via the `model` field."""
    import uvicorn

    unit = ModelUnit(model, mock=mock, adapters=adapters, **engine_kwargs)
    if eager_load:
        print(f"[server] loading {model} (mock={mock}) ...")
        unit.load()
        print(f"[server] loaded in {unit.load_seconds:.2f}s")

    app = create_app(unit)
    print(f"[server] {model} serving on http://{host}:{port} "
          f"(/v1/completions, /v1/chat/completions, /health, /ready)")
    uvicorn.run(app, host=host, port=port, log_level="info")
