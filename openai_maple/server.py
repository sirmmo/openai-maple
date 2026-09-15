"""FastAPI application exposing Maple over an OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__, translate
from .config import Settings
from .engine import (
    EngineError,
    EngineNotReady,
    EngineOverloaded,
    GenerationParams,
    MapleEngine,
    Piece,
    PromptTooLong,
)
from .schemas import ChatCompletionRequest, CompletionRequest, error_body
from .streaming import SSE_DONE, ChatChunker, CompletionChunker
from .translate import StreamParser, TranslationError

log = logging.getLogger(__name__)

_DONE = object()


def create_app(settings: Settings | None = None, engine: Any | None = None) -> FastAPI:
    """Build the app. ``engine`` may be injected for testing."""
    settings = settings or Settings.from_env()
    owns_engine = engine is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if owns_engine:
            app.state.engine = MapleEngine(
                settings.model_path,
                device=settings.device,
                dtype=settings.dtype,
                torch_threads=settings.torch_threads,
                pack_experts=settings.pack_experts,
                max_queue_depth=settings.max_queue_depth,
                queue_timeout=settings.queue_timeout,
                max_prompt_tokens=settings.max_prompt_tokens,
            )
            # Load in the background so /health answers while weights stream in.
            task = asyncio.create_task(asyncio.to_thread(_load, app.state.engine))
            app.state.load_task = task
        else:
            app.state.engine = engine
        yield

    app = FastAPI(title="openai-maple", version=__version__, lifespan=lifespan, docs_url="/docs")
    app.state.settings = settings

    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # -- errors ---------------------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        msg = first.get("msg", "invalid request")
        return JSONResponse(
            status_code=400, content=error_body(f"{loc}: {msg}" if loc else msg, param=loc or None)
        )

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail, headers=exc.headers)
        type_ = "authentication_error" if exc.status_code == 401 else "invalid_request_error"
        if exc.status_code >= 500:
            type_ = "server_error"
        return JSONResponse(
            status_code=exc.status_code, content=error_body(str(exc.detail), type_=type_), headers=exc.headers
        )

    # -- auth -----------------------------------------------------------------

    async def require_auth(authorization: str | None = Header(default=None)) -> None:
        if settings.api_key is None:
            return
        if authorization is None or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})
        if authorization[7:].strip() != settings.api_key:
            raise HTTPException(401, "invalid api key", headers={"WWW-Authenticate": "Bearer"})

    def get_engine(request: Request) -> Any:
        eng = request.app.state.engine
        if not getattr(eng, "ready", False):
            raise HTTPException(
                503, error_body("model is loading; retry shortly", type_="server_error", code="model_loading")
            )
        return eng

    def check_model(name: str | None) -> str:
        if settings.strict_model and name and name != settings.model_id:
            raise HTTPException(
                404,
                error_body(
                    f"model {name!r} not found; this server serves {settings.model_id!r}",
                    code="model_not_found",
                    param="model",
                ),
            )
        return settings.model_id

    # -- info -----------------------------------------------------------------

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        eng = request.app.state.engine
        return {
            "status": "ok" if getattr(eng, "ready", False) else "loading",
            "model": settings.model_id,
            "model_path": settings.model_path,
            "device": getattr(eng, "device", None),
            "dtype": getattr(eng, "dtype", None),
            "attention": getattr(eng, "attention_backend", None),
            "load_seconds": getattr(eng, "load_seconds", None),
            "queue_depth": getattr(eng, "queue_depth", 0),
            "active": getattr(eng, "active", 0),
            "version": __version__,
        }

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    async def list_models() -> dict[str, Any]:
        return {"object": "list", "data": [_model_card(settings)]}

    @app.get("/v1/models/{model_id}", dependencies=[Depends(require_auth)])
    async def get_model(model_id: str) -> dict[str, Any]:
        if model_id != settings.model_id and settings.strict_model:
            raise HTTPException(404, error_body(f"model {model_id!r} not found", code="model_not_found"))
        return _model_card(settings)

    # -- chat completions -----------------------------------------------------

    @app.post("/v1/chat/completions", dependencies=[Depends(require_auth)])
    async def chat_completions(body: ChatCompletionRequest, request: Request, eng: Any = Depends(get_engine)):
        model = check_model(body.model)
        _reject_unsupported_chat(body)
        params = _params(
            settings,
            body.temperature,
            body.top_p,
            body.top_k,
            body.max_completion_tokens or body.max_tokens,
            body.stop,
            body.seed,
            body.repetition_penalty,
        )

        thinking = settings.enable_thinking
        if body.chat_template_kwargs and "enable_thinking" in body.chat_template_kwargs:
            thinking = bool(body.chat_template_kwargs["enable_thinking"])
        include_reasoning = settings.include_reasoning
        if body.include_reasoning is not None:
            include_reasoning = body.include_reasoning

        try:
            messages = translate.normalize_messages([m.model_dump(exclude_none=True) for m in body.messages])
            tools = translate.normalize_tools(body.tools, body.tool_choice)
            prompt = eng.apply_chat_template(messages, tools=tools, enable_thinking=thinking)
            prompt_ids = eng.encode(prompt)
        except TranslationError as exc:
            raise HTTPException(400, error_body(str(exc), param=exc.param)) from exc
        except PromptTooLong as exc:
            raise HTTPException(
                400, error_body(str(exc), param="messages", code="context_length_exceeded")
            ) from exc
        except EngineNotReady as exc:
            raise HTTPException(503, error_body(str(exc), type_="server_error")) from exc

        if body.stream:
            include_usage = bool(body.stream_options and body.stream_options.include_usage)
            return StreamingResponse(
                _stream_chat(
                    request, eng, prompt_ids, params, model, thinking, include_reasoning, include_usage
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        raw, finish, n_tokens = await _collect(request, eng, prompt_ids, params)
        reasoning, content, calls = translate.parse_output(raw, thinking_open=thinking)
        usage = translate.usage_block(len(prompt_ids), n_tokens, eng.count_tokens(reasoning))
        return translate.chat_completion(
            model=model,
            reasoning=reasoning,
            content=content,
            tool_calls=calls,
            finish_reason=finish,
            usage=usage,
            include_reasoning=include_reasoning,
        )

    # -- legacy completions ---------------------------------------------------

    @app.post("/v1/completions", dependencies=[Depends(require_auth)])
    async def completions(body: CompletionRequest, request: Request, eng: Any = Depends(get_engine)):
        model = check_model(body.model)
        if body.n not in (None, 1):
            raise HTTPException(400, error_body("n must be 1", param="n"))
        if body.logprobs:
            raise HTTPException(400, error_body("logprobs are not supported", param="logprobs"))
        if body.echo:
            raise HTTPException(400, error_body("echo is not supported", param="echo"))
        if body.suffix:
            raise HTTPException(400, error_body("suffix is not supported", param="suffix"))
        params = _params(
            settings,
            body.temperature,
            body.top_p,
            body.top_k,
            body.max_tokens,
            body.stop,
            body.seed,
            body.repetition_penalty,
        )

        prompt = body.prompt
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise HTTPException(400, error_body("prompt must be a single string", param="prompt"))
            prompt = prompt[0]
        try:
            if isinstance(prompt, list):  # token ids
                prompt_ids = [int(t) for t in prompt]
                if len(prompt_ids) > eng.max_prompt_tokens:
                    raise PromptTooLong(
                        f"prompt is {len(prompt_ids)} tokens; the server limit is {eng.max_prompt_tokens}"
                    )
            else:
                prompt_ids = eng.encode(str(prompt))
        except PromptTooLong as exc:
            raise HTTPException(
                400, error_body(str(exc), param="prompt", code="context_length_exceeded")
            ) from exc
        if not prompt_ids:
            raise HTTPException(400, error_body("prompt must not be empty", param="prompt"))

        if body.stream:
            include_usage = bool(body.stream_options and body.stream_options.include_usage)
            return StreamingResponse(
                _stream_completion(request, eng, prompt_ids, params, model, include_usage),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        raw, finish, n_tokens = await _collect(request, eng, prompt_ids, params)
        usage = translate.usage_block(len(prompt_ids), n_tokens)
        return translate.text_completion(model=model, text=raw, finish_reason=finish, usage=usage)

    return app


# --- helpers -----------------------------------------------------------------


def _load(engine: MapleEngine) -> None:
    try:
        engine.start()
    except Exception:  # noqa: BLE001
        log.exception("model failed to load")


def _model_card(settings: Settings) -> dict[str, Any]:
    return {
        "id": settings.model_id,
        "object": "model",
        "created": 1767225600,
        "owned_by": "deepgrove",
        "root": settings.model_path,
    }


def _reject_unsupported_chat(body: ChatCompletionRequest) -> None:
    if body.n not in (None, 1):
        raise HTTPException(400, error_body("n must be 1", param="n"))
    if body.logprobs:
        raise HTTPException(400, error_body("logprobs are not supported", param="logprobs"))
    fmt = (body.response_format or {}).get("type")
    if fmt == "json_schema":
        raise HTTPException(
            400,
            error_body(
                "response_format json_schema is not supported; ask for JSON in the prompt",
                param="response_format",
            ),
        )
    if body.tool_choice not in (None, "none", "auto", "required") and not isinstance(body.tool_choice, dict):
        raise HTTPException(400, error_body("invalid tool_choice", param="tool_choice"))


def _params(
    settings: Settings, temperature, top_p, top_k, max_tokens, stop, seed, repetition_penalty
) -> GenerationParams:
    if max_tokens is not None and max_tokens < 1:
        raise HTTPException(400, error_body("max_tokens must be >= 1", param="max_tokens"))
    max_new = min(max_tokens or settings.max_new_tokens, settings.max_new_tokens_limit)
    if temperature is not None and temperature < 0:
        raise HTTPException(400, error_body("temperature must be >= 0", param="temperature"))
    if top_p is not None and not (0 < top_p <= 1):
        raise HTTPException(400, error_body("top_p must be in (0, 1]", param="top_p"))
    stops: list[str] = []
    if isinstance(stop, str):
        stops = [stop]
    elif stop:
        stops = [s for s in stop if s]
    return GenerationParams(
        max_new_tokens=max_new,
        temperature=settings.default_temperature if temperature is None else temperature,
        top_p=settings.default_top_p if top_p is None else top_p,
        top_k=settings.default_top_k if top_k is None else top_k,
        repetition_penalty=repetition_penalty or 1.0,
        seed=seed,
        stop=stops,
    )


async def _pieces(
    request: Request, eng: Any, prompt_ids: list[int], params: GenerationParams
) -> AsyncIterator[Piece]:
    """Run the engine's blocking generator in a thread; yield pieces on the loop.

    Client disconnects (detected while streaming) cancel the generation.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    cancel = threading.Event()

    def pump() -> None:
        try:
            for piece in eng.stream(prompt_ids, params, cancel):
                loop.call_soon_threadsafe(queue.put_nowait, piece)
        except BaseException as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    thread = threading.Thread(target=pump, name="maple-pump", daemon=True)
    thread.start()
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    cancel.set()
                continue
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        cancel.set()


def _engine_http_error(exc: BaseException) -> HTTPException:
    if isinstance(exc, EngineOverloaded):
        return HTTPException(
            503,
            error_body(str(exc), type_="rate_limit_error", code="engine_overloaded"),
            headers={"Retry-After": "5"},
        )
    if isinstance(exc, EngineNotReady):
        return HTTPException(503, error_body(str(exc), type_="server_error", code="model_loading"))
    if isinstance(exc, EngineError):
        return HTTPException(500, error_body(str(exc), type_="server_error"))
    log.exception("generation failed")
    return HTTPException(500, error_body(f"generation failed: {exc}", type_="server_error"))


async def _collect(
    request: Request, eng: Any, prompt_ids: list[int], params: GenerationParams
) -> tuple[str, str, int]:
    text: list[str] = []
    finish = "stop"
    n_tokens = 0
    try:
        async for piece in _pieces(request, eng, prompt_ids, params):
            text.append(piece.text)
            n_tokens = piece.completion_tokens
            if piece.finish_reason:
                finish = piece.finish_reason
    except HTTPException:
        raise
    except BaseException as exc:  # noqa: BLE001
        raise _engine_http_error(exc) from exc
    return "".join(text), finish, n_tokens


async def _stream_chat(
    request, eng, prompt_ids, params, model, thinking, include_reasoning, include_usage
) -> AsyncIterator[str]:
    chunker = ChatChunker(model)
    parser = StreamParser(thinking_open=thinking)
    finish = "stop"
    n_tokens = 0
    yield chunker.role()
    try:
        async for piece in _pieces(request, eng, prompt_ids, params):
            n_tokens = piece.completion_tokens
            if piece.finish_reason:
                finish = piece.finish_reason
                events = parser.flush()
            else:
                events = parser.feed(piece.text)
            for kind, payload in events:
                out = _chat_event(chunker, kind, payload, include_reasoning)
                if out:
                    yield out
    except BaseException as exc:  # noqa: BLE001
        err = _engine_http_error(exc)
        yield f"data: {_json(err.detail)}\n\n"
        yield SSE_DONE
        return

    if parser.tool_calls and finish == "stop":
        finish = "tool_calls"
    if finish == "cancelled":
        finish = "stop"
    yield chunker.finish(finish)
    if include_usage:
        yield chunker.usage(
            translate.usage_block(len(prompt_ids), n_tokens, eng.count_tokens(parser.reasoning))
        )
    yield SSE_DONE


def _chat_event(chunker: ChatChunker, kind: str, payload: Any, include_reasoning: bool) -> str | None:
    if kind == "reasoning":
        return chunker.reasoning(payload) if include_reasoning else None
    if kind == "content":
        return chunker.content(payload)
    if kind == "tool_call":
        return chunker.tool_call(payload)
    return None


async def _stream_completion(request, eng, prompt_ids, params, model, include_usage) -> AsyncIterator[str]:
    chunker = CompletionChunker(model)
    finish = "stop"
    n_tokens = 0
    try:
        async for piece in _pieces(request, eng, prompt_ids, params):
            n_tokens = piece.completion_tokens
            if piece.finish_reason:
                finish = piece.finish_reason
            elif piece.text:
                yield chunker.text(piece.text)
    except BaseException as exc:  # noqa: BLE001
        err = _engine_http_error(exc)
        yield f"data: {_json(err.detail)}\n\n"
        yield SSE_DONE
        return
    if finish == "cancelled":
        finish = "stop"
    yield chunker.text("", finish_reason=finish)
    if include_usage:
        yield chunker.usage(translate.usage_block(len(prompt_ids), n_tokens))
    yield SSE_DONE


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
