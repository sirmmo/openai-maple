"""Pydantic request models. Unknown fields are accepted and ignored so that
clients sending newer OpenAI parameters keep working."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="allow")


class ChatMessage(_Lenient):
    role: str
    content: str | list[Any] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None


class StreamOptions(_Lenient):
    include_usage: bool = False


class ChatCompletionRequest(_Lenient):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    n: int | None = 1
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    logprobs: bool | None = None
    response_format: dict[str, Any] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    user: str | None = None

    #: vLLM / SGLang convention: ``{"enable_thinking": false}``.
    chat_template_kwargs: dict[str, Any] | None = None
    #: OpenRouter convention; ``False`` strips reasoning from the response.
    include_reasoning: bool | None = None


class CompletionRequest(_Lenient):
    model: str | None = None
    prompt: str | list[str] | list[int] | list[list[int]] = ""
    suffix: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    n: int | None = 1
    echo: bool | None = False
    logprobs: int | None = None
    repetition_penalty: float | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    user: str | None = None


ErrorType = Literal["invalid_request_error", "server_error", "authentication_error", "rate_limit_error"]


def error_body(
    message: str,
    *,
    type_: ErrorType = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    return {"error": {"message": message, "type": type_, "param": param, "code": code}}
