"""Mapping between the OpenAI wire format and Maple's prompt/output format.

Maple uses the Qwen-style chat template: an assistant turn is
``<think>\\n{reasoning}\\n</think>\\n\\n{answer}`` and function calls are emitted as
``<tool_call>\\n{json}\\n</tool_call>`` blocks inside the answer. Because the
generation prompt already opens ``<think>``, the model's raw output *starts
inside* the reasoning block.
"""

from __future__ import annotations

import contextlib
import json
import re
import secrets
import time
from collections.abc import Iterator
from typing import Any

THINK_CLOSE = "</think>"
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"

_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class TranslationError(ValueError):
    """The request cannot be expressed to the model. Maps to HTTP 400."""

    def __init__(self, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.param = param


def new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(12)}"


def new_call_id() -> str:
    return f"call_{secrets.token_hex(12)}"


# --- request side ------------------------------------------------------------


def content_to_text(content: Any, *, param: str) -> str:
    """Flatten OpenAI ``content`` (string or parts list) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for i, part in enumerate(content):
            if isinstance(part, str):
                out.append(part)
                continue
            if not isinstance(part, dict):
                raise TranslationError(f"unsupported content part at index {i}", param)
            ptype = part.get("type")
            if ptype in ("text", "input_text"):
                out.append(str(part.get("text", "")))
            else:
                raise TranslationError(
                    f"content part type {ptype!r} is not supported; maple-preview is text-only", param
                )
        return "".join(out)
    raise TranslationError("content must be a string or a list of parts", param)


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce OpenAI messages into what the Jinja chat template expects."""
    if not messages:
        raise TranslationError("messages must not be empty", "messages")
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        role = msg.get("role")
        param = f"messages[{i}]"
        if role == "developer":
            role = "system"
        if role not in ("system", "user", "assistant", "tool"):
            raise TranslationError(f"unsupported role {role!r}", param)
        item: dict[str, Any] = {"role": role, "content": content_to_text(msg.get("content"), param=param)}

        if role == "assistant":
            reasoning = msg.get("reasoning_content") or msg.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                item["reasoning_content"] = reasoning
            calls = msg.get("tool_calls") or []
            norm_calls = []
            for call in calls:
                fn = call.get("function") or {}
                name = fn.get("name")
                if not name:
                    raise TranslationError("tool_calls[].function.name is required", param)
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    # The template prints strings verbatim, so a non-JSON
                    # argument string is passed through unchanged.
                    with contextlib.suppress(json.JSONDecodeError):
                        args = json.loads(args) if args.strip() else {}
                norm_calls.append({"type": "function", "function": {"name": name, "arguments": args}})
            if norm_calls:
                item["tool_calls"] = norm_calls
        elif role == "tool":
            if msg.get("name"):
                item["name"] = msg["name"]
            if msg.get("tool_call_id"):
                item["tool_call_id"] = msg["tool_call_id"]
        out.append(item)
    return out


def normalize_tools(tools: list[dict[str, Any]] | None, tool_choice: Any) -> list[dict[str, Any]] | None:
    """Return the tool list to hand to the template, honouring ``tool_choice``."""
    if not tools:
        return None
    if tool_choice == "none":
        return None
    selected = tools
    if isinstance(tool_choice, dict):
        wanted = (tool_choice.get("function") or {}).get("name")
        if wanted:
            selected = [t for t in tools if (t.get("function") or {}).get("name") == wanted]
            if not selected:
                raise TranslationError(f"tool_choice names unknown function {wanted!r}", "tool_choice")
    out = []
    for i, tool in enumerate(selected):
        if tool.get("type", "function") != "function" or not tool.get("function"):
            raise TranslationError("only function tools are supported", f"tools[{i}]")
        out.append({"type": "function", "function": tool["function"]})
    return out


# --- response side -----------------------------------------------------------


def split_reasoning(raw: str, *, thinking_open: bool) -> tuple[str, str]:
    """Split raw model output into ``(reasoning, content)``."""
    if not thinking_open:
        return "", raw.lstrip("\n")
    idx = raw.find(THINK_CLOSE)
    if idx == -1:
        return raw.strip("\n"), ""
    return raw[:idx].strip("\n"), raw[idx + len(THINK_CLOSE) :].lstrip("\n")


def extract_tool_calls(content: str) -> tuple[str, list[dict[str, Any]]]:
    """Pull ``<tool_call>`` blocks out of ``content``.

    Returns the remaining text and OpenAI-shaped tool call dicts. Blocks whose
    body is not valid JSON are left in the text untouched.
    """
    calls: list[dict[str, Any]] = []

    def repl(match: re.Match[str]) -> str:
        call = parse_tool_call_body(match.group(1))
        if call is None:
            return match.group(0)
        calls.append(call)
        return ""

    remaining = _TOOL_RE.sub(repl, content)
    return remaining.strip(), calls


def parse_tool_call_body(body: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None
    args = obj.get("arguments", {})
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return {
        "id": new_call_id(),
        "type": "function",
        "function": {"name": obj["name"], "arguments": args},
    }


class StreamParser:
    """Incremental version of :func:`split_reasoning` + :func:`extract_tool_calls`.

    ``feed`` yields ``("reasoning", text)``, ``("content", text)`` and
    ``("tool_call", call_dict)`` events. Text that might be the start of a
    marker is held back until it is disambiguated; ``flush`` releases whatever
    is left at end of stream.
    """

    def __init__(self, *, thinking_open: bool) -> None:
        self.phase = "reasoning" if thinking_open else "content"
        self.buf = ""
        self.in_tool = False
        self.reasoning = ""
        self.content = ""
        self.tool_calls: list[dict[str, Any]] = []
        self._content_started = False
        self._reasoning_started = False

    # Longest suffix of ``s`` that is a proper prefix of ``marker``.
    @staticmethod
    def _partial(s: str, marker: str) -> int:
        for n in range(min(len(s), len(marker) - 1), 0, -1):
            if marker.startswith(s[-n:]):
                return n
        return 0

    def feed(self, text: str) -> Iterator[tuple[str, Any]]:
        self.buf += text
        while self.buf:
            if self.phase == "reasoning":
                idx = self.buf.find(THINK_CLOSE)
                if idx != -1:
                    yield from self._emit_reasoning(self.buf[:idx].rstrip("\n"))
                    self.buf = self.buf[idx + len(THINK_CLOSE) :]
                    self.phase = "content"
                    continue
                hold = self._partial(self.buf, THINK_CLOSE)
                release = self.buf[: len(self.buf) - hold]
                # Trailing newlines are held too: the template puts "\n</think>"
                # after the reasoning, and the batch parser strips them.
                stripped = release.rstrip("\n")
                hold += len(release) - len(stripped)
                release, self.buf = stripped, self.buf[len(self.buf) - hold :]
                yield from self._emit_reasoning(release)
                return

            if self.in_tool:
                # The whole body stays in ``buf`` until the close tag arrives,
                # so a tag split across two chunks is still found.
                idx = self.buf.find(TOOL_CLOSE)
                if idx == -1:
                    return
                body = self.buf[:idx]
                self.buf = self.buf[idx + len(TOOL_CLOSE) :]
                self.in_tool = False
                call = parse_tool_call_body(body.strip())
                if call is None:
                    yield from self._emit_content(f"{TOOL_OPEN}{body}{TOOL_CLOSE}")
                else:
                    self.tool_calls.append(call)
                    yield ("tool_call", call)
                continue

            idx = self.buf.find(TOOL_OPEN)
            if idx != -1:
                yield from self._emit_content(self.buf[:idx])
                self.buf = self.buf[idx + len(TOOL_OPEN) :]
                self.in_tool = True
                continue
            hold = self._partial(self.buf, TOOL_OPEN)
            release, self.buf = self.buf[: len(self.buf) - hold], self.buf[len(self.buf) - hold :]
            yield from self._emit_content(release)
            return

    def flush(self) -> Iterator[tuple[str, Any]]:
        if self.in_tool:
            # Unterminated tool call: hand the raw text back as content.
            self.in_tool = False
            self.buf = f"{TOOL_OPEN}{self.buf}"
        if self.buf:
            release, self.buf = self.buf, ""
            if self.phase == "reasoning":
                yield from self._emit_reasoning(release.rstrip("\n"))
            else:
                yield from self._emit_content(release)

    def _emit_reasoning(self, text: str) -> Iterator[tuple[str, Any]]:
        if not self._reasoning_started:
            text = text.lstrip("\n")
            if not text:
                return
            self._reasoning_started = True
        self.reasoning += text
        yield ("reasoning", text)

    def _emit_content(self, text: str) -> Iterator[tuple[str, Any]]:
        if not self._content_started:
            text = text.lstrip("\n")
            if not text:
                return
            self._content_started = True
        self.content += text
        yield ("content", text)


def parse_output(raw: str, *, thinking_open: bool) -> tuple[str, str, list[dict[str, Any]]]:
    """Non-streaming parse: ``(reasoning, content, tool_calls)``."""
    reasoning, content = split_reasoning(raw, thinking_open=thinking_open)
    content, calls = extract_tool_calls(content)
    return reasoning, content, calls


def usage_block(prompt_tokens: int, completion_tokens: int, reasoning_tokens: int = 0) -> dict[str, Any]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }


def chat_completion(
    *,
    model: str,
    reasoning: str,
    content: str,
    tool_calls: list[dict[str, Any]],
    finish_reason: str,
    usage: dict[str, Any],
    include_reasoning: bool = True,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content if content else None}
    if include_reasoning and reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
        if finish_reason == "stop":
            finish_reason = "tool_calls"
    if finish_reason == "cancelled":
        finish_reason = "stop"
    return {
        "id": new_id("chatcmpl"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": finish_reason}],
        "usage": usage,
    }


def text_completion(*, model: str, text: str, finish_reason: str, usage: dict[str, Any]) -> dict[str, Any]:
    if finish_reason == "cancelled":
        finish_reason = "stop"
    return {
        "id": new_id("cmpl"),
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}],
        "usage": usage,
    }
