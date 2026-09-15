"""Server-sent-event framing for streamed responses."""

from __future__ import annotations

import json
import time
from typing import Any


def sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}\n\n"


SSE_DONE = "data: [DONE]\n\n"


class ChatChunker:
    """Builds ``chat.completion.chunk`` payloads sharing one id/created."""

    def __init__(self, model: str) -> None:
        self.id = f"chatcmpl-{time.time_ns():x}"
        self.created = int(time.time())
        self.model = model
        self._tool_index = 0

    def _chunk(self, delta: dict[str, Any], finish_reason: str | None = None) -> str:
        return sse(
            {
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}],
            }
        )

    def role(self) -> str:
        return self._chunk({"role": "assistant", "content": ""})

    def content(self, text: str) -> str:
        return self._chunk({"content": text})

    def reasoning(self, text: str) -> str:
        return self._chunk({"reasoning_content": text})

    def tool_call(self, call: dict[str, Any]) -> str:
        index = self._tool_index
        self._tool_index += 1
        return self._chunk(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["function"]["name"],
                            "arguments": call["function"]["arguments"],
                        },
                    }
                ]
            }
        )

    def finish(self, reason: str) -> str:
        return self._chunk({}, finish_reason=reason)

    def usage(self, usage: dict[str, Any]) -> str:
        return sse(
            {
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [],
                "usage": usage,
            }
        )


class CompletionChunker:
    def __init__(self, model: str) -> None:
        self.id = f"cmpl-{time.time_ns():x}"
        self.created = int(time.time())
        self.model = model

    def text(self, text: str, finish_reason: str | None = None) -> str:
        return sse(
            {
                "id": self.id,
                "object": "text_completion",
                "created": self.created,
                "model": self.model,
                "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}],
            }
        )

    def usage(self, usage: dict[str, Any]) -> str:
        return sse(
            {
                "id": self.id,
                "object": "text_completion",
                "created": self.created,
                "model": self.model,
                "choices": [],
                "usage": usage,
            }
        )
