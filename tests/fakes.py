"""A scripted stand-in for :class:`MapleEngine` so the HTTP surface is testable
without the 40GB of weights."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

from openai_maple.engine import EngineOverloaded, GenerationParams, Piece, PromptTooLong


class FakeEngine:
    def __init__(self, output: str = "<think>", finish: str = "stop") -> None:
        self.ready = True
        self.device = "cpu"
        self.dtype = "bfloat16"
        self.attention_backend = "fake"
        self.load_seconds = 0.0
        self.queue_depth = 0
        self.active = 0
        self.max_prompt_tokens = 4096
        self.output = output
        self.finish = finish
        self.pieces: list[str] | None = None  # explicit chunking, if set
        self.overloaded = False
        self.raise_on_run: Exception | None = None
        self.calls: list[dict[str, Any]] = []
        self.templates: list[dict[str, Any]] = []
        self.honor_stop = True

    # -- prompt helpers -------------------------------------------------------

    def apply_chat_template(self, messages, *, tools=None, enable_thinking=True) -> str:
        self.templates.append({"messages": messages, "tools": tools, "enable_thinking": enable_thinking})
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        text += "<|im_start|>assistant\n<think>\n"
        if not enable_thinking:
            text += "\n</think>\n\n"
        return text

    def encode(self, text: str) -> list[int]:
        ids = [ord(c) for c in text]
        if len(ids) > self.max_prompt_tokens:
            raise PromptTooLong(f"prompt is {len(ids)} tokens; the server limit is {self.max_prompt_tokens}")
        return ids

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    # -- generation -----------------------------------------------------------

    def stream(self, prompt_ids: list[int], params: GenerationParams, cancel=None) -> Iterator[Piece]:
        cancel = cancel or threading.Event()
        self.calls.append({"prompt_ids": prompt_ids, "params": params})
        if self.overloaded:
            raise EngineOverloaded("engine queue is full (16 requests waiting); retry shortly")
        if self.raise_on_run:
            raise self.raise_on_run
        pieces = self.pieces if self.pieces is not None else _chunk(self.output, 5)
        emitted = ""
        n = 0
        finish = self.finish
        for text in pieces:
            if cancel.is_set():
                finish = "cancelled"
                break
            n += 1
            if self.honor_stop and params.stop:
                candidate = emitted + text
                for s in params.stop:
                    idx = candidate.find(s)
                    if idx != -1:
                        tail = candidate[len(emitted) : idx]
                        if tail:
                            yield Piece(tail, n)
                        yield Piece("", n, finish_reason="stop")
                        return
            emitted += text
            yield Piece(text, n)
            if n >= params.max_new_tokens:
                finish = "length"
                break
        yield Piece("", n, finish_reason=finish)


def _chunk(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]
