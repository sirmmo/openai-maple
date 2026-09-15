"""Tests against a running server serving the real model.

    OPENAI_MAPLE_URL=http://127.0.0.1:8000 pytest -m live tests/test_live.py

Only ``pytest`` and ``httpx`` are needed; each call allows minutes because CPU
inference of a reasoning trace is slow.
"""

from __future__ import annotations

import json
import os

import pytest

pytestmark = pytest.mark.live

BASE = os.environ.get("OPENAI_MAPLE_URL")
if not BASE:
    pytest.skip("set OPENAI_MAPLE_URL to run live tests", allow_module_level=True)

httpx = pytest.importorskip("httpx")
TIMEOUT = float(os.environ.get("OPENAI_MAPLE_TIMEOUT", "900"))


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE, timeout=TIMEOUT) as c:
        health = c.get("/health").json()
        if health.get("status") != "ok":
            pytest.skip(f"server not ready: {health}")
        yield c


def test_models(client):
    data = client.get("/v1/models").json()["data"]
    assert data and data[0]["object"] == "model"


def test_no_think_answer(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Reply with exactly one word: hello"}],
            "max_tokens": 16,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert r.status_code == 200, r.text
    msg = r.json()["choices"][0]["message"]
    assert "reasoning_content" not in msg
    assert "hello" in msg["content"].lower()


def test_reasoning_split(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
            "max_tokens": 1024,
            "temperature": 0,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    msg = body["choices"][0]["message"]
    assert msg["reasoning_content"]
    assert "<think>" not in (msg["content"] or "") and "</think>" not in (msg["content"] or "")
    if body["choices"][0]["finish_reason"] == "stop":
        assert "391" in msg["content"]
    assert body["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0


def test_streaming_shape(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 48,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ) as r:
        assert r.status_code == 200
        lines = [line for line in r.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text.strip()
    assert chunks[-1]["usage"]["completion_tokens"] > 0


def test_completions_raw(client):
    r = client.post(
        "/v1/completions",
        json={"prompt": "The capital of France is", "max_tokens": 6, "temperature": 0},
    )
    assert r.status_code == 200, r.text
    assert "Paris" in r.json()["choices"][0]["text"]
