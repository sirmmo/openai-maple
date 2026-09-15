from __future__ import annotations

import json

import pytest


def _sse_events(text: str) -> list[dict | str]:
    out: list[dict | str] = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        out.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return out


def test_health_and_models(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    r = client.get("/v1/models")
    assert r.json()["data"][0]["id"] == "maple-preview"
    assert client.get("/v1/models/maple-preview").status_code == 200


def test_chat_completion_splits_reasoning(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "maple-preview", "messages": [{"role": "user", "content": "17*23?"}]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    msg = body["choices"][0]["message"]
    assert msg["content"] == "The answer is 391."
    assert msg["reasoning_content"] == "Let me think.\nDone."
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == len(client.engine.calls[0]["prompt_ids"])
    assert body["usage"]["completion_tokens"] > 0
    assert body["usage"]["completion_tokens_details"]["reasoning_tokens"] == 4


def test_chat_completion_uses_template_and_defaults(client, settings):
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]},
    )
    tpl = client.engine.templates[0]
    assert [m["role"] for m in tpl["messages"]] == ["system", "user"]
    assert tpl["enable_thinking"] is True
    params = client.engine.calls[0]["params"]
    assert params.temperature == settings.default_temperature
    assert params.max_new_tokens == settings.max_new_tokens


def test_chat_template_kwargs_disable_thinking(client):
    client.engine.output = "\nDirect answer."
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "u"}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert client.engine.templates[0]["enable_thinking"] is False
    msg = r.json()["choices"][0]["message"]
    assert msg["content"] == "Direct answer."
    assert "reasoning_content" not in msg


def test_include_reasoning_false_strips_reasoning(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "u"}], "include_reasoning": False},
    )
    assert "reasoning_content" not in r.json()["choices"][0]["message"]


def test_tool_calls_non_streaming(client):
    client.engine.output = (
        "I should call it.\n</think>\n\n<tool_call>\n"
        '{"name": "get_weather", "arguments": {"city": "Lagos"}}\n</tool_call>'
    )
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "weather in Lagos"}], "tools": tools},
    )
    body = r.json()
    assert client.engine.templates[0]["tools"] == tools
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Lagos"}


def test_streaming_chat(client):
    client.engine.output = 'plan\n</think>\n\nHi! <tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>'
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "u"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(r.text)
    assert events[-1] == "[DONE]"
    chunks = [e for e in events if isinstance(e, dict)]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in chunks if c["choices"])
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert reasoning == "plan"
    assert content.strip() == "Hi!"
    tool_chunks = [c for c in chunks if c["choices"] and "tool_calls" in c["choices"][0]["delta"]]
    assert tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "f"
    finish = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert finish[-1]["choices"][0]["finish_reason"] == "tool_calls"
    usage = [c for c in chunks if not c["choices"]]
    assert usage and usage[0]["usage"]["prompt_tokens"] > 0
    ids = {c["id"] for c in chunks}
    assert len(ids) == 1


def test_streaming_length_finish(client):
    client.engine.output = "a" * 500
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "u"}], "stream": True, "max_tokens": 3},
    )
    chunks = [e for e in _sse_events(r.text) if isinstance(e, dict)]
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_stop_sequence_is_trimmed(client):
    client.engine.output = "\n</think>\n\nOne. Two. Three."
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "u"}], "stop": ["Two"]},
    )
    assert r.json()["choices"][0]["message"]["content"] == "One."
    assert client.engine.calls[0]["params"].stop == ["Two"]


def test_completions_endpoint_raw(client):
    client.engine.output = " world"
    r = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 5})
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == " world"
    assert client.engine.calls[0]["prompt_ids"] == [ord(c) for c in "hello"]


def test_completions_streaming(client):
    client.engine.output = "abcdefghij"
    r = client.post("/v1/completions", json={"prompt": "x", "stream": True})
    events = _sse_events(r.text)
    text = "".join(e["choices"][0]["text"] for e in events if isinstance(e, dict) and e["choices"])
    assert text == "abcdefghij"
    assert events[-1] == "[DONE]"


@pytest.mark.parametrize(
    "body,param",
    [
        ({"messages": [{"role": "user", "content": "u"}], "n": 2}, "n"),
        ({"messages": [{"role": "user", "content": "u"}], "logprobs": True}, "logprobs"),
        ({"messages": [{"role": "user", "content": "u"}], "temperature": -1}, "temperature"),
        ({"messages": [{"role": "user", "content": "u"}], "max_tokens": 0}, "max_tokens"),
        (
            {"messages": [{"role": "user", "content": "u"}], "response_format": {"type": "json_schema"}},
            "response_format",
        ),
        ({"messages": []}, "messages"),
        ({"messages": [{"role": "robot", "content": "u"}]}, "messages[0]"),
    ],
)
def test_bad_requests(client, body, param):
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["param"] == param


def test_prompt_too_long(client):
    client.engine.max_prompt_tokens = 10
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x" * 50}]})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "context_length_exceeded"


def test_overloaded_returns_503(client):
    client.engine.overloaded = True
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "u"}]})
    assert r.status_code == 503
    assert r.headers["retry-after"] == "5"
    assert r.json()["error"]["code"] == "engine_overloaded"


def test_not_ready_returns_503(client):
    client.engine.ready = False
    assert client.get("/health").json()["status"] == "loading"
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "u"}]})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "model_loading"


def test_engine_error_returns_500(client):
    from openai_maple.engine import EngineError

    client.engine.raise_on_run = EngineError("boom")
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "u"}]})
    assert r.status_code == 500
    assert r.json()["error"]["type"] == "server_error"


def test_streaming_error_is_sent_in_band(client):
    from openai_maple.engine import EngineError

    client.engine.raise_on_run = EngineError("boom")
    r = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "u"}], "stream": True}
    )
    assert r.status_code == 200
    events = _sse_events(r.text)
    assert any(isinstance(e, dict) and "error" in e for e in events)
    assert events[-1] == "[DONE]"


def test_api_key_required(engine):
    from fastapi.testclient import TestClient

    from openai_maple.config import Settings
    from openai_maple.server import create_app

    app = create_app(settings=Settings(api_key="sk-test", allowed_origins=[]), engine=engine)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer sk-test"}).status_code == 200


def test_strict_model(engine):
    from fastapi.testclient import TestClient

    from openai_maple.config import Settings
    from openai_maple.server import create_app

    app = create_app(settings=Settings(strict_model=True, allowed_origins=[]), engine=engine)
    with TestClient(app) as c:
        r = c.post(
            "/v1/chat/completions", json={"model": "gpt-4o", "messages": [{"role": "user", "content": "u"}]}
        )
        assert r.status_code == 404
        assert c.get("/v1/models/gpt-4o").status_code == 404


def test_openai_sdk_round_trip(client):
    """The official client parses our responses, including reasoning_content."""
    openai = pytest.importorskip("openai")
    sdk = openai.OpenAI(base_url="http://testserver/v1", api_key="x", http_client=client)
    resp = sdk.chat.completions.create(
        model="maple-preview", messages=[{"role": "user", "content": "17*23?"}]
    )
    assert resp.choices[0].message.content == "The answer is 391."
    assert resp.choices[0].message.model_extra["reasoning_content"] == "Let me think.\nDone."

    stream = sdk.chat.completions.create(
        model="maple-preview", messages=[{"role": "user", "content": "17*23?"}], stream=True
    )
    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
    assert text == "The answer is 391."
