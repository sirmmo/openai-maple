from __future__ import annotations

import json

import pytest

from openai_maple.translate import (
    StreamParser,
    TranslationError,
    extract_tool_calls,
    normalize_messages,
    normalize_tools,
    parse_output,
    split_reasoning,
)


def test_split_reasoning_with_close_tag():
    reasoning, content = split_reasoning("\nthink hard\n</think>\n\nanswer", thinking_open=True)
    assert reasoning == "think hard"
    assert content == "answer"


def test_split_reasoning_unterminated_is_all_reasoning():
    reasoning, content = split_reasoning("still thinking", thinking_open=True)
    assert reasoning == "still thinking"
    assert content == ""


def test_split_reasoning_disabled():
    assert split_reasoning("\nhi", thinking_open=False) == ("", "hi")


def test_extract_tool_calls():
    text = 'Sure.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Lagos"}}\n</tool_call>'
    content, calls = extract_tool_calls(text)
    assert content == "Sure."
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["id"].startswith("call_")
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Lagos"}


def test_extract_tool_calls_keeps_invalid_json_as_text():
    text = "<tool_call>\nnot json\n</tool_call>"
    content, calls = extract_tool_calls(text)
    assert calls == []
    assert content == text


def test_parse_output_end_to_end():
    raw = 'plan\n</think>\n\n<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>'
    reasoning, content, calls = parse_output(raw, thinking_open=True)
    assert reasoning == "plan"
    assert content == ""
    assert calls[0]["function"]["name"] == "f"


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 1000])
def test_stream_parser_matches_batch_parse(chunk):
    raw = (
        "\nthinking about </thin... no\n</think>\n\nHello <tool"
        '_call>\n{"name": "f", "arguments": {"x": 1}}\n</tool_call> tail'
    )
    parser = StreamParser(thinking_open=True)
    events = []
    for i in range(0, len(raw), chunk):
        events.extend(parser.feed(raw[i : i + chunk]))
    events.extend(parser.flush())

    reasoning = "".join(p for k, p in events if k == "reasoning")
    content = "".join(p for k, p in events if k == "content")
    calls = [p for k, p in events if k == "tool_call"]
    b_reasoning, b_content, b_calls = parse_output(raw, thinking_open=True)
    assert reasoning == b_reasoning
    assert content.strip() == b_content
    assert [c["function"] for c in calls] == [c["function"] for c in b_calls]
    assert parser.reasoning == reasoning
    assert parser.tool_calls == calls


def test_stream_parser_unterminated_tool_call_flushes_as_content():
    parser = StreamParser(thinking_open=False)
    events = list(parser.feed('<tool_call>\n{"name": "f"'))
    assert events == []
    events = list(parser.flush())
    assert events == [("content", '<tool_call>\n{"name": "f"')]


def test_stream_parser_thinking_disabled_streams_content_immediately():
    parser = StreamParser(thinking_open=False)
    assert list(parser.feed("\nhi")) == [("content", "hi")]


def test_normalize_messages_flattens_parts_and_developer_role():
    out = normalize_messages(
        [
            {"role": "developer", "content": [{"type": "text", "text": "be brief"}]},
            {"role": "user", "content": "hi"},
        ]
    )
    assert out[0] == {"role": "system", "content": "be brief"}


def test_normalize_messages_rejects_images():
    with pytest.raises(TranslationError) as exc:
        normalize_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])
    assert exc.value.param == "messages[0]"


def test_normalize_messages_assistant_tool_calls_parse_arguments():
    out = normalize_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ]
    )
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    assert out[1]["role"] == "tool"


def test_normalize_tools_choice():
    tools = [
        {"type": "function", "function": {"name": "a", "parameters": {}}},
        {"type": "function", "function": {"name": "b", "parameters": {}}},
    ]
    assert normalize_tools(tools, "none") is None
    assert len(normalize_tools(tools, "auto")) == 2
    only_b = normalize_tools(tools, {"type": "function", "function": {"name": "b"}})
    assert [t["function"]["name"] for t in only_b] == ["b"]
    with pytest.raises(TranslationError):
        normalize_tools(tools, {"type": "function", "function": {"name": "zzz"}})
