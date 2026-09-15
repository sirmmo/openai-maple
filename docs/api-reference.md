# API reference

All routes accept and return JSON. Errors use the OpenAI shape:

```json
{"error": {"message": "n must be 1", "type": "invalid_request_error", "param": "n", "code": null}}
```

| Status | When |
| --- | --- |
| `400` | invalid request, unsupported field, prompt over `MAPLE_MAX_PROMPT_TOKENS` (`code: context_length_exceeded`) |
| `401` | `MAPLE_API_KEY` is set and the bearer token is missing or wrong |
| `404` | `MAPLE_STRICT_MODEL=true` and the request names another model |
| `500` | generation failed inside the model (`type: server_error`) |
| `503` | weights still loading (`code: model_loading`) or queue full (`code: engine_overloaded`, with `Retry-After`) |

In a stream, an error arrives as an in-band `data:` event before `data: [DONE]`.

## `POST /v1/chat/completions`

### Request

| Field | Behaviour |
| --- | --- |
| `model` | echoed back; ignored unless `MAPLE_STRICT_MODEL` |
| `messages` | `system` / `developer`, `user`, `assistant`, `tool`. Content may be a string or a list of `text` parts; images are rejected (text-only model) |
| `tools`, `tool_choice` | rendered through the model's template; see [Tool calling](#tool-calling) |
| `temperature` | `0` is greedy; default `MAPLE_TEMPERATURE` (0.6) |
| `top_p`, `top_k` | defaults 0.95 and 20 |
| `max_tokens` / `max_completion_tokens` | default `MAPLE_MAX_NEW_TOKENS` (4096), capped at `MAPLE_MAX_NEW_TOKENS_LIMIT` |
| `stop` | string or list; trimmed from the output. Applies to the whole raw stream, reasoning included |
| `seed` | seeds torch for that request |
| `stream`, `stream_options.include_usage` | SSE stream; the usage chunk is the last one before `[DONE]` |
| `repetition_penalty` | extra field, passed to Transformers |
| `chat_template_kwargs` | `{"enable_thinking": false}` closes the think block immediately (vLLM / SGLang convention) |
| `include_reasoning` | `false` keeps the model reasoning but strips `reasoning_content` from the response (OpenRouter convention) |
| `presence_penalty`, `frequency_penalty`, `user` | accepted and ignored |
| `n > 1`, `logprobs`, `response_format: json_schema` | rejected with `400` |

Unknown fields are ignored so newer SDKs keep working.

### Response

```json
{
  "id": "chatcmpl-…", "object": "chat.completion", "created": 1789460000, "model": "maple-preview",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "391",
      "reasoning_content": "We need to answer briefly: 17*23 = 391. So final answer: 391."
    },
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 22, "completion_tokens": 33, "total_tokens": 55,
    "prompt_tokens_details": {"cached_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 27}
  }
}
```

`finish_reason` is `stop`, `length`, or `tool_calls`. `content` is `null`
when the whole answer was tool calls.

### Reasoning

Maple's chat template ends the generation prompt with `<think>\n`, so the
model's raw output *starts inside* the reasoning block and closes it with
`</think>` before the answer. The wrapper splits that:

- non-streaming: `message.reasoning_content` next to `message.content`;
- streaming: `delta.reasoning_content` chunks, then `delta.content` chunks;
- `usage.completion_tokens_details.reasoning_tokens` counts the trace.

If `max_tokens` cuts the model off mid-thought, `reasoning_content` holds
everything, `content` is empty, and `finish_reason` is `length`. Raise
`max_tokens`: reasoning traces on hard problems run to thousands of tokens.

This is the shape vLLM, SGLang and DeepSeek use, so clients that already
understand `reasoning_content` work unchanged. With the OpenAI Python SDK the
field is on `message.model_extra["reasoning_content"]`.

### Streaming

```
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}
data: {…"delta":{"reasoning_content":"We need"}…}
data: {…"delta":{"reasoning_content":" to answer"}…}
data: {…"delta":{"content":"391"}…}
data: {…"delta":{},"finish_reason":"stop"…}
data: {…"choices":[],"usage":{…}}          ← only with stream_options.include_usage
data: [DONE]
```

Text that might be the start of a `</think>` or `<tool_call>` marker is held
back for a few characters until it is disambiguated, so markers never leak
into `content`. A client that disconnects mid-stream cancels the generation.

### Tool calling

`tools` are rendered through the model's own template. `<tool_call>` blocks in
the answer become `message.tool_calls`, streamed as one `tool_calls` delta
each once the block closes, and `finish_reason` becomes `tool_calls`.

```json
{"message": {"role": "assistant", "content": null,
  "tool_calls": [{"id": "call_…", "type": "function",
    "function": {"name": "get_weather", "arguments": "{\"city\": \"Lagos\"}"}}]},
 "finish_reason": "tool_calls"}
```

`tool_choice` supports `none` (tools not shown to the model), `auto`,
`required` and a named function (the list handed to the model is filtered to
that one). The choice is not otherwise enforced. Assistant messages carrying
`tool_calls` and `tool` role messages round trip back into the template for
multi-turn use. A block whose body is not valid JSON is left in `content` as
text.

## `POST /v1/completions`

The raw prompt goes to the model untouched: no chat template, no `<think>`,
no parsing. `prompt` may be a string, a one-element list, or a list of token
ids. `stream`, `stop`, `seed`, `max_tokens` and the sampling fields behave as
above. `echo`, `suffix`, `logprobs` and `n > 1` are rejected.

## `GET /v1/models`, `GET /v1/models/{id}`

One model, id `MAPLE_MODEL_ID`, `owned_by: deepgrove`.

## `GET /health`

```json
{"status": "ok", "model": "maple-preview", "model_path": "deepgrove/maple-preview",
 "device": "cpu", "dtype": "float32 + int8 ternary experts (6144 packed)",
 "attention": "torch-sdpa (flash_attn shim)", "load_seconds": 131.9,
 "queue_depth": 0, "active": 1, "version": "0.1.0"}
```

Unauthenticated even when `MAPLE_API_KEY` is set, so orchestrators can probe it.

## Concurrency

One sequence generates at a time. Further requests wait for the slot up to
`MAPLE_QUEUE_TIMEOUT` seconds; beyond `MAPLE_MAX_QUEUE_DEPTH` waiting requests
you get `503` with `Retry-After: 5`.
