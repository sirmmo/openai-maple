# openai-maple

An OpenAI-compatible HTTP API in front of
[deepgrove/maple-preview](https://huggingface.co/deepgrove/maple-preview), the
20B-A1B ternary-weight reasoning model. Point any OpenAI client at it and get
`chat.completions` with `reasoning_content`, streaming, tool calls and the
legacy `completions` endpoint.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="maple-preview",
    messages=[{"role": "user", "content": "What is 17 * 23?"}],
)
print(resp.choices[0].message.content)                           # "391"
print(resp.choices[0].message.model_extra["reasoning_content"])  # the <think> trace
```

!!! tip "Runs without CUDA"
    The model's own Transformers code hard-imports FlashAttention, which only
    builds for CUDA. openai-maple ships a small pure-torch replacement for the
    three `flash_attn` entry points that code uses, plus an int8 packing of the
    ternary expert bank, so the same weights run on CPU, Apple Silicon, or a
    CUDA box without the flash-attn wheel. With a real `flash_attn` installed
    the shim steps aside automatically.

## What you get

| | |
| --- | --- |
| `POST /v1/chat/completions` | streaming and non-streaming, `reasoning_content`, tools and `tool_calls`, `stop`, `seed`, `stream_options.include_usage` |
| `POST /v1/completions` | raw prompt, no chat template, streaming supported |
| `GET /v1/models` | one model, id from `MAPLE_MODEL_ID` |
| `GET /health` | status, device, dtype, attention backend, queue depth |
| `GET /docs` | OpenAPI UI |

## What to expect from the model

Maple-Preview is a reasoning model first. Its chat template opens every
assistant turn with `<think>`, so it always reasons before answering, and
traces on maths or code questions run to hundreds or thousands of tokens.
Tool calling works through the same Qwen-style `<tool_call>` format, but the
preview had minimal agentic post-training, so expect the odd missed call.

Where to go next:

- [Quickstart](quickstart.md): Docker, pip, and the first request.
- [API reference](api-reference.md): every field, and exactly how reasoning and tool calls map onto the OpenAI shape.
- [Configuration](configuration.md): the `MAPLE_*` variables.
- [Architecture](architecture.md): the shim, the packing, and the upstream quirks worked around.
