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
print(resp.choices[0].message.content)  # "17 × 23 = 391."
print(resp.choices[0].message.model_extra["reasoning_content"])  # the <think> trace
```

The model's own Transformers code hard-imports FlashAttention, which only
builds for CUDA. This package ships a small pure-torch replacement for the
three `flash_attn` entry points that code uses, so **the same weights run on
CPU, Apple Silicon, or a CUDA box without the flash-attn wheel**. With a real
`flash_attn` installed the shim steps aside automatically.

## Quickstart

### Docker

Images are published to GHCR on every push to `main` (`latest`, `sha-…`) and
on version tags (`1.2.3`, `1.2`): `ghcr.io/sirmmo/openai-maple` for CPU
(amd64 + arm64) and the `-cuda` suffix (`latest-cuda`, `1.2.3-cuda`) for CUDA 12.4.

```bash
docker run -d -p 8000:8000 -v maple-cache:/cache/huggingface ghcr.io/sirmmo/openai-maple:latest
# GPU:
docker run -d --gpus all -p 8000:8000 -v maple-cache:/cache/huggingface ghcr.io/sirmmo/openai-maple:latest-cuda
curl -s localhost:8000/health   # "loading" until the weights are in RAM, then "ok"
```

Or build locally from a clone:

```bash
docker compose up -d           # CPU image; see docker-compose.yml for the GPU knobs
```

First start downloads ~40 GB of bf16 safetensors into the `maple-cache`
volume. Set `HF_CACHE_DIR=~/.cache/huggingface` in `.env` to reuse weights you
already have.

### Python (>= 3.10)

```bash
# 1. torch for your hardware
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or plain `pip install torch` for CUDA
# 2. the server
pip install -e .
openai-maple --port 8000
```

Or without installing: `python -m openai_maple --port 8000`.

Run `./examples/curl.sh` or `python examples/openai_client.py` against it.

## What you get

| Endpoint | Notes |
| --- | --- |
| `POST /v1/chat/completions` | streaming and non-streaming, `reasoning_content`, tools/`tool_calls`, `stop`, `seed`, `stream_options.include_usage` |
| `POST /v1/completions` | raw prompt, no chat template, streaming supported |
| `GET /v1/models`, `GET /v1/models/{id}` | one model, id from `MAPLE_MODEL_ID` |
| `GET /health` | `status`, device, dtype, attention backend, queue depth |
| `GET /docs` | OpenAPI UI |

### Reasoning

Maple's chat template opens every assistant turn with `<think>`, so the model
always reasons before answering. The wrapper splits the trace off:

* non-streaming: `message.reasoning_content` next to `message.content`;
* streaming: `delta.reasoning_content` chunks, then `delta.content` chunks;
* `usage.completion_tokens_details.reasoning_tokens` counts the trace.

This is the same shape vLLM, SGLang and DeepSeek use, so clients that already
understand `reasoning_content` work unchanged.

To skip reasoning for a request, send the vLLM-style flag:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

or set `MAPLE_ENABLE_THINKING=false` to make that the default. To keep the
model reasoning but hide the trace, send `"include_reasoning": false` or set
`MAPLE_INCLUDE_REASONING=false`.

### Tool calling

`tools` are rendered through the model's own template; `<tool_call>` blocks in
the answer become `message.tool_calls` (streamed as a single `tool_calls`
delta each, once the block closes) and `finish_reason` becomes `tool_calls`.
`tool_choice` supports `none`, `auto`, `required` and a named function
(the list handed to the model is filtered; the choice is not otherwise
enforced). Assistant messages with `tool_calls` and `tool` role messages round
trip back into the template for multi-turn use. Maple-Preview had minimal
agentic post-training, so expect the odd missed call.

### Sampling

| OpenAI field | Behaviour |
| --- | --- |
| `temperature` | `0` is greedy; default `MAPLE_TEMPERATURE` (0.6) |
| `top_p`, `top_k` | defaults 0.95 / 20 (the model card ships no generation config) |
| `max_tokens` / `max_completion_tokens` | default 4096, ceiling `MAPLE_MAX_NEW_TOKENS_LIMIT` |
| `stop` | string or list; trimmed from the output, applies to the whole raw stream including reasoning |
| `seed` | seeds torch for that request |
| `repetition_penalty` | extra field, passed to Transformers |
| `presence_penalty`, `frequency_penalty`, `user` | accepted and ignored |
| `n > 1`, `logprobs`, `response_format: json_schema`, `echo`, `suffix` | rejected with 400 |

Unknown fields are ignored so newer OpenAI SDKs keep working.

### Concurrency and errors

One sequence generates at a time; further requests queue (`MAPLE_MAX_QUEUE_DEPTH`,
`MAPLE_QUEUE_TIMEOUT`). Beyond the queue you get `503` with `Retry-After`.
While the weights load every inference route returns `503` with code
`model_loading` and `/health` reports `"loading"`. A client that disconnects
mid-stream cancels its generation. Errors use the OpenAI
`{"error": {"message", "type", "param", "code"}}` body; in a stream they arrive
as an in-band `data:` event before `[DONE]`.

## Configuration

Everything is an environment variable (see `.env.example`) or a CLI flag
(`openai-maple --help`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAPLE_MODEL_PATH` | `deepgrove/maple-preview` | HF repo id or local directory |
| `MAPLE_MODEL_ID` | `maple-preview` | id shown to clients |
| `MAPLE_DEVICE` | `auto` | `cpu`, `cuda`, `cuda:0`, `mps` |
| `MAPLE_DTYPE` | `auto` (bf16) | `bfloat16`, `float16`, `float32` |
| `MAPLE_TORCH_THREADS` | `0` | CPU threads for torch (0 = all) |
| `MAPLE_PACK_EXPERTS` | `auto` | int8 ternary experts + fp32 compute (`auto` = CPU only) |
| `MAPLE_API_KEY` | unset | require `Authorization: Bearer` |
| `MAPLE_STRICT_MODEL` | `false` | 404 unknown model names |
| `MAPLE_MAX_NEW_TOKENS` | `4096` | default `max_tokens` |
| `MAPLE_ENABLE_THINKING` | `true` | reason before answering |
| `MAPLE_INCLUDE_REASONING` | `true` | expose `reasoning_content` |
| `MAPLE_ALLOWED_ORIGINS` | `*` | CORS |

## Hardware notes

* **Weights are 20B parameters stored as bf16 (~40 GB)** even though only 1B
  are active per token; the "5.31 GB" on the model card is the packed ternary
  format used by DeepGrove's own runtime, not what HuggingFace serves. Budget
  45+ GB of RAM (or VRAM) plus KV cache.
* **CPU**: on by default, the server packs the ternary expert bank as int8 +
  per-row scale and computes in fp32 (`MAPLE_PACK_EXPERTS=auto`). That is
  bit-exact with the bf16 weights, needs ~23 GB of RAM instead of 40 GB, and
  is roughly 10x faster than bf16 on CPUs without AVX-512-BF16/AMX, where
  PyTorch's bf16 GEMV is very slow. Loading still streams the 40 GB
  checkpoint through memory once, and on a box with little free RAM that
  read can take a long time (84 minutes measured on a shared 188 GB host with
  ~50 GB free; 2 minutes with the shards in page cache). Set
  `MAPLE_TORCH_THREADS` to the physical core count.
  Measured on a 2016 Xeon E5-2640 v4 (40 threads, shared, load ~24): 0.17 tok/s
  in bf16, 0.5 tok/s packed, ~40 s to first token on a short prompt. Treat CPU
  as a functional fallback, not a serving target.
* **CUDA**: `pip install torch` (CUDA wheels) and optionally
  `pip install flash-attn --no-build-isolation` for the model's native
  FlashAttention path; without it the shim runs on SDPA, which is fine.
* **Apple Silicon**: `MAPLE_DEVICE=mps` loads, but for real speed use
  DeepGrove's dedicated on-device runtime instead.

## Development

```bash
pip install -e '.[test]'
pytest                    # offline: fake engine + numerical checks of the attention shim
ruff check . && ruff format --check .
```

`tests/test_flash_attn_shim.py` compares the shim against a naive O(L²)
attention for causal, sliding-window, GQA, varlen and mismatched-length cases.
The HTTP suite drives the server through a scripted fake engine and includes an
OpenAI SDK round trip. `pytest -m network` additionally builds a 2-layer
random-init Maple from the HF-hosted custom code and runs real `generate()`
through the engine; `OPENAI_MAPLE_URL=... pytest -m live` hits a running server.

### Upstream quirks this wrapper works around

* `fa3.py` in the model repo computes the query length from the wrong tensor
  axis, which truncates queries to `num_heads` tokens whenever an attention
  mask is passed (Transformers `generate()` always passes one). The engine
  rebinds that helper after loading.
* Transformers builds a window-sized KV cache for the `sliding_attention`
  layers, but the model's own unpad code indexes keys by the full mask. The
  engine hands `generate()` a plain `DynamicCache` and lets the model apply
  its 512-token window itself.
* Transformers 4.57 warns that the tokenizer needs the "Mistral regex fix";
  its output matches `tokenizer.json` exactly, so the flag is set to `False`.

## Layout

```
openai_maple/
  flash_attn_shim.py   pure-torch flash_attn_func / flash_attn_varlen_func / bert_padding
  ternary.py           int8 + row-scale packing of the ternary expert bank (CPU fast path)
  engine.py            model loading, one-slot generation queue, token streaming
  translate.py         messages -> template input; <think>/<tool_call> parsing, batch + streaming
  server.py            FastAPI routes
  streaming.py         SSE chunk framing
  schemas.py           pydantic request models
  config.py            MAPLE_* settings
```

## License

MIT. Maple-Preview itself is MIT-licensed by DeepGrove.
