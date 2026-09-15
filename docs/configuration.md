# Configuration

Everything is an environment variable; the CLI flags of `openai-maple --help`
set the same variables. Copy `.env.example` to `.env` for Docker Compose.

## Model and hardware

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAPLE_MODEL_PATH` | `deepgrove/maple-preview` | HuggingFace repo id or a local directory with the weights |
| `MAPLE_MODEL_ID` | `maple-preview` | id shown by `/v1/models` and echoed in responses |
| `MAPLE_STRICT_MODEL` | `false` | `404` requests naming another model. Off so clients that hard-code `gpt-4o` keep working |
| `MAPLE_DEVICE` | `auto` | `cpu`, `cuda`, `cuda:0`, `mps`. `auto` picks CUDA if available, then MPS, then CPU |
| `MAPLE_DTYPE` | `auto` | `bfloat16` (the default; the weights are ternary so bf16 is exact), `float16`, `float32` |
| `MAPLE_PACK_EXPERTS` | `auto` | Pack the ternary expert bank as int8 + row scale and compute in fp32. `auto` = on for CPU only; `true` / `false` force it. See [Architecture](architecture.md#int8-expert-packing) |
| `MAPLE_TORCH_THREADS` | `0` | torch intra-op threads for CPU inference; `0` leaves torch's default. Set to the physical core count |
| `HF_TOKEN` | unset | avoids anonymous HuggingFace rate limits on the first download |

## Generation

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAPLE_MAX_NEW_TOKENS` | `4096` | default `max_tokens` when a request omits it |
| `MAPLE_MAX_NEW_TOKENS_LIMIT` | `32768` | hard ceiling on `max_tokens` |
| `MAPLE_MAX_PROMPT_TOKENS` | `65536` | longer prompts are rejected with `400` |
| `MAPLE_TEMPERATURE` | `0.6` | default temperature |
| `MAPLE_TOP_P` | `0.95` | default top-p |
| `MAPLE_TOP_K` | `20` | default top-k |
| `MAPLE_ENABLE_THINKING` | `true` | reason in a `<think>` block before answering. Per request: `chat_template_kwargs.enable_thinking` |
| `MAPLE_INCLUDE_REASONING` | `true` | expose the trace as `reasoning_content`. Per request: `include_reasoning` |

The model card ships no `generation_config.json`; the sampling defaults are the
values reasoning models of this generation usually recommend.

## Server

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAPLE_HOST` | `0.0.0.0` | bind address |
| `MAPLE_PORT` | `8000` | port |
| `MAPLE_API_KEY` | unset | require `Authorization: Bearer <key>` on every route except `/health` |
| `MAPLE_MAX_QUEUE_DEPTH` | `16` | requests allowed to wait for the single generation slot |
| `MAPLE_QUEUE_TIMEOUT` | `600` | seconds a request may wait before `503` |
| `MAPLE_ALLOWED_ORIGINS` | `*` | CORS; comma-separated origins or `*` |
| `MAPLE_LOG_LEVEL` | `info` | `debug`, `info`, `warning` |

## Docker Compose

`docker-compose.yml` reads the variables above from `.env`, plus:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TORCH_INDEX` | `https://download.pytorch.org/whl/cpu` | build arg; set to a CUDA index such as `…/whl/cu124` for a GPU image |
| `HF_CACHE_DIR` | `maple-cache` (named volume) | host path or volume mounted at `/cache/huggingface` |

The service has `mem_limit: 64g`; lower it to ~32g if you run packed CPU
inference and the host is tight.
