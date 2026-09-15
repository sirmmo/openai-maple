# Quickstart

## Docker

Images are published to GHCR on every push to `main` and on version tags:

| Tag | Contents |
| --- | --- |
| `ghcr.io/sirmmo/openai-maple:latest` | CPU torch wheels, linux/amd64 and linux/arm64 |
| `ghcr.io/sirmmo/openai-maple:latest-cuda` | CUDA 12.4 torch wheels, linux/amd64 |
| `:1.2.3`, `:1.2`, `:1.2.3-cuda` | the same, pinned to a release |

```bash
docker run -d -p 8000:8000 -v maple-cache:/cache/huggingface ghcr.io/sirmmo/openai-maple:latest
curl -s localhost:8000/health
```

For a GPU:

```bash
docker run -d --gpus all -p 8000:8000 -v maple-cache:/cache/huggingface \
  ghcr.io/sirmmo/openai-maple:latest-cuda
```

`/health` reports `"loading"` until the weights are in memory, then `"ok"`.
Every inference route answers `503` with code `model_loading` in the meantime.

!!! warning "The first start downloads about 40 GB"
    HuggingFace serves the checkpoint as bf16 safetensors, ~40 GB across nine
    shards. The "5.31 GB" on the model card is DeepGrove's packed on-device
    format, which is not what Transformers loads. Mount a volume or you will
    download it on every run. To reuse weights you already have, mount your
    `~/.cache/huggingface` at `/cache/huggingface`.

From a clone, `docker compose up -d` builds the CPU image locally; set
`TORCH_INDEX=https://download.pytorch.org/whl/cu124` in `.env` for a CUDA build.

## Python

Python 3.10 or newer. Install torch for your hardware first, then the server:

=== "CPU"

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install openai-maple
    openai-maple --port 8000
    ```

=== "CUDA"

    ```bash
    pip install torch                # the default index ships CUDA wheels
    pip install openai-maple
    openai-maple --port 8000
    # optional, for the model's native FlashAttention path:
    pip install flash-attn --no-build-isolation
    ```

=== "From a clone"

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -e '.[test]'
    python -m openai_maple --port 8000
    ```

`openai-maple --help` lists the flags; every one also has a `MAPLE_*`
environment variable, see [Configuration](configuration.md).

## First requests

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "maple-preview",
    "messages": [{"role": "user", "content": "What is 17 * 23? Answer briefly."}],
    "max_tokens": 512
  }'
```

The answer arrives in `message.content` and the reasoning trace in
`message.reasoning_content`. To skip the reasoning block:

```bash
curl -sN localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Say hello in French."}],
    "stream": true,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Runnable versions of these, including streaming and a tool call, live in
[`examples/curl.sh`](https://github.com/sirmmo/openai-maple/blob/main/examples/curl.sh)
and
[`examples/openai_client.py`](https://github.com/sirmmo/openai-maple/blob/main/examples/openai_client.py).

## Hardware expectations

| Setup | RAM / VRAM | Speed |
| --- | --- | --- |
| CUDA, bf16 | ~40 GB VRAM plus KV cache | the model's native path; FlashAttention optional |
| CPU, packed (default) | ~23 GB RAM, 40 GB streamed once at load | 0.5 tok/s measured on a shared 2016 Xeon; expect a few tok/s on a modern box |
| CPU, unpacked bf16 (`MAPLE_PACK_EXPERTS=false`) | ~40 GB RAM | 0.17 tok/s on the same Xeon: bf16 GEMV is very slow without AVX-512-BF16 / AMX |
| Apple Silicon (`MAPLE_DEVICE=mps`) | ~40 GB unified memory | loads, but DeepGrove's own on-device runtime is the one that hits 200 tok/s |

Loading is disk-bound when the shards are not in the page cache. On a host
with little free RAM it can take over an hour; with the shards cached it is
about two minutes. Treat CPU as a functional fallback, not a serving target.
