# Changelog

## 0.1.0 - 2026-09-15

First release.

- OpenAI-compatible `/v1/chat/completions` (streaming, `reasoning_content`,
  tool calls, `stop`, `seed`, usage with reasoning token counts),
  `/v1/completions`, `/v1/models`, `/health`.
- `chat_template_kwargs.enable_thinking` and `include_reasoning` controls.
- Pure-torch `flash_attn` shim so the model's Transformers code runs without
  CUDA or the FlashAttention wheel.
- int8 + row-scale packing of the ternary expert bank for CPU inference
  (bit-exact, ~23 GB RAM instead of 40 GB).
- Workarounds for two upstream `generate()` incompatibilities (fa3
  `query_length`, sliding-window KV cache).
- Docker images on GHCR for CPU (amd64, arm64) and CUDA 12.4.
