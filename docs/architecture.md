# Architecture

```
openai_maple/
  server.py            FastAPI routes, auth, error mapping, SSE plumbing
  engine.py            model loading, one-slot generation queue, token streaming
  translate.py         messages -> template input; <think>/<tool_call> parsing, batch + streaming
  streaming.py         chat.completion.chunk / text_completion SSE framing
  schemas.py           pydantic request models (unknown fields allowed)
  config.py            MAPLE_* settings
  flash_attn_shim.py   pure-torch flash_attn_func / flash_attn_varlen_func / bert_padding
  ternary.py           int8 + row-scale packing of the ternary expert bank
```

## Request path

1. `server.py` validates the body, applies defaults, and normalises messages
   and tools into what the model's Jinja template expects (`translate.py`).
2. `engine.py` renders the template, tokenises, and runs Transformers
   `generate()` in a worker thread with a `TextIteratorStreamer`. Pieces of
   text are pumped onto the event loop; a `threading.Event` cancels on client
   disconnect or a matched stop string.
3. For chat, `translate.StreamParser` turns the raw text into `reasoning`,
   `content` and `tool_call` events, holding back partial markers. The
   non-streaming route collects everything and parses once.

One generation slot is shared: CPU inference does not batch usefully and a
single sequence of this size saturates a GPU. Waiting requests are counted;
past `MAPLE_MAX_QUEUE_DEPTH` they get `503` immediately.

## The FlashAttention shim

`fa3.py` in the model repo imports `flash_attn` unconditionally. The shim
registers a fake `flash_attn` package in `sys.modules` before the model code
is imported, providing `flash_attn_func`, `flash_attn_varlen_func` and the
`bert_padding` helpers on top of `torch.nn.functional.scaled_dot_product_attention`.
It reproduces FlashAttention 2 semantics: `(batch, seq, heads, dim)` layout,
implicit grouped-query attention, bottom-right causal alignment (so a single
decode query with a long KV cache sees every key), and `window_size=(left,
right)` local attention. Queries are processed in chunks so the boolean mask
never grows past `chunk × (chunk + window)` on sliding-window layers, and
full-attention prefill uses `is_causal` with no materialised mask.

The shim carries a `ModuleSpec` without a distribution behind it, so
Transformers' own `find_spec("flash_attn")` probe answers "not installed" and
its built-in attention paths are untouched; only the Maple code sees the
shim. With a real `flash_attn` or `flash_attn_interface` importable, `install()`
is a no-op.

`tests/test_flash_attn_shim.py` compares it against a naive O(L²) reference for
causal, windowed, GQA, varlen and mismatched-length cases.

## int8 expert packing

Every expert projection in the checkpoint is row-wise ternary: each row holds
`{-s, 0, +s}` for one bf16 scale `s`. `ternary.py` stores the sign as int8 and
`s` as float32, which reproduces the bf16 tensor bit for bit while halving
memory: the 20B-parameter expert bank drops from 40 GB to about 19 GB. The
~0.9B non-expert parameters (attention, router, embeddings, `lm_head`, norms)
are cast to fp32, so hidden states stay fp32 end to end.

The bigger win is speed. PyTorch's bf16 GEMV on CPUs without AVX-512-BF16 or
AMX is about 100x slower than fp32 at the expert shape (512×2048), and every
token runs 24 layers × 8 experts × 3 projections through it. Dequantising the
active expert into fp32 on the fly costs a fraction of that. A matrix that
turns out not to be exactly ternary is kept as fp32, so the path is safe for
future checkpoints.

Packing is on by default for CPU (`MAPLE_PACK_EXPERTS=auto`). On CUDA the bf16
weights are the fast path and are left alone.

## Upstream quirks worked around

- **`fa3.py` query length.** `flash_attention_forward` computes
  `seq_len = query.shape[1]` on a `(batch, heads, seq, dim)` tensor, i.e. it
  takes the head count for the sequence length. The value only feeds the
  attention-mask branch, so vLLM and SGLang never hit it, but Transformers
  `generate()` always passes a mask and the query then gets truncated to
  `num_heads` tokens. The engine rebinds the inner helper after loading to
  derive the length from the tensor it is handed.
- **Sliding-window KV cache.** Transformers builds a window-sized cache for
  the `sliding_attention` layers when it knows the config, but the model's
  unpad code indexes keys by the full attention mask. The engine hands
  `generate()` a plain `DynamicCache` and lets `fa3.py` apply the 512-token
  window itself.
- **Tokenizer warning.** Transformers 4.57 warns that this Qwen2 tokenizer
  needs the "Mistral regex fix". Its output matches `tokenizer.json` exactly,
  so the engine passes `fix_mistral_regex=False`.
- **Checkpoint size.** HuggingFace serves ~40 GB of bf16 safetensors, not the
  5.31 GB packed format quoted on the model card.

All three code workarounds are verified on a two-layer random-init Maple built
from the same custom code (`tests/test_engine_tiny.py`), where cached, masked
generation agrees with a plain full-sequence forward to 4e-7.
