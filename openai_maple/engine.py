"""Model loading and token streaming on top of ``transformers``.

One :class:`MapleEngine` owns the tokenizer and the model. Requests serialize
through a single generation slot (CPU inference does not batch usefully and a
GPU is saturated by one sequence of this size); callers past ``max_queue_depth``
are refused immediately with :class:`EngineOverloaded`.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

EOS_TOKEN_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|>


class EngineError(RuntimeError):
    """Generation failed inside the model."""


class EngineOverloaded(EngineError):
    """The request queue is full."""


class EngineNotReady(EngineError):
    """The model is still loading."""


class PromptTooLong(EngineError):
    """The prompt exceeds the configured token limit."""


@dataclass
class GenerationParams:
    max_new_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    repetition_penalty: float = 1.0
    seed: int | None = None
    stop: list[str] = field(default_factory=list)


@dataclass
class Piece:
    """One increment of a generation stream.

    ``text`` is the newly decoded text (may be empty on the final piece).
    ``finish_reason`` is set on the last piece only.
    """

    text: str
    completion_tokens: int
    finish_reason: str | None = None


def _patch_query_length_bug(model: Any) -> bool:
    """Work around a bug in the model repo's ``fa3.py``.

    ``flash_attention_forward`` there computes ``seq_len = query.shape[1]``
    on a ``(batch, heads, seq, dim)`` tensor, i.e. it takes the head count
    for the sequence length. The value only feeds the attention-mask branch,
    so vLLM/SGLang never hit it, but ``generate()`` always passes a mask and
    the query then gets truncated to ``num_heads`` tokens. Rebind the inner
    helper to derive the length from the tensor it is handed.
    """
    import sys

    modname = type(model).__module__
    fa3 = sys.modules.get(modname.rsplit(".", 1)[0] + ".fa3")
    inner = getattr(fa3, "_flash_attention_forward", None)
    if inner is None or getattr(inner, "__maple_patched__", False):
        return False

    def fixed(query_states, key_states, value_states, attention_mask, query_length, *args, **kw):
        # ``query_states`` is ``(batch, seq, heads, dim)`` at this point.
        return inner(
            query_states, key_states, value_states, attention_mask, query_states.shape[1], *args, **kw
        )

    fixed.__maple_patched__ = True  # type: ignore[attr-defined]
    fa3._flash_attention_forward = fixed
    log.info("patched fa3._flash_attention_forward query_length (upstream bug)")
    return True


def _resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(requested: str):
    import torch

    table = {
        "auto": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return table[requested.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown dtype {requested!r}; use bfloat16, float16 or float32") from exc


class MapleEngine:
    def __init__(
        self,
        model_path: str = "deepgrove/maple-preview",
        *,
        device: str = "auto",
        dtype: str = "auto",
        torch_threads: int = 0,
        pack_experts: str = "auto",
        max_queue_depth: int = 16,
        queue_timeout: float = 600.0,
        max_prompt_tokens: int = 65536,
    ) -> None:
        self.model_path = model_path
        self.requested_device = device
        self.requested_dtype = dtype
        self.torch_threads = torch_threads
        self.pack_experts = pack_experts
        self.max_queue_depth = max_queue_depth
        self.queue_timeout = queue_timeout
        self.max_prompt_tokens = max_prompt_tokens

        self.ready = False
        self.device = "unknown"
        self.dtype = "unknown"
        self.attention_backend = "unknown"
        self.load_seconds: float | None = None
        self.tokenizer: Any = None
        self.model: Any = None

        self._slot = threading.Lock()
        self._depth_lock = threading.Lock()
        self._waiting = 0
        self._active = 0

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Load tokenizer and weights. Blocking; call from a worker thread."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from . import flash_attn_shim

        started = time.monotonic()
        if self.torch_threads > 0:
            torch.set_num_threads(self.torch_threads)

        self.device = _resolve_device(self.requested_device)
        torch_dtype = _resolve_dtype(self.requested_dtype)
        self.dtype = str(torch_dtype).replace("torch.", "")

        shimmed = flash_attn_shim.install()
        self.attention_backend = "torch-sdpa (flash_attn shim)" if shimmed else "flash_attn"
        log.info("attention backend: %s", self.attention_backend)

        log.info("loading tokenizer from %s", self.model_path)
        # transformers 4.57 flags this Qwen2 tokenizer as needing the Mistral
        # regex fix; its output matches tokenizer.json exactly, so opt out.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, fix_mistral_regex=False
        )

        log.info("loading model from %s (device=%s dtype=%s)", self.model_path, self.device, self.dtype)
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if self.device != "cpu":
            kwargs["device_map"] = self.device
        model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        model.eval()
        _patch_query_length_bug(model)
        if self._should_pack(self.device):
            from .ternary import pack_experts

            log.info("packing ternary experts for CPU inference (this takes a minute)")
            counts = pack_experts(model)
            if counts["packed"]:
                self.dtype = f"float32 + int8 ternary experts ({counts['packed']} packed)"
        self.model = model
        self.load_seconds = time.monotonic() - started
        self.ready = True
        log.info("model ready in %.1fs", self.load_seconds)

    def _should_pack(self, device: str) -> bool:
        mode = (self.pack_experts or "auto").lower()
        if mode in ("1", "true", "yes", "on"):
            return True
        if mode in ("0", "false", "no", "off"):
            return False
        return device == "cpu"

    # -- prompt helpers -------------------------------------------------------

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = True,
    ) -> str:
        """Render the model's chat template.

        The bundled template always ends the generation prompt with
        ``<think>\\n``. With ``enable_thinking=False`` the block is closed
        immediately so the model answers directly.
        """
        self._require_ready()
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            add_generation_prompt=True,
            tokenize=False,
        )
        if not enable_thinking:
            text += "\n</think>\n\n"
        return text

    def encode(self, text: str) -> list[int]:
        self._require_ready()
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > self.max_prompt_tokens:
            raise PromptTooLong(f"prompt is {len(ids)} tokens; the server limit is {self.max_prompt_tokens}")
        return ids

    def count_tokens(self, text: str) -> int:
        if not text or self.tokenizer is None:
            return 0
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    # -- generation -----------------------------------------------------------

    @property
    def queue_depth(self) -> int:
        return self._waiting

    @property
    def active(self) -> int:
        return self._active

    def stream(
        self,
        prompt_ids: list[int],
        params: GenerationParams,
        cancel: threading.Event | None = None,
    ) -> Iterator[Piece]:
        """Generate from ``prompt_ids``; yields :class:`Piece` increments.

        Blocks until the generation slot is free (or ``queue_timeout``).
        """
        self._require_ready()
        cancel = cancel or threading.Event()

        with self._depth_lock:
            if self._waiting >= self.max_queue_depth:
                raise EngineOverloaded(
                    f"engine queue is full ({self._waiting} requests waiting); retry shortly"
                )
            self._waiting += 1
        try:
            if not self._slot.acquire(timeout=self.queue_timeout):
                raise EngineOverloaded("timed out waiting for the generation slot")
        finally:
            with self._depth_lock:
                self._waiting -= 1

        try:
            with self._depth_lock:
                self._active += 1
            yield from self._run(prompt_ids, params, cancel)
        finally:
            with self._depth_lock:
                self._active -= 1
            self._slot.release()

    def _run(
        self, prompt_ids: list[int], params: GenerationParams, cancel: threading.Event
    ) -> Iterator[Piece]:
        import torch
        from transformers import DynamicCache, StoppingCriteria, StoppingCriteriaList
        from transformers.generation.streamers import TextIteratorStreamer

        tokenizer = self.tokenizer
        model = self.model

        class _Streamer(TextIteratorStreamer):
            """TextIteratorStreamer that also counts tokens and remembers the last id."""

            def __init__(self_inner, *a: Any, **kw: Any) -> None:
                super().__init__(*a, **kw)
                self_inner.generated = 0
                self_inner.last_id: int | None = None
                self_inner._seen_prompt = False

            def put(self_inner, value: Any) -> None:
                if not self_inner._seen_prompt:
                    self_inner._seen_prompt = True
                    super().put(value)
                    return
                flat = value.reshape(-1)
                self_inner.generated += int(flat.numel())
                self_inner.last_id = int(flat[-1].item())
                super().put(value)

        class _Cancel(StoppingCriteria):
            def __call__(self_inner, input_ids, scores, **kwargs):  # type: ignore[override]
                return torch.full(
                    (input_ids.shape[0],), cancel.is_set(), dtype=torch.bool, device=input_ids.device
                )

        streamer = _Streamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=None)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self._model_device())

        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "max_new_tokens": params.max_new_tokens,
            "eos_token_id": list(EOS_TOKEN_IDS),
            "pad_token_id": EOS_TOKEN_IDS[1],
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([_Cancel()]),
            "use_cache": True,
            # A config-aware cache would give the sliding_attention layers a
            # window-sized KV store, but the model's unpad code indexes keys by
            # the full attention mask. A plain DynamicCache keeps every key and
            # lets fa3 apply the 512-token window itself.
            "past_key_values": DynamicCache(),
            "repetition_penalty": params.repetition_penalty if params.repetition_penalty else 1.0,
        }
        if params.temperature is not None and params.temperature > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
            )
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        if params.stop:
            gen_kwargs["stop_strings"] = list(params.stop)
            gen_kwargs["tokenizer"] = tokenizer

        seed = params.seed if params.seed is not None else random.getrandbits(62)
        torch.manual_seed(seed)

        failure: list[BaseException] = []

        def worker() -> None:
            try:
                with torch.inference_mode():
                    model.generate(**gen_kwargs)
            except BaseException as exc:  # noqa: BLE001 -- surfaced to the consumer below
                failure.append(exc)
                streamer.end()

        thread = threading.Thread(target=worker, name="maple-generate", daemon=True)
        thread.start()

        stopped_by_string: str | None = None
        emitted = ""
        try:
            for text in streamer:
                if not text:
                    continue
                if params.stop:
                    # HF stops *after* the stop string is produced; trim it here.
                    candidate = emitted + text
                    cut = None
                    for s in params.stop:
                        idx = candidate.find(s, max(0, len(emitted) - len(s)))
                        if idx != -1 and (cut is None or idx < cut):
                            cut, stopped_by_string = idx, s
                    if cut is not None:
                        tail = candidate[len(emitted) : cut]
                        cancel.set()
                        if tail:
                            emitted += tail
                            yield Piece(tail, streamer.generated)
                        break
                emitted += text
                yield Piece(text, streamer.generated)
        finally:
            thread.join()

        if failure:
            exc = failure[0]
            raise EngineError(f"{type(exc).__name__}: {exc}") from exc

        if stopped_by_string is not None or streamer.last_id in EOS_TOKEN_IDS:
            reason = "stop"
        elif cancel.is_set():
            reason = "cancelled"
        elif streamer.generated >= params.max_new_tokens:
            reason = "length"
        else:
            reason = "stop"
        yield Piece("", streamer.generated, finish_reason=reason)

    # -- internals ------------------------------------------------------------

    def _model_device(self):
        return next(self.model.parameters()).device

    def _require_ready(self) -> None:
        if not self.ready:
            raise EngineNotReady("model is still loading")
