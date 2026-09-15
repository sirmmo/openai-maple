"""Runtime configuration, read from ``MAPLE_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_str(name: str, default: str | None) -> str | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


@dataclass
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000

    #: HuggingFace repo id or local directory holding the weights.
    model_path: str = "deepgrove/maple-preview"
    #: Model id advertised by ``GET /v1/models`` and echoed in responses.
    model_id: str = "maple-preview"
    #: Reject requests naming a model other than ``model_id``.
    strict_model: bool = False

    #: ``auto`` picks cuda when available, otherwise cpu.
    device: str = "auto"
    #: ``auto`` picks bfloat16 everywhere (the weights are ternary, so bf16 is exact).
    dtype: str = "auto"
    #: Torch intra-op threads for CPU inference. 0 = torch default.
    torch_threads: int = 0
    #: Pack ternary expert weights as int8 and run fp32 compute. ``auto`` = on
    #: for CPU only; ``true``/``false`` force it. See ``ternary.py``.
    pack_experts: str = "auto"

    #: Optional bearer token. ``None`` serves unauthenticated.
    api_key: str | None = None

    #: Default cap when a request omits ``max_tokens``.
    max_new_tokens: int = 4096
    #: Absolute ceiling on ``max_tokens`` a request may ask for.
    max_new_tokens_limit: int = 32768
    #: Prompts longer than this (in tokens) are rejected with 400.
    max_prompt_tokens: int = 65536

    #: Sampling defaults applied when the request omits the field. The model
    #: card ships no generation_config, so these mirror what reasoning models
    #: of this generation usually recommend.
    default_temperature: float = 0.6
    default_top_p: float = 0.95
    default_top_k: int = 20

    #: Open the assistant turn with an empty think block so the model answers
    #: directly. Per-request override: ``chat_template_kwargs.enable_thinking``.
    enable_thinking: bool = True
    #: Put ``reasoning_content`` on responses / stream deltas.
    include_reasoning: bool = True

    #: Requests serialize through one generation thread; reject past this backlog.
    max_queue_depth: int = 16
    #: Seconds a request may wait in the queue before 503.
    queue_timeout: float = 600.0

    #: CORS. ``*`` or a list of origins.
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])

    log_level: str = "info"

    @classmethod
    def from_env(cls) -> Settings:
        origins = _env_str("MAPLE_ALLOWED_ORIGINS", "*") or "*"
        return cls(
            host=_env_str("MAPLE_HOST", "0.0.0.0") or "0.0.0.0",
            port=_env_int("MAPLE_PORT", 8000),
            model_path=_env_str("MAPLE_MODEL_PATH", "deepgrove/maple-preview") or "deepgrove/maple-preview",
            model_id=_env_str("MAPLE_MODEL_ID", "maple-preview") or "maple-preview",
            strict_model=_env_bool("MAPLE_STRICT_MODEL", False),
            device=_env_str("MAPLE_DEVICE", "auto") or "auto",
            dtype=_env_str("MAPLE_DTYPE", "auto") or "auto",
            torch_threads=_env_int("MAPLE_TORCH_THREADS", 0),
            pack_experts=_env_str("MAPLE_PACK_EXPERTS", "auto") or "auto",
            api_key=_env_str("MAPLE_API_KEY", None),
            max_new_tokens=_env_int("MAPLE_MAX_NEW_TOKENS", 4096),
            max_new_tokens_limit=_env_int("MAPLE_MAX_NEW_TOKENS_LIMIT", 32768),
            max_prompt_tokens=_env_int("MAPLE_MAX_PROMPT_TOKENS", 65536),
            default_temperature=_env_float("MAPLE_TEMPERATURE", 0.6),
            default_top_p=_env_float("MAPLE_TOP_P", 0.95),
            default_top_k=_env_int("MAPLE_TOP_K", 20),
            enable_thinking=_env_bool("MAPLE_ENABLE_THINKING", True),
            include_reasoning=_env_bool("MAPLE_INCLUDE_REASONING", True),
            max_queue_depth=_env_int("MAPLE_MAX_QUEUE_DEPTH", 16),
            queue_timeout=_env_float("MAPLE_QUEUE_TIMEOUT", 600.0),
            allowed_origins=[o.strip() for o in origins.split(",") if o.strip()],
            log_level=_env_str("MAPLE_LOG_LEVEL", "info") or "info",
        )
