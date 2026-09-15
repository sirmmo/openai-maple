"""``python -m openai_maple`` / ``openai-maple`` entry point."""

from __future__ import annotations

import argparse
import logging
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="openai-maple",
        description="OpenAI-compatible HTTP API for deepgrove/maple-preview. "
        "Every flag can also be set through the MAPLE_* environment variables.",
    )
    parser.add_argument("--host", help="bind address (MAPLE_HOST, default 0.0.0.0)")
    parser.add_argument("--port", type=int, help="port (MAPLE_PORT, default 8000)")
    parser.add_argument("--model-path", help="HF repo id or local dir (MAPLE_MODEL_PATH)")
    parser.add_argument("--model-id", help="model id advertised to clients (MAPLE_MODEL_ID)")
    parser.add_argument("--device", help="auto|cpu|cuda|cuda:0|mps (MAPLE_DEVICE)")
    parser.add_argument("--dtype", help="auto|bfloat16|float16|float32 (MAPLE_DTYPE)")
    parser.add_argument("--threads", type=int, help="torch CPU threads (MAPLE_TORCH_THREADS)")
    parser.add_argument("--pack-experts", help="auto|true|false: int8 ternary experts (MAPLE_PACK_EXPERTS)")
    parser.add_argument("--api-key", help="require this bearer token (MAPLE_API_KEY)")
    parser.add_argument("--max-new-tokens", type=int, help="default max_tokens (MAPLE_MAX_NEW_TOKENS)")
    parser.add_argument(
        "--no-thinking", action="store_true", help="answer without a reasoning block by default"
    )
    parser.add_argument("--log-level", help="debug|info|warning (MAPLE_LOG_LEVEL)")
    args = parser.parse_args(argv)

    mapping = {
        "host": "MAPLE_HOST",
        "port": "MAPLE_PORT",
        "model_path": "MAPLE_MODEL_PATH",
        "model_id": "MAPLE_MODEL_ID",
        "device": "MAPLE_DEVICE",
        "dtype": "MAPLE_DTYPE",
        "threads": "MAPLE_TORCH_THREADS",
        "pack_experts": "MAPLE_PACK_EXPERTS",
        "api_key": "MAPLE_API_KEY",
        "max_new_tokens": "MAPLE_MAX_NEW_TOKENS",
        "log_level": "MAPLE_LOG_LEVEL",
    }
    for attr, env in mapping.items():
        value = getattr(args, attr)
        if value is not None:
            os.environ[env] = str(value)
    if args.no_thinking:
        os.environ["MAPLE_ENABLE_THINKING"] = "false"

    from .config import Settings

    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)

    import uvicorn

    from .server import create_app

    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level=settings.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
