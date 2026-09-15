#!/usr/bin/env bash
# curl walkthrough of the openai-maple API. Usage: ./examples/curl.sh [base_url]
set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"

echo "# health"
curl -s "$BASE/health"; echo; echo

echo "# models"
curl -s "$BASE/v1/models"; echo; echo

echo "# chat completion (reasoning comes back as message.reasoning_content)"
curl -s "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "maple-preview",
    "messages": [{"role": "user", "content": "What is 17 * 23? Answer briefly."}],
    "max_tokens": 512
  }'; echo; echo

echo "# streaming, no reasoning block"
curl -sN "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "maple-preview",
    "messages": [{"role": "user", "content": "Say hello in French."}],
    "max_tokens": 64,
    "stream": true,
    "stream_options": {"include_usage": true},
    "chat_template_kwargs": {"enable_thinking": false}
  }'; echo; echo

echo "# legacy completions (raw prompt, no chat template)"
curl -s "$BASE/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model": "maple-preview", "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}'; echo
