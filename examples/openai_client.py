"""Talk to openai-maple with the official OpenAI Python client.

pip install openai
python examples/openai_client.py [http://127.0.0.1:8000/v1]
"""

from __future__ import annotations

import sys

from openai import OpenAI

base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/v1"
client = OpenAI(base_url=base_url, api_key="not-needed")

# 1. Plain answer. Maple reasons first; the trace comes back as reasoning_content.
resp = client.chat.completions.create(
    model="maple-preview",
    messages=[{"role": "user", "content": "What is 17 * 23? Answer briefly."}],
    max_tokens=512,
)
msg = resp.choices[0].message
print("reasoning:", (msg.model_extra or {}).get("reasoning_content", "")[:200], "...")
print("answer:   ", msg.content)
print("usage:    ", resp.usage)

# 2. Streaming: reasoning deltas arrive as `reasoning_content`, then content.
print("\n--- streaming ---")
stream = client.chat.completions.create(
    model="maple-preview",
    messages=[{"role": "user", "content": "Name three prime numbers above 100."}],
    max_tokens=512,
    stream=True,
)
in_reasoning = False
for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    reasoning = (delta.model_extra or {}).get("reasoning_content")
    if reasoning:
        if not in_reasoning:
            print("[thinking] ", end="")
            in_reasoning = True
        print(reasoning, end="", flush=True)
    if delta.content:
        if in_reasoning:
            print("\n[answer] ", end="")
            in_reasoning = False
        print(delta.content, end="", flush=True)
print()

# 3. Skip the reasoning block entirely (vLLM-style chat_template_kwargs).
resp = client.chat.completions.create(
    model="maple-preview",
    messages=[{"role": "user", "content": "Say hello in French."}],
    max_tokens=64,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print("\nno-think:", resp.choices[0].message.content)

# 4. Tool calling. Maple was post-trained lightly for agentic use; expect
# occasional misses.
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
resp = client.chat.completions.create(
    model="maple-preview",
    messages=[{"role": "user", "content": "What's the weather like in Lagos right now?"}],
    tools=tools,
    max_tokens=1024,
)
choice = resp.choices[0]
print("\nfinish_reason:", choice.finish_reason)
for call in choice.message.tool_calls or []:
    print("tool call:", call.function.name, call.function.arguments)
