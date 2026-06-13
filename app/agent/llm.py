"""LLM client abstraction across providers, with an offline stub for dev/CI.

The stub implements a tiny rule-based planner so the agent loop is fully
exercisable without API keys. Real providers plug in behind `complete`.
"""
from __future__ import annotations

import json
from typing import Any

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class BaseLLM:
    def complete(self, system: str, messages: list[dict[str, str]], tools: list[dict] | None = None) -> dict[str, Any]:
        raise NotImplementedError


class OpenAILLM(BaseLLM):
    def __init__(self, model: str, api_key: str) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, system, messages, tools=None):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["input_schema"]
            }} for t in tools]
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            return {"type": "tool_call", "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments)}
        return {"type": "text", "text": msg.content or ""}


class AnthropicLLM(BaseLLM):
    def __init__(self, model: str, api_key: str) -> None:
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system, messages, tools=None):
        kwargs: dict[str, Any] = {
            "model": self.model, "max_tokens": 1024, "system": system,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        }
        if tools:
            kwargs["tools"] = [{"name": t["name"], "description": t["description"],
                                "input_schema": t["input_schema"]} for t in tools]
        resp = self.client.messages.create(**kwargs)
        for block in resp.content:
            if block.type == "tool_use":
                return {"type": "tool_call", "name": block.name, "arguments": block.input}
        text = "".join(b.text for b in resp.content if b.type == "text")
        return {"type": "text", "text": text}


class StubLLM(BaseLLM):
    """Rule-based offline planner. Routes obvious intents to tools, else answers
    from provided context. Good enough to demo + test the agent end-to-end."""

    def complete(self, system, messages, tools=None):
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        low = user.lower()
        # Stop if we've already gathered observations.
        already_acted = any(m["role"] == "tool" for m in messages)
        if not already_acted and tools:
            if any(k in low for k in ["calculate", "compute", "how much", "*", "+", "sum of"]):
                expr = "".join(c for c in user if c in "0123456789.+-*/()% ")
                if expr.strip():
                    return {"type": "tool_call", "name": "calculator",
                            "arguments": {"expression": expr.strip()}}
            if "tkt-" in low or "ticket" in low:
                import re
                m = re.search(r"tkt-\d+", low)
                if m:
                    return {"type": "tool_call", "name": "ticket_lookup",
                            "arguments": {"ticket_id": m.group(0)}}
            if any(k in low for k in ["policy", "vacation", "pto", "expense", "reimburse", "remote"]):
                return {"type": "tool_call", "name": "policy_lookup",
                        "arguments": {"question": user}}
        # Synthesize a final answer from context in the system prompt.
        return {"type": "text",
                "text": "Based on the retrieved evidence and tool results, here is the answer. "
                        "(Offline stub model — configure an API key for full generation.)"}


def get_llm() -> BaseLLM:
    s = get_settings()
    if s.llm_provider == "openai" and s.openai_api_key:
        return OpenAILLM(s.llm_model, s.openai_api_key)
    if s.llm_provider == "anthropic" and s.anthropic_api_key:
        return AnthropicLLM(s.llm_model, s.anthropic_api_key)
    logger.warning("No LLM API key configured; using StubLLM.")
    return StubLLM()
