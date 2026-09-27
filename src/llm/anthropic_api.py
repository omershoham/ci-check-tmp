"""Anthropic Messages API provider (bring your own API key; per-token billing).

For users without a Claude Code subscription. Supports the server-side web-search
tool when the caller asks for "WebSearch"; the CLI's file/shell tools are not
available here, so Investigate mode requires the CLI provider.
"""
from __future__ import annotations

import os

from .. import config
from . import Response


class AnthropicAPIProvider:
    name = "anthropic_api"
    supports_sessions = False
    supports_web_search = True      # server-side web_search tool

    def __init__(self, client=None):
        self._client = client

    @property
    def model_label(self) -> str:
        return str(config.get("claude.model", "claude-opus-4-8"))

    def ready(self) -> tuple[bool, str]:
        if self._client is not None:
            return True, f"injected client · {self.model_label}"
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "llm.provider=anthropic_api needs `pip install anthropic`"
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False, "ANTHROPIC_API_KEY is not set (put it in .env)"
        return True, f"Anthropic API key set · {self.model_label}"

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError("llm.provider=anthropic_api needs `pip install anthropic`") from exc
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set (required for llm.provider=anthropic_api)")
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, prompt: str, *, tools: str | None = None, timeout: float | None = None,
                 extra_args: list[str] | None = None, model: str | None = None) -> Response:
        if extra_args:
            raise RuntimeError("anthropic_api provider does not support CLI sessions/extra args")
        model = model or config.get("claude.model", "claude-opus-4-8")
        want = tools if tools is not None else config.get("claude.allowed_tools", "WebSearch,WebFetch")
        api_tools = []
        if "WebSearch" in (want or ""):
            api_tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 8})
        client = self._get_client()
        kwargs = {"model": model, "max_tokens": 8000,
                  "messages": [{"role": "user", "content": prompt}]}
        if api_tools:
            kwargs["tools"] = api_tools
        if timeout:
            client = client.with_options(timeout=float(timeout))
        msg = client.messages.create(**kwargs)
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
        return Response(text=text, raw=text, session_id=None, model=model,
                        meta={"usage": getattr(msg, "usage", None) and msg.usage.__dict__})
