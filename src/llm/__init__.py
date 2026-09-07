"""LLM provider seam.

Every Claude call in the system goes through `deep_research._run_claude`, which now
delegates to the provider selected by config `llm.provider`:

    claude_cli     headless `claude -p` on the user's Claude Code SUBSCRIPTION (default,
                   the only provider that supports --resume sessions and the CLI tool set)
    anthropic_api  the official SDK with an API key (BYO key; per-token billing)

Both return a `Response`. Tests substitute a fake via `set_provider()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .. import config


@dataclass
class Response:
    text: str                      # the answer (envelope unwrapped)
    raw: str = ""                  # full raw output (for JSON salvage / audit)
    session_id: str | None = None  # CLI sessions only
    model: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    name: str
    supports_sessions: bool

    def complete(self, prompt: str, *, tools: str | None = None, timeout: float | None = None,
                 extra_args: list[str] | None = None, model: str | None = None) -> Response:
        """Run one prompt. `tools` is the comma-separated CLI tool list ("" = none;
        None = config default). Raises RuntimeError on failure/timeout."""


_override: Provider | None = None


def set_provider(p: Provider | None) -> None:
    """Test hook / runtime override."""
    global _override
    _override = p


def get_provider() -> Provider:
    if _override is not None:
        return _override
    name = str(config.get("llm.provider", "claude_cli") or "claude_cli").lower()
    if name == "claude_cli":
        from .claude_cli import ClaudeCLIProvider
        return ClaudeCLIProvider()
    if name in ("anthropic", "anthropic_api"):
        from .anthropic_api import AnthropicAPIProvider
        return AnthropicAPIProvider()
    raise RuntimeError(f"unknown llm.provider '{name}' (claude_cli | anthropic_api)")
