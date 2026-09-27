"""LLM provider seam.

Every research call in the system goes through `deep_research._run_claude`, which
delegates to the provider selected by config `llm.provider`:

    claude_cli     headless `claude -p` on a Claude Code SUBSCRIPTION (default; the only
                   provider with --resume sessions and the CLI tool set, which the
                   dashboard's Investigate mode needs)
    anthropic_api  the official Anthropic SDK with an API key (per-token billing)
    openai_compat  any OpenAI-compatible Chat Completions endpoint: OpenAI, OpenRouter,
                   Gemini, DeepSeek, or a local Ollama / LM Studio server

Every provider exposes the same small surface: `complete()` -> `Response`,
`ready()` -> (ok, detail), `supports_web_search`, `supports_sessions` and
`model_label` (what is stored on each report). Tests substitute a fake via
`set_provider()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .. import config

PROVIDERS = ("claude_cli", "anthropic_api", "openai_compat")


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
    supports_web_search: bool

    def complete(self, prompt: str, *, tools: str | None = None, timeout: float | None = None,
                 extra_args: list[str] | None = None, model: str | None = None) -> Response:
        """Run one prompt. `tools` is the comma-separated CLI tool list ("" = none;
        None = config default). Raises RuntimeError on failure/timeout."""

    def ready(self) -> tuple[bool, str]:
        """Cheap, offline readiness check (no network): (usable?, human detail)."""


_override: Provider | None = None


def set_provider(p: Provider | None) -> None:
    """Test hook / runtime override."""
    global _override
    _override = p


def provider_name() -> str:
    return str(config.get("llm.provider", "claude_cli") or "claude_cli").lower()


def get_provider() -> Provider:
    if _override is not None:
        return _override
    name = provider_name()
    if name == "claude_cli":
        from .claude_cli import ClaudeCLIProvider
        return ClaudeCLIProvider()
    if name in ("anthropic", "anthropic_api"):
        from .anthropic_api import AnthropicAPIProvider
        return AnthropicAPIProvider()
    if name in ("openai", "openai_compat"):
        from .openai_compat import OpenAICompatProvider
        return OpenAICompatProvider()
    raise RuntimeError(f"unknown llm.provider '{name}' ({' | '.join(PROVIDERS)})")


def supports_web_search(p: Provider | None = None) -> bool:
    """Missing attribute = True: a test fake or a legacy provider is treated like the
    web-capable default, so adding this flag changed no existing behaviour."""
    return bool(getattr(p or get_provider(), "supports_web_search", True))


def model_label(p: Provider | None = None) -> str:
    """The model string stored on each report (prompt provenance)."""
    p = p or get_provider()
    label = getattr(p, "model_label", None)
    if isinstance(label, str) and label:
        return label
    return str(config.get("claude.model", "claude-opus-4-8"))


def readiness(p: Provider | None = None) -> tuple[bool, str]:
    try:
        p = p or get_provider()
    except RuntimeError as exc:
        return False, str(exc)
    fn = getattr(p, "ready", None)
    if fn is None:
        return True, getattr(p, "name", "provider")
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — a readiness probe must never crash a run
        return False, f"readiness check failed: {exc}"
