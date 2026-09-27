"""OpenAI-compatible Chat Completions provider.

One implementation for every endpoint that speaks `POST {base_url}/chat/completions`:
OpenAI, OpenRouter, Google Gemini (its OpenAI-compatible surface), DeepSeek, and local
servers such as Ollama (`/v1`) and LM Studio. Plain `requests`, no SDK dependency.

What it deliberately does NOT pretend to do:

  * **web search** — a plain chat endpoint cannot browse. `supports_web_search` is
    False unless you say otherwise in config (e.g. an OpenRouter `:online` model or a
    search-grounded model). When it is False, `deep_research` rewrites the prompts'
    "use web search" instructions into an honest offline version and skips the
    recency-validation step instead of letting the model invent "current" facts.
  * **sessions / CLI tools** — no `--resume`, no file or shell tools, so the
    dashboard's Investigate mode stays CLI-only.

Config (config.yaml):

    llm:
      provider: openai_compat
      openai_compat:
        preset: openai            # openai | openrouter | gemini | deepseek | ollama | lmstudio
        model: <model id>         # REQUIRED — whatever your endpoint serves
        base_url: null            # overrides the preset
        api_key_env: null         # env var holding the key; overrides the preset ("" = no key)
        web_search: false         # true ONLY if this endpoint/model really browses
        temperature: 0.2          # null to omit (some reasoning models reject it)
        max_tokens: 8000
        max_tokens_param: max_tokens   # some newer OpenAI models need max_completion_tokens
        extra_headers: {}         # e.g. OpenRouter's HTTP-Referer / X-Title
"""
from __future__ import annotations

import os
from typing import Any

import requests

from .. import config
from . import Response

# Endpoint presets. Model ids are NOT preset on purpose: they change monthly and a
# silently wrong default is worse than a clear "set llm.openai_compat.model".
PRESETS: dict[str, dict[str, str]] = {
    "openai":     {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    "gemini":     {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                   "api_key_env": "GEMINI_API_KEY"},
    "deepseek":   {"base_url": "https://api.deepseek.com/v1", "api_key_env": "DEEPSEEK_API_KEY"},
    "ollama":     {"base_url": "http://localhost:11434/v1", "api_key_env": ""},
    "lmstudio":   {"base_url": "http://localhost:1234/v1", "api_key_env": ""},
}


def _cfg() -> dict[str, Any]:
    return dict(config.get("llm.openai_compat", {}) or {})


class OpenAICompatProvider:
    name = "openai_compat"
    supports_sessions = False

    def __init__(self, settings: dict[str, Any] | None = None, session: Any | None = None):
        s = dict(settings) if settings is not None else _cfg()
        preset = PRESETS.get(str(s.get("preset") or "").lower(), {})
        self.preset = str(s.get("preset") or "")
        self.base_url = str(s.get("base_url") or preset.get("base_url") or "").rstrip("/")
        env = s.get("api_key_env")
        self.api_key_env = str(preset.get("api_key_env", "") if env is None else env)
        self.model = str(s.get("model") or "")
        self.supports_web_search = bool(s.get("web_search", False))
        self.temperature = s.get("temperature", 0.2)
        self.max_tokens = int(s.get("max_tokens", 8000) or 8000)
        self.max_tokens_param = str(s.get("max_tokens_param") or "max_tokens")
        self.extra_headers = dict(s.get("extra_headers") or {})
        self._http = session or requests

    # ------------------------------------------------------------------ identity
    @property
    def model_label(self) -> str:
        """What gets stored on every report. Offline runs are a different analyst, so
        the tag is part of the label and calibration never pools the two."""
        return self.model + ("" if self.supports_web_search else " [no-web]")

    def _key(self) -> str:
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""

    def ready(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, ("llm.openai_compat needs a preset "
                           f"({' | '.join(PRESETS)}) or a base_url")
        if not self.model:
            return False, "set llm.openai_compat.model to a model your endpoint serves"
        if self.api_key_env and not self._key():
            return False, f"{self.api_key_env} is not set (put it in .env)"
        web = "web search ON" if self.supports_web_search else "no web search"
        return True, f"{self.preset or 'custom'} · {self.base_url} · {self.model} · {web}"

    # ------------------------------------------------------------------ call
    def complete(self, prompt: str, *, tools: str | None = None, timeout: float | None = None,
                 extra_args: list[str] | None = None, model: str | None = None) -> Response:
        if extra_args:
            raise RuntimeError("openai_compat provider does not support CLI sessions/extra args")
        ok, why = self.ready()
        if not ok:
            raise RuntimeError(why)
        body: dict[str, Any] = {"model": model or self.model,
                                "messages": [{"role": "user", "content": prompt}],
                                self.max_tokens_param: self.max_tokens}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        headers = {"Content-Type": "application/json", **self.extra_headers}
        key = self._key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        read_timeout = float(timeout or config.get("claude.timeout_seconds", 600))
        try:
            resp = self._http.post(f"{self.base_url}/chat/completions", json=body,
                                   headers=headers, timeout=(15, read_timeout))
        except requests.Timeout as exc:
            raise RuntimeError(f"{self.base_url} timed out after {read_timeout:.0f}s") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"cannot reach {self.base_url}: {exc}") from exc
        if resp.status_code != 200:
            hint = {401: " (bad or missing API key)", 404: " (wrong base_url or model id)",
                    429: " (rate limited / out of credit)"}.get(resp.status_code, "")
            raise RuntimeError(f"{self.base_url} returned HTTP {resp.status_code}{hint}: "
                               f"{(resp.text or '')[:300]}")
        try:
            data = resp.json()
            msg = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected response shape from {self.base_url}: "
                               f"{(resp.text or '')[:300]}") from exc
        content = msg.get("content")
        if isinstance(content, list):          # some servers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        text = str(content or "")
        if not text.strip():
            raise RuntimeError(f"{self.base_url} returned an empty completion "
                               f"(finish_reason={data['choices'][0].get('finish_reason')})")
        return Response(text=text, raw=text, session_id=None,
                        model=str(data.get("model") or body["model"]),
                        meta={"usage": data.get("usage")})
