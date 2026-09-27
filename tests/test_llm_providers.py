"""Provider surface: the OpenAI-compatible provider, readiness, offline prompt
adaptation — and the guarantee that the web-capable (Claude) path is untouched."""
from __future__ import annotations

import json
import re

import pytest
import requests

from src import deep_research as dr, learning, llm
from src.llm.openai_compat import PRESETS, OpenAICompatProvider


# ------------------------------------------------------------------ fakes
class _Resp:
    def __init__(self, status=200, payload=None, text=None):
        self.status_code = status
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload or {})

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _HTTP:
    """Records the request, returns a canned response (or raises)."""
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if self.exc:
            raise self.exc
        return self.resp


def _ok(content="hello", model="gpt-x"):
    return _Resp(200, {"model": model, "choices": [{"message": {"content": content},
                                                    "finish_reason": "stop"}],
                       "usage": {"total_tokens": 7}})


def _provider(http, **over):
    s = {"preset": "openai", "model": "gpt-x", "api_key_env": "TEST_OAI_KEY"} | over
    return OpenAICompatProvider(settings=s, session=http)


# ------------------------------------------------------------------ openai_compat
def test_request_shape_auth_and_parsing(monkeypatch):
    monkeypatch.setenv("TEST_OAI_KEY", "sk-test")
    http = _HTTP(_ok("{\"a\": 1}", model="gpt-x-2026"))
    r = _provider(http).complete("PROMPT", timeout=30)
    call = http.calls[0]
    assert call["url"] == "https://api.openai.com/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["json"]["model"] == "gpt-x"
    assert call["json"]["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert call["json"]["max_tokens"] == 8000 and call["json"]["temperature"] == 0.2
    assert call["timeout"] == (15, 30.0)
    assert r.text == "{\"a\": 1}" and r.model == "gpt-x-2026" and r.meta["usage"]["total_tokens"] == 7


def test_local_preset_needs_no_key_and_param_overrides(monkeypatch):
    http = _HTTP(_ok())
    p = OpenAICompatProvider(settings={"preset": "ollama", "model": "qwen", "temperature": None,
                                       "max_tokens_param": "max_completion_tokens",
                                       "max_tokens": 123}, session=http)
    assert p.ready()[0] is True
    p.complete("x")
    call = http.calls[0]
    assert call["url"] == "http://localhost:11434/v1/chat/completions"
    assert "Authorization" not in call["headers"]
    assert "temperature" not in call["json"] and call["json"]["max_completion_tokens"] == 123


def test_content_parts_are_joined():
    http = _HTTP(_Resp(200, {"choices": [{"message": {"content": [
        {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]}))
    assert OpenAICompatProvider({"preset": "lmstudio", "model": "m"}, http).complete("x").text == "ab"


@pytest.mark.parametrize("status,hint", [(401, "API key"), (404, "base_url or model"),
                                         (429, "rate limited"), (500, "")])
def test_http_errors_raise_with_a_hint(monkeypatch, status, hint):
    monkeypatch.setenv("TEST_OAI_KEY", "k")
    http = _HTTP(_Resp(status, None, text="upstream says no"))
    with pytest.raises(RuntimeError) as e:
        _provider(http).complete("x")
    assert f"HTTP {status}" in str(e.value) and hint in str(e.value)
    assert "upstream says no" in str(e.value)


def test_network_failures_and_bad_shapes_raise(monkeypatch):
    monkeypatch.setenv("TEST_OAI_KEY", "k")
    with pytest.raises(RuntimeError, match="timed out"):
        _provider(_HTTP(exc=requests.Timeout())).complete("x")
    with pytest.raises(RuntimeError, match="cannot reach"):
        _provider(_HTTP(exc=requests.ConnectionError("refused"))).complete("x")
    with pytest.raises(RuntimeError, match="unexpected response shape"):
        _provider(_HTTP(_Resp(200, {"nope": 1}))).complete("x")
    with pytest.raises(RuntimeError, match="empty completion"):
        _provider(_HTTP(_ok(content="  "))).complete("x")
    with pytest.raises(RuntimeError, match="sessions"):
        _provider(_HTTP(_ok())).complete("x", extra_args=["--resume", "s"])


def test_readiness_explains_what_is_missing(monkeypatch):
    monkeypatch.delenv("TEST_OAI_KEY", raising=False)
    ok, why = OpenAICompatProvider({"preset": "", "model": "m"}).ready()
    assert not ok and "preset" in why
    ok, why = OpenAICompatProvider({"preset": "openai", "model": ""}).ready()
    assert not ok and "model" in why
    ok, why = _provider(_HTTP()).ready()
    assert not ok and "TEST_OAI_KEY" in why
    monkeypatch.setenv("TEST_OAI_KEY", "k")
    ok, why = _provider(_HTTP()).ready()
    assert ok and "no web search" in why
    with pytest.raises(RuntimeError, match="not set"):
        monkeypatch.delenv("TEST_OAI_KEY")
        _provider(_HTTP()).complete("x")


def test_presets_and_model_label():
    assert set(PRESETS) >= {"openai", "openrouter", "gemini", "deepseek", "ollama", "lmstudio"}
    assert all(v["base_url"].startswith("http") for v in PRESETS.values())
    assert OpenAICompatProvider({"preset": "openai", "model": "m"}).model_label == "m [no-web]"
    assert OpenAICompatProvider({"preset": "openai", "model": "m",
                                 "web_search": True}).model_label == "m"


def test_get_provider_selects_openai_compat(monkeypatch):
    cfg = {"llm.provider": "openai_compat", "llm.openai_compat": {"preset": "lmstudio", "model": "m"}}
    monkeypatch.setattr(llm.config, "get", lambda k, d=None: cfg.get(k, d))
    p = llm.get_provider()
    assert p.name == "openai_compat" and llm.supports_web_search(p) is False
    assert llm.readiness(p)[0] is True and llm.model_label(p) == "m [no-web]"


# ------------------------------------------------------------------ the Claude path is untouched
class _WebProvider:
    name, supports_sessions, supports_web_search, model_label = "fake", True, True, "claude-x"

    def __init__(self):
        self.prompts = []

    def complete(self, prompt, **kw):
        self.prompts.append(prompt)
        return llm.Response(text='{"rating": "buy", "conviction": 6}', raw="")

    def ready(self):
        return True, "fake"


class _OfflineProvider(_WebProvider):
    name, supports_web_search, model_label = "offline", False, "local-m [no-web]"


@pytest.fixture
def use(monkeypatch):
    def _set(p):
        llm.set_provider(p)
        return p
    yield _set
    llm.set_provider(None)


def _render_all():
    dr.research_one("NVDA", "NVIDIA", "semis", "CTX-LINE", holding={"shares": 10, "avg_cost": 100})
    dr.deep_dive_holding("NVDA", "NVIDIA", "semis", {"shares": 10, "avg_cost": 100, "last": 120,
                                                    "pnl_pct": 20, "weight_pct": 12},
                         portfolio_block="PF", history_block="HIST")
    dr.validate_recency("NVDA", {"action": "hold", "thesis_health": "ok"})
    dr.red_team("NVDA", "NVIDIA", {"rating": "buy", "conviction": 7, "thesis": "T", "risks": ["r1"]})


def test_web_capable_provider_sends_prompts_unchanged(use):
    """What a web-capable provider receives is exactly the rendered template — the
    offline adaptation must never touch it (so historical calibration stays valid)."""
    p = use(_WebProvider())
    _render_all()
    assert len(p.prompts) == 4
    assert all("web search" in q.lower() or "use web data" in q.lower() for q in p.prompts)
    assert not any(q.startswith("NOTE: you have no web access") for q in p.prompts)
    assert not any("You have NO web access" in q for q in p.prompts)


def test_offline_provider_gets_adapted_prompts_and_skips_validation(use):
    p = use(_OfflineProvider())
    _render_all()
    # research, deep dive, red team were sent; recency validation was NOT (skipped).
    assert len(p.prompts) == 3
    for q in p.prompts:
        assert not re.search(r"(?i)\bweb search\b|\buse web data\b", q), q[:200]
        assert not q.startswith("NOTE: you have no web access")   # specific rewrites hit
    assert "You have NO web access" in p.prompts[0]
    assert "you cannot fetch charts" in p.prompts[0]
    assert "never invent URLs" in p.prompts[1]
    assert "build the STRONGEST bear case" in p.prompts[2]
    v = dr.validate_recency("NVDA", {"action": "hold"})
    assert v["skipped"] is True and v["is_current"] is None and "no web access" in v["note"]


def test_every_offline_rewrite_still_matches_its_template():
    """Template drift guard: each rewrite must find its passage in the RENDERED prompt;
    otherwise a template edit would silently fall back to the blunt notice."""
    import datetime as _dt
    today = _dt.date.today().isoformat()
    corpus = "\n".join([dr._PROMPT_TMPL, dr._DEEPDIVE_TMPL, dr._REDTEAM_TMPL,
                        learning._REFLECT_TMPL]).replace("{today}", today)
    for pat, _ in dr._OFFLINE_REWRITES:
        assert pat.search(corpus), f"offline rewrite no longer matches any template: {pat.pattern[:80]}"


def test_unknown_web_instruction_gets_the_fallback_notice():
    out = dr._offline("Please search the web for the latest price of X.")
    assert out.startswith("NOTE: you have no web access")
    assert dr._offline("No browsing needed here.") == "No browsing needed here."


def test_reports_record_the_model_that_actually_ran(use):
    use(_OfflineProvider())
    rep = dr.research_one("NVDA", "NVIDIA", "semis", "ctx")
    assert rep["model"] == "local-m [no-web]"
    use(_WebProvider())
    assert dr.research_one("NVDA", "NVIDIA", "semis", "ctx")["model"] == "claude-x"


def test_auth_status_is_provider_aware(use):
    use(_OfflineProvider())
    a = dr.auth_status()
    assert a["provider"] and a["ready"] is True and a["web_search"] is False
    assert "claude_on_path" in a            # legacy keys still present
