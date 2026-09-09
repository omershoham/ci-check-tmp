import json

import pytest

from src import chat, deep_research, llm
from src.llm.anthropic_api import AnthropicAPIProvider


class FakeProvider:
    name = "fake"
    supports_sessions = True

    def __init__(self, text="{}", session_id="s1"):
        self.text, self.session_id, self.calls = text, session_id, []

    def complete(self, prompt, *, tools=None, timeout=None, extra_args=None, model=None):
        self.calls.append({"prompt": prompt, "tools": tools, "extra_args": extra_args})
        return llm.Response(text=self.text, raw=json.dumps({"result": self.text,
                                                            "session_id": self.session_id}),
                            session_id=self.session_id, model="fake-model")


@pytest.fixture
def fake():
    p = FakeProvider()
    llm.set_provider(p)
    yield p
    llm.set_provider(None)


def test_run_claude_delegates_to_provider(fake):
    fake.text = '{"a": 1}'
    text, raw = deep_research._run_claude("hi", allowed_tools="", timeout=5)
    assert text == '{"a": 1}' and "session_id" in raw
    assert fake.calls[0]["tools"] == ""


def test_research_one_uses_provider_and_coerces(fake, monkeypatch):
    fake.text = json.dumps({"rating": "Strong Buy", "conviction": 14, "deal_likelihood": "HIGH",
                            "thesis": "t", "catalysts": "one", "risks": None})
    monkeypatch.setattr(deep_research.config, "get", lambda k, d=None: {"claude.model": "m"}.get(k, d))
    rep = deep_research.research_one("NVDA", "NVIDIA", "semis", "ctx")
    assert rep["rating"] == "strong_buy" and rep["conviction"] == 10.0
    assert rep["deal_likelihood"] == "high" and rep["catalysts"] == ["one"] and rep["risks"] == []
    assert rep["prompt_version"] == deep_research.PROMPT_VERSION and rep["model"] == "m"


def test_deal_likelihood_is_optional_legacy(fake):
    """v5 stopped asking for it. Demanding it would make every current report invalid;
    coercing it when volunteered keeps the frozen column readable."""
    r = deep_research._coerce_report({"rating": "buy", "conviction": 5})
    assert "deal_likelihood" not in r
    assert deep_research._coerce_report(
        {"rating": "buy", "conviction": 5, "deal_likelihood": "HIGH"})["deal_likelihood"] == "high"
    assert deep_research._coerce_report(
        {"rating": "buy", "conviction": 5, "deal_likelihood": "maybe"})["deal_likelihood"] is None
    assert "deal_likelihood" not in deep_research._PROMPT_TMPL


def test_research_prompt_is_theme_agnostic(fake, monkeypatch):
    """The template must carry no theme of its own — the role and the actors come from
    the pack, so switching themes cannot silently keep pointing Claude at hyperscalers."""
    from src import theme
    monkeypatch.setattr(theme, "configured_name", lambda: "general")
    deep_research.research_one("XYZ", "Acme", "n/a", "ctx")
    prompt = fake.calls[-1]["prompt"]
    assert "AI-infrastructure" not in prompt
    assert "government / defense primes" in prompt      # the general pack's actors


def test_coerce_rejects_garbage_rating():
    r = deep_research._coerce_report({"rating": "moon", "conviction": "eleven"})
    assert r["rating"] is None and r["conviction"] is None


def test_investigate_is_readonly_by_default(fake, monkeypatch):
    monkeypatch.setattr(chat, "build_context", lambda: "ctx")
    chat.investigate("why?")
    assert "Bash" not in fake.calls[-1]["tools"]
    assert "READ-ONLY SESSION" in fake.calls[-1]["prompt"]
    chat.investigate("why?", allow_corrections=True)
    assert "Bash" in fake.calls[-1]["tools"]
    assert "CORRECTIONS" in fake.calls[-1]["prompt"]
    chat.investigate("more", session_id="abc")
    assert fake.calls[-1]["extra_args"] == ["--resume", "abc"]


def test_investigate_refuses_without_sessions(monkeypatch):
    class NoSess(FakeProvider):
        name = "anthropic_api"
        supports_sessions = False
    llm.set_provider(NoSess())
    try:
        ans, _ = chat.investigate("q")
        assert "needs the Claude Code CLI provider" in ans
    finally:
        llm.set_provider(None)


def test_anthropic_provider_with_fake_client():
    class _Blk:
        type, text = "text", "hello"

    class _Msg:
        content = [_Blk()]
        usage = None

    class _Client:
        def __init__(self):
            self.kwargs = None

        def with_options(self, **kw):
            return self

        class messages:  # noqa: N801
            @staticmethod
            def create(**kw):
                _Client.last = kw
                return _Msg()

    p = AnthropicAPIProvider(client=_Client())
    r = p.complete("hi", tools="WebSearch,WebFetch", timeout=30, model="claude-x")
    assert r.text == "hello" and r.model == "claude-x"
    assert _Client.last["tools"][0]["type"].startswith("web_search")
    with pytest.raises(RuntimeError):
        p.complete("hi", extra_args=["--resume", "x"])


def test_get_provider_reads_config(monkeypatch):
    monkeypatch.setattr(llm.config, "get", lambda k, d=None: "claude_cli" if k == "llm.provider" else d)
    assert llm.get_provider().name == "claude_cli"
    monkeypatch.setattr(llm.config, "get", lambda k, d=None: "nope" if k == "llm.provider" else d)
    with pytest.raises(RuntimeError):
        llm.get_provider()


def test_auth_status_is_callable_and_reports_both_facts(monkeypatch):
    """The pipeline calls this right before the research loop. The offline deep-run test
    replaces it with a stub, so a missing import here (it happened: `shutil`) would crash
    a real deep run 40 minutes in and never fail a test. Call the real function."""
    from src import deep_research
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "x")
    a = deep_research.auth_status()
    assert set(a) >= {"claude_on_path", "oauth_token_set"}
    assert a["oauth_token_set"] is True
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    assert deep_research.auth_status()["oauth_token_set"] is False
