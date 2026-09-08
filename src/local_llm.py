"""Local LLM (Ollama) signal stage: classify recent news into a CATALYST VECTOR
(9 fixed keys, 0-10 each) + extracted catalysts. Free and local — conserves Claude usage.

Degrades gracefully: if Ollama is unreachable, returns a neutral signal so the
pipeline continues (scoring treats a missing key as unknown, not as a zero).
"""
from __future__ import annotations

import json
from typing import Any

import requests

from . import config, theme

# THE KEY SET IS FIXED IN PYTHON, NEVER IN A THEME PACK. A theme varies the prose
# (which actors matter, what to emphasise); if it could vary the KEYS, two themes'
# signals would be different measurements wearing the same column, no information
# coefficient could ever be pooled across them, and the single `hyperscaler_signal`
# column — un-poolable and un-measurable by construction — would simply come back
# under another name.
CATALYST_KEYS: tuple[str, ...] = (
    "customer_deal", "ma_event", "guidance_revision", "analyst_action",
    "product_cycle", "regulatory_policy", "capital_return_insider",
    "management_ownership", "adverse_event",
)
# The one axis that scores BADNESS. Kept separate so `scoring._catalyst_pillar`
# subtracts it instead of rewarding a lawsuit as "a catalyst".
NEGATIVE_KEYS: frozenset[str] = frozenset({"adverse_event"})

# 1 = the single hyperscaler_signal; 2 = this vector. Persisted per row so any
# statistic can partition instead of silently pooling two different signals.
SIGNAL_VERSION = 2

# One line per key, straight into the prompt. Wording is the contract: the model can
# only discriminate keys it can tell apart, and "guidance vs analyst action" is a
# SOURCE distinction (company vs sell-side), not a subject-matter one.
_KEY_DEFS = {
    "customer_deal": "large named customer/partner win, supply agreement, or capex-driven demand ({actors})",
    "ma_event": "the company is an acquisition target or acquirer, an activist takes a stake, or a strategic review starts",
    "guidance_revision": "COMPANY-sourced outlook change: raised/cut guidance, pre-announcement, beat/miss commentary",
    "analyst_action": "SELL-SIDE-sourced: upgrades, downgrades, price-target or estimate revisions",
    "product_cycle": "launch, regulatory/product approval, design win, capacity or node ramp",
    "regulatory_policy": "FAVOURABLE government/regulatory action ONLY — approvals, subsidies, tariff relief, won litigation. Adverse regulatory action (probes, fines, adverse rulings) belongs in adverse_event; score regulatory_policy 0 for those",
    "capital_return_insider": "buyback authorization, dividend initiation/raise, insider buying",
    "management_ownership": "CEO/CFO transition, board change, major ownership shift",
    "adverse_event": "NEGATIVE events: probe, lawsuit, guidance cut, dilution, downgrade, customer loss, accident, recall",
}


def _key_lines() -> str:
    actors = theme.actors_text()
    return "\n".join(f"  - {k}: {_KEY_DEFS[k].format(actors=actors)}"
                     for k in CATALYST_KEYS)


# THE CLASSIFIER IS THEME-NEUTRAL; THE RESEARCH PROMPT IS NOT.
# The division of labour: this stage produces MEASURED FEATURES, so it must score the
# same evidence the same way under every theme — otherwise `customer_deal 8` means
# something different in each pack and the two can never be pooled or IC'd, which is
# the exact defect the single hyperscaler_signal had. The pack's `actors` DO appear,
# in the key definitions only: naming AWS is RECOGNITION ("this counterparty is a
# large customer"), not weighting. What is deliberately absent is the pack's
# `catalyst_emphasis` ("weigh hyperscaler capex heavily") — that is a judgement, and
# judgement belongs in deep_research's prompt, where the output is an opinion rather
# than a feature.
_SYSTEM = (
    "You are a financial news classifier for US stocks. Given recent headlines "
    "about one company, score how strongly the news evidences each of nine "
    "catalyst types over the coming weeks-to-months. Score each key on the evidence "
    "alone; do not privilege any one sector's storyline. Return ONLY JSON."
)


def _system() -> str:
    return _SYSTEM


def _prompt_tmpl() -> str:
    return """Company: {name} ({symbol})

Recent headlines (newest first, with dates):
{headlines}

Return JSON with exactly these keys:
{{"catalyst": {{""" + ", ".join(f'"{k}": <integer 0-10>' for k in CATALYST_KEYS) + """}}, "sentiment": "bullish"|"neutral"|"bearish", "sentiment_score": <integer 0-10>, "catalysts": [<short strings>], "summary": "<one sentence>"}}
- catalyst: score EACH key 0-10 on the evidence in THESE headlines. 0 = no evidence
  of that event type (the common, correct answer — most keys are 0 for most names);
  10 = strong, imminent, well-evidenced. Do NOT spread a single story across keys.
""" + _key_lines() + """
- sentiment: overall tone of the recent news for the stock.
- sentiment_score: 0 = very bearish, 5 = neutral, 10 = very bullish.
- catalysts: up to 4 concrete near-term catalysts from the headlines (empty if none).
- Recency decay (numeric rule): items 0-3 days old get FULL weight, 4-7 days HALF
  weight, older than 7 days QUARTER weight. Yesterday's downgrade outweighs a
  two-week-old price-target raise.
"""


# JSON schema Ollama enforces via constrained decoding (stricter than format:"json").
# Every catalyst key is REQUIRED: the model literally cannot omit one, so a 0 always
# means "the model looked and found nothing", never "the model forgot".
_CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "catalyst": {
            "type": "object",
            "properties": {k: {"type": "integer", "minimum": 0, "maximum": 10}
                           for k in CATALYST_KEYS},
            "required": list(CATALYST_KEYS),
        },
        "sentiment": {"type": "string", "enum": ["bullish", "neutral", "bearish"]},
        "sentiment_score": {"type": "integer", "minimum": 0, "maximum": 10},
        "catalysts": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["catalyst", "sentiment", "sentiment_score", "catalysts", "summary"],
}


def _unknown_vector() -> dict[str, None]:
    """All nine keys UNKNOWN. Distinct from all-zero: zero is measured evidence of
    absence, None is the absence of a measurement, and only the latter may be
    neutralised downstream."""
    return {k: None for k in CATALYST_KEYS}


def _host() -> str:
    return config.get("local_llm.host", "http://localhost:11434").rstrip("/")


def available() -> bool:
    if not config.get("local_llm.enabled", True):
        return False
    try:
        r = requests.get(f"{_host()}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _neutral(symbol: str, reason: str) -> dict[str, Any]:
    cat = _unknown_vector()
    return {"symbol": symbol, "catalyst": cat, "hyperscaler_signal": None,
            "signal_version": SIGNAL_VERSION, "sentiment": None,
            "sentiment_score": None, "catalysts": [], "summary": "", "error": reason}


def classify(symbol: str, name: str, headlines: list[str],
             check_available: bool = True) -> dict[str, Any]:
    if not headlines:
        # No news → neutral tone, and every catalyst key is UNKNOWN (None), not a
        # bearish 0. A hard 0 here would penalize a name we simply never scanned as
        # if we had scanned it and found nothing.
        return {"symbol": symbol, "catalyst": _unknown_vector(),
                "hyperscaler_signal": None, "signal_version": SIGNAL_VERSION,
                "sentiment": "neutral", "sentiment_score": 5, "catalysts": [],
                "summary": "No recent news."}
    if check_available and not available():
        return _neutral(symbol, "ollama_unavailable")

    model = config.get("local_llm.model", "qwen3.5:9b")
    timeout = config.get("local_llm.timeout_seconds", 120)
    hl = "\n".join(f"- {h}" for h in headlines[:12])
    prompt = _prompt_tmpl().format(name=name, symbol=symbol, headlines=hl)

    try:
        resp = requests.post(
            f"{_host()}/api/generate",
            json={
                "model": model,
                "system": _system(),
                "prompt": prompt,
                "format": _CLASSIFY_SCHEMA,  # schema-constrained decoding (valid JSON guaranteed)
                "stream": False,
                "think": config.get("local_llm.think", False),  # disable reasoning models' <think> output
                "options": {"temperature": 0.2},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        data = json.loads(text)
    except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
        return _neutral(symbol, f"error:{type(exc).__name__}")

    def _int0_10(v):
        try:
            return max(0, min(10, int(round(float(v)))))
        except (TypeError, ValueError):
            return None

    raw_cat = data.get("catalyst")
    raw_cat = raw_cat if isinstance(raw_cat, dict) else {}
    # A key the model omitted (impossible under the schema, but the schema is the
    # server's promise, not ours) stays None = unmeasured, never a fabricated 0.
    cat = {k: _int0_10(raw_cat.get(k)) for k in CATALYST_KEYS}
    sent_score = _int0_10(data.get("sentiment_score"))
    sentiment = str(data.get("sentiment", "")).strip().lower() or None
    if sentiment not in ("bullish", "neutral", "bearish"):
        sentiment = None
    cats = data.get("catalysts") or []
    if not isinstance(cats, list):
        cats = [str(cats)]
    return {
        "symbol": symbol,
        "catalyst": cat,
        # DUAL-WRITE, display continuity only: `customer_deal` is the direct heir of
        # hyperscaler_signal, so the legacy column and everything reading it keep
        # rendering while signal_version tells statistics which signal they hold.
        "hyperscaler_signal": cat["customer_deal"],
        "signal_version": SIGNAL_VERSION,
        "sentiment": sentiment,
        "sentiment_score": sent_score,
        "catalysts": [str(c) for c in cats][:4],
        "summary": str(data.get("summary", ""))[:300],
    }


def classify_from_news(symbol: str, name: str, news_items: list[dict],
                       check_available: bool = True) -> dict[str, Any]:
    from datetime import datetime, timezone

    def _age_days(it: dict) -> int | None:
        s = it.get("published", "")
        try:
            from email.utils import parsedate_to_datetime
            d = parsedate_to_datetime(s)
        except Exception:  # noqa: BLE001
            try:
                d = datetime.strptime(str(s)[:10], "%Y-%m-%d")
            except ValueError:
                return None
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return max(0, (datetime.now(timezone.utc) - d).days)

    def _line(it: dict) -> str:
        t = it.get("title", "")
        age = _age_days(it)
        return f"[{age}d ago] {t}" if age is not None else t

    return classify(symbol, name, [_line(it) for it in news_items],
                    check_available=check_available)


def score_headlines(headlines: list[str], timeout: float | None = None) -> list[dict[str, Any]]:
    """Per-headline sentiment in ONE local call. Returns a list aligned to input:
    [{"sentiment": "bullish|neutral|bearish"|None, "score": 0-10|None}, ...].

    `timeout` overrides the config timeout — the dashboard passes a short value so a
    cache-miss on the render path can't freeze the page for the full pipeline timeout.
    """
    n = len(headlines)
    if n == 0:
        return []
    if not available():
        return [{"sentiment": None, "score": None} for _ in headlines]

    numbered = "\n".join(f"{i}: {h}" for i, h in enumerate(headlines[:15]))
    prompt = (
        "Score the sentiment of each stock-news headline (0=very bearish, 5=neutral, "
        "10=very bullish). Return ONLY a JSON object mapping each index to its result, e.g.\n"
        '{"0": {"sentiment": "bullish", "score": 8}, "1": {"sentiment": "bearish", "score": 3}}\n\n'
        f"Headlines:\n{numbered}")
    try:
        resp = requests.post(
            f"{_host()}/api/generate",
            json={"model": config.get("local_llm.model", "qwen3.5:9b"),
                  "prompt": prompt, "format": "json", "stream": False,
                  "think": config.get("local_llm.think", False),
                  "options": {"temperature": 0.1}},
            timeout=timeout or config.get("local_llm.timeout_seconds", 120))
        resp.raise_for_status()
        data = json.loads(resp.json().get("response", "{}"))
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return [{"sentiment": None, "score": None} for _ in headlines]

    out = []
    for i in range(n):
        d = data.get(str(i)) or {}
        try:
            sc = max(0, min(10, int(round(float(d.get("score"))))))
        except (TypeError, ValueError):
            sc = None
        sent = str(d.get("sentiment", "")).strip().lower()
        sent = sent if sent in ("bullish", "neutral", "bearish") else None
        out.append({"sentiment": sent, "score": sc})
    return out


if __name__ == "__main__":
    print("Ollama available:", available())
    demo = [
        "[1d ago] Vertiv lands major data center cooling order tied to AI buildout",
        "[2d ago] Analysts raise Vertiv targets on hyperscaler capex surge",
        "[5d ago] Vertiv expands liquid cooling capacity for NVIDIA GPU racks",
    ]
    print(json.dumps(classify("VRT", "Vertiv", demo), indent=2))
