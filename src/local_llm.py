"""Local LLM (Ollama) signal stage: classify recent news into a hyperscaler
signal score + extracted catalysts. Free and local — conserves Claude usage.

Degrades gracefully: if Ollama is unreachable, returns a neutral signal so the
pipeline continues (scoring treats a missing signal as neutral 5/10).
"""
from __future__ import annotations

import json
from typing import Any

import requests

from . import config

_HYPERSCALERS = ("AWS", "Amazon", "Azure", "Microsoft", "Google Cloud", "GCP",
                 "Oracle", "OCI", "Meta", "NVIDIA", "hyperscaler", "data center",
                 "cloud capex", "AI infrastructure")

_SYSTEM = (
    "You are a financial news classifier for AI-infrastructure stocks. "
    "Given recent headlines about one company, judge how likely the company is "
    "to benefit from large hyperscaler deals/capex (AWS, Azure, Google Cloud, "
    "Oracle, Meta, NVIDIA ecosystem) in the coming weeks-to-months. "
    "Return ONLY JSON."
)

_PROMPT_TMPL = """Company: {name} ({symbol})

Recent headlines (newest first, with dates):
{headlines}

Return JSON with exactly these keys:
{{"hyperscaler_signal": <integer 0-10>, "sentiment": "bullish"|"neutral"|"bearish", "sentiment_score": <integer 0-10>, "catalysts": [<short strings>], "summary": "<one sentence>"}}
- hyperscaler_signal: 0 = no relevance, 10 = strong imminent hyperscaler-deal signal.
- sentiment: overall tone of the recent news for the stock.
- sentiment_score: 0 = very bearish, 5 = neutral, 10 = very bullish.
- catalysts: up to 4 concrete near-term catalysts from the headlines (empty if none).
- Recency decay (numeric rule): items 0-3 days old get FULL weight, 4-7 days HALF
  weight, older than 7 days QUARTER weight. Yesterday's downgrade outweighs a
  two-week-old price-target raise.
"""

# JSON schema Ollama enforces via constrained decoding (stricter than format:"json").
_CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "hyperscaler_signal": {"type": "integer", "minimum": 0, "maximum": 10},
        "sentiment": {"type": "string", "enum": ["bullish", "neutral", "bearish"]},
        "sentiment_score": {"type": "integer", "minimum": 0, "maximum": 10},
        "catalysts": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["hyperscaler_signal", "sentiment", "sentiment_score", "catalysts", "summary"],
}


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
    return {"symbol": symbol, "hyperscaler_signal": None, "sentiment": None,
            "sentiment_score": None, "catalysts": [], "summary": "", "error": reason}


def classify(symbol: str, name: str, headlines: list[str],
             check_available: bool = True) -> dict[str, Any]:
    if not headlines:
        # No news → neutral tone, and hyperscaler relevance is UNKNOWN (None), not a
        # bearish 0. A hard 0 here would wrongly penalize the sentiment pillar and drag
        # the deal signal below a name we simply never scanned.
        return {"symbol": symbol, "hyperscaler_signal": None, "sentiment": "neutral",
                "sentiment_score": 5, "catalysts": [], "summary": "No recent news."}
    if check_available and not available():
        return _neutral(symbol, "ollama_unavailable")

    model = config.get("local_llm.model", "qwen3.5:9b")
    timeout = config.get("local_llm.timeout_seconds", 120)
    hl = "\n".join(f"- {h}" for h in headlines[:12])
    prompt = _PROMPT_TMPL.format(name=name, symbol=symbol, headlines=hl)

    try:
        resp = requests.post(
            f"{_host()}/api/generate",
            json={
                "model": model,
                "system": _SYSTEM,
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

    sig = _int0_10(data.get("hyperscaler_signal"))
    sent_score = _int0_10(data.get("sentiment_score"))
    sentiment = str(data.get("sentiment", "")).strip().lower() or None
    if sentiment not in ("bullish", "neutral", "bearish"):
        sentiment = None
    cats = data.get("catalysts") or []
    if not isinstance(cats, list):
        cats = [str(cats)]
    return {
        "symbol": symbol,
        "hyperscaler_signal": sig,
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
        "Vertiv lands major data center cooling order tied to AI buildout",
        "Analysts raise Vertiv targets on hyperscaler capex surge",
        "Vertiv expands liquid cooling capacity for NVIDIA GPU racks",
    ]
    print(json.dumps(classify("VRT", "Vertiv", demo), indent=2))
