"""Claude Trader dashboard (Streamlit).

Run from the project root:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

try:
    from streamlit_autorefresh import st_autorefresh  # noqa: E402
except Exception:  # noqa: BLE001
    st_autorefresh = None

from src import charts, config, health, news, recommend, regime, storage, theme  # noqa: E402
from src.local_llm import CATALYST_KEYS  # noqa: E402
from src.universe import SUBSECTOR_LABELS  # noqa: E402

st.set_page_config(page_title="Claude Trader", layout="wide", page_icon="📈")
# Idempotent schema migration: the dashboard may be the first process to open a DB
# after an upgrade adds a column (e.g. runs.health), and SELECT * rows must have it.
storage.init_db()

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1480px; }

/* Metric cards */
[data-testid="stMetric"] { background:#161b22; border:1px solid #21262d; border-radius:14px; padding:14px 18px; }
[data-testid="stMetricValue"] { font-family:'JetBrains Mono', monospace; font-weight:600; color:#e6edf3; }
[data-testid="stMetricLabel"] p { color:#8b949e; font-weight:500; }

/* Tabs -> segmented control */
[data-baseweb="tab-list"] { gap:4px; background:#161b22; padding:5px; border-radius:12px; border:1px solid #21262d; }
[data-baseweb="tab"] { border-radius:9px; padding:7px 18px; color:#8b949e; font-weight:600; }
button[aria-selected="true"][data-baseweb="tab"] { background:#0e1117; color:#2dd4bf; box-shadow:inset 0 0 0 1px #2dd4bf33; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display:none; }

/* Expanders & dataframes */
[data-testid="stExpander"] { border:1px solid #21262d; border-radius:14px; background:#13181f; overflow:hidden; }
[data-testid="stExpander"] summary:hover { color:#2dd4bf; }
[data-testid="stDataFrame"] { font-variant-numeric: tabular-nums; border-radius:12px; }

/* Headings */
h1 { font-weight:700; letter-spacing:-0.025em; }
h2, h3 { font-weight:600; letter-spacing:-0.015em; }
hr { border-color:#21262d; }

/* Buttons */
.stButton button { border-radius:10px; border:1px solid #21262d; font-weight:600; }
.stButton button:hover { border-color:#2dd4bf; color:#2dd4bf; }

/* Recommendation pills */
.pill { display:inline-block; padding:4px 14px; border-radius:999px; font-weight:700; font-size:0.95rem; letter-spacing:0.01em; }
.pill-strong_buy{ background:rgba(34,197,94,.18); color:#4ade80; border:1px solid rgba(34,197,94,.45); }
.pill-buy,.pill-add,.pill-add_dip{ background:rgba(45,212,191,.16); color:#2dd4bf; border:1px solid rgba(45,212,191,.45); }
.pill-hold,.pill-trim_wait{ background:rgba(234,179,8,.15); color:#eab308; border:1px solid rgba(234,179,8,.4); }
.pill-sell,.pill-trim{ background:rgba(249,115,22,.16); color:#fb923c; border:1px solid rgba(249,115,22,.45); }
.pill-strong_sell,.pill-exit{ background:rgba(239,68,68,.16); color:#f87171; border:1px solid rgba(239,68,68,.45); }
.pill-avoid,.pill-unrated{ background:rgba(139,148,158,.14); color:#8b949e; border:1px solid rgba(139,148,158,.35); }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


def pill(label_key: str, text: str, screen: bool = False) -> str:
    return f"<span class='pill pill-{label_key}'>{text}{' *' if screen else ''}</span>"


# Tags are IDENTITY, not status — they use neutral glyphs so the traffic-light
# palette stays reserved for genuinely good→bad axes (recommendation/action/regime).
# "🟢 Buy … 🔴 higher-risk" on one row used to read as a contradiction.
TAG_BADGE = {"quality": "◇ quality",
             "higher-risk": "⚠ higher-risk", "earnings-soon": "🗓 earnings-soon",
             # Both are statements about DATA, not about the company: "no trailing P/E"
             # and "too few fundamentals to rank". They used to be silently folded into
             # higher-risk, which flagged every pre-revenue name as risky.
             "no-earnings": "◌ no-earnings", "thin-data": "◌ thin-data"}
ACTION_BADGE = {"add": "🟢 Add", "hold": "🟡 Hold", "trim": "🟠 Trim", "exit": "🔴 Exit",
                "add_dip": "🟢↘ Add on dip", "trim_wait": "🟡 Hold→trim"}
SETUP_EMOJI = {"bullish": "📈", "bearish": "📉", "neutral": "➖"}


# ----------------------------------------------------------------- helpers
def fmt_mcap(v) -> str:
    if not v:
        return "—"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if v >= div:
            return f"${v / div:.1f}{unit}"
    return f"${v:,.0f}"


def fmt_pct(v) -> str:
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "—"


def fmt_num(v, nd=1) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def fmt_bn(v) -> str:
    if not isinstance(v, (int, float)):
        return "—"
    sign = "-" if v < 0 else ""
    a = abs(v)
    for unit, div in (("B", 1e9), ("M", 1e6)):
        if a >= div:
            return f"{sign}${a/div:.1f}{unit}"
    return f"{sign}${a:,.0f}"


def reconciled_action(p) -> tuple[str, str]:
    """Make the deep-dive action consistent with today's price vs its add/trim levels,
    but keep them visually distinct from a genuine Hold."""
    a = (p.get("action") or "").lower()
    last, add, trim = p.get("last"), p.get("add_level"), p.get("trim_level")
    if a == "add" and isinstance(last, (int, float)) and isinstance(add, (int, float)) and last > add * 1.02:
        return "add_dip", f"≤{add:g}"
    if a == "trim" and isinstance(last, (int, float)) and isinstance(trim, (int, float)) and last < trim * 0.98:
        return "trim_wait", f"≥{trim:g}"
    return a, ""


def vs_sector(val, med) -> str:
    if not isinstance(val, (int, float)) or not isinstance(med, (int, float)) or not med:
        return fmt_num(val)
    prem = val / med - 1
    arrow = "▲" if prem > 0 else "▼"
    return f"{val:.1f} vs {med:.1f} ({arrow}{abs(prem)*100:.0f}%)"


def tag_str(tags) -> str:
    return " ".join(TAG_BADGE.get(t, t) for t in (tags or []))


# Short labels for the catalyst column: nine snake_case keys do not fit a table cell.
CATALYST_SHORT = {"customer_deal": "deal", "ma_event": "M&A",
                  "guidance_revision": "guide", "analyst_action": "analyst",
                  "product_cycle": "product", "regulatory_policy": "policy",
                  "capital_return_insider": "buyback",
                  "management_ownership": "mgmt", "adverse_event": "adverse"}


def catalyst_str(row, top: int = 2) -> str:
    """The `top` strongest NON-ZERO catalyst keys, e.g. "deal 8 · product 6".

    Zeros are dropped rather than shown: most keys are legitimately 0 for most names
    on most days, and rendering seven zeros would hide the one key that fired."""
    cat = (row or {}).get("catalyst_json") or ((row or {}).get("metrics") or {}).get("catalyst")
    if not isinstance(cat, dict):
        return ""
    hits = [(k, v) for k, v in cat.items()
            if k in CATALYST_KEYS and isinstance(v, (int, float)) and v]
    hits.sort(key=lambda kv: -kv[1])
    return " · ".join(f"{CATALYST_SHORT.get(k, k)} {int(v)}" for k, v in hits[:top])


def md(text) -> str:
    """Escape $ so Streamlit markdown doesn't render '$...$' spans as LaTeX math."""
    return str(text).replace("$", "\\$") if text not in (None, "") else ""


SENT_EMOJI = {"bullish": "🟢", "neutral": "⚪", "bearish": "🔴"}


def sent_badge(s) -> str:
    s = s or {}
    e = SENT_EMOJI.get(s.get("sentiment"), "•")
    sc = f"{s['score']}/10" if s.get("score") is not None else "?"
    return f"{e} {sc}"


def fmt_date(s) -> str:
    if not s:
        return "—"
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(s).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return str(s)[:10]


def freshness(data_as_of) -> str:
    """Render a research 'data as of' date with a recency badge vs today."""
    if not data_as_of:
        return "data-as-of date not reported"
    try:
        d = datetime.fromisoformat(str(data_as_of)[:10]).date()
    except ValueError:
        return f"data as of {data_as_of}"
    days = (datetime.now(timezone.utc).date() - d).days
    tag = ("✅ today" if days <= 0 else f"✅ {days}d ago" if days <= 2
           else f"🟡 {days}d ago" if days <= 7 else f"⚠️ {days}d old")
    return f"data as of {d.isoformat()} ({tag})"


# Plain-language explanations shown on hovering the ⓘ next to each metric.
GLOSSARY = {
    "Price": "The latest traded price, delayed about 15 minutes (from IBKR). Every valuation ratio and trade level below is computed against this price, so they move as the price moves.",
    "Mkt cap": "Market capitalization = share price x total shares outstanding — the market's price tag on the company's entire equity. It tells you the size class (small / mid / large / mega-cap) and lets you compare valuation ratios fairly: a low P/E on a tiny company is far riskier than the same P/E on a mega-cap.",
    "PE": "Trailing Price/Earnings: share price divided by the last 12 months of earnings per share — roughly what you pay for $1 of current annual profit. Lower looks cheaper, but 'cheap' is relative: fast-growing tech routinely trades at 30-50x, and a very low P/E can signal either a bargain or a declining business. Compare only within the same sector, and note it's blank/negative for unprofitable companies.",
    "Fwd PE": "Forward P/E uses analysts' estimate of NEXT year's earnings instead of the past. If it's well below the trailing P/E, the market expects earnings to grow; if it's higher, earnings are expected to shrink. Estimates are often wrong, so read it as an expectation gauge, not a fact.",
    "P/FCF": "Price divided by free cash flow (operating cash flow minus capital spending). Free cash flow is much harder to manipulate than reported earnings, so this is a stricter value measure — approximately how many years of cash generation it would take to pay back the share price. Lower = cheaper. Capital-heavy names building out (data centers, chip fabs) often show a high P/FCF while they invest.",
    "FCF yield": "Free cash flow divided by market cap — the inverse of P/FCF expressed as a yield. It's the cash 'return' the business throws off relative to its price; higher is better. Above ~5% is healthy for a mature company, while growth names reinvesting aggressively can show low or even negative yields.",
    "Rev growth": "Year-over-year revenue growth — the top-line momentum of the business. For medium-swing ideas you generally want strong, ideally accelerating growth; decelerating growth even at a 'cheap' valuation is a warning sign.",
    "Gross margin": "Gross profit (revenue minus the direct cost of goods/services) as a percent of revenue. It measures pricing power and product economics — software and chip-IP businesses run 60-80%, hardware and equipment makers lower. Rising gross margin is a strong quality signal; falling margin hints at pricing pressure or rising input costs.",
    "ROE": "Return on Equity = net income divided by shareholder equity — how much profit the company produces per $1 of equity capital, a core efficiency/quality measure. Consistently high ROE (>15-20%) points to a strong franchise, but heavy debt can inflate it, so read it alongside the company's leverage.",
    "RSI(14)": "Relative Strength Index over 14 days, scaled 0-100 — a momentum oscillator. Above 70 is 'overbought' (stretched, prone to a pullback), below 30 is 'oversold' (beaten down, prone to a bounce), and 40-65 is a healthy uptrend zone. It's a timing aid, not a standalone buy/sell signal — strong leaders can stay overbought for weeks.",
    "52w position": "Where today's price sits inside its 52-week range: 0% = at the yearly low, 100% = at the yearly high. High readings show momentum and leadership; very high (>95%) can mean the move is extended. Low readings mean the stock is out of favor — which is either value or a falling knife.",
    "ATR %": "Average True Range as a percent of price — the stock's typical daily swing, i.e. its volatility. A 5% ATR name moves far more than a 1.5% one, so it needs wider stops and smaller position sizes. It's what the volatility-based stop below is built from.",
    "Suggested stop (2·ATR)": "A volatility-based stop-loss idea: current price minus two times the ATR. Placing the stop beyond two normal daily moves means random noise is less likely to shake you out while still capping the downside. It's a generic starting reference for sizing — not financial advice and not tuned to your personal risk tolerance.",
    "Support (SMA50)": "The 50-day simple moving average — the average closing price over the last ~10 weeks. It acts as a dynamic medium-term support/resistance line: price holding above a rising SMA50 is a healthy swing-trade backdrop, and pullbacks toward it are common entry zones.",
    "Support (SMA200)": "The 200-day moving average — the long-term trend line (~10 months of price). It's the line most institutions watch: above it = long-term uptrend, below it = downtrend. Breaks of the 200-day often mark major trend changes, making it a key support/resistance level.",
    "52w high": "The highest price over the last 52 weeks — a reference for how far the stock has run and where prior resistance/euphoria sits.",
    "52w low": "The lowest price over the last 52 weeks — a reference for the downside the stock has seen and where prior support/capitulation sits.",
    "Dist to 52w high": "How far below the 52-week high the price currently is. Near 0% means it's at or near new highs (momentum leadership); a large gap means it's well off its peak — either room to recover or broken momentum.",
    "Next earnings": "The next scheduled earnings report date. Earnings are the single biggest scheduled catalyst — stocks routinely gap several percent on the result. For a swing trade, know whether you'd be holding through the print (higher risk and reward) and size the position accordingly.",
    "EV/EBITDA": "Enterprise value (market cap + debt − cash) divided by EBITDA. It values the whole business against pre-depreciation operating profit, so it compares companies with different debt loads or tax situations better than P/E. Lower = cheaper; ~10-15x is typical, higher for fast growers.",
    "EV/Sales": "Enterprise value divided by revenue. The go-to multiple for high-growth or not-yet-profitable names where P/E is meaningless. Lower = cheaper, but fast-growing AI/software names routinely trade at 10x+ — judge it against growth.",
    "PEG": "P/E divided by the earnings growth rate — valuation in the context of growth. Around 1.0 is 'fairly priced for its growth', below ~1 can be cheap, above ~2 expensive. A high P/E with fast growth can still have a reasonable PEG.",
    "Rule of 40": "Revenue growth % + free-cash-flow margin % — a quick health check for growth/software names. 40 or above means the trade-off between growing and making money is healthy; well below 40 is a yellow flag.",
    "Net cash": "Total cash minus total debt. Positive = net cash (balance-sheet strength and flexibility to invest or weather a downturn); negative = net debt (leverage, which adds risk if conditions tighten).",
    "PE vs sector": "This stock's trailing P/E against the median P/E of its sub-sector. ▲ = trading at a premium to peers (quality, or over-valuation), ▼ = a discount (value, or a problem). Relative valuation is more useful than an absolute number.",
    "Fwd PE vs sector": "Forward P/E against the sub-sector median forward P/E — the same premium/discount read, but on next year's expected earnings.",
    "EV/EBITDA vs sector": "EV/EBITDA against the sub-sector median — premium/discount on a debt- and tax-neutral basis.",
}

# Wider explanations of the scores/columns (shown in the glossary panel).
SCORE_GLOSSARY = {
    "Recommendation": "Claude's overall medium-term call after reading the three pillars plus its own web research: 🟩 Strong Buy / 🟢 Buy / 🟡 Hold / 🟠 Sell / 🔴 Strong Sell. A trailing * means it's a provisional screen guess (that name wasn't deep-researched this run), so trust it less.",
    "Composite": "The 0–1 ranking score that orders the board, a weighted blend of the three pillars (Technical 35% / Fundamental 30% / Sentiment 35%). It's relative to the day's peer set, not an absolute grade.",
    "📊 Technical": "A 0–1 score of the chart/price action: trend (price vs the 50/200-day averages and whether they've crossed), RSI, MACD, where price sits in its 52-week range, and volume confirmation. High = healthy uptrend with momentum.",
    "💰 Fundamental": "A 0–1 score of the business and valuation, percentile-ranked against the day's peer set: valuation (P/E, P/FCF, etc.), cash flow, revenue/earnings growth, and quality (margins, ROE, debt). High = cheaper and higher-quality relative to peers.",
    "📰 Sentiment": "A 0–1 score of recent news TONE only (scored bullish/neutral/bearish by the local LLM). Events — deals, approvals, lawsuits — are scored separately in the catalyst vector, so the same news is never counted twice.",
    "Conviction": "Claude's 0–10 confidence in its call after the deep research (8+ = high conviction). Only names that were deep-researched have one; others are blank.",
    "Tags": "Quick labels: ◇ quality (strong cash flow + reasonable valuation) · ⚠ higher-risk (unprofitable or expensively valued) · 🗓 earnings-soon (reports within a week). Tags are neutral identity markers — the colored traffic-light scale is reserved for recommendations/actions.",
    "⚡ Catalyst": "The nine-key catalyst vector the local LLM scores from this week's headlines (customer_deal, ma_event, guidance_revision, analyst_action, product_cycle, regulatory_policy, capital_return_insider, management_ownership, adverse_event), 0–10 each. The column shows the two strongest non-zero keys. SHIPPED DARK: it is recorded every run but carries no weight in the ranking score until forward data earns it one.",
    "Setup": "A rule-based read of the trend: 📈 bullish (price above a rising, stacked 50/200-day MA structure with MACD up), 📉 bearish (the mirror image), ➖ neutral/range. Built from moving-average structure, golden/death crosses, MACD and RSI — robust signals, not fuzzy shape-matching.",
    "Relative strength": "The stock's price return minus the benchmark's (default SPY) over 3 and 6 months, in percentage points. Positive = outperforming the market (leadership); negative = lagging. One of the most predictive momentum signals — leaders tend to keep leading.",
    "Chart read": "Claude's qualitative description of the current chart setup/pattern (breakout, base/consolidation, cup-and-handle, head-and-shoulders, downtrend) from recent price action — with the caveat that chart patterns are probabilistic, not certainties.",
    "Sell signals (invalidation)": "The pre-agreed conditions that would prove the investment thesis WRONG — e.g. 'the Microsoft contract is cancelled' or 'price breaks below $34'. Deciding these in advance (before you're emotionally attached) is what separates a plan from a hope. The 🤖 line shows which of them the system converts into automatic tripwires checked on every run: price levels and news keywords. When one trips, a 🚨 alert appears at the top of the dashboard the same day.",
}


def metric_rows_html(rows: list[tuple]) -> str:
    """Render (label, value) rows as a compact table with a hover-tooltip ⓘ per label."""
    out = ["<table style='width:100%;font-size:0.9em;border-collapse:collapse'>"]
    for label, value in rows:
        tip = GLOSSARY.get(label, "")
        info = (f" <abbr title=\"{tip}\" style='cursor:help;opacity:0.55;text-decoration:none'>ⓘ</abbr>"
                if tip else "")
        out.append(f"<tr><td style='padding:3px 8px 3px 0;opacity:0.85'>{label}{info}</td>"
                   f"<td style='text-align:right;padding:3px 0'><b>{value}</b></td></tr>")
    out.append("</table>")
    return "".join(out)


def ago(iso_ts: str | None) -> str:
    if not iso_ts:
        return "unknown"
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return iso_ts
    mins = int((datetime.now(timezone.utc) - dt).total_seconds() // 60)
    local = dt.astimezone().strftime("%Y-%m-%d %H:%M")
    rel = ("just now" if mins < 1 else f"{mins} min ago" if mins < 60
           else f"{mins // 60}h {mins % 60}m ago" if mins < 1440 else f"{mins // 1440}d ago")
    return f"{local} ({rel})"


@st.cache_data(ttl=120)
def load_board(run_id: int):
    return storage.get_board(run_id)


@st.cache_data(ttl=60)
def load_holdings() -> dict:
    from contextlib import closing
    with closing(storage.get_conn()) as cn:
        return {h["symbol"]: dict(h) for h in cn.execute("SELECT * FROM holdings").fetchall()}


@st.cache_data(ttl=60)
def load_quote_map() -> dict:
    from contextlib import closing
    with closing(storage.get_conn()) as cn:
        return {q["symbol"]: q["last"]
                for q in cn.execute("SELECT symbol,last FROM quotes").fetchall()}


@st.cache_data(ttl=120)
def load_score_history(symbol: str) -> list:
    return storage.score_history(symbol)


@st.cache_data(ttl=900)
def load_regime() -> dict:
    return regime.detect(use_cache=True)  # benchmark-level only; daily cache


@st.cache_data(ttl=120)
def load_fired_triggers() -> list:
    return storage.recent_fired_triggers(days=7)


@st.cache_data(ttl=120)
def load_active_triggers_by_symbol() -> dict:
    out: dict = {}
    for t in storage.active_triggers():
        out.setdefault(t["symbol"], []).append(t)
    return out


@st.cache_data(ttl=3600)
def load_calibration():
    from src import performance
    return performance.rating_calibration(), performance.shadow_vs_actual()


@st.cache_data(ttl=3600)
def load_calibration_extra():
    """Claude-vs-screen attribution, precision-with-CI table, per-prompt-version cut."""
    from src import performance
    return (performance.claude_vs_screen("4w"), performance.calibration_table("4w"),
            performance.calibration_by_prompt_version("4w"))


@st.cache_data(ttl=900)
def get_frame(symbol: str):
    # Fetch 2y so SMA200 / RSI / MACD are valid even when viewing a short window.
    df = charts.price_frame(symbol, period="2y")
    return charts.add_indicators(df) if df is not None else None


CHART_GLOSSARY = {
    "Candlesticks": "Each candle is one day: the body spans open→close (green = up day, red = down), the thin wicks mark the day's high and low. Long bodies = decisive moves; long wicks = rejection/indecision.",
    "SMA 20 / 50 / 200": "Simple moving averages of the close over 20, 50 and 200 days — they smooth noise to reveal the trend. Price above rising MAs = uptrend; the 50 crossing above the 200 is a 'golden cross' (bullish), below is a 'death cross' (bearish). MAs often act as support/resistance.",
    "Volume": "Shares traded per day — the conviction behind a move. Rising volume on up-days confirms strength; a move on light volume is suspect; big spikes often mark breakouts or capitulation.",
    "RSI (14)": "Momentum, 0–100. Above 70 = overbought (stretched, pullback risk), below 30 = oversold (bounce candidate), 40–65 = healthy trend. RSI diverging from price can warn of a reversal.",
    "MACD": "Trend-momentum: the MACD line vs its signal line, with a histogram of the gap. MACD above signal / positive histogram = bullish momentum; crossing below = momentum fading. Useful for catching shifts early.",
}

_TF_DAYS = {"1M": 22, "3M": 66, "6M": 132, "1Y": 252, "2Y": 504}


@st.fragment
def render_price_chart(symbol: str, key: str, chart_note: str | None = None,
                       levels: dict | None = None, earnings: str | None = None,
                       avg_cost: float | None = None, add_level: float | None = None,
                       trim_level: float | None = None) -> None:
    # A fragment: the timeframe radio reruns ONLY this chart, not the whole 800-line app.
    frame = get_frame(symbol)
    if frame is None:
        st.caption("No price history available.")
        return
    tf = st.radio("Timeframe", list(_TF_DAYS), index=2, horizontal=True, key=f"tf_{key}")
    sliced = frame.tail(_TF_DAYS[tf])
    fig = candlestick_fig(sliced, symbol, levels=levels, earnings=earnings,
                          avg_cost=avg_cost, add_level=add_level, trim_level=trim_level)
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": False})
    if chart_note:
        st.info(f"📐 **Claude's read of {symbol}'s chart:** {md(chart_note)}")
    with st.expander("📖 How to read these charts (general guide)"):
        for k, v in CHART_GLOSSARY.items():
            st.markdown(f"- **{k}** — {v}")


@st.cache_data(ttl=900)
def get_levels(symbol: str):
    # Derive from the already-cached 2y frame (a superset of the 1y default) instead of
    # a second yfinance download per ticker.
    df = get_frame(symbol)
    return charts.levels(df) if df is not None else {}


@st.cache_data(ttl=3600)
def get_earnings(symbol: str):
    return charts.earnings_date(symbol)


@st.cache_data(ttl=900)
def get_news(symbol: str, name: str):
    return news.get_news(symbol, name)


@st.cache_data(ttl=900)
def get_headline_sentiments(titles: tuple):
    from src import local_llm
    # Short timeout on the render path: a cache miss must not freeze the page for the
    # full pipeline timeout. Failure degrades to None scores (handled by sent_badge).
    return local_llm.score_headlines(list(titles), timeout=12)


def candlestick_fig(df: pd.DataFrame, symbol: str, levels: dict | None = None,
                    earnings: str | None = None, avg_cost: float | None = None,
                    add_level: float | None = None,
                    trim_level: float | None = None) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.52, 0.13, 0.17, 0.18],
                        subplot_titles=(f"{symbol} price + SMAs", "Volume", "RSI(14)", "MACD"))
    # One green/red pair everywhere (#26a69a/#ef5350); hollow-up candles are the
    # classic CVD-safe convention (shape carries direction, not just hue).
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="price",
                                 increasing_line_color="#26a69a",
                                 increasing_fillcolor="rgba(38,166,154,0.15)",
                                 decreasing_line_color="#ef5350",
                                 decreasing_fillcolor="#ef5350"), row=1, col=1)
    for sma, color in (("SMA20", "#38bdf8"), ("SMA50", "#f59e0b"), ("SMA200", "#f472b6")):
        if sma in df:
            fig.add_trace(go.Scatter(x=df.index, y=df[sma], name=sma,
                                     line=dict(width=1.2, color=color),
                                     hovertemplate="%{y:.2f}<extra>" + sma + "</extra>"),
                          row=1, col=1)
    # Volume colored by up/down day (green/red) — the direction carries the signal.
    up = df["Close"] >= df["Open"]
    vol_color = np.where(up, "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="vol", marker_color=vol_color,
                         showlegend=False, hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                             line=dict(color="#a78bfa"),
                             hovertemplate="%{y:.0f}<extra>RSI</extra>"), row=3, col=1)
    fig.add_hline(y=70, line=dict(color="#ef5350", dash="dot"), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="#26a69a", dash="dot"), row=3, col=1)
    hist_color = np.where(df["MACDhist"] >= 0, "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=df.index, y=df["MACDhist"], name="hist", marker_color=hist_color,
                         showlegend=False, hoverinfo="skip"), row=4, col=1)
    # MACD lines are NEUTRAL colors: green/gold here would collide with the semantic
    # up/down and Hold/earnings meanings those hues carry elsewhere on the page.
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                             line=dict(color="#e6edf3", width=1.4),
                             hovertemplate="%{y:.2f}<extra>MACD</extra>"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACDsig"], name="signal",
                             line=dict(color="#8b949e", width=1.2, dash="dot"),
                             hovertemplate="%{y:.2f}<extra>signal</extra>"), row=4, col=1)

    # Decision-price overlays on the price panel. These are the numbers the user
    # actually trades against — they belong ON the chart, not only in a side table.
    lv = levels or {}
    def _hline(y, color, text):
        if isinstance(y, (int, float)):
            fig.add_hline(y=y, line=dict(color=color, dash="dot", width=1),
                          annotation_text=text, annotation_position="right",
                          annotation_font_size=10, row=1, col=1)
    _hline(lv.get("stop_2atr"), "#f87171", "stop 2·ATR")
    if avg_cost:
        _hline(avg_cost, "#2dd4bf", "avg cost")
    if add_level:
        _hline(add_level, "#26a69a", f"add ≤{add_level:g}")
    if trim_level:
        _hline(trim_level, "#fb923c", f"trim ≥{trim_level:g}")
    # Earnings date as a vertical marker (neutral color — gold is already taken).
    if earnings:
        try:
            ed = pd.to_datetime(earnings)
            if ed >= df.index.min():
                fig.add_vline(x=ed, line=dict(color="#e6edf3", dash="dash", width=1),
                              annotation_text="earnings", annotation_position="top",
                              annotation_font_size=10, row=1, col=1)
        except Exception:  # noqa: BLE001
            pass

    # Skip weekends AND market holidays (any weekday missing from the data).
    all_days = pd.date_range(df.index.min(), df.index.max(), freq="B")
    holidays = all_days.difference(df.index.normalize())
    breaks = [dict(bounds=["sat", "mon"])]
    if len(holidays):
        breaks.append(dict(values=list(holidays)))
    fig.update_xaxes(rangebreaks=breaks, showspikes=True, spikemode="across",
                     spikedash="solid", spikethickness=1)
    fig.update_layout(height=700, xaxis_rangeslider_visible=False, hovermode="x unified",
                      showlegend=True, legend=dict(orientation="h", y=1.02, x=0,
                                                   font=dict(size=11)),
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ----------------------------------------------------------------- chat (all tabs)
@st.dialog("💬 Ask Claude about your portfolio & the analysis", width="large")
def chat_dialog():
    st.caption("Runs on your Claude subscription. Not financial advice.")
    _mode = st.radio("Mode", ["⚡ Quick", "🔬 Investigate"], horizontal=True,
                     key="chat_mode", label_visibility="collapsed",
                     help="Quick: answers in ~20s from a snapshot of the system state. "
                          "Investigate: a full agent with read access to the codebase and "
                          "database — it digs, verifies, searches the web, and can CORRECT "
                          "stored ratings/actions (announced + audit-logged). 1–5 min per "
                          "question.")
    _deep = _mode.startswith("🔬")
    _c1, _c2 = st.columns([3, 1])
    _use_web = (_c1.toggle("🌐 Allow web search (slower)", value=False, key="chat_web")
                if not _deep else True)
    _allow_fix = False
    if _deep:
        # Read-only by default: web headlines flow into the prompt, so a shell-capable
        # agent is a prompt-injection surface. Corrections are an explicit opt-in.
        _allow_fix = _c1.toggle("🛠️ Allow corrections (gives the agent a shell + DB writes)",
                                value=False, key="chat_allow_fix")
        _c1.caption("🔬 Reads code, DB and the web. "
                    + ("Can CORRECT stored ratings/actions (announced + audit-logged in Lessons)."
                       if _allow_fix else "Read-only — it will tell you what it would change."))
    if _c2.button("🗑️ Clear", key="chat_clear"):
        st.session_state.chat_history = []
        st.session_state.pop("chat_session_id", None)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for _m in st.session_state.chat_history:
        with st.chat_message(_m["role"]):
            st.markdown(md(_m["content"]) if _m["role"] == "assistant" else _m["content"])

    _pending = None
    if not st.session_state.chat_history:
        # Seed questions are DERIVED from the current state (largest sell-action,
        # highest conviction, fired triggers) — never hardcoded tickers that would
        # go stale after a sale.
        _chips = ["Which holding is my biggest risk right now, and why?"]
        try:
            _pfl = storage.latest_portfolio_run("ok")
            if _pfl:
                _pos = storage.get_portfolio(_pfl["id"])["positions"]
                _held_now = set(holdings.keys())  # only still-held names
                _sell = [p for p in _pos if (p.get("action") or "") in ("trim", "exit")
                         and p["symbol"] in _held_now]
                if _sell:
                    _w = max(_sell, key=lambda x: x.get("weight_pct") or 0)
                    _chips.append(f"Why is {_w['symbol']} rated {_w['action']}? "
                                  f"Steelman the case for keeping it.")
                _live = [p for p in _pos if p["symbol"] in _held_now]
                _best = max(_live, key=lambda x: x.get("conviction") or 0, default=None)
                if _best and (_best.get("conviction") or 0) >= 7:
                    _chips.append(f"What would make the system turn bearish on "
                                  f"{_best['symbol']}?")
        except Exception:  # noqa: BLE001
            pass
        _fired_c = load_fired_triggers()
        if _fired_c:
            _chips.insert(1, f"A sell-signal fired for {_fired_c[0]['symbol']} — "
                             f"how serious is it and what should I do?")
        _chips.append("Summarize how my portfolio's risk profile changed this week.")
        st.markdown("**Try asking:**")
        for _q in _chips[:4]:
            if st.button(_q, key=f"chip_{hash(_q)}"):
                _pending = _q
    # A form (not st.chat_input) — chat_input isn't supported inside dialogs.
    # Enter in the text box submits the form.
    with st.form("chat_form", clear_on_submit=True, border=False):
        _fc1, _fc2 = st.columns([5, 1])
        _typed = _fc1.text_input("question", label_visibility="collapsed",
                                 placeholder="Ask about the board, a holding, a rating, "
                                             "the risk numbers…")
        _sent = _fc2.form_submit_button("Send", type="primary", width="stretch")
    if _sent and _typed.strip():
        _pending = _typed.strip()

    if _pending:
        from src import chat as _chat
        st.session_state.chat_history.append({"role": "user", "content": _pending})
        with st.chat_message("user"):
            st.markdown(_pending)
        with st.chat_message("assistant"):
            if _deep:
                with st.spinner("🔬 Investigating — reading code, querying the DB, "
                                "searching… (1–5 min)"):
                    _ans, _sid = _chat.investigate(
                        _pending, session_id=st.session_state.get("chat_session_id"),
                        allow_corrections=_allow_fix)
                if _sid:
                    st.session_state.chat_session_id = _sid
                # An investigation may have CORRECTED stored data — drop the data
                # caches so any change is visible on the next rerun.
                load_board.clear()
                load_fired_triggers.clear()
            else:
                with st.spinner("Reading the system state and thinking"
                                + (" + searching the web" if _use_web else "") + "…"):
                    _ans = _chat.ask(_pending,
                                     history=st.session_state.chat_history[:-1],
                                     use_web=_use_web)
            st.markdown(md(_ans))
        st.session_state.chat_history.append({"role": "assistant", "content": _ans})


# ----------------------------------------------------------------- sidebar
st.sidebar.title("📈 Claude Trader")
if st.sidebar.button("💬 Ask Claude", type="primary", width="stretch",
                     help="Chat grounded in the full dashboard state — works from any tab"):
    chat_dialog()
_n_chat = len(st.session_state.get("chat_history", []))
if _n_chat:
    st.sidebar.caption(f"💬 {_n_chat // 2} question(s) in this session — reopen to continue")
runs = storage.list_runs(50)
if not runs:
    st.warning("No runs yet. Run:  `python -m scripts.run_daily --mode deep`")
    st.stop()

def _row_health(row) -> str | None:
    """runs.health for a sqlite3.Row; None when the column/value is absent."""
    try:
        return row["health"]
    except (IndexError, KeyError):
        return None


run_labels = {f"{health.GLYPH.get(_row_health(r), '⚪')} #{r['id']} · {r['mode']} · "
              f"{(r['started_at'] or '')[:16]} · {r['status']}": r["id"]
              for r in runs}
# Default to the most recent COMPLETED run that has deep research (so the analyses
# are visible); fall back to the latest completed run otherwise.
_research_runs = storage.runs_with_research()
default_idx = next((i for i, r in enumerate(runs)
                    if r["status"] == "ok" and r["id"] in _research_runs),
                   next((i for i, r in enumerate(runs) if r["status"] == "ok"), 0))
sel = st.sidebar.selectbox("Run", list(run_labels.keys()), index=default_idx)
run_id = run_labels[sel]
watch = storage.get_watch()

rec_filter = st.sidebar.multiselect(
    "Filter by recommendation", ["strong_buy", "buy", "hold", "sell", "strong_sell", "avoid"],
    format_func=lambda k: recommend.BADGE.get(k, k))
tag_filter = st.sidebar.multiselect(
    "Filter by tag",
    ["quality", "higher-risk", "earnings-soon", "no-earnings", "thin-data"])
# Options come from the tickers table, not the four curated labels: in broad mode
# `subsector` carries the SIC industry group and a hardcoded list offers nothing.
_sub_options = storage.distinct_subsectors() or list(SUBSECTOR_LABELS.keys())
sub_filter = st.sidebar.multiselect("Sub-sector", _sub_options,
                                    format_func=lambda k: SUBSECTOR_LABELS.get(k, k))
hide_excluded = st.sidebar.checkbox("Hide excluded watchlist names", value=True)
top_n = st.sidebar.slider("Show top N", 5, 300, 25)   # 300: a broad board is 250
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear(); st.rerun()
st.sidebar.divider()
if st_autorefresh is None:
    st.sidebar.warning("`streamlit-autorefresh` not installed — auto-refresh disabled. "
                       "`pip install streamlit-autorefresh`")
    auto = False
else:
    auto = st.sidebar.checkbox("Auto-refresh", value=True)
    interval_min = st.sidebar.select_slider("every (min)", options=[1, 5, 10, 15, 30], value=5)
    if auto:
        st_autorefresh(interval=interval_min * 60 * 1000, key="auto_refresh")

# ----------------------------------------------------------------- header
st.title(theme.board_name())
sel_run = next((r for r in runs if r["id"] == run_id), None)
latest_any = runs[0] if runs else None
if sel_run:
    stamp = sel_run["finished_at"] or sel_run["started_at"]
    bc1, bc2 = st.columns([3, 2])
    bc1.markdown(f"🕒 **Data as of:** {ago(stamp)}  ·  run #{sel_run['id']} ({sel_run['mode']})")
    # Data health: 'ok' status only ever meant "didn't crash". This says whether the
    # inputs were whole, and names the stage that wasn't.
    _h = _row_health(sel_run)
    _hj = sel_run["health_json"] if "health_json" in sel_run.keys() else None
    _probs = health.RunHealth.from_json(_hj).problems()
    if _h in (health.DEGRADED, health.FAILED):
        _fn = st.error if _h == health.FAILED else st.warning
        _fn(f"{health.badge(_h)} — **this run's inputs were not whole.** "
            + " · ".join(f"**{n}** {s}: {note}" for n, s, note in _probs))
    elif _h == health.OK:
        bc1.caption(f"{health.badge(_h)} · all stages ran on complete inputs")
    else:
        bc1.caption("⚪ health unknown · run predates health tracking (2026-09-07)")
    if latest_any and latest_any["id"] != run_id:
        bc2.markdown(f"⚠️ Newer run exists: #{latest_any['id']} "
                     f"({latest_any['mode']}, {ago(latest_any['finished_at'] or latest_any['started_at'])})")
# Live progress banner for an in-flight daily pipeline run
_running_run = next((dict(r) for r in runs if r["status"] == "running"), None)
if _running_run:
    _rp = _running_run.get("progress") or "starting…"
    _rhb = _running_run.get("heartbeat")
    _rhb_min = None
    if _rhb:
        try:
            _rhb_min = int((datetime.now(timezone.utc)
                            - datetime.fromisoformat(_rhb)).total_seconds() // 60)
        except ValueError:
            pass
    _stall = (f" · ⚠️ quiet {_rhb_min} min — may be stalled"
              if _rhb_min is not None and _rhb_min >= 15 else
              (f" · active {_rhb_min} min ago" if _rhb_min is not None else ""))
    st.info(f"🏃 **{_running_run['mode'].capitalize()} run #{_running_run['id']} in "
            f"progress** — {_rp}{_stall}")
st.caption("⚠️ **Not financial advice.** A personal research aid — verify everything and do your own due diligence.")

# Market regime chip + thesis-break alerts (the day they happen, not weeks later)
_rg = load_regime()
if _rg.get("regime"):
    _rg_icon = {"risk_on": "🟢", "neutral": "🟡", "stress": "🟠", "risk_off": "🔴"}
    st.caption(f"{_rg_icon.get(_rg['regime'], '•')} **Regime: {_rg['regime'].replace('_', '-')}** · "
               f"{_rg.get('benchmark', 'SPY')} {_rg.get('bench_vs_200dma_pct')}% vs 200dma · "
               f"20d vol {_rg.get('bench_vol20_ann_pct')}%"
               + (f" · breadth {_rg['breadth_pct']}%" if _rg.get("breadth_pct") is not None else "")
               + (f" · ⚠️ weak: {', '.join(_rg['weak_subsectors'])}" if _rg.get("weak_subsectors") else ""))
_fired = load_fired_triggers()
if _fired:
    st.error("🚨 **Thesis-invalidation triggers fired (last 7 days):**\n" + "\n".join(
        f"- **{t['symbol']}** — {t.get('trigger_detail','')} · _{t.get('description','')}_"
        f" ({str(t.get('triggered_at',''))[:10]})" for t in _fired[:8]))

board = load_board(run_id)
if not board:
    st.warning("⏳ This run has no scored data yet (it may still be running). "
               "Pick a completed run in the sidebar.")
    st.stop()
board_by_sym = {r["symbol"]: r for r in board}
holdings = load_holdings()
quote_map = load_quote_map()

tab_board, tab_port, tab_chg, tab_cal, tab_acc = st.tabs(
    ["📋 Board", "💼 Portfolio", "🔄 What changed", "📅 Calendar", "🎯 Accuracy"])

# ================================================================ BOARD TAB
with tab_board:
    with st.expander("ℹ️ What do these columns mean?"):
        st.markdown(
            "- **Recommendation** — 🟩 Strong Buy · 🟢 Buy · 🟡 Hold · 🟠 Sell · 🔴 Strong Sell, "
            "from Claude's research. Trailing **\\*** = provisional screen guess (not deep-researched).\n"
            "- **📊tech / 💰fund / 📰sent** — the three pillar scores (0–1). **Composite** ranks the board.\n"
            "- **Conviction** — Claude's 0–10 confidence. **Quote** — 15-min delayed price (IBKR).\n"
            "- **⚡ catalyst** — the strongest catalyst keys the local LLM found in this "
            "week's headlines (0–10 each). Dark: not in the ranking score yet.\n"
            "- **Tags** — ◇ quality · ⚠ higher-risk · 🗓 earnings-soon.")

    # Sidebar filters FIRST, so the tile counts describe the list you're actually
    # looking at (they used to count the raw board — "All · 104" over a 25-row table).
    filtered = board
    if rec_filter:
        filtered = [r for r in filtered if recommend.recommendation(r)[0] in rec_filter]
    if tag_filter:
        filtered = [r for r in filtered if set(tag_filter) & set(r.get("tags") or [])]
    if sub_filter:
        filtered = [r for r in filtered if r.get("subsector") in sub_filter]
    if hide_excluded:
        filtered = [r for r in filtered if watch.get(r["symbol"]) != "excluded"]

    n_buys = sum(1 for r in filtered
                 if recommend.recommendation(r)[0] in ("buy", "strong_buy"))
    n_qual = sum(1 for r in filtered if "quality" in (r.get("tags") or []))
    n_held = sum(1 for r in filtered if r.get("held_position"))
    n_res = sum(1 for r in filtered if r.get("conviction") is not None)
    if "quick" not in st.session_state:
        st.session_state.quick = "all"
    tiles = [("all", "📋 All", len(filtered)), ("buys", "🟢 Buys", n_buys),
             ("quality", "◇ Quality", n_qual), ("held", "📌 Held", n_held),
             ("research", "🔬 Deep-researched", n_res)]
    for col, (k, lbl, n) in zip(st.columns(len(tiles)), tiles):
        if col.button(f"{lbl} · {n}", key=f"q_{k}", width="stretch",
                      type="primary" if st.session_state.quick == k else "secondary"):
            st.session_state.quick = "all" if st.session_state.quick == k else k
            st.rerun()
    st.caption("Tap a tile to filter the board · 🔬 = deep-researched by Claude")
    if not holdings:
        st.info("IBKR not connected for this run — holdings/quotes overlay unavailable.")

    q = st.session_state.get("quick", "all")
    if q == "buys":
        rows = [r for r in filtered
                if recommend.recommendation(r)[0] in ("buy", "strong_buy")]
    elif q == "quality":
        rows = [r for r in filtered if "quality" in (r.get("tags") or [])]
    elif q == "held":
        rows = [r for r in filtered if r.get("held_position")]
    elif q == "research":
        rows = [r for r in filtered if r.get("conviction") is not None]
    else:
        rows = filtered
    n_matching = len(rows)
    rows = rows[:top_n]

    st.subheader("Board")
    if n_matching > len(rows):
        st.caption(f"Showing the top **{len(rows)} of {n_matching}** matching names — "
                   f"raise **'Show top N'** in the sidebar to see more.")
    table = []
    for r in rows:
        m = r.get("metrics") or {}
        _lab, _ = recommend.recommendation(r)
        table.append({
            "rank": r["rank"],
            "recommendation": recommend.badge(r),
            "rec#": recommend.WEIGHT.get(_lab, 0),  # numeric so the column SORTS by strength
            "symbol": r["symbol"] + (" ⭐" if watch.get(r["symbol"]) == "pinned" else "")
                      + (" 📌" if r.get("held_position") else "")
                      + (" 🔬" if r.get("conviction") is not None else ""),
            "name": (r.get("name") or "")[:26],
            "sub": SUBSECTOR_LABELS.get(r.get("subsector"), r.get("subsector")),
            "setup": SETUP_EMOJI.get((m.get("setup") or ""), ""),
            "composite": r["blended"],
            "📊": m.get("pillar_technical"), "💰": m.get("pillar_fundamental"), "📰": m.get("pillar_sentiment"),
            "RS 3m": m.get("rs_3m"),
            "⚡ catalyst": catalyst_str(r),
            "conv": r.get("conviction"), "quote": r.get("quote_last"),
            "tags": tag_str(r.get("tags")),
        })
    st.caption("Click a row to jump to that stock's analysis below.")
    # One ProgressColumn only (the ranking score); pillars are inputs — small numbers,
    # inspectable in Details. Four identical teal bars gave inputs the same visual
    # weight as the output.
    _small_num = lambda lbl: st.column_config.NumberColumn(lbl, format="%.2f", width="small")
    _evt = st.dataframe(
        pd.DataFrame(table), width="stretch", hide_index=True,
        height=min(620, 60 + 35 * len(table)),
        on_select="rerun", selection_mode="single-row",
        column_config={
            "composite": st.column_config.ProgressColumn(
                "composite", min_value=0.0, max_value=1.0, format="%.2f"),
            "rec#": st.column_config.NumberColumn(
                "rec#", format="%.0f", width="small",
                help="Numeric recommendation strength (5=Strong Buy … 1=Strong Sell) — click to sort"),
            "📊": _small_num("📊 tech"), "💰": _small_num("💰 fund"), "📰": _small_num("📰 sent"),
            "conv": st.column_config.NumberColumn("conv", format="%.1f", width="small"),
            "quote": st.column_config.NumberColumn("quote", format="$%.2f"),
            "RS 3m": st.column_config.NumberColumn("RS 3m", format="%+.0f", width="small"),
            "name": st.column_config.TextColumn(width="medium"),
            "⚡ catalyst": st.column_config.TextColumn(
                "⚡ catalyst", width="small",
                help="Strongest catalyst keys from this run's headline scan (0–10). "
                     "Dark: not in the ranking score."),
            "tags": st.column_config.TextColumn(width="small"),
        })
    st.caption(f"RS 3m = 3-month return vs {config.get('technical.rs_benchmark', 'SMH')} "
               "(the AI-theme peer set) · quotes are ~15-min delayed, as of the last "
               "pipeline run · sizing hints in Details are priced off these quotes.")
    # Clicking a table row drives the Details selector and scrolls down to it.
    _sel = getattr(_evt, "selection", None)
    if _sel and _sel.get("rows"):
        _chosen = rows[_sel["rows"][0]]["symbol"]
        if st.session_state.get("detail_select") != _chosen:
            st.session_state.detail_select = _chosen
            st.session_state.do_scroll = True
            st.rerun()

    st.markdown("<div id='stock-details'></div>", unsafe_allow_html=True)
    if st.session_state.pop("do_scroll", False):
        components.html(
            "<script>window.parent.document.getElementById('stock-details')"
            "?.scrollIntoView({behavior:'smooth', block:'start'});</script>", height=0)
    st.subheader("Details")
    symbols = [r["symbol"] for r in rows]
    if symbols:
        if st.session_state.get("detail_select") not in symbols:
            st.session_state.detail_select = symbols[0]
        pick = st.selectbox("Inspect a ticker", symbols, key="detail_select")
        r = board_by_sym.get(pick)
        if r:
            m = r.get("metrics") or {}
            left, right = st.columns([3, 2])
            with left:
                st.markdown(f"### {r['symbol']} — {r.get('name','')}")
                _lab, _basis = recommend.recommendation(r)
                st.markdown(pill(_lab, recommend.BADGE.get(_lab, _lab), _basis == "screen"),
                            unsafe_allow_html=True)
                st.markdown(f"**{SUBSECTOR_LABELS.get(r.get('subsector'), r.get('subsector'))}** · {tag_str(r.get('tags'))}")
                p1, p2, p3 = st.columns(3)
                p1.metric("📊 Technical", fmt_num(m.get("pillar_technical"), 2),
                          help="Technical pillar (0–1) from RSI, MACD, moving-average trend/crosses, 52-week position and volume.")
                p1.caption(m.get("tech_state") or "—")
                p2.metric("💰 Fundamental", fmt_num(m.get("pillar_fundamental"), 2),
                          help="Fundamental pillar (0–1) from valuation, cash flow, growth and quality, ranked vs peers.")
                p2.caption(f"PE {fmt_num(m.get('trailingPE'))}")
                p3.metric("📰 Sentiment", fmt_num(m.get("pillar_sentiment"), 2),
                          help="Sentiment pillar (0–1) from local-LLM news TONE only. Events are scored separately in the ⚡ catalyst vector.")
                p3.caption(m.get("sentiment_label") or "—")
                _cat = catalyst_str(r, top=9)
                if _cat:
                    st.markdown(f"**⚡ Catalysts (local LLM, 0–10):** {_cat}")
                if m.get("setup_label"):
                    st.markdown(f"**{SETUP_EMOJI.get(m.get('setup'),'')} Setup:** {m['setup_label']}"
                                + (f" — {' · '.join(m.get('signals') or [])}" if m.get("signals") else ""))
                rs3, rs6, bench = m.get("rs_3m"), m.get("rs_6m"), m.get("benchmark") or "SPY"
                if rs3 is not None or rs6 is not None:
                    st.markdown(f"**Relative strength** (vs {bench}): "
                                f"3mo {fmt_num(rs3,1)}pp · 6mo {fmt_num(rs6,1)}pp")
                # Composite-score trend across recent runs — "improving or fading?"
                _hist = load_score_history(pick)
                if len(_hist) >= 2:
                    _vals = [v for _, v in _hist]
                    _spark = go.Figure(go.Scatter(
                        y=_vals, mode="lines+markers",
                        line=dict(color="#2dd4bf", width=2), marker=dict(size=4)))
                    _delta = _hist[-1][1] - _hist[0][1]
                    # Padded fixed range so an autoscaled axis can't exaggerate noise
                    # into a dramatic-looking "trend".
                    _pad = max(0.05, (max(_vals) - min(_vals)) * 0.3)
                    _spark.update_layout(height=70, margin=dict(l=0, r=0, t=0, b=0),
                                         showlegend=False, xaxis=dict(visible=False),
                                         yaxis=dict(visible=False,
                                                    range=[min(_vals) - _pad,
                                                           max(_vals) + _pad]),
                                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    _arrow = "▲" if _delta > 0.005 else "▼" if _delta < -0.005 else "→"
                    st.caption(f"Composite over last {len(_hist)} runs {_arrow} {_delta:+.3f}")
                    st.plotly_chart(_spark, width="stretch",
                                    config={"displayModeBar": False, "staticPlot": True})
                wc1, wc2, wc3 = st.columns(3)
                # The watchlist isn't cached (re-read every run), so a rerun is enough —
                # no need to blow away every price/news/LLM cache.
                if wc1.button("⭐ Pin", key=f"pin{pick}"):
                    storage.set_watch(pick, "pinned"); st.rerun()
                if wc2.button("🚫 Exclude", key=f"exc{pick}"):
                    storage.set_watch(pick, "excluded"); st.rerun()
                if wc3.button("↩ Clear", key=f"clr{pick}"):
                    from contextlib import closing
                    with closing(storage.get_conn()) as cn:
                        cn.execute("DELETE FROM watchlist WHERE symbol=?", (pick,))
                        cn.commit()
                    st.rerun()

                researched = r.get("conviction") is not None
                _marker = config.cache_dir() / f"research_inflight_{run_id}_{pick}.marker"
                _inflight = _marker.exists()
                if st.button("🔬 Refresh deep analysis" if researched else "🔬 Run deep analysis",
                             key=f"deep_{pick}", type="primary", width="stretch",
                             disabled=_inflight):
                    # Launch detached so the page never freezes for the ~1–2 min research
                    # call; the marker file drives the in-progress state below.
                    env = os.environ.copy(); env["PYTHONIOENCODING"] = "utf-8"
                    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
                    # sys.executable = the interpreter serving this dashboard (the venv's),
                    # so the launch works wherever the venv lives, not only .venv\Scripts.
                    subprocess.Popen([sys.executable,
                                      "-m", "scripts.research_one",
                                      "--run-id", str(run_id), "--symbol", pick],
                                     cwd=str(ROOT), env=env, creationflags=flags,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    st.toast(f"Deep analysis started for {pick} (~1–2 min)", icon="🔬")
                    st.rerun()
                if _inflight:
                    st.info(f"⏳ Researching {pick} in the background — refresh in ~1–2 min.")
                    load_board.clear()  # so the finished report shows up on the next tick
                if r.get("conviction") is not None:
                    _dl = r.get("deal_likelihood")   # legacy: prompts <= v4 only
                    st.markdown(f"**Claude conviction:** {r['conviction']}/10"
                                + (f" · **deal likelihood:** {_dl}" if _dl else ""))
                if r.get("rating") is not None or r.get("conviction") is not None:
                    st.caption(f"🔬 Analyzed {ago(r.get('research_at'))} · {freshness(r.get('data_as_of'))}")
                if r.get("thesis"):
                    st.markdown(f"**Thesis.** {md(r['thesis'])}")
                if r.get("bull") or r.get("bear"):
                    st.markdown(f"🐂 **Bull:** {md(r.get('bull','—'))}")
                    st.markdown(f"🐻 **Bear:** {md(r.get('bear','—'))}")
                _rt = r.get("red_team_json") or {}
                if isinstance(_rt, dict) and _rt.get("bear_case"):
                    _surv = _rt.get("survives")
                    _badge = ("✅ rating survived" if _surv else
                              f"⛔ downgraded from {_rt.get('original_rating','?')}")
                    with st.expander(f"😈 Red team ({_badge}"
                                     + (f", conviction −{_rt['confidence_haircut']}"
                                        if _rt.get("confidence_haircut") else "") + ")"):
                        st.markdown(f"**Strongest bear case:** {md(_rt['bear_case'])}")
                        for ev in (_rt.get("new_evidence") or [])[:5]:
                            st.markdown(f"- {md(ev)}")
                        if _rt.get("note"):
                            st.caption(f"Biggest vulnerability: {md(_rt['note'])}")
                cats = r.get("res_catalysts") or r.get("sig_catalysts") or []
                if cats:
                    st.markdown("**Catalysts:** " + md("; ".join(cats)))
                if r.get("risks_json"):
                    st.markdown("**Risks:** " + md("; ".join(r["risks_json"])))
                if r.get("raw") and r.get("conviction") is None:
                    with st.expander("Raw Claude output"):
                        st.code(r["raw"][:4000])
            with right:
                lv = get_levels(pick)
                ed = get_earnings(pick)
                # Position sizing from the owner's policy: risk budget / (2·ATR stop)
                _sz = ""
                _pmv = sum((quote_map.get(s) or 0) * (h_.get("position") or 0)
                           for s, h_ in holdings.items())
                if _pmv and lv.get("atr") and lv.get("price"):
                    _risk_pct = config.get("policy.risk_per_position_pct", 1.0)
                    _risk_dollars = _pmv * _risk_pct / 100
                    _shares = int(_risk_dollars / (2 * lv["atr"]))
                    _cost = _shares * lv["price"]
                    _sz = (f"{_shares} shares (≈${_cost:,.0f}, {(_cost / _pmv * 100):.0f}% of pf) "
                           f"at {_risk_pct:g}% risk to a 2·ATR stop")
                st.markdown("**Trade levels** — hover ⓘ for explanations")
                st.markdown(metric_rows_html([
                    ("Price", fmt_num(lv.get("price"), 2)),
                    ("Suggested stop (2·ATR)", fmt_num(lv.get("stop_2atr"), 2)),
                    ("Support (SMA50)", fmt_num(lv.get("support_sma50"), 2)),
                    ("Support (SMA200)", fmt_num(lv.get("support_sma200"), 2)),
                    ("52w high", fmt_num(lv.get("high_52w"), 2)),
                    ("52w low", fmt_num(lv.get("low_52w"), 2)),
                    ("Dist to 52w high", fmt_pct(lv.get("dist_to_high"))),
                    ("ATR %", fmt_pct(lv.get("atr_pct"))),
                    ("Next earnings", ed or "—"),
                ]), unsafe_allow_html=True)
                if _sz:
                    st.caption(f"📐 **Sizing hint (policy):** {_sz}")
                st.markdown("**Fundamentals**")
                st.markdown(metric_rows_html([
                    ("Mkt cap", fmt_mcap(m.get("marketCap"))),
                    ("PE", fmt_num(m.get("trailingPE"))),
                    ("Fwd PE", fmt_num(m.get("forwardPE"))),
                    ("PEG", fmt_num(m.get("peg"))),
                    ("EV/EBITDA", fmt_num(m.get("ev_ebitda"))),
                    ("EV/Sales", fmt_num(m.get("ev_sales"))),
                    ("P/FCF", fmt_num(m.get("p_fcf"))),
                    ("FCF yield", fmt_pct(m.get("fcf_yield"))),
                    ("Rev growth", fmt_pct(m.get("revenueGrowth"))),
                    ("Rule of 40", fmt_num(m.get("rule_of_40"))),
                    ("Gross margin", fmt_pct(m.get("grossMargins"))),
                    ("ROE", fmt_pct(m.get("returnOnEquity"))),
                    ("Net cash", fmt_bn(m.get("net_cash"))),
                    ("RSI(14)", fmt_num(m.get("rsi"), 1)),
                ]), unsafe_allow_html=True)
                st.markdown("**Valuation vs sector**")
                st.markdown(metric_rows_html([
                    ("PE vs sector", vs_sector(m.get("trailingPE"), m.get("sector_pe"))),
                    ("Fwd PE vs sector", vs_sector(m.get("forwardPE"), m.get("sector_fwdpe"))),
                    ("EV/EBITDA vs sector", vs_sector(m.get("ev_ebitda"), m.get("sector_evebitda"))),
                ]), unsafe_allow_html=True)
                if r.get("metrics_read"):
                    st.info(f"🔬 **Claude's read of {pick}'s numbers:** {md(r['metrics_read'])}")
                if r.get("held_position"):
                    h = holdings.get(pick)
                    if h:
                        st.success(f"📌 Held: {h['position']:g} @ avg {fmt_num(h['avg_cost'],2)}")

            # Candlestick + indicators (timeframe selector + how-to-read explainer,
            # with Claude's stock-specific chart read + trade-level overlays)
            _held = holdings.get(pick)
            render_price_chart(pick, key="board", chart_note=r.get("chart_pattern"),
                               levels=lv, earnings=ed,
                               avg_cost=(_held.get("avg_cost") if _held else None))
            # News headlines
            items = get_news(pick, r.get("name") or pick)
            if items:
                sents = get_headline_sentiments(tuple(it["title"] for it in items[:10]))
                with st.expander(f"📰 Recent news ({len(items)}) — date · sentiment", expanded=False):
                    for it, s in zip(items[:10], sents):
                        st.markdown(
                            f"- {sent_badge(s)} · `{fmt_date(it.get('published'))}` · "
                            f"[{md(it['title'])}]({it['link']}) <small>({it.get('source','')})</small>",
                            unsafe_allow_html=True)

            with st.expander("📖 Metric glossary (general guide)"):
                st.markdown("##### Scores & columns")
                for k, v in SCORE_GLOSSARY.items():
                    st.markdown(f"- **{k}** — {v}")
                st.markdown("##### Fundamentals & trade levels")
                for k, v in GLOSSARY.items():
                    st.markdown(f"- **{k}** — {v}")

# ================================================================ PORTFOLIO TAB
with tab_port:
    st.subheader("💼 Your IBKR portfolio")
    if not holdings:
        st.info("No holdings loaded. Connect IB Gateway and re-run to populate.")
    else:
        # The Portfolio tab uses the latest portfolio DEEP-DIVE's holding actions as the
        # single source of truth (the Board tab uses the screening rating).
        _pf_latest = storage.latest_portfolio_run("ok")
        pf_pos = ({p["symbol"]: p for p in storage.get_portfolio(_pf_latest["id"])["positions"]}
                  if _pf_latest else {})
        st.caption("Recommendations here come from the latest **portfolio deep-dive** "
                   "(cost-basis aware). The Board tab's rating is the screen — they can differ.")
        hrows, total_mv, total_cost = [], 0.0, 0.0
        by_sub: dict[str, float] = {}
        for sym, h in sorted(holdings.items()):
            last = quote_map.get(sym)
            pos, cost = h["position"], h["avg_cost"]
            mv = (last * pos) if last else None
            pnl_abs = ((last - cost) * pos) if (last and cost) else None
            pnl_pct = ((last - cost) / cost * 100) if (last and cost) else None
            b = board_by_sym.get(sym)
            dd = pf_pos.get(sym)
            if dd:
                eff, note = reconciled_action(dd)
                rec = ACTION_BADGE.get(eff, eff) + (f" ({note})" if note else "")
            elif b:
                rec = recommend.badge(b)
            else:
                rec = "— (no deep-dive yet)"
            sub = SUBSECTOR_LABELS.get(b.get("subsector"), "Other") if b else "Other"
            if mv:
                total_mv += mv; by_sub[sub] = by_sub.get(sub, 0) + mv
            if last and cost:
                total_cost += cost * pos
            # Keep raw numbers (not pre-formatted strings) so the table sorts numerically
            # and can be colored — column_config handles the display formatting.
            hrows.append({"symbol": sym, "recommendation": rec, "shares": pos,
                          "avg cost": cost, "last": last, "mkt value": mv,
                          "P&L $": pnl_abs, "P&L %": pnl_pct, "sub-sector": sub})
        m1, m2, m3 = st.columns(3)
        m1.metric("Holdings", len(holdings))
        m2.metric("Market value", f"${total_mv:,.0f}" if total_mv else "—")
        tot_pnl = total_mv - total_cost if total_cost else None
        m3.metric("Unrealized P&L", f"${tot_pnl:,.0f}" if tot_pnl is not None else "—",
                  delta=f"{(tot_pnl/total_cost*100):.1f}%" if tot_pnl is not None and total_cost else None)
        # P&L is THE number a holder scans — color it (sign glyph stays as the
        # colorblind-safe redundant channel via the %+ formats).
        _pnl_css = (lambda v: "color:#4ade80" if isinstance(v, (int, float)) and v > 0
                    else "color:#f87171" if isinstance(v, (int, float)) and v < 0 else "")
        _hdf = pd.DataFrame(hrows)
        st.dataframe(
            _hdf.style.map(_pnl_css, subset=["P&L $", "P&L %"]),
            width="stretch", hide_index=True,
            column_config={
                "shares": st.column_config.NumberColumn("shares", format="%g"),
                "avg cost": st.column_config.NumberColumn("avg cost", format="$%.2f"),
                "last": st.column_config.NumberColumn("last", format="$%.2f"),
                "mkt value": st.column_config.NumberColumn("mkt value", format="$%,.0f"),
                "P&L $": st.column_config.NumberColumn("P&L $", format="$%,.0f"),
                "P&L %": st.column_config.NumberColumn("P&L %", format="%+.1f%%"),
            })
        if by_sub:
            st.markdown("**Allocation by sub-sector**")
            _alloc = sorted(by_sub.items(), key=lambda kv: kv[1], reverse=True)
            afig = go.Figure(go.Bar(
                x=[k for k, _ in _alloc], y=[v for _, v in _alloc], marker_color="#2dd4bf",
                text=[f"{v/total_mv*100:.0f}%" if total_mv else "" for _, v in _alloc],
                textposition="outside", cliponaxis=False))
            afig.update_xaxes(fixedrange=True)
            afig.update_yaxes(fixedrange=True, title="market value")
            afig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(afig, width="stretch",
                            config={"scrollZoom": False, "displayModeBar": False})
            top = max(by_sub.values()) / total_mv if total_mv else 0
            _max_sub = config.get("policy.max_subsector_weight", 40) / 100
            if top > _max_sub:
                st.warning(f"⚠️ Concentration: one sub-sector is {top*100:.0f}% of your "
                           f"portfolio (policy limit: {_max_sub*100:.0f}%).")
        _hupd = max((h.get("updated_at") or "" for h in holdings.values()), default="")
        st.caption(f"Quotes as of last pipeline run ({ago(_hupd)}), ~15-min delayed. "
                   "Class shares (GOOG vs GOOGL) shown as held separately.")

    # ---- On-demand portfolio deep-dive ----
    st.divider()
    st.subheader("🔬 Portfolio deep-dive (on-demand)")
    PY = sys.executable   # the venv interpreter serving this page
    running = storage.running_portfolio_run()
    latest_pf = storage.latest_portfolio_run("ok")
    bcol1, bcol2 = st.columns([1, 2])
    with bcol1:
        if st.button("🔬 Deep-dive my portfolio", disabled=bool(running)):
            env = os.environ.copy(); env["PYTHONIOENCODING"] = "utf-8"
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
            subprocess.Popen([str(PY), "-m", "scripts.portfolio_deepdive"], cwd=str(ROOT), env=env,
                             creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            st.success("Started — runs in the background (~10–15 min). The page auto-updates when done.")
    with bcol2:
        if running:
            _rd = dict(running)
            _prog = _rd.get("progress") or "starting…"
            _hb = _rd.get("heartbeat")
            _hb_min = None
            if _hb:
                try:
                    _hb_min = int((datetime.now(timezone.utc)
                                   - datetime.fromisoformat(_hb)).total_seconds() // 60)
                except ValueError:
                    pass
            st.info(f"⏳ **Deep-dive #{_rd['id']} in progress** — {_prog}\n\n"
                    f"started {ago(_rd['started_at'])}"
                    + (f" · last activity {_hb_min} min ago" if _hb_min is not None else ""))
            # Each holding takes ~2-4 min (research + validation). Quiet for much
            # longer than that = probably stalled (hung subprocess, dead network).
            if _hb_min is not None and _hb_min >= 12:
                st.warning(f"⚠️ No progress for {_hb_min} min — the dive may be stalled. "
                           "It will be auto-marked failed after 3h; check data/pf_*.log, "
                           "or kill the python process and relaunch.")
            elif _hb_min is None and _rd.get("progress") is None:
                st.caption("(no live progress reported — dive started before the progress "
                           "feature, or still initializing)")
            st.caption("🔄 This panel updates on each auto-refresh tick (sidebar).")
        elif latest_pf:
            st.caption(f"Last deep-dive: #{latest_pf['id']} · {ago(latest_pf['finished_at'])}")
        else:
            st.caption("No deep-dive yet — press the button (needs IB Gateway running).")

    pf_runs = [r for r in storage.list_portfolio_runs(20) if r["status"] == "ok"]
    if pf_runs:
        pf_labels = {f"#{r['id']} · {ago(r['finished_at'])}": r["id"] for r in pf_runs}
        pf_id = pf_labels[st.selectbox("Deep-dive run", list(pf_labels.keys()))]
        pf = storage.get_portfolio(pf_id)
        # Staleness check: did holdings change since this deep-dive was run?
        _dd_syms = {p["symbol"] for p in pf["positions"]}
        _cur_syms = set(holdings.keys())
        _added, _removed = _cur_syms - _dd_syms, _dd_syms - _cur_syms
        if _added or _removed:
            _parts = []
            if _added:
                _parts.append("added: " + ", ".join(sorted(_added)))
            if _removed:
                _parts.append("removed: " + ", ".join(sorted(_removed)))
            st.warning("⚠️ Your holdings changed since this deep-dive (" + " · ".join(_parts)
                       + "). Click **🔬 Deep-dive my portfolio** above to refresh.")
        summ = pf["summary"]
        if summ:
            st.markdown("#### Portfolio assessment")
            # Computed risk metrics (numbers, not adjectives)
            _rj = summ.get("risk_json")
            try:
                _rm = json.loads(_rj) if isinstance(_rj, str) else (_rj or {})
            except Exception:  # noqa: BLE001
                _rm = {}
            if _rm.get("effective_bets") is not None:
                rc1, rc2, rc3, rc4 = st.columns(4)
                _vab = _rm.get("vol_aware_bets")
                rc1.metric("Effective bets",
                           f"{_rm.get('effective_bets')}" + (f" / {_vab}" if _vab else ""),
                           help="Correlation-adjusted / volatility-aware number of independent "
                                "positions. 8 uncorrelated names = 8; one big correlated theme ≈ 2. "
                                "The second (vol-aware) number counts a high-vol position for its "
                                "real risk share.")
                rc2.metric("Avg pairwise corr", _rm.get("avg_pairwise_corr"),
                           help="Weighted average 90-day return correlation between holdings.")
                _bm = _rm.get("benchmark") or "benchmark"
                _tm = _rm.get("theme_benchmark") or "theme"
                rc3.metric(f"Beta vs {_bm}", _rm.get("beta_bench"))
                rc4.metric(f"{_tm} −20% scenario", f"{_rm.get('scenario_drawdown_pct', '—')}%",
                           help=f"Estimated portfolio move if {_tm} drops 20%, "
                                f"via the portfolio's beta to {_tm}. Stressed (betas rise in "
                                f"selloffs): {_rm.get('scenario_stressed_pct', '—')}%. Worst "
                                f"ACTUAL 20-day stretch in the lookback: "
                                f"{_rm.get('worst_20d_pct', '—')}%.")
            sa, sb = st.columns(2)
            sa.markdown(f"**Overall risk:** {md(summ.get('overall_risk','—'))}")
            sa.markdown(f"**Concentration:** {md(summ.get('concentration','—'))}")
            sa.markdown(f"**Correlation / theme risk:** {md(summ.get('correlation','—'))}")
            sb.markdown(f"**Diversification gaps:** {md(summ.get('diversification_gaps','—'))}")
            sb.markdown(f"**Weakest link:** {md(summ.get('weakest_link','—'))}")
            if summ.get("summary"):
                st.info(md(summ["summary"]))
            why_map = {a.get("symbol"): a.get("why") for a in (summ.get("ranked_actions_json") or [])
                       if isinstance(a, dict)}
            st.markdown("**Ranked actions** (final PM decisions: weight-aware, "
                        "conviction-calibrated — sorted by what to DO first)")
            # Sort by action severity (sell-side first), then conviction: an exit must
            # not hide below a high-conviction hold. Δ pp is the trade to execute.
            _SEV = {"exit": 0, "trim": 1, "trim_wait": 2, "hold": 3, "add_dip": 4, "add": 5}
            ra_rows = []
            for p in sorted(pf["positions"],
                            key=lambda x: (_SEV.get(reconciled_action(x)[0], 3),
                                           -(x.get("conviction") or 0))):
                eff, note = reconciled_action(p)
                _w, _t = p.get("weight_pct"), p.get("target_weight_pct")
                ra_rows.append({
                    "symbol": p["symbol"],
                    "action": ACTION_BADGE.get(eff, eff) + (f" ({note})" if note else ""),
                    "conviction": p.get("conviction"),
                    "weight %": _w,
                    "target %": _t,
                    "Δ pp": (round(_t - _w, 1) if _w is not None and _t is not None
                             else None),
                    "why": p.get("action_note") or why_map.get(p["symbol"])
                           or (p.get("thesis_health") or "")[:90],
                })
            _dpp_css = (lambda v: "color:#4ade80" if isinstance(v, (int, float)) and v > 0
                        else "color:#f87171" if isinstance(v, (int, float)) and v < 0 else "")
            _radf = pd.DataFrame(ra_rows)
            st.dataframe(_radf.style.map(_dpp_css, subset=["Δ pp"]),
                         width="stretch", hide_index=True,
                         column_config={
                             "weight %": st.column_config.NumberColumn("weight %", format="%.1f%%"),
                             "target %": st.column_config.NumberColumn("target %", format="%.1f%%"),
                             "Δ pp": st.column_config.NumberColumn(
                                 "Δ to trade", format="%+.1f pp",
                                 help="Percentage points to buy (+) or sell (−) to reach the PM target"),
                         })
            _bj = summ.get("ballast_json")
            try:
                _ballast = json.loads(_bj) if isinstance(_bj, str) else (_bj or [])
            except Exception:  # noqa: BLE001
                _ballast = []
            if _ballast:
                st.markdown("**🛡️ Suggested ballast** (non-AI diversifiers, policy targets "
                            f"~{config.get('policy.min_ballast_weight', 10)}%)")
                for b in _ballast[:4]:
                    if isinstance(b, dict):
                        st.markdown(f"- **{b.get('symbol')}** → {b.get('target_weight_pct')}% — "
                                    f"{md(b.get('why', ''))}")

        st.markdown("#### Per-holding deep dives")
        for p in pf["positions"]:
            val = p.get("validation_json") or {}
            cur = val.get("is_current")
            vbadge = ("✅ verified current" if cur is True else
                      "⚠️ may be stale" if cur is False else "🟡 not validated")
            eff_act, note = reconciled_action(p)
            act = ACTION_BADGE.get(eff_act, eff_act or "—")
            note_s = f" ({note})" if note else ""
            _w = p.get("weight_pct")
            _tw = p.get("target_weight_pct")
            _wstr = (f"  · {_w:.0f}%→{_tw:.0f}% of pf" if (_w is not None and _tw is not None)
                     else f"  · {_w:.0f}% of pf" if _w is not None else "")
            with st.expander(f"{act}{note_s}  **{p['symbol']}**  · conviction {p.get('conviction','—')}{_wstr}  · {vbadge}"):
                if p.get("action_note"):
                    st.info(f"🎯 {md(p['action_note'])}")
                st.markdown(f"**Thesis health:** {md(p.get('thesis_health','—'))}")
                st.markdown(f"**Valuation:** {md(p.get('valuation','—'))}")
                lv = []
                if p.get("add_level"):
                    lv.append(f"add ≤ {fmt_num(p['add_level'],2)}")
                if p.get("trim_level"):
                    lv.append(f"trim ≥ {fmt_num(p['trim_level'],2)}")
                if lv:
                    st.markdown("**Levels:** " + " · ".join(lv))
                # "Invalidation" in plain English: the pre-agreed sell signals.
                st.markdown("**🚫 Sell signals — this thesis is WRONG if:**")
                _inv = str(p.get("invalidation") or "").strip()
                if _inv and _inv not in ("—",) and not _inv.startswith("("):
                    for _part in _inv.replace("; ", ";").split(";"):
                        if _part.strip():
                            st.markdown(f"- {md(_part.strip().rstrip('.'))}")
                else:
                    st.caption("— none recorded")
                _sym_trigs = load_active_triggers_by_symbol().get(p["symbol"], [])
                _sym_fired = [t for t in load_fired_triggers() if t["symbol"] == p["symbol"]]
                if _sym_trigs or _sym_fired:
                    _px_now = quote_map.get(p["symbol"]) or p.get("last")
                    _bits = []
                    for t in _sym_fired:
                        _bits.append(f"🚨 **TRIPPED:** {t.get('trigger_detail','')}")
                    for t in _sym_trigs:
                        if t["kind"] == "price_below" and t.get("threshold"):
                            _dist = (f", {abs(t['threshold']/_px_now-1)*100:.0f}% below now"
                                     if _px_now else "")
                            _bits.append(f"🔻 price under **${t['threshold']:g}**{_dist}")
                        elif t["kind"] == "price_above" and t.get("threshold"):
                            _bits.append(f"🔺 price over **${t['threshold']:g}**")
                        elif t["kind"] == "news_keyword" and t.get("keyword"):
                            _bits.append(f"📰 news says *“{t['keyword']}”*")
                    st.caption("🤖 Watched automatically every run: " + " · ".join(_bits))
                if p.get("risks_json"):
                    st.markdown("**Risks:** " + md("; ".join(p["risks_json"])))
                cats = p.get("catalysts_json") or []
                if cats:
                    st.markdown("**Catalysts:** " + md("; ".join(
                        f"{c.get('date','')}: {c.get('event','')}" if isinstance(c, dict) else str(c)
                        for c in cats)))
                stale = val.get("stale_items") or []
                st.caption(f"🕒 {freshness(p.get('data_as_of'))} · checked {val.get('checked_as_of','?')}"
                           f" · {val.get('note','')}"
                           + (f" · ⚠️ stale: {'; '.join(stale)}" if stale else ""))

        st.markdown("#### Holding chart")
        if pf["positions"]:
            chart_sym = st.selectbox("Chart for holding",
                                     [p["symbol"] for p in pf["positions"]], key="dd_chart_sym")
            # This is the chart where decision prices matter MOST: draw the PM's own
            # add/trim levels, the stop, avg cost, and the earnings date on it.
            _cp = next((p for p in pf["positions"] if p["symbol"] == chart_sym), {})
            _ch = holdings.get(chart_sym)
            render_price_chart(chart_sym, key="deepdive",
                               levels=get_levels(chart_sym),
                               earnings=get_earnings(chart_sym),
                               avg_cost=(_ch.get("avg_cost") if _ch else None),
                               add_level=_cp.get("add_level"),
                               trim_level=_cp.get("trim_level"))

        if len(pf_runs) >= 2 and pf_id == pf_runs[0]["id"]:
            prev = storage.get_portfolio(pf_runs[1]["id"])
            prev_act = {p["symbol"]: p.get("action") for p in prev["positions"]}
            chg = [{"symbol": p["symbol"], "from": prev_act[p["symbol"]], "to": p.get("action")}
                   for p in pf["positions"]
                   if prev_act.get(p["symbol"]) and prev_act[p["symbol"]] != p.get("action")]
            st.markdown("#### Action changes vs previous deep-dive")
            st.dataframe(pd.DataFrame(chg or [{"symbol": "— no changes —", "from": "", "to": ""}]),
                         width="stretch", hide_index=True)

# ================================================================ WHAT CHANGED TAB
with tab_chg:
    st.subheader("🔄 What changed vs the previous deep run")
    deep_runs = [r for r in runs if r["mode"] == "deep" and r["status"] == "ok"]
    if len(deep_runs) < 2:
        st.info("Need at least two completed deep runs to compare.")
    else:
        cur, prev = deep_runs[0], deep_runs[1]
        st.caption(f"Comparing #{cur['id']} ({ago(cur['finished_at'])}) vs #{prev['id']} ({ago(prev['finished_at'])})")
        cur_b = {r["symbol"]: r for r in load_board(cur["id"])}
        prev_b = {r["symbol"]: r for r in load_board(prev["id"])}
        ups, downs, movers = [], [], []
        for sym, r in cur_b.items():
            cl, _ = recommend.recommendation(r)
            cw = recommend.WEIGHT.get(cl, 0)
            if sym in prev_b:
                pl, _ = recommend.recommendation(prev_b[sym])
                pw = recommend.WEIGHT.get(pl, 0)
                if cw > pw:
                    ups.append((sym, pl, cl))
                elif cw < pw:
                    downs.append((sym, pl, cl))
                dblend = (r.get("blended") or 0) - (prev_b[sym].get("blended") or 0)
                movers.append((sym, dblend, r.get("blended")))
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**⬆️ Upgrades**")
            if ups:
                st.dataframe(pd.DataFrame([{"symbol": s, "from": recommend.BADGE.get(p, p),
                                            "to": recommend.BADGE.get(c, c)}
                                           for s, p, c in ups]),
                             width="stretch", hide_index=True)
            else:
                st.caption("No upgrades.")  # a caption, not a fake data row
        with cc2:
            st.markdown("**⬇️ Downgrades**")
            if downs:
                st.dataframe(pd.DataFrame([{"symbol": s, "from": recommend.BADGE.get(p, p),
                                            "to": recommend.BADGE.get(c, c)}
                                           for s, p, c in downs]),
                             width="stretch", hide_index=True)
            else:
                st.caption("No downgrades.")
        new_syms = [s for s in cur_b if s not in prev_b]
        gone_syms = [s for s in prev_b if s not in cur_b]
        st.markdown(f"**🆕 New to universe:** {', '.join(new_syms) or '—'}")
        st.markdown(f"**➖ Dropped:** {', '.join(gone_syms) or '—'}")
        movers.sort(key=lambda x: abs(x[1]), reverse=True)
        st.markdown("**📈 Biggest composite moves**")
        # This tab is entirely about polarity — encode it (sign format + color).
        _mv_css = (lambda v: "color:#4ade80" if isinstance(v, (int, float)) and v > 0
                   else "color:#f87171" if isinstance(v, (int, float)) and v < 0 else "")
        _mvdf = pd.DataFrame([{"symbol": s, "Δ composite": round(d, 3), "now": b}
                              for s, d, b in movers[:12]])
        if not _mvdf.empty:
            st.dataframe(_mvdf.style.map(_mv_css, subset=["Δ composite"]),
                         width="stretch", hide_index=True,
                         column_config={"Δ composite": st.column_config.NumberColumn(
                             "Δ composite", format="%+.3f")})
        else:
            st.caption("No movers.")

# ================================================================ CALENDAR TAB
with tab_cal:
    st.subheader("📅 Upcoming catalysts & earnings")
    st.caption("Dated catalysts from the latest research + earnings dates, next ~60 days. "
               "Know what you're holding into.")
    cal_rows = []
    # Earnings dates from the board metrics (already fetched by the pipeline)
    for r in board:
        m = r.get("metrics") or {}
        dte, edate = m.get("days_to_earnings"), m.get("earnings_date")
        if edate and isinstance(dte, (int, float)) and 0 <= dte <= 60:
            cal_rows.append({"date": edate, "symbol": r["symbol"],
                             "event": "📊 Earnings report",
                             "held": "📌" if r.get("held_position") else "",
                             "source": "screen"})
    # Research catalysts (deep runs + latest portfolio dive)
    def _add_cats(sym, cats, held):
        for c in cats or []:
            if isinstance(c, dict) and c.get("date"):
                cal_rows.append({"date": str(c["date"])[:10], "symbol": sym,
                                 "event": str(c.get("event", ""))[:110],
                                 "held": "📌" if held else "", "source": "research"})
    _pf_latest2 = storage.latest_portfolio_run("ok")
    if _pf_latest2:
        for p in storage.get_portfolio(_pf_latest2["id"])["positions"]:
            _add_cats(p["symbol"], p.get("catalysts_json"), True)
    if cal_rows:
        import calendar as _pycal
        from datetime import timedelta as _td

        caldf = pd.DataFrame(cal_rows).drop_duplicates(subset=["date", "symbol", "event"])
        today_d = datetime.now(timezone.utc).date()

        # Split into day-precise events and month-only ("YYYY-MM") events.
        by_day: dict = {}      # date -> [event dicts]
        by_month: dict = {}    # (y, m) -> [event dicts]
        for _, ev in caldf.iterrows():
            ds = str(ev["date"]).strip()
            try:
                if len(ds) >= 10:
                    d = datetime.fromisoformat(ds[:10]).date()
                    if d >= today_d - _td(days=1):
                        by_day.setdefault(d, []).append(ev)
                elif len(ds) == 7:  # YYYY-MM
                    y, m = int(ds[:4]), int(ds[5:7])
                    if (y, m) >= (today_d.year, today_d.month):
                        by_month.setdefault((y, m), []).append(ev)
            except (ValueError, TypeError):
                continue

        def _chip(ev) -> str:
            is_earn = "Earnings" in str(ev["event"])
            color = "#eab308" if is_earn else "#2dd4bf"
            bg = "rgba(234,179,8,.14)" if is_earn else "rgba(45,212,191,.12)"
            icon = "📊" if is_earn else "⭐"
            held = "📌" if ev["held"] else ""
            tip = f"{ev['symbol']}: {ev['event']}"
            return (f"<div title=\"{tip}\" style='background:{bg};color:{color};"
                    f"border:1px solid {color}44;border-radius:6px;padding:1px 5px;"
                    f"margin:2px 0;font-size:0.72em;font-weight:600;white-space:nowrap;"
                    f"overflow:hidden;text-overflow:ellipsis;cursor:default'>"
                    f"{held}{icon} {ev['symbol']}</div>")

        def _month_html(y: int, m: int) -> str:
            cal = _pycal.Calendar(firstweekday=6)  # Sunday-first
            weeks = cal.monthdatescalendar(y, m)
            out = [f"<div style='margin:14px 0 6px;font-weight:700;font-size:1.05em'>"
                   f"{_pycal.month_name[m]} {y}</div>"]
            # month-only events banner
            mev = by_month.get((y, m), [])
            if len(mev):
                chips = "".join(_chip(e) for e in mev[:8])
                out.append(f"<div style='margin:4px 0 8px;padding:6px 10px;background:#13181f;"
                           f"border:1px dashed #21262d;border-radius:8px'>"
                           f"<span style='opacity:.6;font-size:.8em'>sometime this month:</span>"
                           f"<div style='display:flex;gap:6px;flex-wrap:wrap'>{chips}</div></div>")
            out.append("<table style='width:100%;table-layout:fixed;border-collapse:separate;"
                       "border-spacing:3px'>")
            out.append("<tr>" + "".join(
                f"<th style='opacity:.55;font-size:.75em;font-weight:600;padding:2px'>{d}</th>"
                for d in ("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")) + "</tr>")
            for week in weeks:
                out.append("<tr>")
                for d in week:
                    in_month = d.month == m
                    is_today = d == today_d
                    is_wknd = d.weekday() >= 5
                    bg = "#0e1117" if not in_month else ("#1a2f2c" if is_today else
                                                         "#11151c" if is_wknd else "#161b22")
                    border = "#2dd4bf" if is_today else "#21262d"
                    op = "0.35" if not in_month else "1"
                    chips = "".join(_chip(e) for e in by_day.get(d, [])[:3]) if in_month else ""
                    more = len(by_day.get(d, [])) - 3 if in_month else 0
                    if more > 0:
                        chips += f"<div style='opacity:.5;font-size:.7em'>+{more} more</div>"
                    daynum = (f"<div style='font-size:.75em;font-weight:{700 if is_today else 400};"
                              f"color:{'#2dd4bf' if is_today else '#8b949e'}'>{d.day}</div>")
                    out.append(f"<td style='vertical-align:top;height:64px;background:{bg};"
                               f"border:1px solid {border};border-radius:8px;padding:4px 5px;"
                               f"opacity:{op}'>{daynum}{chips}</td>")
                out.append("</tr>")
            out.append("</table>")
            return "".join(out)

        # Render the months that carry events (current month first, up to 3).
        months = []
        y, m = today_d.year, today_d.month
        for _ in range(3):
            months.append((y, m))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        shown = [mm for mm in months
                 if mm in by_month or any((d.year, d.month) == mm for d in by_day)]
        for (yy, mm) in (shown or months[:2]):
            # overflow-x wrapper: the 7-column fixed grid must scroll, not crush,
            # on narrow screens.
            st.markdown("<div style='overflow-x:auto'><div style='min-width:640px'>"
                        + _month_html(yy, mm) + "</div></div>", unsafe_allow_html=True)
        st.caption("📊 earnings · ⭐ research catalyst · 📌 currently held · hover a chip for details")

        # List twin: chip text is hover-only (dead on touch) and '+N more' hides
        # overflow — the chronological list is the accessible full record.
        _n_all = (sum(len(v) for v in by_day.values())
                  + sum(len(v) for v in by_month.values()))
        with st.expander(f"📜 All events as a list ({_n_all})"):
            for d in sorted(by_day):
                for ev in by_day[d]:
                    st.markdown(f"- `{d}` {ev['held']} **{ev['symbol']}** — {md(ev['event'])}")
            for (yy, mm) in sorted(by_month):
                for ev in by_month[(yy, mm)]:
                    st.markdown(f"- `{yy}-{mm:02d} (month)` {ev['held']} **{ev['symbol']}** — "
                                f"{md(ev['event'])}")

        _earn_held = [(ev["symbol"], d) for d, evs in sorted(by_day.items()) for ev in evs
                      if ev["held"] and "Earnings" in str(ev["event"])]
        if _earn_held:
            st.warning("📌 You are holding into: " + " · ".join(
                f"{s} ({d})" for s, d in _earn_held))
    else:
        st.info("No dated catalysts yet — run a deep run / portfolio deep-dive to populate.")

# ================================================================ ACCURACY TAB
with tab_acc:
    st.subheader("🎯 Does the system actually predict anything?")
    st.caption("Forward excess returns **vs the screened universe average** (selection skill, "
               "not theme beta) of past picks, grouped by the rating given at pick time — "
               "plus the PM's target-weight portfolio vs your actual weights. "
               "Judge the tool before trusting it.")
    try:
        cal_df, shadow = load_calibration()
    except Exception as exc:  # noqa: BLE001
        cal_df, shadow = pd.DataFrame(), []
        st.warning(f"Calibration unavailable: {exc}")
    if cal_df is not None and not cal_df.empty:
        st.markdown("**Rating calibration** — in a calibrated system the bars should step "
                    "down from strong_buy to sell:")
        # Shape question -> chart first (sign-colored, n= labels bake in the sample-size
        # caveat); the raw table follows as its backup view.
        _calfig = go.Figure()
        _shown = False
        for w, op in (("1w", 0.55), ("4w", 0.8), ("13w", 1.0)):
            col = f"avg_excess_{w}"
            if col not in cal_df or cal_df[col].dropna().empty:
                continue
            _vals = cal_df[col]
            _calfig.add_bar(
                name=w, x=cal_df["rating"], y=_vals, opacity=op,
                marker_color=["#26a69a" if isinstance(v, (int, float)) and v >= 0
                              else "#ef5350" for v in _vals],
                text=[f"n={int(n)}" if n == n else "" for n in cal_df[f"n_{w}"]],
                textposition="outside", cliponaxis=False)
            _shown = True
        if _shown:
            _calfig.add_hline(y=0, line=dict(color="#8b949e", width=1))
            _calfig.update_layout(barmode="group", height=320,
                                  yaxis_title="avg excess vs universe, %",
                                  legend=dict(orientation="h"),
                                  margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(_calfig, width="stretch",
                            config={"displayModeBar": False, "staticPlot": True})
        with st.expander("Table view"):
            st.dataframe(cal_df, width="stretch", hide_index=True)
        st.caption("Only complete windows are graded (a 4-week bar appears once picks are "
                   "4 weeks old). Small n = noise — judge shapes, not single bars.")
    else:
        st.info("No graded picks yet — windows fill as picks age (first 1-week grades ~5 "
                "trading days after a deep run).")
    # ---- Does the Claude layer add value over the quant screen? -------------------
    try:
        _attr, _cal_tbl, _by_pv = load_calibration_extra()
    except Exception as exc:  # noqa: BLE001
        _attr, _cal_tbl, _by_pv = {}, pd.DataFrame(), pd.DataFrame()
        st.warning(f"Attribution unavailable: {exc}")
    st.markdown("**Does the Claude layer add value over the quant screen?** — buy-rated picks "
                "vs. finalists that reached the same slots without research (screen-only), "
                "both graded on 4-week excess vs the universe, **net of a "
                f"{config.get('evaluation.cost_bps', 30)} bp round-trip cost** on the buys.")
    if _attr.get("edge_pp") is not None:
        _ac1, _ac2, _ac3 = st.columns(3)
        _ac1.metric("Claude buys (net)", f"{_attr['claude_avg']:+.2f}%", f"n={_attr['n_claude']}",
                    delta_color="off")
        _ac2.metric("Screen-only control", f"{_attr['screen_avg']:+.2f}%", f"n={_attr['n_screen']}",
                    delta_color="off")
        _ac3.metric("Edge (95% CI)", f"{_attr['edge_pp']:+.2f} pp",
                    f"{_attr['ci_lo']:+.2f} … {_attr['ci_hi']:+.2f}", delta_color="off")
        if _attr.get("significant"):
            _msg = ("✅ The interval excludes zero — the research layer is "
                    + ("adding" if _attr["edge_pp"] > 0 else "**subtracting**") + " value.")
        else:
            _msg = ("⚖️ The interval includes zero — no evidence either way yet. Keep "
                    "accumulating; do not re-weight on this.")
        st.caption(_msg + " Survivorship note: both cohorts come from the same hand-picked "
                   "universe, so only the *difference* is meaningful, not the levels.")
    else:
        st.info(_attr.get("note", "Attribution needs graded picks in both cohorts."))

    with st.expander("🎚️ Precision by call type (rating × conviction × regime, Wilson 95% CI)"):
        st.caption("How often a call of this kind beat the universe, net of costs. Cells with "
                   f"n < {config.get('evaluation.min_n_for_ci', 8)} are marked unreliable — read "
                   "the CI width, not the point. This is the raw material a meta-labeling model "
                   "would learn from; it stays a table until there are ≥1,000 graded calls.")
        if _cal_tbl is not None and not _cal_tbl.empty:
            st.dataframe(_cal_tbl, width="stretch", hide_index=True, column_config={
                "precision_pct": st.column_config.NumberColumn("precision", format="%.0f%%"),
                "ci_lo": st.column_config.NumberColumn("CI lo", format="%.0f%%"),
                "ci_hi": st.column_config.NumberColumn("CI hi", format="%.0f%%"),
                "avg_net_excess": st.column_config.NumberColumn("avg net excess", format="%+.2f%%"),
            })
        else:
            st.info("No 4-week-graded picks yet.")
    with st.expander("🧬 By prompt version (is a prompt change helping?)"):
        st.caption("Ratings grouped by the analyst prompt that produced them. 'pre-v4' = before "
                   "versioning was recorded (2026-09-07). A new version earns trust on its own "
                   "outcomes, not on the blended history.")
        if _by_pv is not None and not _by_pv.empty:
            st.dataframe(_by_pv, width="stretch", hide_index=True)
        else:
            st.info("Nothing graded yet.")
        try:
            from src import performance as _perf
            _rt = _perf.red_team_stats()
        except Exception:  # noqa: BLE001
            _rt = pd.DataFrame()
        st.markdown("**Red-team monitor** — kill rate and haircut mix per prompt version. A "
                    "healthy stress test kills few and haircuts mostly 0–1; a kill rate near "
                    "the July-2026 overkill (85%) or a haircut mix stuck at 2+ means the prompt "
                    "is grading its own eloquence again.")
        if _rt is not None and not _rt.empty:
            st.dataframe(_rt, width="stretch", hide_index=True)
        else:
            st.caption("No red-teamed reports yet.")

    # ---- Learned ranking pillar (dark until its kill criteria pass) -------------------
    with st.expander("🤖 Learned ranking pillar — walk-forward verdict"):
        try:
            from src.ml import infer as _ml_infer
            _meta = _ml_infer.latest_meta()
        except Exception:  # noqa: BLE001
            _meta = None
        _wL = config.get("scoring.pillars.learned", 0.0) or 0.0
        if not _meta:
            st.info("No trained model. `python -m scripts.data_refresh` then "
                    "`python -m scripts.train_ranker` builds one; it ships at weight 0 until every "
                    "kill criterion passes.")
        else:
            _s = _meta.get("summary") or {}
            _v = _meta.get("verdict") or {}
            _m1, _m2, _m3, _m4 = st.columns(4)
            _m1.metric("OOS rank IC", f"{_s.get('ic_mean')}", f"t = {_s.get('ic_t')}", delta_color="off")
            _m2.metric("vs composite IC", f"{_meta.get('baseline_ic')}", delta_color="off")
            _m3.metric("Deflated Sharpe", f"{_s.get('dsr')}", f"{_s.get('n_trials')} trials", delta_color="off")
            _m4.metric("Verdict", "SHIP-ABLE" if _v.get("pass") else "DO NOT SHIP",
                       f"weight now {_wL}", delta_color="off")
            for _n, _c in (_v.get("checks") or {}).items():
                st.markdown(f"- {'✅' if _c.get('ok') else '❌'} **{_n}** — {_c.get('detail')}")
            st.caption(f"Model fold {_meta.get('test_date')} · trained {_meta.get('trained_at')} · "
                       f"{_s.get('n_dates')} out-of-sample months · {_meta.get('spec', {}).get('model')}. "
                       "Universe is hindsight-curated: read the ranking metrics, not the returns.")

    st.markdown("**Shadow portfolio** — if you had rebalanced to the PM's target weights at "
                "each deep-dive, vs what your actual weights did since "
                "(undeployed target weight counts as cash at 0%):")
    if shadow:
        _ek = "edge_net_pct" if all("edge_net_pct" in r for r in shadow) else "edge_pct"
        _shfig = go.Figure(go.Bar(
            x=[f"#{r['run_id']} {r['date']}" for r in shadow],
            y=[r.get(_ek) for r in shadow],
            marker_color=["#26a69a" if (r.get(_ek) or 0) >= 0 else "#ef5350"
                          for r in shadow],
            text=[f"{r.get(_ek):+.2f}%" if r.get(_ek) is not None else ""
                  for r in shadow], textposition="outside", cliponaxis=False))
        _shfig.add_hline(y=0, line=dict(color="#8b949e", width=1))
        _shfig.update_layout(height=260, yaxis_title="PM edge vs actual, net of costs, %",
                             margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(_shfig, width="stretch",
                        config={"displayModeBar": False, "staticPlot": True})
        with st.expander("Table view"):
            st.dataframe(pd.DataFrame(shadow), width="stretch", hide_index=True, column_config={
                "actual_ret_pct": st.column_config.NumberColumn("actual %", format="%+.2f%%"),
                "target_ret_pct": st.column_config.NumberColumn("PM target %", format="%+.2f%%"),
                "edge_pct": st.column_config.NumberColumn("edge (gross)", format="%+.2f%%"),
                "turnover_pct": st.column_config.NumberColumn("weight moved", format="%.0f%%"),
                "cost_pct": st.column_config.NumberColumn("cost", format="%.3f%%"),
                "edge_net_pct": st.column_config.NumberColumn("edge (net)", format="%+.2f%%"),
            })
        _edges = [r[_ek] for r in shadow if r.get(_ek) is not None]
        if _edges:
            st.metric("Avg PM edge per dive (net of costs)", f"{sum(_edges)/len(_edges):+.2f}%")
            st.caption(f"Costs: one-way {config.get('evaluation.cost_bps', 30) / 2:g} bp on every "
                       "point of weight the targets would have moved.")
    else:
        st.info("Populates once a deep-dive with target weights (run #10+) is a few days old.")

