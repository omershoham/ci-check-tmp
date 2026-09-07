"""Dual scoring: Quality score + Deal score -> blended conviction + tags.

Scores are cross-sectional percentile ranks within the day's universe (0..1),
so "good" is always relative to the current AI-infra peer set. Missing metrics
degrade to a neutral 0.5 rather than dropping the name.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from . import config

# Bump whenever a pillar's DEFINITION changes (components, ranking method, signs).
# pillar_ic filters to the current version so pre-change scores can't contaminate
# the information-coefficient statistics.
SCORE_VERSION = 3  # v3: 12-1 momentum, sector-relative ranks, deduped fundamentals,
                   #     sentiment-only sentiment pillar
# v4 = v3 + the learned (ML) ranking pillar carrying non-zero weight. The version is
# derived from config so turning the pillar on is what bumps it — rows scored with and
# without the pillar never mix inside pillar_ic.


def learned_weight() -> float:
    try:
        return float((config.get("scoring.pillars", {}) or {}).get("learned", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def effective_score_version() -> int:
    return 4 if learned_weight() > 0 else SCORE_VERSION

# Static base relevance of each sub-sector to hyperscaler capex (feeds Deal score).
SUBSECTOR_RELEVANCE = {
    "semis_equipment": 0.95,
    "networking_optical": 0.85,
    "datacenter_power": 0.80,
    "cloud_ai_software": 0.70,
}

# Component metrics per subscore. (column, higher_is_better)
# Deduped per CR#2: p_fcf removed (reciprocal of fcf_yield — one number was ~20% of
# the pillar); pegRatio removed (mixes valuation with growth, which has its own
# subscore); operatingMargins moved from CASHFLOW (it isn't a cash-flow metric) to
# QUALITY; returnOnAssets added as the profitability proxy (GP/A in spirit);
# fcf_margin gives cashflow a genuine second leg.
_VALUATION = [("trailingPE", False), ("forwardPE", False),
              ("enterpriseToEbitda", False), ("priceToSalesTrailing12Months", False)]
_CASHFLOW = [("fcf_yield", True), ("fcf_margin", True)]
_GROWTH = [("revenueGrowth", True), ("earningsGrowth", True)]
_QUALITY = [("returnOnEquity", True), ("returnOnAssets", True),
            ("grossMargins", True), ("operatingMargins", True), ("debtToEquity", False)]

# "Lower is better" VALUATION multiples where a non-positive value means the company
# is loss-making, not "cheap". These must be masked to NaN (→ neutral) before ranking,
# otherwise a negative P/E ranks as the cheapest name in the universe. (debtToEquity is
# deliberately NOT here: zero debt is legitimately best, so it must keep ranking.)
_MASK_NONPOS = {"trailingPE", "forwardPE", "pegRatio", "enterpriseToEbitda",
                "priceToSalesTrailing12Months", "p_fcf", "ev_sales"}


def _pct(series: pd.Series, higher_is_better: bool, mask_nonpositive: bool = False,
         groups: pd.Series | None = None) -> pd.Series:
    """Percentile rank; when `groups` is given, blend 50% universe-wide rank with 50%
    within-subsector rank. Cloud software structurally out-multiples semis, so a pure
    universe valuation rank is largely a SUBSECTOR bet — the within-group half turns
    it back into a stock-selection signal. Small groups are naturally shrunk by the
    universe half."""
    s = series.where(series > 0) if mask_nonpositive else series
    r = s.rank(pct=True)
    if groups is not None:
        rg = s.groupby(groups).rank(pct=True)
        r = 0.5 * r + 0.5 * rg
    if not higher_is_better:
        r = 1.0 - r
    return r.fillna(0.5)


def _subscore(df: pd.DataFrame, components: list[tuple[str, bool]],
              groups: pd.Series | None = None) -> pd.Series:
    cols = []
    for col, hib in components:
        if col in df.columns:
            cols.append(_pct(pd.to_numeric(df[col], errors="coerce"), hib,
                             mask_nonpositive=(col in _MASK_NONPOS), groups=groups))
    if not cols:
        return pd.Series(0.5, index=df.index)
    return pd.concat(cols, axis=1).mean(axis=1)


def fundamental_pillar(df: pd.DataFrame, subsectors: dict[str, str]) -> pd.Series:
    """The fundamental pillar (0..1) for a metrics frame indexed by symbol — the exact
    computation `score_universe` uses, exposed so the backtest harness can replay the
    live pillar on point-in-time snapshots with the same code path."""
    qw = config.get("scoring.quality.subweights", {})
    grp = pd.Series({s: subsectors.get(s, "") for s in df.index})
    val = _subscore(df, _VALUATION, groups=grp)
    cash = _subscore(df, _CASHFLOW, groups=grp)
    grow = _subscore(df, _GROWTH, groups=grp)
    qual = _subscore(df, _QUALITY, groups=grp)
    return (val * qw.get("valuation", 0.3) + cash * qw.get("cashflow", 0.3)
            + grow * qw.get("growth", 0.2) + qual * qw.get("quality", 0.2))


def _sentiment_pillar(sig: dict | None) -> float:
    """0..1 news-sentiment pillar: TONE ONLY. hyperscaler_signal is thesis-RELEVANCE,
    not tone, and already carries 45% of the deal score — blending it here would
    double-count relevance across the composite and make the pillar's information
    coefficient uninterpretable (which signal is being measured?)."""
    if not sig:
        return 0.5
    ss = sig.get("sentiment_score")
    return 0.5 if ss is None else float(ss) / 10.0


def score_universe(metrics: dict[str, dict],
                   subsectors: dict[str, str],
                   signals: dict[str, dict] | None = None,
                   technicals: dict[str, float] | None = None,
                   ml_scores: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Three-pillar scoring (+ an optional fourth, learned pillar).

    metrics: symbol -> fundamentals dict.
    signals: symbol -> local_llm dict (hyperscaler_signal, sentiment_score, ...).
    technicals: symbol -> technical_score (0..1).
    ml_scores: symbol -> raw score from the learned ranker (src/ml). Weighted only by
        config scoring.pillars.learned (default 0 = shipped dark); still persisted so the
        dashboard can show it. Missing/None -> neutral 0.5 for every name.
    """
    if not metrics:
        return []
    signals = signals or {}
    technicals = technicals or {}
    ml_scores = ml_scores or {}
    df = pd.DataFrame(list(metrics.values()))
    df = df.set_index("symbol", drop=False)

    qw = config.get("scoring.quality.subweights", {})
    pillars = config.get("scoring.pillars", {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.35})
    tagcfg = config.get("scoring.tags", {})

    # --- Fundamental pillar (valuation/cashflow/growth/quality, percentile-ranked;
    #     50% universe + 50% within-subsector so multiples compare within peer groups)
    grp = pd.Series({s: subsectors.get(s, "") for s in df.index})
    val = _subscore(df, _VALUATION, groups=grp)
    cash = _subscore(df, _CASHFLOW, groups=grp)
    grow = _subscore(df, _GROWTH, groups=grp)
    qual = _subscore(df, _QUALITY, groups=grp)
    fundamental = (val * qw.get("valuation", 0.3) + cash * qw.get("cashflow", 0.3)
                   + grow * qw.get("growth", 0.2) + qual * qw.get("quality", 0.2))

    # --- Deal-relevance (feeds the "deal" tag): sub-sector + hyperscaler signal + momentum.
    # Weights come from config (scoring.deal.subweights) so tuning the YAML is live.
    relevance = pd.Series({s: SUBSECTOR_RELEVANCE.get(subsectors.get(s, ""), 0.6) for s in df.index})
    mom_raw = pd.concat([
        pd.to_numeric(df.get("mom_50", pd.Series(index=df.index)), errors="coerce"),
        pd.to_numeric(df.get("mom_200", pd.Series(index=df.index)), errors="coerce"),
    ], axis=1).mean(axis=1)
    momentum = _pct(mom_raw, higher_is_better=True)

    def _hs(sym: str) -> float:
        # A genuine 0 ("no hyperscaler relevance") must NOT be coerced to neutral 5;
        # only a truly missing/unknown signal defaults to 5.
        v = (signals.get(sym) or {}).get("hyperscaler_signal")
        return (5.0 if v is None else float(v)) / 10.0

    hsignal = pd.Series({s: _hs(s) for s in df.index})
    dw = config.get("scoring.deal.subweights", {}) or {}
    w_rel = dw.get("subsector_relevance", 0.35)
    w_hs = dw.get("hyperscaler_signal", 0.45)
    w_mom = dw.get("momentum", 0.20)
    deal = relevance * w_rel + hsignal * w_hs + momentum * w_mom

    # --- Technical & Sentiment pillars
    # technicals: symbol -> dict from technical.compute_many (has technical_score, state, rsi).
    technical = pd.Series({s: float((technicals.get(s) or {}).get("technical_score", 0.5))
                           for s in df.index})
    sentiment = pd.Series({s: _sentiment_pillar(signals.get(s)) for s in df.index})

    # --- Composite (the ranking score).
    # Each pillar is percentile-ranked across the universe BEFORE weighting. Without this
    # the three pillars sit on different scales (fundamental ~uniform, technical an absolute
    # rule score, sentiment a compressed LLM rescale), so the widest-dispersion pillar would
    # dominate regardless of the configured weights. Ranking makes the weights mean what they say.
    wT = pillars.get("technical", 0.35)
    wF = pillars.get("fundamental", 0.30)
    wS = pillars.get("sentiment", 0.35)
    wL = learned_weight()
    technical_r = _pct(technical, higher_is_better=True)
    fundamental_r = _pct(fundamental, higher_is_better=True)
    sentiment_r = _pct(sentiment, higher_is_better=True)
    learned = pd.Series({s: ml_scores.get(s) for s in df.index}, dtype=float)
    learned_r = _pct(learned, higher_is_better=True) if learned.notna().any() \
        else pd.Series(0.5, index=df.index)
    # Normalize by the weight sum so the composite stays on the 0..1 scale when a user
    # turns the learned pillar on without re-balancing the other three (identical to
    # the old formula whenever the weights already sum to 1).
    wsum = (wT + wF + wS + wL) or 1.0
    composite = (technical_r * wT + fundamental_r * wF + sentiment_r * wS + learned_r * wL) / wsum

    # Sector-relative valuation: median PE / fwd PE / EV-EBITDA per sub-sector.
    df["_sub"] = [subsectors.get(s, "") for s in df.index]

    def _sec_med(col: str) -> dict:
        if col not in df.columns:
            return {}
        v = pd.to_numeric(df[col], errors="coerce")
        v = v.where(v > 0)  # ignore negative/zero (loss-making) valuations
        return v.groupby(df["_sub"]).median().to_dict()

    pe_med = _sec_med("trailingPE")
    fpe_med = _sec_med("forwardPE")
    ev_med = _sec_med("enterpriseToEbitda")

    rows: list[dict[str, Any]] = []
    q_thresh = tagcfg.get("quality_threshold", 0.6)
    d_thresh = tagcfg.get("deal_threshold", 0.6)
    require_fcf = config.get("scoring.quality.require_positive_fcf_for_tag", True)
    hr = tagcfg.get("higher_risk", {})
    pe_ceiling = hr.get("pe_risk_ceiling", 60)
    earnings_soon_days = int(config.get("screen.earnings_soon_days", 7))

    for s in df.index:
        m = metrics[s]
        fcf = m.get("freeCashflow")
        positive_fcf = bool(fcf and fcf > 0)
        pe = m.get("trailingPE")
        tech_info = technicals.get(s) or {}

        tags: list[str] = []
        is_quality = fundamental[s] >= q_thresh and (positive_fcf or not require_fcf)
        is_deal = deal[s] >= d_thresh
        if is_quality:
            tags.append("quality")
        if is_deal:
            tags.append("deal")
        if is_quality and is_deal:
            tags.append("both")
        higher_risk = False
        if hr.get("flag_if_unprofitable", True) and not m.get("profitable", False):
            higher_risk = True
        if pe is None or (isinstance(pe, (int, float)) and pe > pe_ceiling):
            higher_risk = True
        if higher_risk:
            tags.append("higher-risk")
        dte = m.get("days_to_earnings")
        if isinstance(dte, (int, float)) and 0 <= dte <= earnings_soon_days:
            tags.append("earnings-soon")

        sig = signals.get(s) or {}
        rows.append({
            "symbol": s,
            "name": m.get("name", s),
            "subsector": subsectors.get(s, ""),
            # pillar scores (0..1)
            "technical": round(float(technical[s]), 4),
            "fundamental": round(float(fundamental[s]), 4),
            "sentiment": round(float(sentiment[s]), 4),
            "blended": round(float(composite[s]), 4),   # composite = ranking score
            # kept for back-compat / tags
            "quality": round(float(fundamental[s]), 4),
            "deal": round(float(deal[s]), 4),
            "tags": tags,
            "metrics": {
                "price": m.get("price"),
                "marketCap": m.get("marketCap"),
                "trailingPE": pe,
                "forwardPE": m.get("forwardPE"),
                "p_fcf": m.get("p_fcf"),
                "fcf_yield": m.get("fcf_yield"),
                "revenueGrowth": m.get("revenueGrowth"),
                "grossMargins": m.get("grossMargins"),
                "returnOnEquity": m.get("returnOnEquity"),
                # pillar scores (persisted here so the dashboard can show them)
                "pillar_technical": round(float(technical[s]), 4),
                "pillar_fundamental": round(float(fundamental[s]), 4),
                "pillar_sentiment": round(float(sentiment[s]), 4),
                # learned pillar (dark until scoring.pillars.learned > 0)
                "pillar_learned": (round(float(learned[s]), 4)
                                   if s in learned.index and learned[s] == learned[s] else None),
                "pillar_learned_rank": round(float(learned_r[s]), 4) if learned.notna().any() else None,
                "tech_state": tech_info.get("state"),
                "rsi": tech_info.get("rsi"),
                "macd_hist": tech_info.get("macd_hist"),
                "pos_52w": tech_info.get("pos_52w"),
                "atr_pct": tech_info.get("atr_pct"),
                # trend setup + relative strength
                "setup": tech_info.get("setup"),
                "setup_label": tech_info.get("setup_label"),
                "signals": tech_info.get("signals"),
                "rs_3m": tech_info.get("rs_3m"),
                "rs_6m": tech_info.get("rs_6m"),
                "benchmark": tech_info.get("benchmark"),
                # modern valuation metrics
                "ev_ebitda": m.get("enterpriseToEbitda"),
                "ev_sales": m.get("ev_sales"),
                "ps": m.get("priceToSalesTrailing12Months"),
                "peg": m.get("pegRatio"),
                "rule_of_40": m.get("rule_of_40"),
                "net_cash": m.get("net_cash"),
                "earnings_date": m.get("earnings_date"),
                "days_to_earnings": m.get("days_to_earnings"),
                # sector-relative valuation (sub-sector medians)
                "sector_pe": pe_med.get(subsectors.get(s, "")),
                "sector_fwdpe": fpe_med.get(subsectors.get(s, "")),
                "sector_evebitda": ev_med.get(subsectors.get(s, "")),
                "sentiment_label": sig.get("sentiment"),
                "sub_valuation": round(float(val[s]), 3),
                "sub_cashflow": round(float(cash[s]), 3),
                "sub_growth": round(float(grow[s]), 3),
                "sub_quality": round(float(qual[s]), 3),
            },
        })

    rows.sort(key=lambda r: r["blended"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


if __name__ == "__main__":
    from . import universe, fundamentals, technical
    syms = universe.symbols()[:12]
    print(f"Fetching {len(syms)} names for a scoring demo...")
    metr = fundamentals.fetch_many(syms)
    sub = universe.subsector_map()
    metr = {s: m for s, m in metr.items() if fundamentals.passes_floor(m)}
    techs = technical.compute_many(list(metr.keys()))
    print(f"{len(metr)} passed the liquidity floor")
    scored = score_universe(metr, sub, technicals=techs)
    print(f"\n{'rank':<4} {'sym':<6} {'comp':<6} {'tech':<6} {'fund':<6} {'sent':<6} tags")
    for r in scored[:15]:
        print(f"{r['rank']:<4} {r['symbol']:<6} {r['blended']:<6} "
              f"{r['technical']:<6} {r['fundamental']:<6} {r['sentiment']:<6} {','.join(r['tags'])}")
