"""ONE definition of every derived fundamental metric.

`derive(raw)` turns a dict of raw company figures into the metrics the scoring
pillar consumes (fcf_yield, ev_sales, fcf_margin, rule_of_40, net_cash, margins,
growth, PE ...). It is called by BOTH:

  * the live path  — `fundamentals.fetch_one`, raw = yfinance `.info` fields
  * the historical path — `data.sec_facts`, raw = the same field names rebuilt from
    SEC XBRL facts as they stood on a past date

Sharing the function is what makes a model trained on historical snapshots see the
same definitions at inference that the live screen produces. If a ratio changes,
it changes in exactly one place and `SCORE_VERSION` gets bumped.

Raw field names follow yfinance's `.info` vocabulary so the live path needs no
translation; the historical builder maps XBRL tags onto these names.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

# Raw fields `derive` reads (all optional; missing -> the dependent metric is None).
RAW_FIELDS = (
    "price", "marketCap", "freeCashflow", "totalRevenue", "grossProfits", "ebitda",
    "operatingIncome", "netIncome", "totalAssets", "totalStockholderEquity",
    "totalCash", "totalDebt", "sharesOutstanding", "trailingEps",
    "revenueGrowth", "earningsGrowth", "grossMargins", "operatingMargins",
    "returnOnEquity", "returnOnAssets", "debtToEquity",
    "trailingPE", "forwardPE", "pegRatio", "enterpriseToEbitda", "enterpriseToRevenue",
    "priceToSalesTrailing12Months", "priceToBook", "fiftyDayAverage",
    "twoHundredDayAverage", "cfo", "operatingCashflow", "dna",
    "financialCurrency", "earningsTimestamp", "earningsTimestampStart",
)

# Ratios that mix a USD market figure with a local-currency statement figure are
# meaningless for non-USD reporters (ADRs); per-share USD ratios and margins survive.
_CCY_SENSITIVE = ("net_cash", "fcf_yield", "p_fcf", "ev_sales", "enterpriseToEbitda",
                  "priceToSalesTrailing12Months", "priceToBook", "p_ffo", "ffo_yield",
                  "trailingPE")
# ...but only when WE built them out of a market cap and a statement figure. A ratio the
# source handed us whole (yfinance's `trailingPE`, `priceToBook`) is already internally
# consistent — Yahoo prices the ADR in USD against USD-converted per-share figures — so
# masking it would throw away the only correct multiple a foreign filer has. TSM is the
# case that forced this: SEC facts give USD cap / TWD income = P/E 206, which read as a
# risk flag, while `.info` says 31.8 and is right.
_CCY_FALLBACK_ONLY = ("trailingPE", "priceToBook")


def _pos(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None   # NaN -> None


def _ratio(num, den) -> float | None:
    n, d = _pos(num), _pos(den)
    if n is None or d is None or d == 0:
        return None
    return n / d


def derive(raw: dict[str, Any], as_of: date | None = None) -> dict[str, Any]:
    """Return the derived-metrics dict. `raw` is NOT mutated. `as_of` (default today)
    anchors days_to_earnings when an earnings timestamp is present."""
    m: dict[str, Any] = {}
    price = _pos(raw.get("price") or raw.get("currentPrice") or raw.get("regularMarketPrice"))
    mcap = _pos(raw.get("marketCap"))
    if mcap is None and price is not None and _pos(raw.get("sharesOutstanding")):
        mcap = price * float(raw["sharesOutstanding"])
    fcf = _pos(raw.get("freeCashflow"))
    rev = _pos(raw.get("totalRevenue"))
    cash, debt = _pos(raw.get("totalCash")), _pos(raw.get("totalDebt"))
    ni = _pos(raw.get("netIncome"))

    m["price"] = price
    m["marketCap"] = mcap
    m["freeCashflow"] = fcf
    m["totalRevenue"] = rev

    # Cash-flow & valuation ratios
    m["fcf_yield"] = _ratio(fcf, mcap)
    m["p_fcf"] = _ratio(mcap, fcf) if (fcf or 0) > 0 else None
    m["fcf_margin"] = _ratio(fcf, rev)
    ev = (mcap + (debt or 0) - (cash or 0)) if mcap is not None else None
    m["enterpriseValue"] = ev
    m["ev_sales"] = _pos(raw.get("enterpriseToRevenue"))
    if m["ev_sales"] is None:
        m["ev_sales"] = _ratio(ev, rev)
    m["enterpriseToEbitda"] = _pos(raw.get("enterpriseToEbitda"))
    if m["enterpriseToEbitda"] is None:
        m["enterpriseToEbitda"] = _ratio(ev, raw.get("ebitda"))
    m["priceToSalesTrailing12Months"] = _pos(raw.get("priceToSalesTrailing12Months"))
    if m["priceToSalesTrailing12Months"] is None:
        m["priceToSalesTrailing12Months"] = _ratio(mcap, rev)
    # `from_source` records which of the currency-fragile multiples the caller supplied
    # whole versus which we rebuilt from a market cap; only the rebuilt ones are masked
    # for a non-USD reporter (see _CCY_FALLBACK_ONLY).
    from_source: set[str] = set()
    m["trailingPE"] = _pos(raw.get("trailingPE"))
    if m["trailingPE"] is not None:
        from_source.add("trailingPE")
    elif ni is not None and ni > 0:
        m["trailingPE"] = _ratio(mcap, ni)
    m["forwardPE"] = _pos(raw.get("forwardPE"))          # live-only (analyst estimates)
    m["pegRatio"] = _pos(raw.get("pegRatio"))            # live-only
    m["net_cash"] = (cash - debt) if (cash is not None and debt is not None) else None
    # Book value: the valuation multiple for banks and insurers, whose EV/EBITDA does
    # not exist (interest IS the revenue) and whose P/S is meaningless. Without it a
    # whole sector could only ever be ranked on P/E.
    equity = _pos(raw.get("totalStockholderEquity"))
    m["priceToBook"] = _pos(raw.get("priceToBook"))
    if m["priceToBook"] is not None:
        from_source.add("priceToBook")
    elif (equity or 0) > 0:
        m["priceToBook"] = _ratio(mcap, equity)
    # FFO: the REIT earnings measure. Depreciation on buildings is the single largest
    # charge in a REIT's P&L and is largely fictional, so P/E and every margin built on
    # net income mis-rank the sector; P/FFO is what the sector is actually priced on.
    # For the same reason FFO — not free cash flow — is a REIT's cash-flow measure:
    # development capex makes fcf_yield negative for a healthy REIT by construction.
    dna = _pos(raw.get("dna"))
    m["ffo"] = (ni + dna) if (ni is not None and dna is not None) else None
    m["p_ffo"] = _ratio(mcap, m["ffo"]) if (m["ffo"] or 0) > 0 else None
    m["ffo_yield"] = _ratio(m["ffo"], mcap)
    m["ffo_margin"] = _ratio(m["ffo"], rev)
    # Quarters of operating cash burn the balance sheet covers. None for a cash
    # GENERATOR (cfo >= 0): "infinite runway" is not a number and must not rank.
    cfo = _pos(raw.get("cfo"))
    if cfo is None:
        cfo = _pos(raw.get("operatingCashflow"))
    m["cash_runway_quarters"] = (_ratio(cash, max(-cfo, 1e-9) / 4.0)
                                 if (cash is not None and cfo is not None and cfo < 0)
                                 else None)

    # Growth
    m["revenueGrowth"] = _pos(raw.get("revenueGrowth"))
    m["earningsGrowth"] = _pos(raw.get("earningsGrowth"))
    growth = m["revenueGrowth"]
    m["rule_of_40"] = (growth * 100 + m["fcf_margin"] * 100
                       if growth is not None and m["fcf_margin"] is not None else None)

    # Quality
    m["grossMargins"] = _pos(raw.get("grossMargins"))
    if m["grossMargins"] is None:
        m["grossMargins"] = _ratio(raw.get("grossProfits"), rev)
    m["operatingMargins"] = _pos(raw.get("operatingMargins"))
    if m["operatingMargins"] is None:
        m["operatingMargins"] = _ratio(raw.get("operatingIncome"), rev)
    m["returnOnEquity"] = _pos(raw.get("returnOnEquity"))
    if m["returnOnEquity"] is None:
        m["returnOnEquity"] = _ratio(ni, raw.get("totalStockholderEquity"))
    m["returnOnAssets"] = _pos(raw.get("returnOnAssets"))
    if m["returnOnAssets"] is None:
        m["returnOnAssets"] = _ratio(ni, raw.get("totalAssets"))
    m["debtToEquity"] = _pos(raw.get("debtToEquity"))
    if m["debtToEquity"] is None:
        r = _ratio(debt, raw.get("totalStockholderEquity"))
        m["debtToEquity"] = r * 100 if r is not None else None   # yfinance reports percent

    # Momentum vs moving averages (price-based; the historical builder supplies the MAs)
    d50, d200 = _pos(raw.get("fiftyDayAverage")), _pos(raw.get("twoHundredDayAverage"))
    m["mom_50"] = _ratio(price, d50)
    m["mom_200"] = _ratio(price, d200)

    eps = _pos(raw.get("trailingEps"))
    m["profitable"] = bool((fcf or 0) > 0 or (eps or 0) > 0 or (ni or 0) > 0)

    # ADR masking: USD market cap over local-currency statements is not a ratio.
    fin_ccy = raw.get("financialCurrency")
    m["financial_currency"] = fin_ccy
    if fin_ccy and fin_ccy != "USD":
        for k in _CCY_SENSITIVE:
            if k in _CCY_FALLBACK_ONLY and k in from_source:
                continue
            m[k] = None

    # Next earnings date (live-only; historical snapshots have none)
    et = raw.get("earningsTimestamp") or raw.get("earningsTimestampStart")
    if et:
        try:
            ed = datetime.fromtimestamp(int(et), tz=timezone.utc).date()
            m["earnings_date"] = ed.isoformat()
            m["days_to_earnings"] = (ed - (as_of or date.today())).days
        except (ValueError, OSError, OverflowError, TypeError):
            pass
    return m
