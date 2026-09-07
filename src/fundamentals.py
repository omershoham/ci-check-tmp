"""Fetch fundamentals/price metrics via yfinance, with on-disk daily caching.

yfinance is unofficial and rate-limited, so we cache one JSON per symbol per
calendar day and throttle between calls. Missing fields degrade to None.
"""
from __future__ import annotations

import json
import time
from datetime import date
from typing import Any

import yfinance as yf

from . import config
from .analytics.metrics import derive

# Fields we pull from yfinance .info (best-effort; many may be absent).
_INFO_FIELDS = [
    "shortName", "longName", "marketCap", "currentPrice", "regularMarketPrice",
    "averageVolume", "averageDailyVolume10Day",
    "trailingPE", "forwardPE", "pegRatio", "enterpriseToEbitda", "enterpriseToRevenue",
    "enterpriseValue", "priceToSalesTrailing12Months", "priceToBook",
    "profitMargins", "grossMargins", "operatingMargins",
    "returnOnEquity", "returnOnAssets", "debtToEquity",
    "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
    "freeCashflow", "operatingCashflow", "ebitda", "totalRevenue",
    "totalCash", "totalDebt", "financialCurrency",
    "fiftyDayAverage", "twoHundredDayAverage", "trailingEps",
    "earningsTimestamp", "earningsTimestampStart",
]


def _cache_file(symbol: str) -> Any:
    return config.cache_dir() / f"fund_{symbol}_{date.today().isoformat()}.json"


def _price(info: dict) -> float | None:
    return info.get("currentPrice") or info.get("regularMarketPrice")


def _avg_volume(info: dict) -> float | None:
    return info.get("averageVolume") or info.get("averageDailyVolume10Day")


def fetch_one(symbol: str, use_cache: bool = True) -> dict[str, Any]:
    """Return a metrics dict for one symbol. Always returns a dict (may be sparse)."""
    cf = _cache_file(symbol)
    if use_cache and cf.exists():
        try:
            return json.loads(cf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    metrics: dict[str, Any] = {"symbol": symbol}
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as exc:  # noqa: BLE001 - yfinance raises many types
        metrics["error"] = str(exc)[:200]
        return metrics

    for f in _INFO_FIELDS:
        if f in info:
            metrics[f] = info[f]

    price = _price(info)
    metrics["name"] = info.get("longName") or info.get("shortName") or symbol
    metrics["avg_volume"] = _avg_volume(info)
    metrics["avg_dollar_volume"] = (
        (metrics["avg_volume"] or 0) * price if price and metrics["avg_volume"] else None
    )
    # Every derived ratio comes from the ONE shared definition (analytics.metrics.derive),
    # the same function the historical/backtest path uses on SEC facts. yfinance's
    # `.info` already uses the raw field names derive() expects.
    raw = dict(info)
    raw["price"] = price
    raw.setdefault("netIncome", info.get("netIncomeToCommon"))
    metrics.update(derive(raw))

    try:
        cf.write_text(json.dumps(metrics), encoding="utf-8")
    except OSError:
        pass
    return metrics


def fetch_many(symbols: list[str], throttle: float = 0.4,
               use_cache: bool = True, progress: bool = True) -> dict[str, dict]:
    out: dict[str, dict] = {}
    n = len(symbols)
    for i, sym in enumerate(symbols, 1):
        # Decide BEFORE the call whether this is a network hit: fetch_one writes the
        # cache file on success, so checking existence afterwards always says "cached"
        # and the throttle never fires — hammering Yahoo on cold-cache mornings.
        will_hit_network = not (use_cache and _cache_file(sym).exists())
        m = fetch_one(sym, use_cache=use_cache)
        out[sym] = m
        if progress and (i % 10 == 0 or i == n):
            print(f"  fundamentals {i}/{n}")
        if will_hit_network:
            time.sleep(throttle)
    return out


def passes_floor(m: dict) -> bool:
    """Hard liquidity/size floor from config (Q&A: standard floor)."""
    floor = config.get("universe.liquidity_floor", {}) or {}
    mcap = m.get("marketCap")
    price = m.get("price")
    adv = m.get("avg_dollar_volume")
    if mcap is None or price is None:
        return False  # can't verify → exclude (defensive)
    if mcap < floor.get("min_market_cap", 0):
        return False
    if price < floor.get("min_price", 0):
        return False
    if adv is not None and adv < floor.get("min_avg_dollar_volume", 0):
        return False
    return True


if __name__ == "__main__":
    import sys
    test = sys.argv[1:] or ["NVDA", "VRT", "ANET", "ORCL", "AAOI"]
    data = fetch_many(test, use_cache=True)
    for s, m in data.items():
        print(f"{s:6} price={m.get('price')} mcap={m.get('marketCap')} "
              f"PE={m.get('trailingPE')} P/FCF={m.get('p_fcf')} "
              f"floor={'OK' if passes_floor(m) else 'FAIL'}")
