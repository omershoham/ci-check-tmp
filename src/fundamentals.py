"""Fetch fundamentals/price metrics via yfinance, with on-disk daily caching.

yfinance is unofficial and rate-limited, so we cache one JSON per symbol per
calendar day and throttle between calls. Missing fields degrade to None.
"""
from __future__ import annotations

import json
import time
from datetime import date
from typing import Any

import pandas as pd
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


def from_store(store, symbols: list[str], as_of: date | None = None,
               extra: dict[str, dict] | None = None) -> dict[str, dict]:
    """The screen's metrics for `symbols`, read from the point-in-time SEC store.

    THIS REPLACES `fetch_many` AS THE SCREEN'S INPUT. The old stage 1 made one
    throttled `.info` request per name (5,000 sequential requests to score a broad
    universe); the snapshots here are a parquet read, and they are the SAME rows the
    backtest scores, so the live screen and the composite backtest finally consume
    identical inputs instead of two sources that merely resemble each other.

    Price and dollar volume come from the local price store, market cap from the
    snapshot (price x as-filed shares), the next earnings date from filing cadence
    (`sec_facts.next_earnings_estimate`). `extra` overrides per symbol - that is how
    `enrich()`'s forward P/E and exact earnings timestamp get in for the shortlist.

    Always returns one dict per symbol; a name the store cannot serve comes back with
    `error` set, so the floor drops it instead of it vanishing without a trace."""
    from . import universe
    from .data import sec_facts
    d = as_of or date.today()
    snap = store.fundamentals_asof(symbols, d)
    liq = universe.price_liquidity(store, symbols, d)
    try:
        earn = sec_facts.next_earnings_estimate(store, symbols, d)
    except Exception:  # noqa: BLE001 - an estimate is a nicety, never a reason to fail
        earn = pd.Series(dtype="datetime64[ns]")
    snap_rows = snap.to_dict("index") if not snap.empty else {}
    liq_rows = liq.to_dict("index") if not liq.empty else {}
    out: dict[str, dict] = {}
    for sym in symbols:
        m: dict[str, Any] = {"symbol": sym, "name": sym}
        row = snap_rows.get(sym)
        if row:
            m.update({k: (None if v != v else v) for k, v in row.items()
                      if k not in ("as_of", "symbol")})
        else:
            m["error"] = "no SEC snapshot within the freshness window"
        lq = liq_rows.get(sym) or {}
        if lq.get("price") == lq.get("price") and lq.get("price"):
            m["price"] = float(lq["price"])
        m["avg_dollar_volume"] = (float(lq["adv"]) if lq.get("adv") == lq.get("adv")
                                  and lq.get("adv") is not None else None)
        # Market cap must be priced off the SAME close the floor is: shares are as
        # filed, the price moved since the snapshot was written.
        if m.get("sharesOutstanding") and m.get("price"):
            m["marketCap"] = float(m["price"]) * float(m["sharesOutstanding"])
        nxt = earn.get(sym) if len(earn) else None
        if nxt is not None and nxt == nxt:
            m["earnings_date"] = pd.Timestamp(nxt).date().isoformat()
            m["days_to_earnings"] = (pd.Timestamp(nxt).date() - d).days
        if extra and sym in extra:
            m.update({k: v for k, v in extra[sym].items() if k != "error"})
        out[sym] = m
    return out


# The only fields `.info` still exists for. Everything else the screen ranks on now
# comes from the local SEC store (`HistoryStore.fundamentals_asof` -> `derive`), which
# is a parquet read rather than one throttled Yahoo request per name.
# `trailingPE`/`priceToBook` are here for FOREIGN filers only, but are carried for
# everyone because it costs nothing: SEC facts give a 20-F filer a USD market cap over
# a local-currency statement, so `derive()` masks the rebuilt multiple to None. Yahoo
# quotes the ADR against USD-converted per-share figures, so its number is the one
# correct multiple such a name has. For a domestic filer the two agree to ~4%.
_ENRICH_FIELDS = ("forwardPE", "pegRatio", "trailingPE", "priceToBook",
                  "earningsTimestamp", "earningsTimestampStart",
                  "earnings_date", "days_to_earnings", "name")


def enrich(symbols: list[str], throttle: float = 0.4,
           progress: bool = True) -> dict[str, dict]:
    """`.info` for a BOUNDED set of names — the shortlist plus what you hold (~40-60).

    This is the entire remaining surface of the per-symbol Yahoo path. It exists for
    the two things SEC filings genuinely do not contain — forward P/E and the analyst
    PEG, both estimates — plus the exact next-earnings timestamp and the company's
    display name. Everything else is served from the store, which is why the universe
    can grow 15x without the fundamentals stage growing at all.

    Returns symbol -> only the enrichable keys (plus `error` when the call failed), so
    a caller can merge it over store-derived metrics without a `.info` field silently
    overwriting a point-in-time one."""
    out: dict[str, dict] = {}
    fetched = fetch_many(symbols, throttle=throttle, use_cache=True, progress=progress)
    for sym, m in fetched.items():
        keep = {k: m[k] for k in _ENRICH_FIELDS if m.get(k) is not None}
        if m.get("error"):
            keep["error"] = m["error"]
        out[sym] = keep
    return out


def fill_missing_market_caps(metrics: dict[str, dict], max_names: int = 25,
                             fetch=None) -> list[str]:
    """Yahoo's live market cap for names the SEC store could not size. Returns them.

    For a HAND-PICKED list only. Some filers publish no usable undimensioned share
    count at all -- Visa has tagged its cover-page count per share class since 2010,
    and companyfacts drops dimensioned facts -- so the store has no market cap, the
    floor cannot be verified, and a name the user chose on purpose vanishes. The live
    screen can ask Yahoo; the backtest never does (this is not point-in-time), and the
    broad universe keeps failing such names because there the PIT floor is the point.

    Bounded: more than `max_names` unsized names means the store is thin, which a
    Yahoo loop would hide rather than fix."""
    todo = [s for s, m in metrics.items()
            if m.get("price") and not m.get("marketCap")]
    if not todo or len(todo) > max_names:
        return []
    got = (fetch or fetch_many)(todo, use_cache=True, progress=False)
    filled = []
    for sym in todo:
        cap = (got.get(sym) or {}).get("marketCap")
        if cap and cap == cap and float(cap) > 0:
            metrics[sym]["marketCap"] = float(cap)
            metrics[sym]["marketCap_source"] = "yahoo"
            filled.append(sym)
    return filled


def floor_thresholds() -> dict[str, float]:
    """The three config numbers the floor is made of, in one place."""
    floor = config.get("universe.liquidity_floor", {}) or {}
    return {"min_market_cap": float(floor.get("min_market_cap", 0) or 0),
            "min_price": float(floor.get("min_price", 0) or 0),
            "min_avg_dollar_volume": float(floor.get("min_avg_dollar_volume", 0) or 0)}


def passes_floor_pit(price, mcap, adv) -> bool:
    """THE hard liquidity/size floor (Q&A: standard floor), on bare numbers.

    Taking numbers rather than a `.info` dict is what lets the broad universe apply the
    SAME floor from bulk data - last raw close, mean close*volume, SEC share count -
    with zero per-name network calls, instead of a second copy of the rule that has to
    be kept in step with this one.

    Missing price or market cap FAILS (the floor could not be verified); a missing ADV
    passes, because absence of a volume series is not evidence of illiquidity."""
    f = floor_thresholds()
    if pd.isna(mcap) or pd.isna(price):
        return False
    if float(mcap) < f["min_market_cap"] or float(price) < f["min_price"]:
        return False
    return bool(pd.isna(adv) or float(adv) >= f["min_avg_dollar_volume"])


def passes_floor(m: dict) -> bool:
    """Floor for a `.info` metrics dict — a field-name shim over `passes_floor_pit`."""
    return passes_floor_pit(m.get("price"), m.get("marketCap"), m.get("avg_dollar_volume"))


if __name__ == "__main__":
    import sys
    test = sys.argv[1:] or ["NVDA", "VRT", "ANET", "ORCL", "AAOI"]
    data = fetch_many(test, use_cache=True)
    for s, m in data.items():
        print(f"{s:6} price={m.get('price')} mcap={m.get('marketCap')} "
              f"PE={m.get('trailingPE')} P/FCF={m.get('p_fcf')} "
              f"floor={'OK' if passes_floor(m) else 'FAIL'}")
