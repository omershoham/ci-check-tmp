"""Build the ticker universe: the curated AI-infra list, or a broad SEC-derived one.

Two sources, chosen by `universe.source` in config:

  curated  The per-sub-sector lists in the active THEME PACK (themes/*.yaml) are the
           single source of truth — transparent and stable. (ETF-holdings augmentation
           was considered but dropped: issuer CSV scraping is too fragile to rely on.)
           A `universe.sectors:` block in config.yaml still overrides the pack. The
           shipped default.

  broad    Every US-listed SEC registrant that files financial statements
           (`data/registrants.py`), narrowed to what is actually investable on a given
           date by `investable()`.

WHY THE BROAD PATH EXISTS. The curated list was picked in June 2026 knowing which
names had worked, so every backtest run on it is upward-biased and no amount of care
downstream removes that. It is also 105 names: at ~12 independent dates a year,
detecting a true rank IC of 0.03 at |t|>=2 takes five years, which is why `pillar_ic`
keeps answering "not enough data". The broad path fixes both at once.

THE FLOOR IS THE POINT. `investable(as_of)` computes price, dollar volume and market
cap from BULK data only — the local price store and SEC share counts — and applies
`fundamentals.passes_floor_pit` before anything makes a per-name network call. The old
order (fetch `.info` for everyone, then floor) costs 5,000 sequential Yahoo requests to
throw away 4,750 of them. Here the floor runs on parquet in seconds and `.info` is
asked about the survivors.

`max_symbols` is the phasing knob: 250 while the broad path is being proven, then 750,
then 1,500.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Callable

import pandas as pd

from . import config, fundamentals, theme

# Friendly sub-sector labels for display, from the active theme pack. Resolved at
# import so the dashboard can keep importing it as a constant; `subsector_labels()`
# re-reads the pack for callers that switch themes inside one process (the tests do).
SUBSECTOR_LABELS = theme.subsector_labels()


def subsector_labels() -> dict[str, str]:
    return theme.subsector_labels()

_BROAD_DEFAULTS = {
    "max_symbols": 250,
    "rank_by": "adv",
    "min_history_days": 200,
    "adv_window": 20,
    "screen_price_days": 400,
    "recency_trading_days": 10,
}


def _log_noop(msg: str) -> None:
    pass


def configured_source() -> str:
    """`universe.source` from config: 'curated' (shipped default) or 'broad'."""
    return str(config.get("universe.source", "curated") or "curated").lower()


def broad_cfg(key: str) -> Any:
    cfg = config.get("universe.broad", {}) or {}
    return cfg.get(key, _BROAD_DEFAULTS.get(key))


def _curated() -> list[dict[str, str]]:
    """Return [{symbol, subsector}] deduped, first sub-sector wins on overlap.

    The lists come from the theme pack (or a `universe.sectors:` override), so the
    universe and the prose that describes it can never drift apart."""
    sectors = theme.curated_sectors()
    seen: dict[str, str] = {}
    for subsector, symbols in sectors.items():
        for sym in symbols or []:
            sym = str(sym).strip().upper()
            if sym and sym not in seen:
                seen[sym] = subsector
    return [{"symbol": s, "subsector": ss} for s, ss in seen.items()]


def _broad_candidates(store=None, log: Callable[[str], None] = _log_noop) -> list[dict[str, str]]:
    """The registrant table as universe rows — CANDIDATES, before the floor.

    `subsector` carries the industry group so every existing consumer (tickers table,
    `scoring._pct(groups=)`, `regime.subsector_breadth`, `ml.features.sector_rel_mom`,
    the dashboard filters) keeps working on a field that still means "peer group"."""
    from .data.history_store import HistoryStore
    from .data import registrants
    reg = registrants.load_registrants(store or HistoryStore())
    if reg.empty:
        log("universe.source=broad but no registrants table — run "
            "`python -m scripts.data_refresh --registrants`")
        return []
    return [{"symbol": r.symbol, "subsector": r.industry_group, "sector": r.sector,
             "industry_group": r.industry_group, "name": r.name}
            for r in reg.itertuples(index=False)]


def build_universe(source: str | None = None) -> list[dict[str, str]]:
    """Universe rows for the configured source. In broad mode these are CANDIDATES —
    `investable(as_of)` is what applies the point-in-time floor."""
    src = (source or configured_source()).lower()
    if src == "broad":
        return _broad_candidates()
    return _curated()


def symbols() -> list[str]:
    return [r["symbol"] for r in build_universe()]


def subsector_map() -> dict[str, str]:
    return {r["symbol"]: r["subsector"] for r in build_universe()}


def sector_map(rows: list[dict[str, Any]] | None = None, store=None,
               symbols: list[str] | None = None) -> dict[str, str]:
    """symbol -> GICS-like sector, the peer set VALUATION is ranked inside.

    The broad path already carries it on each row; the curated path does not, but
    every curated name is an SEC registrant too, so the same SIC-derived table serves
    both. A name the table does not know keeps "Unknown" — its own group, never
    silently folded into a real sector's peer comparison."""
    from .data import registrants
    from .data.history_store import HistoryStore
    out: dict[str, str] = {}
    if rows:
        out = {r["symbol"]: r["sector"] for r in rows if r.get("sector")}
        symbols = symbols or [r["symbol"] for r in rows]
    symbols = symbols or [r["symbol"] for r in build_universe()]
    missing = [s for s in symbols if not out.get(s)]
    if missing:
        try:
            reg = registrants.load_registrants(store or HistoryStore())
        except Exception:  # noqa: BLE001 - no registrant table is not a run-stopper
            reg = pd.DataFrame()
        if not reg.empty:
            m = dict(zip(reg["symbol"], reg["sector"]))
            for s in missing:
                if m.get(s):
                    out[s] = m[s]
    return {s: out.get(s) or "Unknown" for s in symbols}


# --------------------------------------------------------------- the PIT floor
def price_liquidity(store, symbols: list[str], as_of: date) -> pd.DataFrame:
    """Per symbol: closes in the screen window, last close date, last close, and mean
    dollar volume over the trailing `adv_window` sessions. Indexed by symbol.

    ONE definition of the price half of the floor. `investable()` uses it for the
    broad path; the curated path needs the same numbers now that stage 1 reads
    fundamentals from the store and no longer receives `.info`'s averageVolume."""
    ts = pd.Timestamp(as_of).normalize()
    window = int(broad_cfg("screen_price_days"))
    adv_win = int(broad_cfg("adv_window"))
    px = store.prices(symbols, start=(ts - pd.Timedelta(days=window)).date(),
                      end=ts.date(), columns=["close", "volume"])
    if px.empty:
        return pd.DataFrame(columns=["n_obs", "last_date", "price", "adv"])
    # store.prices() already returns (symbol, date)-sorted rows, so `last`/`tail` mean
    # "most recent" without a second sort of a multi-million-row frame.
    g = px.groupby("symbol", sort=False)
    tail = g.tail(adv_win)
    out = pd.DataFrame({
        "n_obs": g["date"].size(),
        "last_date": g["date"].max(),
        "price": g["close"].last(),
        "adv": tail.assign(_dv=tail["close"] * tail["volume"]).groupby("symbol")["_dv"].mean(),
    })
    out.attrs["sessions"] = pd.Index(px["date"].unique()).sort_values()
    return out


def investable(as_of: date, store=None, candidates: list[str] | None = None,
               max_symbols: int | None = None,
               log: Callable[[str], None] = _log_noop) -> list[dict[str, Any]]:
    """The names that were investable on `as_of`, ranked by dollar volume.

    Point-in-time by construction and computed entirely from bulk data:
      price       last raw close on or before `as_of`
      adv         mean(close * volume) over the trailing `adv_window` sessions
      shares      `store.shares_asof` — filed on or before `as_of`, period ended
                  on or before it
      marketCap   price * shares

    Two things here are easy to get wrong and both are load-bearing:

    * MULTI-CLASS ISSUERS. GOOG and GOOGL are one CIK. Market cap is summed at the CIK
      level and attributed to the class that actually trades (highest ADV); the other
      class is dropped, because it is the same company and ranking it twice both
      double-counts the issuer and understates its size by ~50%.
    * RECENCY. A name whose last close predates `as_of` by more than
      `recency_trading_days` sessions was delisted or halted; carrying it forward is
      exactly the survivorship error this universe exists to remove.
    """
    from .data.history_store import HistoryStore
    from .data import registrants
    store = store or HistoryStore()
    reg = registrants.load_registrants(store)
    if reg.empty:
        return []
    if candidates is not None:
        reg = reg[reg["symbol"].isin(set(candidates))]
    if reg.empty:
        return []

    ts = pd.Timestamp(as_of).normalize()
    min_hist = int(broad_cfg("min_history_days"))
    recency = int(broad_cfg("recency_trading_days"))
    syms = reg["symbol"].tolist()

    stat = price_liquidity(store, syms, as_of)
    if stat.empty:
        log(f"  investable({ts.date()}): no prices for {len(syms)} candidates")
        return []
    sessions = stat.attrs["sessions"]
    cutoff = sessions[-min(recency, len(sessions))]

    meta = reg.set_index("symbol")
    stat = stat.join(meta[["cik", "name", "sector", "industry_group"]], how="inner")

    shares = store.shares_asof(syms, as_of)
    if shares.empty:
        log(f"  investable({ts.date()}): no SEC share counts in the store")
        return []
    sh = shares.to_frame("shares").join(meta["cik"], how="inner")
    by_cik = sh.groupby("cik")["shares"].sum()
    stat["shares"] = stat["cik"].map(by_cik)
    stat["marketCap"] = stat["price"] * stat["shares"]

    n0 = len(stat)
    stat = stat[(stat["n_obs"] >= min_hist) & (stat["last_date"] >= cutoff)]
    log(f"  ladder: {len(stat)}/{n0} with >={min_hist} closes ending within "
        f"{recency} sessions of {ts.date()}")

    # ONE definition of the floor, applied row by row. The frame is thousands of rows,
    # not millions, and a vectorised copy of the rule is a second implementation that
    # would have to be kept in step with the live screen's forever.
    keep = [fundamentals.passes_floor_pit(p, m, a) for p, m, a
            in zip(stat["price"], stat["marketCap"], stat["adv"])]
    # dtype=bool is load-bearing: an EMPTY list becomes a float64 Series, which pandas
    # then reads as a list of column LABELS rather than a row mask, and the frame comes
    # back with no columns at all. Early month-ends of a backfill are exactly that case.
    stat = stat[pd.Series(keep, index=stat.index, dtype=bool)]
    log(f"  ladder: {len(stat)} passed the PIT liquidity floor "
        f"({fundamentals.floor_thresholds()})")

    if stat.empty:
        return []
    rank_by = str(broad_cfg("rank_by") or "adv")
    stat = stat.sort_values(rank_by if rank_by in stat.columns else "adv", ascending=False)
    stat = stat.drop_duplicates("cik", keep="first")
    cap = int(max_symbols if max_symbols is not None else (broad_cfg("max_symbols") or 0))
    if cap:
        stat = stat.head(cap)
    log(f"  ladder: {len(stat)} investable after de-duping share classes and capping "
        f"at {cap or 'none'}")

    return [{"symbol": s, "subsector": r.industry_group, "sector": r.sector,
             "industry_group": r.industry_group, "name": r.name,
             "price": float(r.price), "adv": float(r.adv),
             "marketCap": float(r.marketCap)}
            for s, r in zip(stat.index, stat.itertuples(index=False))]


def current_universe(as_of: date | None = None, store=None,
                     log: Callable[[str], None] = _log_noop) -> list[dict[str, Any]]:
    """What the pipeline actually scores: the curated list, or today's floored broad
    universe. Callers that must not download thousands of candidate price histories
    (the regime CLI, the backtest driver) want THIS, not `build_universe()`."""
    if configured_source() != "broad":
        return _curated()
    return investable(as_of or date.today(), store=store, log=log)


def refresh_screen_prices(store, candidates: list[str] | None = None,
                          days: int | None = None,
                          log: Callable[[str], None] = _log_noop) -> dict[str, Any]:
    """Top up the ~400 trailing days of closes the PIT floor needs for every candidate.

    This is the one genuinely expensive step of the broad path (~125 chunks), so it is
    a separate, resumable call rather than something a pipeline run does inline."""
    from .data import yfinance_live
    days = int(days or broad_cfg("screen_price_days"))
    syms = candidates if candidates is not None else [r["symbol"] for r in build_universe()]
    start = pd.Timestamp(date.today()) - pd.Timedelta(days=days)
    log(f"screen prices: {len(syms)} candidates, {days}d window from {start.date()}")
    df = yfinance_live.download_daily(syms, start=start.date(), log=log)
    stats = {"symbols": len(syms), "rows": len(df),
             "downloaded": int(df["symbol"].nunique()) if not df.empty else 0,
             **{k: v for k, v in df.attrs.items()}}
    if not df.empty:
        stats["store_rows"] = store.upsert_prices(df)
        from .data import registrants
        stats["with_prices"] = registrants.update_price_coverage(store)
    return stats


if __name__ == "__main__":
    u = build_universe()
    by_sub: dict[str, int] = {}
    for r in u:
        by_sub[r["subsector"]] = by_sub.get(r["subsector"], 0) + 1
    labels = subsector_labels()
    print(f"Universe ({configured_source()}, theme {theme.configured_name()}): "
          f"{len(u)} tickers")
    for ss, n in sorted(by_sub.items(), key=lambda kv: -kv[1]):
        print(f"  {labels.get(ss, ss)}: {n}")
