"""yfinance-backed loaders for the history store (prices) and the live screen.

Prices are the ONE thing yfinance supplies that is legitimately historical: a daily
bar is what it was. Everything else it offers (`.info`) is current-state and must
never be written into the history store.

HANG-PROOFING. At 105 names a throttled chunk costs a retry; at the 5,000 candidates
the broad universe screens it is 125 chunks, and Yahoo throttles the long ones. A
`yf.download` that blocks forever is exactly how runs #46-#52 died (see
`technical._download_with_deadline`), so every chunk here runs in a worker thread with
a hard deadline, retries once un-threaded, and on failure is SKIPPED - never raised.
One bad chunk must cost 40 symbols, not the run.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout
from datetime import date, timedelta
from typing import Callable, Iterable

import pandas as pd
import yfinance as yf

_DEADLINE_S = 120.0        # per chunk, threaded attempt
_RETRY_DEADLINE_S = 90.0   # per chunk, un-threaded retry
_SLEEP_S = 0.5             # between chunks; Yahoo throttles bursts, not volume
# How stale the store may be before a refresh gives up on the incremental window and
# re-downloads the full period. A gap wider than this is a re-seed, not a top-up.
_INCREMENTAL_MAX_DAYS = 400


def _log_noop(msg: str) -> None:
    pass


def _download_once(part: list[str], period: str, start, end, threads: bool,
                   deadline_s: float):
    """yf.download in a worker thread with a hard deadline. Returns None on timeout."""
    kw = dict(interval="1d", group_by="ticker", auto_adjust=False,
              progress=False, threads=threads)
    if start is not None:
        kw["start"] = pd.Timestamp(start).date()
        if end is not None:
            kw["end"] = pd.Timestamp(end).date() + timedelta(days=1)
    else:
        kw["period"] = period
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(yf.download, part, **kw).result(timeout=deadline_s)
    except _FutTimeout:
        return None
    finally:
        ex.shutdown(wait=False)


def _rows(raw, part: list[str]) -> list[pd.DataFrame]:
    frames = []
    for s in part:
        try:
            df = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        df = df.dropna(how="all")
        if df.empty or "Close" not in df:
            continue
        idx = pd.to_datetime(df.index)
        out = pd.DataFrame({
            "date": idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx,
            "symbol": s,
            "open": df.get("Open"), "high": df.get("High"), "low": df.get("Low"),
            "close": df["Close"],
            "adj_close": df["Adj Close"] if "Adj Close" in df else df["Close"],
            "volume": df.get("Volume"),
        })
        frames.append(out.dropna(subset=["close"]))
    return frames


def download_daily(symbols: Iterable[str], period: str = "10y", chunk: int = 40,
                   start: date | None = None, end: date | None = None,
                   log: Callable[[str], None] = _log_noop) -> pd.DataFrame:
    """Long frame (date, symbol, o/h/l/c, adj_close, volume) for `symbols`.

    `start`/`end` select an incremental window; without them the whole `period` is
    fetched. auto_adjust=False so BOTH the raw close (for historical $-floors and
    market cap with as-filed share counts) and the adjusted close (for returns) are
    kept. `df.attrs` carries the chunk tally so callers can report throttling instead
    of silently shipping a short universe.
    """
    syms = list(dict.fromkeys(s.upper() for s in symbols))
    frames: list[pd.DataFrame] = []
    n_chunks = (len(syms) + chunk - 1) // chunk
    failed, retried = 0, 0
    for i in range(0, len(syms), chunk):
        part = syms[i:i + chunk]
        k = i // chunk + 1
        try:
            raw = _download_once(part, period, start, end, True, _DEADLINE_S)
            if raw is None or getattr(raw, "empty", True):
                retried += 1
                log(f"  prices: chunk {k}/{n_chunks} empty/timed out — retrying un-threaded")
                raw = _download_once(part, period, start, end, False, _RETRY_DEADLINE_S)
        except Exception as exc:  # noqa: BLE001 - yfinance raises many types
            log(f"  prices: chunk {k}/{n_chunks} failed: {str(exc)[:120]}")
            raw = None
        if raw is None or getattr(raw, "empty", True):
            failed += 1
            log(f"  prices: chunk {k}/{n_chunks} gave nothing — {len(part)} symbols skipped")
        else:
            frames.extend(_rows(raw, part))
        if k % 10 == 0 or k == n_chunks:
            log(f"  prices: chunk {k}/{n_chunks} ({sum(len(f) for f in frames)} rows)")
        if k < n_chunks:
            time.sleep(_SLEEP_S)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"])
    out.attrs["chunks"] = n_chunks
    out.attrs["chunks_failed"] = failed
    out.attrs["chunks_retried"] = retried
    return out


def refresh_prices(store, symbols: Iterable[str], period: str = "10y",
                   start: date | None = None, end: date | None = None,
                   log: Callable[[str], None] = _log_noop) -> int:
    """Download and merge into the store; returns the price table's total row count.

    When no explicit window is given and EVERY requested symbol already has prices in
    a store whose last bar is recent, only the missing tail is downloaded. A daily
    refresh used to re-fetch ten years for every name - fine at 105, and 125 chunks of
    ten-year history at 5,000."""
    syms = list(dict.fromkeys(s.upper() for s in symbols))
    if start is None and syms:
        last = store.last_date()
        have = set(store.symbols_with_prices())
        gap = (pd.Timestamp(date.today()) - last).days if last is not None else None
        if last is not None and gap is not None and gap <= _INCREMENTAL_MAX_DAYS \
                and set(syms) <= have:
            # Overlap by a few days: `compact_prices` keeps the last write per
            # (symbol, date), so re-fetching the boundary costs nothing and covers a
            # bar that was provisional when it was first stored.
            start = (last - pd.Timedelta(days=5)).date()
            log(f"  prices: incremental window from {start} (store last {last.date()})")
    df = download_daily(symbols, period=period, start=start, end=end, log=log)
    if df.empty:
        return 0
    return store.upsert_prices(df)


def month_ends(start: date, end: date) -> list[pd.Timestamp]:
    """Business month-end dates in [start, end] — the snapshot grid."""
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="BME")
    return [d for d in idx]
