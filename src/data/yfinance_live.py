"""yfinance-backed loaders for the history store (prices) and the live screen.

Prices are the ONE thing yfinance supplies that is legitimately historical: a daily
bar is what it was. Everything else it offers (`.info`) is current-state and must
never be written into the history store.
"""
from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd
import yfinance as yf


def download_daily(symbols: Iterable[str], period: str = "10y",
                   chunk: int = 40) -> pd.DataFrame:
    """Long frame (date, symbol, o/h/l/c, adj_close, volume) for `symbols`.

    auto_adjust=False so BOTH the raw close (for historical $-floors and market cap
    with as-filed share counts) and the adjusted close (for returns) are kept.
    Downloaded in chunks so one throttled name doesn't sink the batch."""
    syms = list(dict.fromkeys(s.upper() for s in symbols))
    frames = []
    for i in range(0, len(syms), chunk):
        part = syms[i:i + chunk]
        try:
            raw = yf.download(part, period=period, interval="1d", group_by="ticker",
                              auto_adjust=False, progress=False, threads=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[yfinance_live] chunk {i // chunk + 1} failed: {exc}", flush=True)
            continue
        if raw is None or raw.empty:
            continue
        for s in part:
            try:
                df = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            df = df.dropna(how="all")
            if df.empty or "Close" not in df:
                continue
            out = pd.DataFrame({
                "date": pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) is not None else pd.to_datetime(df.index),
                "symbol": s,
                "open": df.get("Open"), "high": df.get("High"), "low": df.get("Low"),
                "close": df["Close"],
                "adj_close": df["Adj Close"] if "Adj Close" in df else df["Close"],
                "volume": df.get("Volume"),
            })
            frames.append(out.dropna(subset=["close"]))
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close",
                                     "adj_close", "volume"])
    return pd.concat(frames, ignore_index=True)


def refresh_prices(store, symbols: Iterable[str], period: str = "10y") -> int:
    df = download_daily(symbols, period=period)
    if df.empty:
        return 0
    return store.upsert_prices(df)


def month_ends(start: date, end: date) -> list[pd.Timestamp]:
    """Business month-end dates in [start, end] — the snapshot grid."""
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="BME")
    return [d for d in idx]
