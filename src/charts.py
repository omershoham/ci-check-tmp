"""Helpers for the dashboard: OHLC history, indicator series, trade levels,
earnings dates. Kept separate from technical.py (which returns scores) because
the chart needs full series, not just the latest value.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf


def price_frame(symbol: str, period: str = "1y") -> pd.DataFrame | None:
    try:
        df = yf.Ticker(symbol).history(period=period)
        return df if df is not None and not df.empty else None
    except Exception:  # noqa: BLE001
        return None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    df["SMA20"] = c.rolling(20).mean()
    df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    delta = c.diff()
    # Wilder's RSI (matches src/technical.py and standard charting platforms).
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = (100 - (100 / (1 + rs))).where(avg_loss != 0, 100.0)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACDsig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACDhist"] = df["MACD"] - df["MACDsig"]
    return df


def levels(df: pd.DataFrame) -> dict[str, Any]:
    """Entry/stop/support/resistance from price action."""
    c = df["Close"].dropna()
    if c.empty:
        return {}
    price = float(c.iloc[-1])
    hi, lo, prev = df["High"], df["Low"], c.shift(1)
    tr = pd.concat([hi - lo, (hi - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    w = c.tail(252)
    hi52, lo52 = float(w.max()), float(w.min())
    sma50 = float(c.tail(50).mean()) if len(c) >= 50 else None
    sma200 = float(c.tail(200).mean()) if len(c) >= 200 else None
    return {
        "price": price,
        "atr": atr,
        "atr_pct": atr / price if price else None,
        "stop_2atr": price - 2 * atr,
        "support_sma50": sma50,
        "support_sma200": sma200,
        "high_52w": hi52,
        "low_52w": lo52,
        "dist_to_high": (price / hi52 - 1) if hi52 else None,
    }


def earnings_date(symbol: str) -> str | None:
    """Next earnings date as YYYY-MM-DD, best-effort."""
    try:
        cal = yf.Ticker(symbol).calendar
    except Exception:  # noqa: BLE001
        return None
    try:
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if isinstance(ed, (list, tuple)) and ed:
                return str(ed[0])
            if ed:
                return str(ed)
        elif cal is not None and hasattr(cal, "loc") and "Earnings Date" in getattr(cal, "index", []):
            return str(cal.loc["Earnings Date"][0])
    except Exception:  # noqa: BLE001
        return None
    return None
