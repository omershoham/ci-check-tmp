"""Technical-analysis pillar: indicators + a 0..1 technical score from price history.

Pure pandas math on free yfinance daily history (batch-downloaded). Produces per
symbol: RSI(14), MACD(12/26/9), SMA 20/50/200 trend + crosses, 52-week position,
volume trend, ATR% volatility, a composite technical_score (0..1), and a plain
state label ("Uptrend · Overbought", etc.).
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from . import config


# Bump the cache version whenever indicator definitions / score weights change, so a
# day's earlier cache (computed under old rules) can't mix with new-rule entries.
_CACHE_VER = "v3"


def _cache_file(symbol: str):
    return config.cache_dir() / f"tech_{_CACHE_VER}_{symbol}_{date.today().isoformat()}.json"


def _rsi(close: pd.Series, period: int = 14) -> float | None:
    """Wilder's RSI (the standard used by charting platforms/brokers).

    Uses Wilder smoothing (ewm alpha=1/period), not a simple rolling mean, so values
    line up with what TradingView/IBKR show — important because the score bins on exact
    RSI thresholds (70/45/35). A window with no down-closes yields RSI 100 (overbought)
    rather than a NaN that would silently fall back to neutral.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)  # all-gains window → fully overbought
    v = rsi.iloc[-1]
    return float(v) if v == v else None


def _macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])


def _atr_pct(df: pd.DataFrame, period: int = 14) -> float | None:
    try:
        hi, lo, cl = df["High"], df["Low"], df["Close"]
        prev = cl.shift(1)
        tr = pd.concat([hi - lo, (hi - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        price = cl.iloc[-1]
        return float(atr / price) if price else None
    except Exception:  # noqa: BLE001
        return None


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _score_and_state(ind: dict[str, Any]) -> tuple[float, str]:
    price = ind.get("price")
    sma20, sma50, sma200 = ind.get("sma20"), ind.get("sma50"), ind.get("sma200")
    rsi = ind.get("rsi")
    macd_hist = ind.get("macd_hist")
    pos52 = ind.get("pos_52w")
    vol_trend = ind.get("vol_trend")

    states: list[str] = []

    # Trend (SMA alignment) — 0..1
    checks, total = 0, 0
    for cond in [price and sma50 and price > sma50,
                 price and sma200 and price > sma200,
                 sma50 and sma200 and sma50 > sma200]:
        total += 1
        checks += 1 if cond else 0
    trend = checks / total if total else 0.5
    if price and sma50 and sma200:
        states.append("Uptrend" if (price > sma50 > sma200) else
                      ("Downtrend" if (price < sma50 < sma200) else "Mixed trend"))

    # RSI — reward the healthy-trend zone; at a weeks-to-months horizon overbought
    # usually means CONTINUATION, not reversal, so the penalty is deliberately mild
    # (a hard fade of RSI>70 would systematically sell the strongest momentum names,
    # fighting the trend/momentum components). Only a blow-off >=80 gets real caution.
    if rsi is None:
        rsi_s = 0.5
    elif rsi >= 80:
        rsi_s = 0.45; states.append("Overbought")
    elif rsi >= 70:
        rsi_s = 0.6; states.append("Overbought")
    elif rsi >= 45:
        rsi_s = 0.9
    elif rsi >= 35:
        rsi_s = 0.6
    elif rsi >= 25:
        rsi_s = 0.45; states.append("Oversold")
    else:
        rsi_s = 0.35; states.append("Oversold")

    # MACD histogram sign
    macd_s = 0.75 if (macd_hist is not None and macd_hist > 0) else 0.35
    if macd_hist is not None:
        states.append("MACD+" if macd_hist > 0 else "MACD-")

    # 52-week position (George-Hwang anchor) — reward upper range, slight penalty if blown off
    if pos52 is None:
        pos_s = 0.5
    elif pos52 >= 0.98:
        pos_s = 0.7   # extended
    else:
        pos_s = _clip01(pos52)

    # 12-1 momentum (Jegadeesh-Titman), volatility-adjusted (Barroso-Santa-Clara):
    # the single best-documented cross-sectional signal at this exact horizon.
    # mom_va ≈ 12-1 return / annualized ATR-vol → squashed to 0..1. Absolute scale is
    # fine here because the whole pillar is percentile-ranked in scoring anyway.
    mom_va = ind.get("mom_va")
    mom_s = 0.5 if mom_va is None else _clip01(0.5 + 0.5 * np.tanh(mom_va))

    # Volume confirmation — SIGNED by recent price direction: rising volume into a rising
    # price confirms strength, but rising volume into a falling price is distribution and
    # should lower (not raise) the score.
    ret20 = ind.get("ret_20")
    if vol_trend is None:
        vol_s = 0.5
    else:
        expansion = np.tanh((vol_trend - 1) * 2)          # -1..1
        direction = 1.0 if (ret20 is None or ret20 >= 0) else -1.0
        vol_s = _clip01(0.5 + 0.5 * expansion * direction)

    # Trend and MACD are collinear (both smoothed price direction), so their combined
    # weight is deliberately modest; momentum carries the weight the literature earns it.
    score = (0.25 * trend + 0.25 * mom_s + 0.10 * rsi_s + 0.10 * macd_s
             + 0.15 * pos_s + 0.15 * vol_s)
    return round(_clip01(score), 4), " · ".join(states) if states else "n/a"


def _detect_cross(sma_fast: pd.Series, sma_slow: pd.Series, lookback: int = 30):
    """Detect a golden/death cross within `lookback` days. Returns (type, days_ago)."""
    diff = (sma_fast - sma_slow).dropna()
    if len(diff) < 2:
        return None, None
    sign = diff.apply(lambda x: 1 if x > 0 else -1)
    recent = sign.tail(lookback + 1).to_list()
    for i in range(len(recent) - 1, 0, -1):
        if recent[i] != recent[i - 1]:
            days_ago = len(recent) - 1 - i
            return ("golden" if recent[i] > 0 else "death"), days_ago
    return None, None


def _ret(close: pd.Series, days: int) -> float | None:
    c = close.dropna()
    if len(c) <= days:
        return None
    return float((c.iloc[-1] / c.iloc[-days - 1] - 1) * 100)


def _classify_setup(ind: dict, cross: tuple) -> tuple[str, str, list[str]]:
    """Rule-based trend/setup label. Returns (direction, label, signals)."""
    price, s50, s200 = ind.get("price"), ind.get("sma50"), ind.get("sma200")
    rsi, hist, pos = ind.get("rsi"), ind.get("macd_hist"), ind.get("pos_52w")
    sigs: list[str] = []
    if price and s50 and s200:
        if price > s50 > s200:
            sigs.append("Above all MAs (stacked up)")
        elif price < s50 < s200:
            sigs.append("Below all MAs (stacked down)")
        elif price > s200:
            sigs.append("Above 200-day")
        else:
            sigs.append("Below 200-day")
    ctype, cdays = cross
    if ctype:
        sigs.append(f"{ctype.title()} cross {cdays}d ago")
    if hist is not None:
        sigs.append("MACD+" if hist > 0 else "MACD-")
    if rsi is not None:
        sigs.append("RSI overbought" if rsi >= 70 else "RSI oversold" if rsi <= 30
                    else f"RSI {rsi:.0f}")
    if pos is not None and pos >= 0.9:
        sigs.append("Near 52w high")

    stack_up = bool(price and s50 and s200 and price > s50 > s200)
    stack_dn = bool(price and s50 and s200 and price < s50 < s200)
    up200 = bool(price and s200 and price > s200)
    if stack_up and (hist or 0) > 0 and (pos or 0) >= 0.6:
        return "bullish", "Strong uptrend", sigs
    if up200 and (s50 or 0) >= (s200 or 0):
        return "bullish", "Uptrend", sigs
    if stack_dn and (hist or 0) < 0 and (rsi or 50) < 45:
        return "bearish", "Strong downtrend", sigs
    if (not up200) and (s50 or 0) < (s200 or 1):
        return "bearish", "Downtrend", sigs
    return "neutral", "Range / mixed", sigs


def _compute_one(df: pd.DataFrame) -> dict[str, Any] | None:
    df = df.dropna(how="all")
    if df is None or df.empty or "Close" not in df or len(df) < 60:
        return None
    close = df["Close"].dropna()
    if close.empty:
        return None
    price = float(close.iloc[-1])
    macd, signal, hist = _macd(close)
    sma50_s = close.rolling(50).mean()
    sma200_s = close.rolling(200).mean()
    window52 = close.tail(252)
    lo52, hi52 = float(window52.min()), float(window52.max())
    pos_52w = (price - lo52) / (hi52 - lo52) if hi52 > lo52 else None
    vol_trend = None
    if "Volume" in df:
        vol = df["Volume"].dropna()
        if len(vol) >= 60:
            recent = vol.tail(20).mean()
            base = vol.iloc[-60:-20].mean()  # prior 40 days, NOT overlapping the recent 20
            vol_trend = float(recent / base) if base else None

    atr_pct = _atr_pct(df)
    # 12-1 momentum: return from t-252 to t-21 (skip the most recent month, which
    # mean-reverts). Requires ~13 months of history — the download is 2y.
    ret_252_21 = None
    c = close.dropna()
    if len(c) > 253:
        base, end = float(c.iloc[-253]), float(c.iloc[-22])
        ret_252_21 = (end / base - 1) * 100 if base else None
    mom_va = None
    if ret_252_21 is not None and atr_pct:
        ann_vol_pct = atr_pct * 100 * np.sqrt(252)  # rough annualized vol from ATR%
        mom_va = ret_252_21 / ann_vol_pct if ann_vol_pct else None

    ind = {
        "price": price,
        "sma20": float(close.tail(20).mean()) if len(close) >= 20 else None,
        "sma50": float(sma50_s.iloc[-1]) if sma50_s.notna().any() else None,
        "sma200": float(sma200_s.iloc[-1]) if sma200_s.notna().any() else None,
        "rsi": _rsi(close),
        "macd": macd, "macd_signal": signal, "macd_hist": hist,
        "pos_52w": pos_52w,
        "vol_trend": vol_trend,
        "atr_pct": atr_pct,
        "ret_20": _ret(close, 20),    # ~1-month return (%), signs the volume score
        "ret_63": _ret(close, 63),    # 3-month return (%)
        "ret_126": _ret(close, 126),  # 6-month return (%)
        "ret_252_21": ret_252_21,     # 12-1 momentum (%)
        "mom_va": round(mom_va, 3) if mom_va is not None else None,
    }
    score, state = _score_and_state(ind)
    cross = _detect_cross(sma50_s, sma200_s)
    direction, label, sigs = _classify_setup(ind, cross)
    ind["technical_score"] = score
    ind["state"] = state
    ind["setup"] = direction
    ind["setup_label"] = label
    ind["signals"] = sigs
    return ind


def _download_with_deadline(syms: list[str], period: str = "2y",
                            deadline_s: float = 150.0):
    """yf.download that CANNOT hang the pipeline.

    Yahoo intermittently throttles the 14:00 burst and the threaded batch download
    can then block forever — runs #46-#52 (Jul 13-16) all died this way, stuck at
    'stage 1/4: technicals' until the Task Scheduler's 2h kill. The download runs
    in a worker thread with a hard deadline; on expiry we retry once in chunked
    non-threaded mode, and otherwise give up gracefully (callers already degrade
    to neutral technicals rather than dying)."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout

    def _dl(symbols_, threads_):
        return yf.download(symbols_, period=period, interval="1d", group_by="ticker",
                           auto_adjust=True, progress=False, threads=threads_)

    ex = ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(_dl, syms, True)
        try:
            return fut.result(timeout=deadline_s)
        except _FutTimeout:
            print(f"[technical] batch download exceeded {deadline_s:.0f}s "
                  f"(Yahoo throttling?) — retrying in chunked mode", flush=True)
        except Exception:  # noqa: BLE001
            pass
    finally:
        ex.shutdown(wait=False)
    # Fallback: small sequential chunks, each with its own deadline.
    frames = []
    for i in range(0, len(syms), 25):
        chunk = syms[i:i + 25]
        ex2 = ThreadPoolExecutor(max_workers=1)
        try:
            fut = ex2.submit(_dl, chunk, False)
            frames.append(fut.result(timeout=90))
        except Exception:  # noqa: BLE001
            print(f"[technical] chunk {i // 25 + 1} failed/timed out — skipping "
                  f"{len(chunk)} names", flush=True)
        finally:
            ex2.shutdown(wait=False)
    if not frames:
        return None
    return pd.concat(frames, axis=1)


def compute_many(symbols: list[str], use_cache: bool = True) -> dict[str, dict]:
    out: dict[str, dict] = {}
    missing: list[str] = []
    for s in symbols:
        cf = _cache_file(s)
        if use_cache and cf.exists():
            try:
                out[s] = json.loads(cf.read_text(encoding="utf-8"))
                continue
            except (json.JSONDecodeError, OSError):
                pass
        missing.append(s)

    if missing:
        # Relative strength vs the THEME (SMH), not the broad market: "leadership"
        # for an AI-infra name means beating its peers, not beating SPY via beta.
        bench = config.get("technical.rs_benchmark", "SMH")
        dl_syms = list(dict.fromkeys(missing + [bench]))
        try:
            # 2y so the 12-1 momentum window (~13 months) is fully covered.
            # Deadline-guarded: a throttled Yahoo batch must degrade, not hang the run.
            data = _download_with_deadline(dl_syms, period="2y")
        except Exception:  # noqa: BLE001
            data = None

        def _frame(sym):
            try:
                if data is not None and isinstance(data.columns, pd.MultiIndex):
                    if sym in data.columns.get_level_values(0):
                        return data[sym]
                elif data is not None:
                    return data
            except Exception:  # noqa: BLE001
                return None
            return None

        # Benchmark returns for relative strength
        bench_frame = _frame(bench)
        b63 = b126 = None
        if bench_frame is not None and "Close" in bench_frame:
            bc = bench_frame["Close"].dropna()
            b63, b126 = _ret(bc, 63), _ret(bc, 126)

        for s in missing:
            computed = None
            try:
                computed = _compute_one(_frame(s))
            except Exception:  # noqa: BLE001
                computed = None
            # A transient download/compute failure must NOT be cached — otherwise the
            # neutral stub is read back for the rest of the day and the whole technical
            # pillar flatlines at 0.5 with no symptom (light run, dashboard, re-runs).
            ind = computed or {"technical_score": 0.5, "state": "n/a", "insufficient_data": True}
            # Relative strength = stock return minus benchmark return (percentage points)
            if ind.get("ret_63") is not None and b63 is not None:
                ind["rs_3m"] = round(ind["ret_63"] - b63, 1)
            if ind.get("ret_126") is not None and b126 is not None:
                ind["rs_6m"] = round(ind["ret_126"] - b126, 1)
            ind["benchmark"] = bench
            out[s] = ind
            if computed is not None:
                try:
                    _cache_file(s).write_text(json.dumps(out[s]), encoding="utf-8")
                except OSError:
                    pass
    return out


def yf_last_prices(symbols: list[str]) -> dict[str, float]:
    """Latest close per symbol via one batched yfinance call (~15-min delayed intraday).

    Fallback quote source when IB Gateway is offline, so P&L math and the deep-dive
    context don't silently run on days-old prices.
    """
    if not symbols:
        return {}
    try:
        from concurrent.futures import ThreadPoolExecutor
        ex = ThreadPoolExecutor(max_workers=1)
        try:
            fut = ex.submit(yf.download, list(dict.fromkeys(symbols)), period="5d",
                            interval="1d", auto_adjust=True, progress=False, threads=True)
            df = fut.result(timeout=60)["Close"]  # deadline: degrade, don't hang
        finally:
            ex.shutdown(wait=False)
        if isinstance(df, pd.Series):
            df = df.to_frame(symbols[0])
        out = {}
        for s in symbols:
            if s in df.columns:
                c = df[s].dropna()
                if not c.empty:
                    out[s] = float(c.iloc[-1])
        return out
    except Exception:  # noqa: BLE001
        return {}


def benchmark_last_price(bench: str | None = None) -> float | None:
    """Latest close of the benchmark (default SPY), for the pick backtest snapshot.

    Cheap best-effort 5-day download; returns None if unavailable.
    """
    bench = bench or config.get("technical.benchmark", "SPY")
    try:
        df = yf.download(bench, period="5d", interval="1d",
                         auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):  # MultiIndex when yfinance groups by ticker
            close = close.iloc[:, 0]
        close = close.dropna()
        return float(close.iloc[-1]) if not close.empty else None
    except Exception:  # noqa: BLE001
        return None


def scores(symbols: list[str], use_cache: bool = True) -> dict[str, float]:
    return {s: (v.get("technical_score", 0.5)) for s, v in compute_many(symbols, use_cache).items()}


if __name__ == "__main__":
    import sys
    syms = sys.argv[1:] or ["NVDA", "VRT", "ANET", "INTC", "WOLF"]
    res = compute_many(syms)
    for s, v in res.items():
        print(f"{s:6} score={v.get('technical_score')} state={v.get('state'):<28} "
              f"RSI={v.get('rsi') and round(v['rsi'],1)} pos52={v.get('pos_52w') and round(v['pos_52w'],2)}")
