"""Point-in-time feature frame for the learned pillar.

Every feature is computable from the history store alone (prices + SEC snapshots +
regime recomputed from prices). No `.info`, no `metrics_json`, no news — trap #1 in
the plan is enforced structurally: this module imports only the store.

Rows = (as_of, symbol). Columns = FEATURES + LABELS + ids. Features are stored RAW;
`rank_transform` (used at train and inference time) turns each into its cross-sectional
percentile per date so level drift never reaches the model.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .. import parallel, technical

FEATURE_VERSION = 1

TECH_FEATURES = ["rsi", "macd_hist", "atr_pct", "pos_52w", "vol_trend", "ret_20", "ret_63",
                 "ret_126", "ret_252_21", "mom_va", "px_sma50", "px_sma200", "sma50_200",
                 "technical_score", "vol_20", "vol_60", "mdd_63", "dollar_vol", "rs_3m", "rs_6m",
                 "sector_rel_mom", "golden_cross_days", "death_cross_days"]
FUND_FEATURES = ["fcf_yield", "ev_sales", "grossMargins", "operatingMargins", "revenueGrowth",
                 "earningsGrowth", "returnOnAssets", "returnOnEquity", "debtToEquity",
                 "priceToSalesTrailing12Months", "trailingPE", "net_cash_mcap", "fcf_margin",
                 "snapshot_age_days"]
REGIME_FEATURES = ["spy_vs_200", "spy_vol20", "breadth_50"]
FEATURES = TECH_FEATURES + FUND_FEATURES + REGIME_FEATURES
LABELS = ["fwd_21", "fwd_excess_21", "label_rank"]


def _cross_days(sma_fast: pd.Series, sma_slow: pd.Series, lookback: int = 250) -> tuple[float, float]:
    kind, days = technical._detect_cross(sma_fast, sma_slow, lookback=lookback)
    if kind == "golden":
        return float(days), 999.0
    if kind == "death":
        return 999.0, float(days)
    return 999.0, 999.0


def _technical_features_one(payload: tuple) -> list[dict]:
    """One symbol's feature rows across every as-of date. Module-level and frame-only
    so a spawned worker can run it (see src/parallel.py)."""
    sym, g, dates, bench_close = payload
    rows = []
    b63 = b126 = None
    f = (g["adj_close"] / g["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    frame = pd.DataFrame({"Open": g["open"] * f, "High": g["high"] * f, "Low": g["low"] * f,
                          "Close": g["adj_close"], "Volume": g["volume"]})
    for d in dates:
        win = frame.loc[:d].tail(520)
        if len(win) < 260:
            continue
        try:
            ind = technical._compute_one(win)
        except Exception:  # noqa: BLE001
            ind = None
        if not ind:
            continue
        close = win["Close"]
        rets = close.pct_change().dropna()
        rec = {"as_of": d, "symbol": sym,
               "rsi": ind.get("rsi"), "macd_hist": ind.get("macd_hist"),
               "atr_pct": ind.get("atr_pct"), "pos_52w": ind.get("pos_52w"),
               "vol_trend": ind.get("vol_trend"), "ret_20": ind.get("ret_20"),
               "ret_63": ind.get("ret_63"), "ret_126": ind.get("ret_126"),
               "ret_252_21": ind.get("ret_252_21"), "mom_va": ind.get("mom_va"),
               "technical_score": ind.get("technical_score"),
               "px_sma50": (ind["price"] / ind["sma50"]) if ind.get("sma50") else None,
               "px_sma200": (ind["price"] / ind["sma200"]) if ind.get("sma200") else None,
               "sma50_200": (ind["sma50"] / ind["sma200"]) if ind.get("sma50") and ind.get("sma200") else None,
               "vol_20": float(rets.tail(20).std() * np.sqrt(252)) if len(rets) >= 20 else None,
               "vol_60": float(rets.tail(60).std() * np.sqrt(252)) if len(rets) >= 60 else None,
               "mdd_63": float((close.tail(63) / close.tail(63).cummax() - 1).min()),
               "dollar_vol": float(np.log1p((close.tail(20) * win["Volume"].tail(20)).mean()))
               if "Volume" in win and win["Volume"].notna().any() else None}
        g_days, d_days = _cross_days(close.rolling(50).mean(), close.rolling(200).mean())
        rec["golden_cross_days"], rec["death_cross_days"] = g_days, d_days
        if bench_close is not None:
            bc = bench_close.loc[:d].dropna()
            b63, b126 = technical._ret(bc, 63), technical._ret(bc, 126)
        rec["rs_3m"] = (rec["ret_63"] - b63) if (rec["ret_63"] is not None and b63 is not None) else None
        rec["rs_6m"] = (rec["ret_126"] - b126) if (rec["ret_126"] is not None and b126 is not None) else None
        rows.append(rec)
    return rows


def technical_replay(ohlcv: pd.DataFrame, dates: list[pd.Timestamp], bench_close: pd.Series | None,
                     log: Callable[[str], None] | None = None,
                     workers: int | None = None) -> pd.DataFrame:
    """Long frame (as_of, symbol, <TECH_FEATURES minus sector_rel_mom>) from trailing
    windows of adjusted OHLCV. Shared with backtest.signals.technical_pillar."""
    by_sym = {s: g.set_index("date").sort_index() for s, g in ohlcv.groupby("symbol")}
    chunks = parallel.replay_symbols(
        _technical_features_one,
        [(sym, g, dates, bench_close) for sym, g in by_sym.items()],
        workers=workers, log=log, label="technical replay",
        work=len(by_sym) * len(dates))
    return pd.DataFrame(parallel.flatten(chunks))


def regime_replay(closes_adj: pd.DataFrame, bench: str, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """SPY vs 200dma, 20d vol, universe breadth above 50dma — per date."""
    rows = []
    b = closes_adj[bench].dropna() if bench in closes_adj else None
    uni = closes_adj.drop(columns=[c for c in closes_adj.columns if c in (bench, "SMH", "QQQ", "SPY")],
                          errors="ignore")
    sma50 = uni.rolling(50).mean()
    for d in dates:
        rec = {"as_of": d}
        if b is not None:
            bb = b.loc[:d]
            if len(bb) >= 200:
                rec["spy_vs_200"] = float(bb.iloc[-1] / bb.rolling(200).mean().iloc[-1] - 1) * 100
                rec["spy_vol20"] = float(bb.pct_change().tail(20).std() * np.sqrt(252) * 100)
        px = uni.loc[:d]
        if len(px) >= 50:
            last, s50 = px.iloc[-1], sma50.loc[:d].iloc[-1]
            ok = last.notna() & s50.notna()
            if ok.sum() >= 10:
                rec["breadth_50"] = float((last[ok] > s50[ok]).mean() * 100)
        rows.append(rec)
    return pd.DataFrame(rows)


def fundamentals_features(store, symbols: list[str], dates: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for d in dates:
        snap = store.fundamentals_asof(symbols, d.date())
        if snap.empty:
            continue
        for sym, r in snap.iterrows():
            mcap = r.get("marketCap")
            pe = r.get("trailingPE")
            rows.append({"as_of": d, "symbol": sym,
                         "fcf_yield": r.get("fcf_yield"), "ev_sales": r.get("ev_sales"),
                         "grossMargins": r.get("grossMargins"), "operatingMargins": r.get("operatingMargins"),
                         "revenueGrowth": r.get("revenueGrowth"), "earningsGrowth": r.get("earningsGrowth"),
                         "returnOnAssets": r.get("returnOnAssets"), "returnOnEquity": r.get("returnOnEquity"),
                         "debtToEquity": r.get("debtToEquity"),
                         "priceToSalesTrailing12Months": r.get("priceToSalesTrailing12Months"),
                         "trailingPE": pe if (pe is not None and pe == pe and pe > 0) else None,
                         "net_cash_mcap": (r.get("net_cash") / mcap) if (mcap and r.get("net_cash") is not None) else None,
                         "fcf_margin": r.get("fcf_margin"),
                         "snapshot_age_days": float((d - pd.Timestamp(r.get("as_of"))).days)
                         if r.get("as_of") is not None else None})
    return pd.DataFrame(rows)


def labels(closes_adj: pd.DataFrame, dates: list[pd.Timestamp], horizon: int = 21) -> pd.DataFrame:
    """Forward `horizon`-day return from the NEXT close after as_of (same convention as
    the harness), minus the same-date universe mean, rank-transformed per date."""
    idx = closes_adj.index
    rows = []
    for d in dates:
        i = idx.searchsorted(d, side="right")
        if i + horizon >= len(idx):
            continue
        f0, f1 = idx[i], idx[i + horizon]
        fwd = (closes_adj.loc[f1] / closes_adj.loc[f0] - 1) * 100
        fwd = fwd.dropna()
        if len(fwd) < 10:
            continue
        exc = fwd - fwd.mean()
        rk = exc.rank(pct=True)
        for s in fwd.index:
            rows.append({"as_of": d, "symbol": s, "fwd_21": float(fwd[s]),
                         "fwd_excess_21": float(exc[s]), "label_rank": float(rk[s])})
    return pd.DataFrame(rows)


def build_feature_frame(store, symbols: list[str], subsectors: dict[str, str],
                        dates: list[pd.Timestamp], bench: str = "SMH", regime_bench: str = "SPY",
                        with_labels: bool = True, log: Callable[[str], None] | None = None) -> pd.DataFrame:
    # Bound both reads to what the replays actually need: technical_replay looks back
    # ~520 trading days, regime_replay 200, and labels look forward `horizon`. Reading
    # the entire store was fine at 105 symbols and is not at 1,500.
    _lo = (min(dates) - pd.Timedelta(days=800)).date() if dates else None
    _hi = (max(dates) + pd.Timedelta(days=120)).date() if dates else None
    ohlcv = store.prices(symbols, start=_lo, end=_hi)
    closes = store.close_matrix(None, adjusted=True, start=_lo, end=_hi)
    if ohlcv.empty or closes.empty:
        return pd.DataFrame()
    bench_close = closes[bench] if bench in closes else None
    tech = technical_replay(ohlcv, dates, bench_close, log=log)
    if tech.empty:
        return pd.DataFrame()
    fund = fundamentals_features(store, symbols, dates)
    reg = regime_replay(closes, regime_bench, dates)
    df = tech.merge(fund, on=["as_of", "symbol"], how="left").merge(reg, on="as_of", how="left")
    df["subsector"] = df["symbol"].map(subsectors).fillna("")
    med = df.groupby(["as_of", "subsector"])["ret_126"].transform("median")
    df["sector_rel_mom"] = df["ret_126"] - med
    if with_labels:
        lab = labels(closes[[c for c in closes.columns if c in set(symbols)]], dates)
        df = df.merge(lab, on=["as_of", "symbol"], how="left")
    df["feature_version"] = FEATURE_VERSION
    return df.sort_values(["as_of", "symbol"]).reset_index(drop=True)


def rank_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cross-sectional percentile per date for each feature column (NaN stays NaN —
    LightGBM handles missing natively). Regime columns are date-constant and would rank
    to 0.5 everywhere, so they are passed through as-is."""
    out = df.copy()
    per_date = [c for c in cols if c in out.columns and c not in REGIME_FEATURES]
    out[per_date] = out.groupby("as_of")[per_date].rank(pct=True)
    return out
