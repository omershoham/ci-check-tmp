"""Baseline signals for the harness, built ONLY from the point-in-time store.

  momentum_12_1        Jegadeesh-Titman 12-1 (the literature's yardstick)
  technical_pillar     src/technical._compute_one replayed on each date's trailing window
  fundamental_pillar   src/scoring.fundamental_pillar on the PIT snapshot for each date
  composite_3pillar    the live screen replayed: percentile-ranked pillars × config weights,
                       sentiment fixed at 0.5 (no news history exists)

Each returns a date × symbol score frame on the requested rebalance dates.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .. import config, parallel, scoring, technical


def _rank01(row: pd.Series) -> pd.Series:
    return row.rank(pct=True)


def momentum_12_1(closes_adj: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    out = {}
    for d in dates:
        px = closes_adj.loc[:d]
        if len(px) < 260:
            continue
        end = px.iloc[-22]            # skip the most recent month
        base = px.iloc[-253]
        out[d] = (end / base - 1)
    return pd.DataFrame(out).T.sort_index()


def _technical_scores_one(payload: tuple) -> dict:
    """One symbol's technical_score per date. Module-level and frame-only so a spawned
    worker can run it (see src/parallel.py)."""
    g, dates = payload
    # adjust OHLC by the same factor as close so indicators see a consistent series
    f = (g["adj_close"] / g["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    frame = pd.DataFrame({"Open": g["open"] * f, "High": g["high"] * f, "Low": g["low"] * f,
                          "Close": g["adj_close"], "Volume": g["volume"]})
    out: dict = {}
    for d in dates:
        win = frame.loc[:d].tail(520)
        if len(win) < 260:
            continue
        try:
            ind = technical._compute_one(win)
        except Exception:  # noqa: BLE001
            ind = None
        if ind:
            out[d] = ind["technical_score"]
    return out


def technical_pillar(ohlcv: pd.DataFrame, dates: list[pd.Timestamp],
                     log: Callable[[str], None] | None = None,
                     workers: int | None = None) -> pd.DataFrame:
    """Replay `technical._compute_one` per symbol per date on the trailing 2y window.
    `ohlcv` is the long price frame from the store (adjusted OHLC used for indicators).

    Symbols are independent; the pool keeps results in symbol order so the score
    frame's column order — which breaks ties when the engine cuts the top decile —
    is exactly what the serial loop produced."""
    by_sym = {s: g.set_index("date").sort_index() for s, g in ohlcv.groupby("symbol")}
    scored = parallel.replay_symbols(_technical_scores_one,
                                     [(g, dates) for g in by_sym.values()],
                                     workers=workers, log=log, label="technical replay",
                                     work=len(by_sym) * len(dates))
    out: dict[pd.Timestamp, dict[str, float]] = {}
    for sym, per_date in zip(by_sym, scored):
        for d, v in per_date.items():
            out.setdefault(d, {})[sym] = v
    return pd.DataFrame(out).T.sort_index()


def fundamental_pillar(store, symbols: list[str], subsectors: dict[str, str],
                       dates: list[pd.Timestamp]) -> pd.DataFrame:
    out = {}
    for d in dates:
        snap = store.fundamentals_asof(symbols, d.date())
        if snap.empty or len(snap) < 10:
            continue
        out[d] = scoring.fundamental_pillar(snap, subsectors)
    return pd.DataFrame(out).T.sort_index()


def composite_3pillar(tech: pd.DataFrame, fund: pd.DataFrame,
                      weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Percentile-rank each pillar per date and blend with config weights; the missing
    sentiment pillar is a constant 0.5 so its weight simply shrinks the others' scale
    (ranks are unaffected)."""
    w = weights or config.get("scoring.pillars",
                              {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.35})
    dates = sorted(set(tech.index) & set(fund.index))
    out = {}
    for d in dates:
        t = tech.loc[d].dropna()
        f = fund.loc[d].dropna()
        names = t.index.intersection(f.index)
        if len(names) < 10:
            continue
        out[d] = (_rank01(t.loc[names]) * w.get("technical", 0.35)
                  + _rank01(f.loc[names]) * w.get("fundamental", 0.30)
                  + 0.5 * w.get("sentiment", 0.35))
    return pd.DataFrame(out).T.sort_index()
