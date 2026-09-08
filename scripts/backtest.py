"""Run a walk-forward rank backtest of a baseline signal on the point-in-time store.

    python -m scripts.backtest --signal momentum
    python -m scripts.backtest --signal technical --start 2016-01-01
    python -m scripts.backtest --signal composite --top-frac 0.1 --cost-bps 15
    python -m scripts.backtest --all             # every baseline, one report each

Reports land under $CLAUDE_TRADER_HOME/data/history/backtests/<name>_<timestamp>/ and
every run is appended to trials.jsonl (the count deflates the Sharpe of whatever you keep).
"""
from __future__ import annotations

import argparse
from datetime import date, datetime

import pandas as pd

from src import universe
from src.backtest import signals
from src.backtest.engine import BacktestSpec, log_trial, n_trials, run_rank_backtest
from src.data.history_store import HistoryStore
from src.data.yfinance_live import month_ends


def _log(msg: str) -> None:
    print(f"[backtest] {msg}", flush=True)


def build_signal(name: str, store: HistoryStore, syms: list[str], subs: dict[str, str],
                 dates: list[pd.Timestamp], closes_adj: pd.DataFrame) -> pd.DataFrame:
    if name == "momentum":
        return signals.momentum_12_1(closes_adj[syms], dates)
    if name == "technical":
        return signals.technical_pillar(store.prices(syms), dates, log=_log)
    if name == "fundamental":
        return signals.fundamental_pillar(store, syms, subs, dates)
    if name == "composite":
        tech = signals.technical_pillar(store.prices(syms), dates, log=_log)
        fund = signals.fundamental_pillar(store, syms, subs, dates)
        return signals.composite_3pillar(tech, fund)
    raise SystemExit(f"unknown signal {name}")


def run_one(name: str, store: HistoryStore, start: date, top_frac: float, top_n: int | None,
            cost_bps: float) -> None:
    uni = universe.build_universe()
    syms = [u["symbol"] for u in uni]
    subs = {u["symbol"]: u["subsector"] for u in uni}
    have = set(store.symbols_with_prices())
    if not have:
        raise SystemExit("history store has no prices — run scripts.data_refresh first")
    # Membership is a set question; it used to be answered by pivoting the whole store
    # into a date x symbol matrix and reading `.columns`.
    syms = [s for s in syms if s in have]
    closes = store.close_matrix(None, adjusted=True)
    dates = month_ends(start, date.today())
    _log(f"{name}: {len(syms)} symbols, {len(dates)} month-ends from {start}")
    scores = build_signal(name, store, syms, subs, dates, closes)
    if scores.empty:
        raise SystemExit(f"{name}: signal produced no scores (missing data?)")
    # PIT universe where recorded; otherwise the current curated list (disclosed).
    eligible = {d: set(store.universe_asof(d.date())) for d in scores.index}
    note = (f"curated list; membership {store.universe_source(scores.index[0].date())} at start, "
            f"{store.universe_source(scores.index[-1].date())} at end. Absolute returns are "
            f"upward-biased by hindsight selection — read the RANKING metrics.")
    spec = BacktestSpec(name=name, top_frac=top_frac, top_n=top_n, cost_bps_side=cost_bps,
                        params={"start": str(start), "rebalance": "BME"})
    trials_root = store.root
    n_prior = n_trials(trials_root)
    res = run_rank_backtest(scores, closes, spec, eligible=eligible, n_trials=n_prior + 1,
                            universe_note=note)
    log_trial(trials_root, name, {**spec.params, "top_frac": top_frac, "top_n": top_n,
                                  "cost_bps": cost_bps}, res.summary)
    out = store.root / "backtests" / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    res.save(out)
    print(res.to_markdown())
    _log(f"saved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward rank backtest")
    ap.add_argument("--signal", choices=["momentum", "technical", "fundamental", "composite"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--cost-bps", type=float, default=15.0)
    args = ap.parse_args()
    store = HistoryStore()
    start = date.fromisoformat(args.start)
    names = ["momentum", "technical", "fundamental", "composite"] if args.all else [args.signal]
    if not names or names == [None]:
        ap.error("--signal or --all required")
    for n in names:
        run_one(n, store, start, args.top_frac, args.top_n, args.cost_bps)


if __name__ == "__main__":
    main()
