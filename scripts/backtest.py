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


def universe_note(store: HistoryStore, eligible: dict, source: str) -> str:
    """The disclosure line printed at the top of every report.

    The curated wording is unchanged on purpose: nothing ever LEAVES a hand-written
    config list, so exit counts would be zero by construction and the only honest thing
    to say about it is that it was chosen with hindsight. The broad wording reports the
    membership actually recorded — how many names at each end and how many left — plus
    the bias that survives even so."""
    dates = sorted(eligible)
    first, last = dates[0].date(), dates[-1].date()
    if source != "broad":
        return (f"curated list; membership {store.universe_source(first)} at start, "
                f"{store.universe_source(last)} at end. Absolute returns are "
                f"upward-biased by hindsight selection — read the RANKING metrics.")
    ever = set().union(*eligible.values()) if eligible else set()
    m0, m1 = eligible[dates[0]], eligible[dates[-1]]
    return (f"broad SEC-registrant universe, membership recomputed point-in-time at "
            f"each month-end: {len(m0)} names at {first}, {len(m1)} at {last}, "
            f"{len(ever - m1)} of the {len(ever)} names ever a member had exited by the "
            f"end. Membership is {store.universe_source(first, source='broad')} at start, "
            f"{store.universe_source(last, source='broad')} at end. Names delisted "
            f"before the price store's first download are absent entirely, so residual "
            f"survivorship bias remains — read the RANKING metrics.")


def run_one(name: str, store: HistoryStore, start: date, top_frac: float, top_n: int | None,
            cost_bps: float) -> None:
    src = universe.configured_source()
    uni = universe.current_universe(store=store)
    syms = [u["symbol"] for u in uni]
    subs = {u["symbol"]: u["subsector"] for u in uni}
    have = set(store.symbols_with_prices())
    if not have:
        raise SystemExit("history store has no prices — run scripts.data_refresh first")
    dates = month_ends(start, date.today())
    if src == "broad":
        # Score every name that was EVER a member, not only today's. Restricting the
        # scored set to the current universe would re-introduce exactly the survivorship
        # bias the recorded membership exists to remove: the eligibility mask could then
        # only ever select from the survivors.
        from src.data import registrants
        reg = registrants.load_registrants(store)
        subs.update(dict(zip(reg["symbol"], reg["industry_group"])))
        ever = set().union(*(set(store.universe_asof(d.date(), source=src)) for d in dates))             if dates else set()
        syms = sorted(set(syms) | ever)
    # Membership is a set question; it used to be answered by pivoting the whole store
    # into a date x symbol matrix and reading `.columns`.
    syms = [s for s in syms if s in have]
    spec = BacktestSpec(name=name, top_frac=top_frac, top_n=top_n, cost_bps_side=cost_bps,
                        params={"start": str(start), "rebalance": "BME"})
    # Read ONLY the scored names plus the report benchmarks. `close_matrix(None)` looked
    # harmless and was not: the engine takes its rebalance calendar from this frame's
    # index, so every unrelated symbol in the store could move a curated backtest's
    # dates and every metric with them. A result must be a function of the universe it
    # claims to test, not of what else happens to have been downloaded.
    closes = store.close_matrix(sorted(set(syms) | set(spec.benchmarks)), adjusted=True)
    _log(f"{name}: {len(syms)} symbols, {len(dates)} month-ends from {start}")
    scores = build_signal(name, store, syms, subs, dates, closes)
    if scores.empty:
        raise SystemExit(f"{name}: signal produced no scores (missing data?)")
    # PIT universe where recorded; otherwise the current curated list (disclosed).
    eligible = {d: set(store.universe_asof(d.date(), source=src)) for d in scores.index}
    note = universe_note(store, eligible, src)
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
