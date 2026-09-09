"""Run a walk-forward rank backtest of a baseline signal on the point-in-time store.

    python -m scripts.backtest --signal momentum
    python -m scripts.backtest --signal technical --start 2016-01-01
    python -m scripts.backtest --signal composite --top-frac 0.1 --cost-bps 15
    python -m scripts.backtest --all             # every baseline, one report each

Reports land under $CLAUDE_TRADER_HOME/data/history/backtests/<name>_<timestamp>/ and
every run is appended to trials.jsonl. The number of DISTINCT configurations in that
file (not the number of runs) deflates the Sharpe of whatever you keep: re-running a
config you already ran is verification, not another lottery ticket.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime

import pandas as pd

from src import config, universe
from src.backtest import signals
from src.backtest.engine import (BacktestSpec, config_hash, log_trial, n_trials,
                                 run_rank_backtest, trial_hashes)
from src.data.history_store import HistoryStore
from src.data.yfinance_live import every_n_trading_days, month_ends


def _log(msg: str) -> None:
    print(f"[backtest] {msg}", flush=True)


def build_signal(name: str, store: HistoryStore, syms: list[str], subs: dict[str, str],
                 dates: list[pd.Timestamp], closes_adj: pd.DataFrame,
                 secs: dict[str, str] | None = None) -> pd.DataFrame:
    if name == "momentum":
        return signals.momentum_12_1(closes_adj[syms], dates)
    if name == "technical":
        return signals.technical_pillar(store.prices(syms), dates, log=_log)
    if name == "fundamental":
        return signals.fundamental_pillar(store, syms, subs, dates, secs)
    if name == "composite":
        # Checked BEFORE the ~10-minute technical replay, not only inside
        # composite_3pillar: a run that is going to be refused should be refused
        # while the user is still watching.
        signals.assert_replayable()
        tech = signals.technical_pillar(store.prices(syms), dates, log=_log)
        # `scored` is what keeps the replay's cross-section identical to the live
        # screen's: thin-data names stay in and are re-weighted, names the liquidity
        # floor would have dropped stay out.
        fund, scored = signals.fundamental_frames(store, syms, subs, dates, secs)
        return signals.composite_3pillar(tech, fund, scored)
    raise SystemExit(f"unknown signal {name}")


def universe_note(store: HistoryStore, eligible: dict, source: str,
                  mask: str | None = None) -> str:
    """The disclosure line printed at the top of every report.

    The curated wording is unchanged on purpose: nothing ever LEAVES a hand-written
    config list, so exit counts would be zero by construction and the only honest thing
    to say about it is that it was chosen with hindsight. The broad wording reports the
    membership actually recorded — how many names at each end and how many left — plus
    the bias that survives even so.

    `mask` is the tag the eligibility history is keyed by (`broad:1500`). Naming it is
    the point: broad-250 and broad-1500 are different populations, and a report that
    said only "broad" could not be reproduced once the cap moved."""
    dates = sorted(eligible)
    first, last = dates[0].date(), dates[-1].date()
    mask = mask or source
    if source != "broad":
        return (f"curated list; membership {store.universe_source(first, source=mask)} "
                f"at start, {store.universe_source(last, source=mask)} at end. Absolute "
                f"returns are upward-biased by hindsight selection — read the RANKING "
                f"metrics.")
    ever = set().union(*eligible.values()) if eligible else set()
    m0, m1 = eligible[dates[0]], eligible[dates[-1]]
    return (f"broad SEC-registrant universe (eligibility mask `{mask}`), membership "
            f"recomputed point-in-time at "
            f"each month-end: {len(m0)} names at {first}, {len(m1)} at {last}, "
            f"{len(ever - m1)} of the {len(ever)} names ever a member had exited by the "
            f"end. Membership is {store.universe_source(first, source=mask)} at start, "
            f"{store.universe_source(last, source=mask)} at end. Names delisted "
            f"before the price store's first download are absent entirely, so residual "
            f"survivorship bias remains — read the RANKING metrics.")


def run_one(name: str, store: HistoryStore, start: date, top_frac: float, top_n: int | None,
            cost_bps: float, horizon_days: int | None = None) -> None:
    src = universe.configured_source()
    # The eligibility mask is keyed by source AND cap, so a report always runs on the
    # membership recorded for the cap it is configured at. Reading a bare `broad` mask
    # is what silently ran the first broad-750 backtests on a 250-name eligibility
    # history in Phase 7.
    mask = universe.membership_source(src)
    uni = universe.current_universe(store=store)
    syms = [u["symbol"] for u in uni]
    subs = {u["symbol"]: u["subsector"] for u in uni}
    have = set(store.symbols_with_prices())
    if not have:
        raise SystemExit("history store has no prices — run scripts.data_refresh first")
    # The rebalance grid IS the label horizon: the engine's forward return runs from one
    # grid date to the next. So `--horizon-days 10` measures the signal on a swing clock
    # and `126` on a long one — the non-overlapping version of learning.pillar_ic's
    # question. Default (None) = month-ends, byte-identical to every prior report.
    dates = (list(every_n_trading_days(start, date.today(), horizon_days))
             if horizon_days else month_ends(start, date.today()))
    if src == "broad":
        # Score every name that was EVER a member, not only today's. Restricting the
        # scored set to the current universe would re-introduce exactly the survivorship
        # bias the recorded membership exists to remove: the eligibility mask could then
        # only ever select from the survivors.
        from src.data import registrants
        reg = registrants.load_registrants(store)
        subs.update(dict(zip(reg["symbol"], reg["industry_group"])))
        ever = set().union(*(set(store.universe_asof(d.date(), source=mask)) for d in dates))             if dates else set()
        syms = sorted(set(syms) | ever)
    # Membership is a set question; it used to be answered by pivoting the whole store
    # into a date x symbol matrix and reading `.columns`.
    syms = [s for s in syms if s in have]
    # The VALUATION peer set. Sourced from the registrant table for BOTH universe
    # sources — the curated names are SEC registrants too — so a replay ranks multiples
    # inside the same sectors the live screen does.
    secs = universe.sector_map(store=store, symbols=syms)
    spec = BacktestSpec(name=name, top_frac=top_frac, top_n=top_n, cost_bps_side=cost_bps,
                        params={"start": str(start),
                                "rebalance": f"{horizon_days}d" if horizon_days else "BME"})
    # Read ONLY the scored names plus the report benchmarks. `close_matrix(None)` looked
    # harmless and was not: the engine takes its rebalance calendar from this frame's
    # index, so every unrelated symbol in the store could move a curated backtest's
    # dates and every metric with them. A result must be a function of the universe it
    # claims to test, not of what else happens to have been downloaded.
    closes = store.close_matrix(sorted(set(syms) | set(spec.benchmarks)), adjusted=True)
    _log(f"{name}: {len(syms)} symbols, {len(dates)} "
         f"{f'{horizon_days}-day rebalances' if horizon_days else 'month-ends'} from {start}")
    scores = build_signal(name, store, syms, subs, dates, closes, secs)
    if scores.empty:
        raise SystemExit(f"{name}: signal produced no scores (missing data?)")
    # PIT universe where recorded; otherwise the current curated list (disclosed).
    eligible = {d: set(store.universe_asof(d.date(), source=mask)) for d in scores.index}
    note = universe_note(store, eligible, src, mask=mask)
    trials_root = store.root
    params = {**spec.params, "top_frac": top_frac, "top_n": top_n, "cost_bps": cost_bps}
    # DISTINCT configurations, not runs. Re-running this exact config (a refactor
    # verification, a byte-identical gate) is not a new hypothesis and must not deflate
    # the Sharpe of the one you keep - see engine.config_hash. The +1 covers THIS run
    # when it is a configuration never tried before; when it is a repeat the hash is
    # already in the set and the count does not move.
    # The universe tag carries the SIZE CAP too: broad-250, broad-750 and broad-1500
    # are three different populations, so the same signal on them is three hypotheses,
    # not one re-run. `curated` has no cap and needs none.
    uni_tag = mask   # same tag the eligibility mask is keyed by: one name for one population
    n_prior = n_trials(trials_root)
    this_hash = config_hash(name, params, uni_tag)
    n_this = n_prior + (0 if this_hash in trial_hashes(trials_root) else 1)
    res = run_rank_backtest(scores, closes, spec, eligible=eligible, n_trials=n_this,
                            universe_note=note)
    log_trial(trials_root, name, params, res.summary, universe=uni_tag)
    suffix = f"_h{horizon_days}" if horizon_days else ""
    out = (store.root / "backtests"
           / f"{name}{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
    # Comparing two universes must not mean editing the shipped config.yaml and
    # editing it back: an interrupted comparison used to leave the file describing a
    # universe nobody chose. These override the config FOR THIS PROCESS ONLY.
    ap.add_argument("--universe", choices=["curated", "broad"], default=None,
                    help="override universe.source for this run only")
    ap.add_argument("--theme", default=None,
                    help="override the theme pack for this run only (e.g. ai_infra). "
                         "A curated universe needs the pack that owns its ticker list.")
    ap.add_argument("--horizon-days", type=int, default=None,
                    help="rebalance AND label horizon in trading days (e.g. 10 swing, "
                         "42 position, 126 long). Omit for the default month-end grid.")
    args = ap.parse_args()
    if args.universe:
        config.set_override("universe.source", args.universe)
    if args.theme:
        config.set_override("theme", args.theme)
    if args.universe or args.theme:
        _log(f"config overrides (this process only): {config.overrides()}")
    if args.horizon_days is not None and args.horizon_days < 1:
        ap.error("--horizon-days must be >= 1")
    store = HistoryStore()
    start = date.fromisoformat(args.start)
    names = ["momentum", "technical", "fundamental", "composite"] if args.all else [args.signal]
    if not names or names == [None]:
        ap.error("--signal or --all required")
    for n in names:
        run_one(n, store, start, args.top_frac, args.top_n, args.cost_bps,
                horizon_days=args.horizon_days)


if __name__ == "__main__":
    main()
