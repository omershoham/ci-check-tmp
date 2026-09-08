"""Build / refresh the point-in-time history store.

    python -m scripts.data_refresh                 # prices + facts + snapshots for the universe
    python -m scripts.data_refresh --prices        # just prices (10y daily, incl. benchmarks)
    python -m scripts.data_refresh --facts         # just SEC companyfacts (per CIK)
    python -m scripts.data_refresh --facts-bulk    # SEC FSDS quarterly ZIPs (all filers)
    python -m scripts.data_refresh --snapshots     # rebuild monthly fundamentals snapshots
    python -m scripts.data_refresh --symbols NVDA AMD --start 2018-01-01

Everything lands under $CLAUDE_TRADER_HOME/data/history (see data/history_store.py).
"""
from __future__ import annotations

import argparse
from datetime import date

from src import config, universe
from src.data import sec_facts, sec_fsds, yfinance_live
from src.data.history_store import HistoryStore


def _log(msg: str) -> None:
    print(f"[data] {msg}", flush=True)


def benchmarks() -> list[str]:
    b = {config.get("technical.benchmark", "SPY"), config.get("technical.rs_benchmark", "SMH"),
         config.get("regime.benchmark", "SPY"), "QQQ"}
    return sorted(s for s in b if s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh the point-in-time history store")
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--facts", action="store_true",
                    help="SEC companyfacts, one request per CIK (fine for a curated universe)")
    ap.add_argument("--facts-bulk", action="store_true",
                    help="SEC Financial Statement Data Sets: one ZIP per quarter covering "
                         "every filer. O(1) in universe size; use for a broad universe")
    ap.add_argument("--from-quarter", default="2015q1",
                    help="first FSDS quarter to ingest, e.g. 2015q1")
    ap.add_argument("--snapshots", action="store_true")
    ap.add_argument("--symbols", nargs="*", help="subset (default: whole universe)")
    ap.add_argument("--period", default="10y", help="yfinance period for prices")
    ap.add_argument("--start", default="2015-01-01", help="first snapshot month-end")
    args = ap.parse_args()
    do_all = not (args.prices or args.facts or args.facts_bulk or args.snapshots)

    store = HistoryStore()
    syms = [s.upper() for s in args.symbols] if args.symbols else universe.symbols()
    _log(f"store: {store.root}")
    _log(f"universe: {len(syms)} symbols")

    if do_all or args.prices:
        _log(f"prices: downloading {args.period} daily bars for {len(syms)} + benchmarks")
        n = yfinance_live.refresh_prices(store, syms + benchmarks(), period=args.period)
        _log(f"prices: {n} rows in store")

    if do_all or args.facts:
        _log("facts: fetching SEC companyfacts (one request per CIK)")
        res = sec_facts.refresh_facts(store, syms, log=_log)
        _log(f"facts: {res}")

    if args.facts_bulk:
        _log(f"facts (bulk): FSDS quarters from {args.from_quarter}")
        # `symbols` restricts the ingest; pass None once the universe is broad.
        res = sec_fsds.refresh_facts_bulk(store, syms, start_quarter=args.from_quarter,
                                          log=_log)
        _log(f"facts (bulk): {res}")

    if do_all or args.snapshots:
        dates = yfinance_live.month_ends(date.fromisoformat(args.start), date.today())
        _log(f"snapshots: {len(dates)} month-ends from {dates[0].date()} for {len(syms)} symbols")
        df = sec_facts.build_snapshots(store, syms, dates, log=_log)
        _log(f"snapshots: {len(df)} rows written")

    # Universe membership as of today — the going-forward survivorship record.
    store.record_universe(date.today(), universe.build_universe())
    _log(f"manifest: {store.manifest()}")


if __name__ == "__main__":
    main()
