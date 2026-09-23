"""Build / refresh the point-in-time history store.

    python -m scripts.data_refresh                 # prices + facts + snapshots for the universe
    python -m scripts.data_refresh --prices        # just prices (incremental if the store is current)
    python -m scripts.data_refresh --facts         # just SEC companyfacts (per CIK)
    python -m scripts.data_refresh --facts-bulk    # SEC FSDS quarterly ZIPs (all filers)
    python -m scripts.data_refresh --snapshots     # rebuild monthly fundamentals snapshots
    python -m scripts.data_refresh --symbols NVDA AMD --start 2018-01-01

Broad universe (universe.source: broad), in the order they depend on each other:

    python -m scripts.data_refresh --registrants      # SEC exchange file + FSDS sub.txt -> candidates
    python -m scripts.data_refresh --shares-bulk      # share counts for every candidate CIK
    python -m scripts.data_refresh --screen-prices    # 400d of closes for every candidate
    python -m scripts.data_refresh --universe-history # month-end membership, 2016 -> today

Everything lands under $CLAUDE_TRADER_HOME/data/history (see data/history_store.py).
"""
from __future__ import annotations

import argparse
from datetime import date

from src import config, news, universe
from src.data import registrants, sec_facts, sec_fsds, yfinance_live
from src.data.history_store import HistoryStore


def _log(msg: str) -> None:
    print(f"[data] {msg}", flush=True)


def benchmarks() -> list[str]:
    b = {config.get("technical.benchmark", "SPY"), config.get("technical.rs_benchmark", "SMH"),
         config.get("regime.benchmark", "SPY"), "QQQ"}
    return sorted(s for s in b if s)


def universe_history(store, start: date, source: str) -> int:
    """Recompute and record membership at every month-end from `start`.

    This is what makes `scripts.backtest`'s eligibility mask survivorship-aware without
    the engine changing at all: each date's members are re-derived from the store's own
    prices and SEC filings as of THAT date, so a company that was delisted in 2019 is a
    member up to 2019 and absent after — instead of the whole history being scored on
    the list of survivors that exist today.

    `source` carries the CAP (`broad:1500`). Membership at cap 250 and cap 1500 are two
    different populations; recording both under a bare `broad` overwrote the smaller
    history and made every report built on it unreproducible (Phase 7). Re-run this
    once per cap change."""
    dates = yfinance_live.month_ends(start, date.today())
    n = 0
    for i, d in enumerate(dates, 1):
        members = universe.investable(d.date(), store=store)
        if members:
            store.record_universe(d.date(), members, source=source)
        n += len(members)
        if i % 12 == 0 or i == len(dates):
            _log(f"  universe history {i}/{len(dates)}: {d.date()} -> {len(members)} names")
    return n


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
    ap.add_argument("--facts-source", default="fsds",
                    help="marker recording which FSDS ingest pass covered a quarter; "
                         "use a new name to ingest a WIDER symbol set into quarters an "
                         "earlier pass already marked")
    ap.add_argument("--refetch-quarters", action="store_true",
                    help="re-ingest FSDS quarters already in the store (only needed "
                         "when the PARSE changed; a published ZIP never does)")
    ap.add_argument("--snapshots", action="store_true")
    ap.add_argument("--registrants", action="store_true",
                    help="rebuild the SEC registrant table (broad universe candidates)")
    ap.add_argument("--shares-bulk", action="store_true",
                    help="FSDS share counts for every candidate CIK — what the broad "
                         "point-in-time market-cap floor is computed from")
    ap.add_argument("--screen-prices", action="store_true",
                    help="download the trailing price window the broad floor needs")
    ap.add_argument("--universe-history", action="store_true",
                    help="recompute and record month-end membership from --start")
    ap.add_argument("--symbols", nargs="*", help="subset (default: whole universe)")
    ap.add_argument("--period", default="10y", help="yfinance period for prices")
    ap.add_argument("--price-start", default=None,
                    help="explicit first price date; overrides the incremental window "
                         "(use it to backfill full history for a symbol subset)")
    ap.add_argument("--start", default="2015-01-01", help="first snapshot month-end")
    ap.add_argument("--no-quality", action="store_true",
                    help="skip the data-quality gate that normally runs after a rebuild")
    args = ap.parse_args()
    broad_only = (args.registrants or args.shares_bulk or args.screen_prices
                  or args.universe_history)
    do_all = not (args.prices or args.facts or args.facts_bulk or args.snapshots or broad_only)

    store = HistoryStore()
    src = universe.configured_source()
    mask = universe.membership_source(src)
    _log(f"store: {store.root}  (universe.source={src}, membership mask={mask})")
    # A store written before Phase 2 is one file per table; convert it before anything
    # reads it, so the first refresh after an upgrade is where the cost is paid and
    # logged rather than inside whatever happened to touch the store first.
    store.migrate_layout(log=_log)

    if args.registrants:
        # The registrant ladder reads `sub.txt` from the CACHED FSDS quarterly ZIPs to
        # decide who files financial statements. On a fresh machine that cache is empty,
        # every registrant fails the "files financials" rung, and the step crashed with
        # KeyError 'sic' on an empty frame. Fetch any missing quarters first (a no-op
        # when the cache is warm; ~50-100 MB per quarter the first time).
        qs = sec_fsds.quarters_between(args.from_quarter)
        missing = [q for q in qs if not (sec_fsds.cache_dir() / f"{q}.zip").exists()]
        if missing:
            _log(f"registrants: downloading {len(missing)} FSDS quarters first "
                 f"({missing[0]} .. {missing[-1]})")
            for i, q in enumerate(missing, 1):
                ok = sec_fsds.download_quarter(q) is not None
                _log(f"  fsds {q} {'ok' if ok else 'not published'} ({i}/{len(missing)})")
        res = registrants.refresh_registrants(store, log=_log)
        _log(f"registrants: {res['rows']} rows -> {res['path']}")
        _log(f"registrants funnel: {res['funnel']}")

    # `--symbols` present but EMPTY is a shell expansion that matched nothing, never a
    # request to refresh the entire universe. Falling through to "everything" turned a
    # 3-symbol repair into a full re-ingest twice during Phase 3.
    if args.symbols is not None and not args.symbols:
        ap.error("--symbols was given with no tickers (an empty expansion?). "
                 "Omit --symbols entirely to mean the whole universe.")
    syms = [s.upper() for s in args.symbols] if args.symbols else universe.symbols()
    _log(f"universe: {len(syms)} symbols")

    if args.shares_bulk:
        # Restricted to candidate CIKs, but the parse itself is O(1) in universe size.
        cikmap = news._ticker_cik_map()
        want = set(syms)
        cik_to_symbol: dict[str, str] = {}
        for sym, cik in cikmap.items():
            if sym in want:
                cik_to_symbol.setdefault(str(cik).zfill(10), sym)
        _log(f"shares (bulk): {len(cik_to_symbol)} CIKs from {args.from_quarter}")
        res = sec_fsds.refresh_shares_bulk(store, cik_to_symbol,
                                           start_quarter=args.from_quarter,
                                           force=args.refetch_quarters, log=_log)
        _log(f"shares (bulk): {res['rows']} rows for {res['symbols']} symbols")

    if args.screen_prices:
        res = universe.refresh_screen_prices(store, syms + benchmarks(), log=_log)
        _log(f"screen prices: {res}")

    if do_all or args.prices:
        _log(f"prices: refreshing daily bars for {len(syms)} + benchmarks")
        n = yfinance_live.refresh_prices(
            store, syms + benchmarks(), period=args.period, log=_log,
            start=date.fromisoformat(args.price_start) if args.price_start else None)
        _log(f"prices: {n} rows in store")

    if do_all or args.facts:
        _log("facts: fetching SEC companyfacts (one request per CIK)")
        res = sec_facts.refresh_facts(store, syms, log=_log)
        _log(f"facts: {res}")

    if args.facts_bulk:
        _log(f"facts (bulk): FSDS quarters from {args.from_quarter}")
        # `symbols` restricts the ingest; pass None once the universe is broad.
        res = sec_fsds.refresh_facts_bulk(store, syms, start_quarter=args.from_quarter,
                                          force=args.refetch_quarters,
                                          source=args.facts_source, log=_log)
        _log(f"facts (bulk): {res}")

    if do_all or args.snapshots:
        dates = yfinance_live.month_ends(date.fromisoformat(args.start), date.today())
        _log(f"snapshots: {len(dates)} month-ends from {dates[0].date()} for {len(syms)} symbols")
        df = sec_facts.build_snapshots(store, syms, dates, log=_log)
        _log(f"snapshots: {len(df)} rows written")

    if args.universe_history:
        start = date.fromisoformat(args.start)
        _log(f"universe history: month-ends from {start} (mask={mask})")
        n = universe_history(store, start, mask)
        _log(f"universe history: {n} membership rows recorded")
    else:
        # Universe membership as of today — the going-forward survivorship record.
        # Never record an EMPTY membership: in broad mode that just means the price or
        # share data this date needs has not been downloaded yet, and writing it would
        # overwrite a real snapshot with "nothing was investable".
        members = universe.current_universe(store=store)
        if members:
            store.record_universe(date.today(), members, source=mask)
    _log(f"manifest: {store.manifest()}")

    # A refresh is exactly when the store can silently get worse — a tag list changed,
    # a quarter parsed differently, a cohort stopped resolving — so the gate runs HERE,
    # unprompted, rather than waiting for someone to notice a number on the board. It
    # reports and does not raise: a refresh that already wrote good data must not be
    # undone by a check, and the exit code belongs to `scripts.data_quality`.
    if not args.no_quality:
        from src.data import quality
        rep = quality.run(store)
        c = rep.counts()
        _log(f"quality: {c['critical']} critical, {c['error']} error, {c['warn']} warn "
             f"({rep.seconds:.1f}s)")
        for f in rep.blocking():
            _log(f"quality BLOCKING: {f.line()}")
        for check, info in list(rep.by_check().items())[:12]:
            _log(f"  {info['severity']:8} {check:28} n={info['n']}")


if __name__ == "__main__":
    main()
