"""Entrypoint for scheduled runs.

Usage:
    python -m scripts.run_daily --mode deep
    python -m scripts.run_daily --mode light
"""
from __future__ import annotations

import argparse

from datetime import date

from src import config, market_calendar, pipeline, universe


def _refresh_today_snapshot() -> None:
    """Write today's fundamentals snapshot BEFORE the pipeline reads it.

    Order matters. Stage 1 now serves the fundamental pillar from
    `fundamentals_asof(today)`, which falls back to the last month-end snapshot — so
    without this the screen would rank today's technicals against multiples priced off
    a close up to a month old. The facts have not changed since the month-end; only
    the price has, which is why this is seconds rather than a rebuild."""
    from src.data import sec_facts
    from src.data.history_store import HistoryStore
    store = HistoryStore()
    syms = [r["symbol"] for r in universe.current_universe(store=store)]
    if not syms:
        print("[run_daily] no universe members; skipping today's snapshot")
        return
    try:
        df = sec_facts.build_daily_snapshot(store, syms, date.today(),
                                            log=lambda m: print(f"[run_daily] {m}"))
        print(f"[run_daily] today's fundamentals snapshot: {len(df)}/{len(syms)} names")
    except Exception as exc:  # noqa: BLE001 - a stale snapshot still scores; a crash does not
        print(f"[run_daily] today's snapshot failed ({exc}); the pipeline will fall "
              f"back to the last month-end")


def main() -> None:
    ap = argparse.ArgumentParser(description="Claude Trader daily run")
    ap.add_argument("--mode", choices=["deep", "light"], default="deep")
    ap.add_argument("--force", action="store_true",
                    help="run even if the US market is closed today (weekend/holiday)")
    args = ap.parse_args()
    if (config.get("schedule.skip_when_market_closed", True) and not args.force
            and not market_calendar.is_trading_day()):
        print(f"[run_daily] US market closed today ({market_calendar.reason_closed()}); "
              f"skipping {args.mode} run. Use --force to override.")
        return
    _refresh_today_snapshot()
    pipeline.run(args.mode)


if __name__ == "__main__":
    main()
