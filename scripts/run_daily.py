"""Entrypoint for scheduled runs.

Usage:
    python -m scripts.run_daily --mode deep
    python -m scripts.run_daily --mode light
"""
from __future__ import annotations

import argparse

from src import config, market_calendar, pipeline


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
    pipeline.run(args.mode)


if __name__ == "__main__":
    main()
