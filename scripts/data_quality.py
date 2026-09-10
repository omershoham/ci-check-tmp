"""Run the point-in-time store's data-quality gate.

    python -m scripts.data_quality                    # check, print, exit 1 on critical
    python -m scripts.data_quality --json out.json    # machine-readable findings
    python -m scripts.data_quality --update-baseline  # freeze today's coverage/totals
    python -m scripts.data_quality --cross-source 40  # also compare 40 names to .info

Exits non-zero when anything CRITICAL fires, so it can gate a refresh or a run.
The baseline it compares coverage against lives in the store
(`<store>/quality_baseline.json`); update it deliberately, never automatically —
a gate that re-baselines itself can only ratify a regression, never catch one.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from src.data import quality
from src.data.history_store import HistoryStore


def main() -> int:
    ap = argparse.ArgumentParser(description="Data-quality gate over the PIT store")
    ap.add_argument("--symbols", nargs="*", help="subset (default: the newest snapshot)")
    ap.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--json", default=None, help="write the full report here")
    ap.add_argument("--update-baseline", action="store_true",
                    help="record today's coverage/totals as the regression baseline")
    ap.add_argument("--cross-source", type=int, default=0, metavar="N",
                    help="also compare N names against yfinance .info (network)")
    ap.add_argument("--limit", type=int, default=40,
                    help="findings printed per check (0 = all)")
    args = ap.parse_args()

    store = HistoryStore()
    rep = quality.run(store,
                      symbols=[s.upper() for s in args.symbols] if args.symbols else None,
                      as_of=date.fromisoformat(args.as_of) if args.as_of else None,
                      cross_source=args.cross_source)

    print(f"\n  store: {store.root}")
    print(f"  cross-section: {rep.stats.get('as_of')} · {rep.stats.get('symbols')} symbols "
          f"· {rep.stats.get('scored')} with a fundamental pillar "
          f"· {rep.stats.get('nonusd_symbols')} non-USD reporters")
    print(f"  runtime: {rep.seconds:.1f}s\n")

    for check, info in rep.by_check().items():
        print(f"  {info['severity'].upper():8} {check:28} n={info['n']:<6} "
              f"e.g. {', '.join(str(x) for x in info['examples'])}")
    print()
    shown = 0
    for f in sorted(rep.findings, key=lambda x: (quality._ORDER.get(x.severity, 9), x.check)):
        if f.severity != quality.CRITICAL:
            continue
        if args.limit and shown >= args.limit:
            print(f"  ... ({args.limit} critical findings shown)")
            break
        print("  " + f.line())
        shown += 1

    c = rep.counts()
    print(f"\n  {len(rep.findings)} findings: {c['critical']} CRITICAL, "
          f"{c['error']} ERROR, {c['warn']} WARN\n")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep.to_dict(), fh, indent=2, default=str)
        print(f"  full report -> {args.json}\n")
    if args.update_baseline:
        data = quality.write_baseline(store, rep)
        print(f"  baseline updated -> {quality.baseline_path(store)} "
              f"({data['scored']}/{data['symbols']} scored)\n")
    return 1 if c["critical"] else 0


if __name__ == "__main__":
    sys.exit(main())
