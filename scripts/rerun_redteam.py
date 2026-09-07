"""One-off: re-adjudicate a run's red-teamed ratings under the recalibrated contract.

The first live outing of the red team (run #53) killed 11 of 13 buy ratings with a
uniform 2-point haircut — grading its own eloquence, not evidence. The contract now
requires concrete NEW dated evidence to kill. This re-runs ONLY the red-team pass on
the stored research reports (the expensive web research is reused, not repeated).

Usage: python -m scripts.rerun_redteam --run-id 53
"""
from __future__ import annotations

import argparse
import json

from src import deep_research, storage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=int, required=True)
    args = ap.parse_args()
    with storage.get_conn() as cn:
        rows = [dict(r) for r in cn.execute(
            "SELECT * FROM research_reports WHERE run_id=?", (args.run_id,))]
    for r in rows:
        try:
            rt_old = json.loads(r.get("red_team_json") or "{}")
        except json.JSONDecodeError:
            rt_old = {}
        if not rt_old or rt_old.get("survives") is not False:
            continue  # only re-judge the killed ones
        # Reconstruct the pre-red-team report.
        report = {
            "symbol": r["symbol"],
            "rating": rt_old.get("original_rating") or r.get("rating"),
            "conviction": rt_old.get("original_conviction") or r.get("conviction"),
            "thesis": r.get("thesis"),
            "risks": json.loads(r.get("risks_json") or "[]"),
        }
        print(f"[rerun] {r['symbol']}: re-judging "
              f"{report['rating']} conv={report['conviction']}", flush=True)
        rt_new = deep_research.red_team(r["symbol"], r["symbol"], report)
        if not rt_new:
            print(f"[rerun]   {r['symbol']}: red team returned nothing — keeping as-is",
                  flush=True)
            continue
        report = deep_research.apply_red_team(report, rt_new)
        # Merge back into the stored row (keep everything else).
        merged = {**r,
                  "rating": report.get("rating"),
                  "conviction": report.get("conviction"),
                  "red_team": report.get("red_team", {}),
                  "catalysts": json.loads(r.get("catalysts_json") or "[]"),
                  "risks": json.loads(r.get("risks_json") or "[]")}
        storage.upsert_research(args.run_id, merged)
        print(f"[rerun]   {r['symbol']}: survives={rt_new.get('survives')} "
              f"haircut={rt_new.get('confidence_haircut')} -> "
              f"{report.get('rating')} conv={report.get('conviction')}", flush=True)
    print("[rerun] done", flush=True)


if __name__ == "__main__":
    main()
