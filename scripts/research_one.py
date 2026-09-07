"""Background single-ticker deep research, launched detached by the dashboard.

Usage:
    python -m scripts.research_one --run-id 42 --symbol NVDA

Rebuilds the Claude context from the stored board row, runs one deep-research call, and
upserts the report. A marker file under the cache dir signals "in progress" to the
dashboard so the button can show status without freezing the page.
"""
from __future__ import annotations

import argparse

from src import config, deep_research, learning, news, storage


def marker_path(run_id: int, symbol: str):
    return config.cache_dir() / f"research_inflight_{run_id}_{symbol.upper()}.marker"


def main() -> None:
    ap = argparse.ArgumentParser(description="Background single-ticker deep research")
    ap.add_argument("--run-id", type=int, required=True)
    ap.add_argument("--symbol", required=True)
    args = ap.parse_args()
    sym = args.symbol.upper()
    marker = marker_path(args.run_id, sym)
    try:
        marker.write_text("running", encoding="utf-8")
    except OSError:
        pass
    try:
        board = {r["symbol"]: r for r in storage.get_board(args.run_id)}
        row = board.get(sym)
        if not row:
            print(f"[research_one] {sym} not on run #{args.run_id}")
            return
        # Fetch fresh dated headlines (incl. the adverse-news query) so Claude sees the
        # actual current news, not just the catalysts stored at pipeline time.
        try:
            items = news.get_news(sym, row.get("name") or sym)
            heads = [f"[{news._published_dt(it).strftime('%Y-%m-%d')}] {it.get('title','')}"
                     for it in items[:10]]
        except Exception:  # noqa: BLE001
            heads = []
        ctx = deep_research.build_context(row, {"catalysts": row.get("sig_catalysts") or [],
                                                "headlines": heads})
        lb = learning.learning_block(sym)
        if lb:
            ctx = lb + "\n\n" + ctx
        hold = ({"shares": row.get("held_position"), "avg_cost": row.get("held_avg_cost")}
                if row.get("held_position") else None)
        report = deep_research.research_one(sym, row.get("name") or sym,
                                            row.get("subsector", ""), ctx, holding=hold)
        # Same red-team discipline as the scheduled pipeline — otherwise a
        # button-refreshed rating is systematically more bullish than a pipeline one.
        if (config.get("claude.red_team", True)
                and str(report.get("rating", "")).lower() in ("buy", "strong_buy")):
            rt = deep_research.red_team(sym, row.get("name") or sym, report)
            report = deep_research.apply_red_team(report, rt)
        storage.upsert_research(args.run_id, report)
        print(f"[research_one] {sym} rating={report.get('rating')} "
              f"conviction={report.get('conviction')}")
    finally:
        try:
            marker.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    main()
