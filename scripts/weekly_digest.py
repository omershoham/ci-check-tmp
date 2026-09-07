"""Monday-morning digest: one markdown file summarizing the week for the owner.

Sections: fired invalidation triggers, current regime, board rating changes vs a week
ago, upcoming catalysts/earnings (14 days), calibration snapshot, latest PM targets.

Usage:  python -m scripts.weekly_digest
Output: data/digests/digest_YYYY-MM-DD.md  (also printed to stdout for the task log)
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from src import config, learning, performance, recommend, regime, storage


def _section(lines: list[str], title: str) -> None:
    lines.append(f"\n## {title}\n")


def build() -> str:
    lines: list[str] = [f"# Claude Trader — weekly digest, {date.today().isoformat()}",
                        "_Advisory only; not financial advice._"]

    # 1. Fired triggers (the most important news first)
    fired = storage.recent_fired_triggers(days=7)
    _section(lines, "🚨 Thesis-invalidation triggers fired (7 days)")
    if fired:
        for t in fired:
            lines.append(f"- **{t['symbol']}** — {t.get('trigger_detail','')} · "
                         f"_{t.get('description','')}_ ({str(t.get('triggered_at',''))[:10]})")
    else:
        lines.append("- none 🎉")

    # 2. Regime
    r = regime.detect(use_cache=True)
    _section(lines, "🌡️ Market regime")
    if r.get("regime"):
        lines.append(f"- **{r['regime'].replace('_','-')}** · {r.get('benchmark','SPY')} "
                     f"{r.get('bench_vs_200dma_pct')}% vs 200dma · 20d vol "
                     f"{r.get('bench_vol20_ann_pct')}%")
        if r.get("weak_subsectors"):
            lines.append(f"- weak sub-sectors: {', '.join(r['weak_subsectors'])}")
    else:
        lines.append("- unavailable")

    # 3. Rating changes vs ~a week ago (compare latest two deep runs >=5 days apart)
    _section(lines, "🔄 Board rating changes (vs ~1 week ago)")
    runs = [x for x in storage.list_runs(60) if x["mode"] == "deep" and x["status"] == "ok"]
    changed_any = False
    if len(runs) >= 2:
        cur = runs[0]
        prev = next((x for x in runs[1:] if (datetime.fromisoformat(cur["started_at"])
                     - datetime.fromisoformat(x["started_at"])).days >= 5), runs[-1])
        cur_b = {x["symbol"]: x for x in storage.get_board(cur["id"])}
        prev_b = {x["symbol"]: x for x in storage.get_board(prev["id"])}
        for sym, row in cur_b.items():
            if sym not in prev_b:
                continue
            cl, _ = recommend.recommendation(row)
            pl, _ = recommend.recommendation(prev_b[sym])
            if cl != pl:
                lines.append(f"- **{sym}**: {pl} → **{cl}**")
                changed_any = True
    if not changed_any:
        lines.append("- no rating changes")

    # 4. Upcoming catalysts / earnings (14 days)
    _section(lines, "📅 Next 14 days")
    latest = storage.latest_run("deep")
    added = 0
    if latest:
        for row in storage.get_board(latest["id"]):
            m = row.get("metrics") or {}
            dte = m.get("days_to_earnings")
            if isinstance(dte, (int, float)) and 0 <= dte <= 14:
                held = " 📌held" if row.get("held_position") else ""
                lines.append(f"- {m.get('earnings_date')}: **{row['symbol']}** earnings{held}")
                added += 1
    pf = storage.latest_portfolio_run("ok")
    if pf:
        for p in storage.get_portfolio(pf["id"])["positions"]:
            for c in (p.get("catalysts_json") or []):
                if not isinstance(c, dict):
                    continue
                ds = str(c.get("date", ""))
                try:  # "YYYY-MM-DD", or "YYYY-MM" treated as month start
                    d = (datetime.strptime(ds[:10], "%Y-%m-%d").date() if len(ds) >= 10
                         else datetime.strptime(ds[:7] + "-01", "%Y-%m-%d").date())
                except ValueError:
                    continue
                if 0 <= (d - date.today()).days <= 14:  # a real 14-day window,
                    lines.append(f"- {ds}: **{p['symbol']}** — "  # not "this month"
                                 f"{str(c.get('event',''))[:100]} 📌held")
                    added += 1
    if not added:
        lines.append("- nothing dated in the window")

    # 5. Latest PM targets (what the tool wants your portfolio to look like)
    _section(lines, "🎯 Latest PM rebalance targets")
    if pf:
        with storage.get_conn() as cn:
            rows = cn.execute(
                """SELECT symbol, action, conviction, weight_pct, target_weight_pct
                   FROM portfolio_positions WHERE run_id=? ORDER BY weight_pct DESC""",
                (pf["id"],)).fetchall()
        for x in rows:
            if x["weight_pct"] is None:
                continue
            lines.append(f"- **{x['symbol']}**: {x['action']} · conv {x['conviction']} · "
                         f"{x['weight_pct']:.0f}% → {x['target_weight_pct'] or 0:.0f}%")
    else:
        lines.append("- no portfolio deep-dive yet")

    # 6. Calibration snapshot
    _section(lines, "📏 Calibration (excess vs SPY, complete windows only)")
    try:
        cal = performance.rating_calibration()
        if cal is not None and not cal.empty:
            for _, row in cal.iterrows():
                bits = [f"{row['rating']}: {row['picks']} picks"]
                for w in ("1w", "4w", "13w"):
                    if row.get(f"avg_excess_{w}") is not None:
                        bits.append(f"{w} avg {row[f'avg_excess_{w}']:+.1f}% "
                                    f"(hit {row[f'hit_{w}']}%)")
                lines.append("- " + " · ".join(bits))
        else:
            lines.append("- no graded picks yet")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"- unavailable: {exc}")

    # 7. Learning loop: weekly reflection over graded calls -> durable lessons
    _section(lines, "🧠 Learning (reflection on graded calls)")
    try:
        refl = learning.reflect()
        if refl.get("lessons"):
            lines.append(f"_{refl.get('summary', '')}_")
            for ls in refl["lessons"]:
                tag = f"[{ls['symbol']}] " if ls.get("symbol") else ""
                lines.append(f"- {tag}{ls['lesson']}")
        else:
            lines.append(f"- {refl.get('summary', 'no new lessons')}")
        active = storage.active_lessons(scope="global", limit=8)
        if active:
            lines.append("\n**Active lesson set (injected into all research prompts):**")
            lines.extend(f"- {ls['lesson']}" for ls in active)
    except Exception as exc:  # noqa: BLE001
        lines.append(f"- reflection unavailable: {exc}")

    # 8. Pillar predictive power (the light run's learning signal)
    _section(lines, "⚖️ Pillar information coefficients (4w fwd excess)")
    try:
        ic = learning.pillar_ic()
        if ic.get("ic"):
            lines.append(f"- n={ic['n']} graded rows · IC: technical {ic['ic']['technical']}, "
                         f"fundamental {ic['ic']['fundamental']}, "
                         f"sentiment {ic['ic']['sentiment']}")
            lines.append(f"- current weights: {ic['current_weights']}")
            lines.append(f"- **suggested weights**: {ic['suggested_weights']} "
                         f"({ic.get('note','')})")
        else:
            lines.append(f"- {ic.get('note', 'not enough graded data yet')}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"- unavailable: {exc}")

    return "\n".join(lines) + "\n"


def main() -> None:
    storage.init_db()
    text = build()
    outdir = Path(config.abs_path("data")) / "digests"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"digest_{date.today().isoformat()}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"[digest] written to {out}")


if __name__ == "__main__":
    main()
