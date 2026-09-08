"""Outcome-feedback learning loop ("RL for prompts, statistics for the quant screen").

The models' weights can't be trained, but the SYSTEM can learn:
  1. grade_calls()          — every past rating gets its realized forward excess return
  2. symbol_track_record()  — a name's own prior calls + outcomes, injected when
                              re-researching it (self-consistency pressure)
  3. calibration_line()     — global "how biased have I been" line for every prompt
  4. reflect()              — weekly Claude pass over the worst/best graded calls,
                              distilling durable LESSONS (stored, injected into prompts)
  5. pillar_ic()            — information coefficient of each quant pillar vs realized
                              returns; suggests evidence-based pillar re-weights

The light run learns through (5); the deep run through (2)+(3)+(4).
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from . import config, performance, storage

_GRADE_WINDOW = "4w"       # preferred grading window
_FALLBACK_WINDOW = "1w"    # used while 4w windows are still filling


def graded_calls(symbol: str | None = None) -> list[dict[str, Any]]:
    """All research calls with realized outcomes, newest first.

    Grade = excess vs the EQUAL-WEIGHT UNIVERSE MEAN (selection skill), falling back
    to SMH then SPY when the universe benchmark is unavailable. Grading vs SPY alone
    on an SMH-beta>1 universe would teach the loop beta timing, not stock picking.
    """
    rows = performance.pick_outcomes()
    out = []
    for r in rows:
        if symbol and r["symbol"] != symbol.upper():
            continue
        grade, window, basis = None, None, None
        for w in (_GRADE_WINDOW, _FALLBACK_WINDOW):
            for pref, b in (("excess_uni_", "universe"),
                            ("excess_theme_", performance.risk._theme()),
                            ("excess_", "SPY")):
                v = r.get(f"{pref}{w}")
                if v is not None:
                    grade, window, basis = v, w, b
                    break
            if grade is not None:
                break
        if grade is None:
            continue  # too fresh to grade
        out.append({**r, "grade_excess": grade, "grade_window": window,
                    "grade_basis": basis})
    out.sort(key=lambda x: x["run_id"], reverse=True)
    return out


def symbol_track_record(symbol: str, max_n: int = 5) -> str:
    """Prompt block: this name's own prior calls and what happened after each."""
    calls = graded_calls(symbol)[:max_n]
    if not calls:
        return ""
    lines = [f"YOUR PRIOR CALLS ON {symbol.upper()} (excess vs the screened universe "
             f"mean — i.e. selection skill, not market beta):"]
    for c in calls:
        when = str(c.get("surfaced_at", ""))[:10]
        rating = c.get("rating") or "screen-only"
        conv = f" conv {c['conviction']}" if c.get("conviction") is not None else ""
        lines.append(f"  - {when}: {rating}{conv} @ ${c.get('price_at_surface')} -> "
                     f"{c['grade_excess']:+.1f}% vs {c.get('grade_basis','universe')} "
                     f"over {c['grade_window']}")
    lines.append("Weigh what your previous theses got right or wrong before rating again.")
    return "\n".join(lines)


def portfolio_action_outcomes(symbol: str | None = None) -> list[dict[str, Any]]:
    """Grade past portfolio deep-dive ACTIONS (add/hold/trim/exit): the holding's
    excess return vs SMH (theme benchmark; SPY fallback) after each dive. 'Add'
    graded well = positive excess; 'trim/exit' graded well = NEGATIVE excess (you
    avoided a laggard). 'Hold' is graded with a ±3pp dead band — a hold that merely
    drifts with the theme is neither right nor wrong."""
    with storage.get_conn() as cn:
        rows = [dict(r) for r in cn.execute(
            """SELECT p.run_id, p.symbol, p.action, p.conviction, p.last,
                      p.weight_pct, p.target_weight_pct, r.finished_at
               FROM portfolio_positions p
               JOIN portfolio_runs r ON r.id = p.run_id
               WHERE r.status='ok' AND p.action IS NOT NULL
               ORDER BY p.run_id""")]
    rows = [r for r in rows if not symbol or r["symbol"] == symbol.upper()]
    if not rows:
        return []
    syms = sorted({r["symbol"] for r in rows} | {"SPY", "SMH"})
    t_all = []
    for r in rows:
        try:
            t_all.append(pd.Timestamp(datetime.fromisoformat(r["finished_at"])
                                      .astimezone(timezone.utc).date()))
        except (ValueError, TypeError):
            t_all.append(None)
    oldest = min((t for t in t_all if t is not None), default=None)
    px = performance._price_history(syms, period=performance._period_for(oldest))
    if px is None:
        return []
    out = []
    for r, t0 in zip(rows, t_all):
        if t0 is None:
            continue
        graded = None
        window = None
        for label, days in (("4w", 21), ("1w", 5)):
            fr = performance._fwd_return(px, r["symbol"], t0, days)
            # Theme benchmark first (holdings are AI-cluster); SPY as fallback.
            br = performance._fwd_return(px, "SMH", t0, days)
            if br is None:
                br = performance._fwd_return(px, "SPY", t0, days)
            if fr is not None and br is not None:
                graded, window = round(fr - br, 2), label
                break
        if graded is None:
            continue
        act = (r.get("action") or "").lower()
        # add wants positive excess; trim/exit want negative (avoided a laggard);
        # hold uses a ±3pp dead band — theme drift is neither right nor wrong.
        if act == "hold":
            right = abs(graded) < 3.0 or graded >= 0
        elif act == "add":
            right = graded >= 0
        else:  # trim / exit
            right = graded <= 0
        out.append({**r, "grade_excess": graded, "grade_window": window, "was_right": right})
    out.sort(key=lambda x: x["run_id"], reverse=True)
    return out


def portfolio_track_record(symbol: str, max_n: int = 4) -> str:
    """Prompt block: prior PORTFOLIO actions on this holding and how they graded."""
    calls = portfolio_action_outcomes(symbol)[:max_n]
    if not calls:
        return ""
    lines = [f"YOUR PRIOR PORTFOLIO ACTIONS ON {symbol.upper()} (graded vs SPY):"]
    for c in calls:
        when = str(c.get("finished_at", ""))[:10]
        verdict = "RIGHT" if c["was_right"] else "WRONG"
        tw = (f" (target {c['weight_pct']:.0f}%->{c['target_weight_pct']:.0f}%)"
              if c.get("weight_pct") is not None and c.get("target_weight_pct") is not None
              else "")
        lines.append(f"  - {when}: {c['action']} conv {c['conviction']}{tw} -> "
                     f"{c['grade_excess']:+.1f}% vs SPY over {c['grade_window']} "
                     f"[{verdict} call]")
    return "\n".join(lines)


def portfolio_batting_average() -> str:
    """One line: how often past portfolio actions were right, by action type."""
    calls = portfolio_action_outcomes()
    if len(calls) < 4:
        return ""
    by_act: dict[str, list[bool]] = {}
    for c in calls:
        by_act.setdefault(c["action"], []).append(c["was_right"])
    bits = [f"{a}: {sum(v)}/{len(v)} right" for a, v in sorted(by_act.items())]
    return "YOUR PORTFOLIO-ACTION TRACK RECORD - " + "; ".join(bits) + "."


def calibration_line() -> str:
    """One honest line about the system's own recent bias, for every research prompt.

    Uses the UNIVERSE-relative calibration (selection skill). The explicit bias
    instruction is gated on 4-week windows only: 1-week grades are mostly noise and
    must not flip the prompt's tone week to week.
    """
    try:
        cal = performance.rating_calibration("uni")
    except Exception:  # noqa: BLE001
        return ""
    if cal is None or cal.empty:
        return ""
    bits = []
    for _, row in cal.iterrows():
        for w in ("4w", "1w"):
            avg, hit, n = row.get(f"avg_excess_{w}"), row.get(f"hit_{w}"), row.get(f"n_{w}")
            if avg is not None and n:
                bits.append(f"{row['rating']}: {avg:+.1f}% vs universe ({w}, n={n}, "
                            f"hit {hit}%)")
                break
    if not bits:
        return ""
    line = ("YOUR RECENT TRACK RECORD (excess vs the screened-universe average, i.e. "
            "selection skill) - " + "; ".join(bits) + ".")
    # Bias instruction ONLY on 4w+ evidence with a real sample.
    buy_rows = cal[cal["rating"].isin(["buy", "strong_buy"])]
    if "avg_excess_4w" in buy_rows:
        vals = buy_rows["avg_excess_4w"].dropna()
        n4 = buy_rows["n_4w"].fillna(0).sum() if "n_4w" in buy_rows else 0
        if len(vals) and n4 >= 20 and vals.mean() < -1:
            line += (" Your buy ratings have been UNDERPERFORMING the universe - be more "
                     "selective, demand stronger evidence, and prefer hold when uncertain.")
    return line


def learning_block(symbol: str | None = None) -> str:
    """Composite injection block: calibration + global lessons + symbol history
    (both research calls AND portfolio actions, when the name has them)."""
    parts = []
    cal = calibration_line()
    if cal:
        parts.append(cal)
    lessons = storage.active_lessons(scope="global", limit=5)
    if symbol:
        lessons += storage.active_lessons(scope="symbol", symbol=symbol, limit=3)
    if lessons:
        parts.append("LESSONS FROM YOUR PAST MISSES (apply them):")
        parts.extend(f"  - {ls['lesson']}" for ls in lessons)
    if symbol:
        tr = symbol_track_record(symbol)
        if tr:
            parts.append(tr)
        ptr = portfolio_track_record(symbol)
        if ptr:
            parts.append(ptr)
    return "\n".join(parts)


def portfolio_learning_block() -> str:
    """Portfolio-level injection for the synthesis / PM-reconcile stages:
    global calibration + action batting average + lessons (no single-symbol focus)."""
    parts = []
    ba = portfolio_batting_average()
    if ba:
        parts.append(ba)
    cal = calibration_line()
    if cal:
        parts.append(cal)
    lessons = storage.active_lessons(scope="global", limit=5)
    if lessons:
        parts.append("LESSONS FROM YOUR PAST MISSES (apply them):")
        parts.extend(f"  - {ls['lesson']}" for ls in lessons)
    return "\n".join(parts)


# ---------------------------------------------------------------- reflection
_REFLECT_TMPL = """Today is {today}. You are reviewing YOUR OWN past stock calls to extract
durable lessons. Below are your worst and best graded calls (excess return vs SPY over the
stated window), with the thesis you wrote at the time.

WORST CALLS:
{worst_block}

BEST CALLS:
{best_block}

For the worst calls, use web search to understand what ACTUALLY happened after your call
(what did the thesis miss - a risk, a timing signal, crowding, valuation, macro?). Then
distill 2-4 DURABLE, ACTIONABLE lessons - rules you should apply to future analysis. Good
lessons are specific and testable ("Parabolic names >80% above their 200dma have poor
forward returns even with strong fundamentals - cap conviction at 6"), not platitudes
("do better research"). Also write symbol-specific lessons where a name has a repeated
pattern.

Output ONLY one JSON object:
{{"lessons": [{{"scope": "global"|"symbol", "symbol": "<SYM or null>",
   "lesson": "<one actionable rule>", "evidence": "<which calls support it>"}}],
  "summary": "<one sentence on the main systematic error>"}}"""


def _call_block(calls: list[dict]) -> str:
    lines = []
    for c in calls:
        with storage.get_conn() as cn:
            r = cn.execute("SELECT thesis FROM research_reports WHERE run_id=? AND symbol=?",
                           (c["run_id"], c["symbol"])).fetchone()
        thesis = (r["thesis"] if r and r["thesis"] else "(screen-only, no thesis)")[:280]
        lines.append(f"- {c['symbol']} on {str(c.get('surfaced_at',''))[:10]}: "
                     f"{c.get('rating') or 'screen-only'} conv {c.get('conviction')} -> "
                     f"{c['grade_excess']:+.1f}% vs SPY ({c['grade_window']}). "
                     f"Thesis then: {thesis}")
    return "\n".join(lines) or "(none)"


def reflect(n_worst: int = 4, n_best: int = 2) -> dict[str, Any]:
    """Weekly self-review: distill lessons from the most-wrong and most-right calls."""
    from . import deep_research
    calls = graded_calls()
    # Prefer 4w-graded calls (1w grades are mostly noise); dedupe per symbol on the
    # most extreme graded call.
    solid = [c for c in calls if c["grade_window"] == _GRADE_WINDOW] or calls
    best_by_sym: dict[str, dict] = {}
    for c in solid:
        cur = best_by_sym.get(c["symbol"])
        if cur is None or abs(c["grade_excess"]) > abs(cur["grade_excess"]):
            best_by_sym[c["symbol"]] = c
    ranked = sorted(best_by_sym.values(), key=lambda x: x["grade_excess"])
    # Guard the right bound: worst and best must never overlap.
    worst = ranked[:n_worst]
    best = ranked[-n_best:] if len(ranked) >= n_worst + n_best else []
    if not worst:
        return {"lessons": [], "summary": "not enough graded calls yet"}
    prompt = _REFLECT_TMPL.format(today=date.today().isoformat(),
                                  worst_block=_call_block(worst),
                                  best_block=_call_block(best))
    parsed, _ = deep_research._run_json(prompt)
    if not parsed or not isinstance(parsed.get("lessons"), list):
        return {"lessons": [], "summary": "reflection parse failed"}
    clean = [{"scope": ls.get("scope", "global"),
              "symbol": ls.get("symbol"),
              "lesson": str(ls.get("lesson", ""))[:400],
              "evidence": str(ls.get("evidence", ""))[:300]}
             for ls in parsed["lessons"] if isinstance(ls, dict) and ls.get("lesson")]
    if clean:
        # Tag with the regime they were learned in — a lesson from a rout may not
        # generalize to a rally; the tag travels with it and age-expiry retires it.
        from . import regime as _regime
        reg = (_regime.detect(use_cache=True) or {}).get("regime")
        storage.add_lessons(clean, regime=reg)
        storage.retire_old_lessons(keep_latest=12)
    return {"lessons": clean, "summary": parsed.get("summary", "")}


# ---------------------------------------------------------------- pillar IC
_PILLARS = ("technical", "fundamental", "sentiment")


def pillar_ic() -> dict[str, Any]:
    """Information coefficient per pillar, computed Fama-MacBeth style:
    cross-sectional Spearman per run DATE (full universe, one deep run per day),
    on NON-OVERLAPPING dates >=21 trading days apart, then averaged across dates
    with a t-stat. Only rows from the CURRENT score version count (a pillar whose
    definition changed is a different signal). Suggested re-weights are printed
    only when at least one pillar's |t| >= 2 — otherwise the honest answer is
    'not enough independent data yet'."""
    from .scoring import effective_score_version, pillar_set
    sv = effective_score_version()
    ps = pillar_set()
    with storage.get_conn() as cn:
        # (score_version, pillar_set) is the partition key. The version catches a
        # changed pillar DEFINITION; pillar_set catches a changed pillar COUNT, which
        # moves the composite's dispersion without changing any pillar. Rows written
        # before the column existed have pillar_set NULL and are matched only when the
        # current set is the historical three-pillar one.
        rows = [dict(r) for r in cn.execute(
            """SELECT s.run_id, s.symbol, s.metrics_json, r.started_at
               FROM screen_scores s JOIN runs r ON r.id=s.run_id
               WHERE r.status='ok' AND r.mode='deep' AND s.score_version=?
                 AND (s.pillar_set=? OR (s.pillar_set IS NULL AND ?=?))
               ORDER BY s.run_id""",
            (sv, ps, ps, "technical,fundamental,sentiment"))]
    if not rows:
        return {"note": f"no rows at score_version={sv} / pillars '{ps}' yet "
                        f"(accumulates from the next deep runs)"}
    recs = []
    for r in rows:
        try:
            m = json.loads(r["metrics_json"] or "{}")
        except json.JSONDecodeError:
            continue
        recs.append({"run_id": r["run_id"], "symbol": r["symbol"],
                     "started_at": str(r["started_at"])[:10],
                     **{pl: m.get(f"pillar_{pl}") for pl in _PILLARS}})
    df = pd.DataFrame(recs)
    # Require only the pillars that actually carry data. A newly added pillar has no
    # history, and dropna() on it would discard every pre-existing row and silently
    # report an empty study rather than an honest "this pillar has no history yet".
    present = [pl for pl in _PILLARS if pl in df.columns and df[pl].notna().any()]
    skipped = [pl for pl in _PILLARS if pl not in present]
    df = df.dropna(subset=present) if present else df.iloc[0:0]
    if df.empty:
        return {"note": "no scored rows with pillar values yet"}

    # Non-overlapping run dates: keep dates >= 30 calendar days (~21 trading) apart.
    dates = sorted(df["started_at"].unique())
    kept, last = [], None
    for d in dates:
        dd = pd.Timestamp(d)
        if last is None or (dd - last).days >= 30:
            kept.append(d)
            last = dd
    df = df[df["started_at"].isin(kept)]

    syms = sorted(df["symbol"].unique().tolist())
    px = performance._price_history(
        syms, period=performance._period_for(pd.Timestamp(kept[0]) if kept else None))
    if px is None:
        return {"note": "price history unavailable"}

    # Per-date cross-sectional Spearman IC (raw fwd returns: subtracting the
    # universe-mean constant per date does not change ranks).
    per_date: dict[str, list[float]] = {p: [] for p in present}
    dates_used = 0
    for d, g in df.groupby("started_at"):
        t0 = pd.Timestamp(d)
        fwd = {row.symbol: performance._fwd_return(px, row.symbol, t0, 21)
               for row in g.itertuples()}
        gg = g.assign(fwd=[fwd.get(s) for s in g["symbol"]]).dropna(subset=["fwd"])
        if len(gg) < 20:
            continue  # too few names for a meaningful cross-section
        fwd_rank = gg["fwd"].rank()
        for p in present:
            ic = float(gg[p].rank().corr(fwd_rank))
            if ic == ic:  # not NaN
                per_date[p].append(ic)
        dates_used += 1
    if dates_used < 2:
        return {"n_dates": dates_used,
                "note": "need >=2 non-overlapping run dates with complete 4w windows "
                        "(dates must be ~1 month apart) - keep accumulating"}

    import math
    ics, tstats = {}, {}
    for p in present:
        v = per_date[p]
        if not v:
            ics[p], tstats[p] = None, None
            continue
        mean = sum(v) / len(v)
        sd = (sum((x - mean) ** 2 for x in v) / max(len(v) - 1, 1)) ** 0.5
        ics[p] = round(mean, 3)
        tstats[p] = round(mean / (sd / math.sqrt(len(v))), 2) if sd > 0 else None

    cur = config.get("scoring.pillars",
                     {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.35})
    out: dict[str, Any] = {"n_dates": dates_used, "ic": ics, "t_stat": tstats,
                           "current_weights": cur, "score_version": sv,
                           "pillar_set": ps, "pillars_measured": present}
    if skipped:
        out["pillars_no_history"] = skipped
    if any(t is not None and abs(t) >= 2 for t in tstats.values()):
        pos = {p: max(v, 0.05) for p, v in ics.items() if v is not None}
        tot = sum(pos.values())
        out["suggested_weights"] = {p: round(0.6 * cur.get(p, 0.33) + 0.4 * (pos[p] / tot), 2)
                                    for p in pos}
        out["note"] = "suggestion only - update config.yaml scoring.pillars manually"
    else:
        out["note"] = (f"no pillar is statistically significant yet "
                       f"(|t|<2 on {dates_used} independent dates) - no re-weight suggested")
    return out


if __name__ == "__main__":
    print("=== calibration line ===")
    print(calibration_line() or "(none yet)")
    print("\n=== NVDA track record ===")
    print(symbol_track_record("NVDA") or "(none yet)")
    print("\n=== pillar IC ===")
    print(json.dumps(pillar_ic(), indent=2))
    print("\n=== active lessons ===")
    for ls in storage.active_lessons():
        print(f"  [{ls['scope']}] {ls['lesson']}")
