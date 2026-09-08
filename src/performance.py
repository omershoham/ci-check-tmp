"""Calibration tracking: does the system actually predict anything?

Grading design (post CR#2):
  * BASE PRICE — a pick surfaces pre-market, so it is graded from `price_at_surface`
    (the snapshot taken at pick time ≈ previous close); benchmarks are graded from the
    previous close before the surface date. Grading from the surface-day close would
    exclude the day-0 move the pick most predicts (lookahead-ish).
  * BENCHMARK — primary excess is vs the EQUAL-WEIGHT MEAN of that run's own scored
    universe (`excess_uni_*`): pure cross-sectional selection skill, immune to theme
    beta. The theme proxy (`excess_theme_*`, config-driven) and SPY (`excess_*`, legacy
    name kept for the dashboard)
    are computed as secondary references. Grading vs SPY alone on an SMH-beta>1
    universe would teach the learning loop beta timing and call it skill.
  * WINDOW GUARD — if the price history doesn't cover the surface date (searchsorted
    clamping to bar 0), the grade is rejected instead of silently wrong.
  * All yfinance downloads are memoized per (symbols, period, day) — the learning
    blocks call into this per finalist and must not trigger dozens of batch downloads.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf

from . import config, risk, storage

_WINDOWS = {"1w": 5, "4w": 21, "13w": 65}  # trading days
# NEVER change _WINDOWS. rating_calibration, calibration_table, claude_vs_screen,
# calibration_by_prompt_version, learning.graded_calls and learning.reflect all read
# these three keys; a pick's own-horizon grade is ADDED next to them (the *_h keys),
# never in place of them, so the fixed-window series stays unbroken across the cutover.
# The theme proxy is config-driven; risk._theme() is the single resolver
# (risk.theme_benchmark -> technical.rs_benchmark -> SMH).

# ------------------------------------------------------------- memoized data
_MEMO: dict[tuple, Any] = {}


def _memo_key(*parts) -> tuple:
    """Day-scoped memo key. Also evicts yesterday's entries: the dashboard is a
    long-lived process, and without eviction every day's price frames stayed
    resident forever."""
    from datetime import date
    today = date.today().isoformat()
    stale = [k for k in _MEMO if k and k[-1] != today]
    for k in stale:
        _MEMO.pop(k, None)
    return (*parts, today)


def _cost_pct(rating: str | None) -> float:
    """Round-trip transaction cost charged to a graded call, in % of position.

    A buy/strong_buy implies a trade (and its eventual exit); hold and screen-only
    picks are the do-nothing alternative and cost nothing. Config `evaluation.cost_bps`
    (default 30 = 0.30% round trip: spread + slippage on a liquid US large-cap)."""
    if str(rating or "").lower() in ("buy", "strong_buy"):
        return float(config.get("evaluation.cost_bps", 30)) / 100.0
    return 0.0


def _horizon_days(rec: dict) -> int | None:
    """Trading days for a pick's OWN stated horizon, from `evaluation.horizons`.

    None when the pick never claimed one (every pre-Phase-6 pick, and the benchmark
    snapshot row). Those are graded on the fixed windows only — assigning them a
    default clock here would fabricate a decision the analyst never made."""
    h = str(rec.get("horizon") or "").strip().lower()
    if not h:
        return None
    try:
        d = int((config.get("evaluation.horizons", {}) or {}).get(h))
    except (TypeError, ValueError):
        return None
    return d if d > 0 else None


def _price_history(symbols: list[str], period: str = "1y") -> pd.DataFrame | None:
    key = _memo_key("px", tuple(sorted(set(symbols))), period)
    if key in _MEMO:
        return _MEMO[key]
    try:
        px = yf.download(list(dict.fromkeys(symbols)), period=period, interval="1d",
                         auto_adjust=True, progress=False, threads=True)["Close"]
        if isinstance(px, pd.Series):
            px = px.to_frame(symbols[0])
        px = px.dropna(how="all")
        _MEMO[key] = px
        return px
    except Exception:  # noqa: BLE001
        return None


def _period_for(oldest: pd.Timestamp | None) -> str:
    """Pick a yfinance period that safely covers the oldest date being graded."""
    if oldest is None:
        return "1y"
    days = (pd.Timestamp.now() - oldest).days + 100  # 13w window + margin
    for period, cap in (("6mo", 170), ("1y", 350), ("2y", 715), ("5y", 1800)):
        if days <= cap:
            return period
    return "10y"


def _fwd_return(px: pd.DataFrame, sym: str, start: pd.Timestamp, days: int,
                base_override: float | None = None) -> float | None:
    """Forward return over `days` trading days from the previous close before `start`
    (or from `base_override`, e.g. the pick-time snapshot). None if the window is
    incomplete OR the history doesn't actually cover `start` (clamp guard)."""
    if sym not in px.columns:
        return None
    s = px[sym].dropna()
    if s.empty:
        return None
    idx = s.index.searchsorted(start)
    if idx >= len(s):
        return None
    # Clamp guard: the landed bar must be near the surface date, not the window edge.
    if abs((s.index[idx] - start).days) > 7:
        return None
    base_idx = max(idx - 1, 0)
    base = base_override if base_override else float(s.iloc[base_idx])
    j = base_idx + days
    if j >= len(s):
        return None  # window not complete yet — don't grade in-flight picks
    return (float(s.iloc[j]) / base - 1) * 100 if base else None


# Public name for the ONE forward-return convention in the system. The walk-forward
# backtest harness (plan Track C) must grade with this same function so a signal
# validated there and consumed here is measured identically (prior-close base,
# clamp guard, incomplete windows rejected).
label_return = _fwd_return


# ------------------------------------------------------------- pick grading
def _universe_by_run() -> dict[int, list[str]]:
    key = _memo_key("uni_by_run")
    if key in _MEMO:
        return _MEMO[key]
    with storage.get_conn() as cn:
        rows = cn.execute("SELECT run_id, symbol FROM screen_scores").fetchall()
    out: dict[int, list[str]] = {}
    for r in rows:
        out.setdefault(r["run_id"], []).append(r["symbol"])
    _MEMO[key] = out
    return out


def _uni_mean_fwd(px: pd.DataFrame, run_id: int, t0: pd.Timestamp,
                  days: int) -> float | None:
    """Equal-weight mean forward return of the run's scored universe (memoized)."""
    key = _memo_key("uni_mean", run_id, days)
    if key in _MEMO:
        return _MEMO[key]
    syms = _universe_by_run().get(run_id, [])
    rets = [r for r in (_fwd_return(px, s, t0, days) for s in syms) if r is not None]
    val = sum(rets) / len(rets) if len(rets) >= 10 else None
    _MEMO[key] = val
    return val


def pick_outcomes() -> list[dict[str, Any]]:
    """Per-pick forward returns with excess vs universe mean / theme / SPY.

    Two clocks per pick, always both: the three FIXED windows (unchanged forever) and,
    when the pick stated a horizon, the same measurements over that horizon's own day
    count (`*_h`). Legacy picks carry horizon=None and every `*_h` key None."""
    key = _memo_key("pick_outcomes")
    if key in _MEMO:
        return _MEMO[key]
    bench = config.get("technical.benchmark", "SPY")
    with storage.get_conn() as cn:
        rows = [dict(r) for r in cn.execute(
            """SELECT p.run_id, p.symbol, p.surfaced_at, p.price_at_surface,
                      p.horizon, p.horizon_source,
                      r.rating, r.conviction, r.prompt_version, r.model,
                      s.blended, s.rank, ru.regime
               FROM picks p
               LEFT JOIN research_reports r ON r.run_id=p.run_id AND r.symbol=p.symbol
               LEFT JOIN screen_scores s ON s.run_id=p.run_id AND s.symbol=p.symbol
               LEFT JOIN runs ru ON ru.id=p.run_id
               ORDER BY p.run_id""")]
    picks = [r for r in rows if r["symbol"] != bench]
    if not picks:
        _MEMO[key] = []
        return []
    # Universe symbols are needed for the equal-weight benchmark.
    uni_syms = {s for syms in _universe_by_run().values() for s in syms}
    theme = risk._theme()
    all_syms = sorted({r["symbol"] for r in picks} | uni_syms | {bench, theme})
    t_all = []
    for r in picks:
        try:
            t_all.append(pd.Timestamp(datetime.fromisoformat(r["surfaced_at"])
                                      .astimezone(timezone.utc).date()))
        except (ValueError, TypeError):
            t_all.append(None)
    oldest = min((t for t in t_all if t is not None), default=None)
    px = _price_history(all_syms, period=_period_for(oldest))
    if px is None:
        _MEMO[key] = []
        return []
    out = []
    for r, t0 in zip(picks, t_all):
        if t0 is None:
            continue
        rec = {**r}
        base = r.get("price_at_surface")
        base = float(base) if isinstance(base, (int, float)) and base > 0 else None
        for label, days in _WINDOWS.items():
            fr = _fwd_return(px, r["symbol"], t0, days, base_override=base)
            spy = _fwd_return(px, bench, t0, days)
            thm = _fwd_return(px, theme, t0, days)
            uni = _uni_mean_fwd(px, r["run_id"], t0, days)
            rec[f"ret_{label}"] = round(fr, 2) if fr is not None else None
            rec[f"excess_{label}"] = (round(fr - spy, 2)
                                      if fr is not None and spy is not None else None)
            rec[f"excess_theme_{label}"] = (round(fr - thm, 2)
                                          if fr is not None and thm is not None else None)
            rec[f"excess_uni_{label}"] = (round(fr - uni, 2)
                                          if fr is not None and uni is not None else None)
        # Net of the cost a real trade would have paid; hold/screen-only cost 0.
        rec["cost_pct"] = _cost_pct(r.get("rating"))
        for label in _WINDOWS:
            e = rec.get(f"excess_uni_{label}")
            rec[f"net_excess_uni_{label}"] = round(e - rec["cost_pct"], 2) if e is not None else None
        # ---- the pick's OWN clock, computed alongside the fixed windows.
        # _uni_mean_fwd memoizes on (run_id, days), so an arbitrary horizon costs one
        # extra universe pass per (run, horizon) and nothing thereafter.
        rec["horizon_source"] = r.get("horizon_source") or "legacy"
        hd = _horizon_days(rec)
        rec["horizon_days"] = hd
        for k in ("ret_h", "excess_uni_h", "excess_bench_h", "excess_theme_h",
                  "net_excess_uni_h"):
            rec[k] = None
        if hd:
            frh = _fwd_return(px, r["symbol"], t0, hd, base_override=base)
            spyh = _fwd_return(px, bench, t0, hd)
            thmh = _fwd_return(px, theme, t0, hd)
            unih = _uni_mean_fwd(px, r["run_id"], t0, hd)
            if frh is not None:
                rec["ret_h"] = round(frh, 2)
                if spyh is not None:
                    rec["excess_bench_h"] = round(frh - spyh, 2)
                if thmh is not None:
                    rec["excess_theme_h"] = round(frh - thmh, 2)
                if unih is not None:
                    rec["excess_uni_h"] = round(frh - unih, 2)
                    rec["net_excess_uni_h"] = round(frh - unih - rec["cost_pct"], 2)
        out.append(rec)
    _MEMO[key] = out
    return out


def _bucket(r: dict) -> str:
    rating = r.get("rating")
    rating = rating.lower() if isinstance(rating, str) else ""  # NULL -> NaN in pandas
    if rating in ("strong_buy", "buy", "hold", "sell", "strong_sell"):
        return rating
    return "screen_only"


_HORIZON_ORDER = ("swing", "position", "long")


def rating_calibration(basis: str = "uni", clock: str = "fixed") -> pd.DataFrame:
    """Aggregate pick outcomes by rating bucket. basis: 'uni' (selection skill,
    default), 'smh' (theme-relative) or 'spy' (market-relative).

    clock='fixed' (default) reports the three fixed windows and is byte-identical to
    the pre-Phase-6 output. clock='horizon' grades each pick on the horizon it CLAIMED
    and reports MARGINALLY - by horizon alone first, then horizon x rating. The deeper
    cuts wait for sample size: horizon x rating x conviction x regime is ~81 cells,
    which needs ~650 graded picks per horizon before any of them means anything."""
    rows = pick_outcomes()
    if not rows:
        return pd.DataFrame()
    if clock == "horizon":
        return _horizon_calibration(rows, basis)
    # "smh" accepted as a legacy alias for the now-configurable theme basis.
    prefix = {"uni": "excess_uni_", "theme": "excess_theme_",
              "smh": "excess_theme_", "spy": "excess_"}[basis]
    df = pd.DataFrame(rows)
    df["bucket"] = df.apply(_bucket, axis=1)
    agg_rows = []
    order = ["strong_buy", "buy", "hold", "sell", "strong_sell", "screen_only"]
    for b in order:
        g = df[df["bucket"] == b]
        if g.empty:
            continue
        row: dict[str, Any] = {"rating": b, "picks": len(g)}
        for label in _WINDOWS:
            col = f"{prefix}{label}"
            e = g[col].dropna() if col in g else pd.Series(dtype=float)
            row[f"n_{label}"] = len(e)
            row[f"avg_excess_{label}"] = round(e.mean(), 2) if len(e) else None
            row[f"hit_{label}"] = round((e > 0).mean() * 100) if len(e) else None
            if basis == "uni":
                ncol = f"net_excess_uni_{label}"
                ne = g[ncol].dropna() if ncol in g else pd.Series(dtype=float)
                row[f"net_avg_excess_{label}"] = round(ne.mean(), 2) if len(ne) else None
        agg_rows.append(row)
    return pd.DataFrame(agg_rows)


# The own-clock counterpart of the excess_* prefixes above. One mapping, so the two
# clocks can never drift into different benchmark conventions.
_HORIZON_BASIS = {"uni": "excess_uni_h", "theme": "excess_theme_h",
                  "smh": "excess_theme_h", "spy": "excess_bench_h"}


def _horizon_calibration(rows: list[dict], basis: str) -> pd.DataFrame:
    """Own-clock calibration: one row per horizon ('(all)' ratings), then per rating."""
    col = _HORIZON_BASIS[basis]
    df = pd.DataFrame(rows)
    if col not in df or "horizon" not in df:
        return pd.DataFrame()
    df = df[df["horizon"].notna()]
    if df.empty:
        return pd.DataFrame()
    df["bucket"] = df.apply(_bucket, axis=1)
    order = ["strong_buy", "buy", "hold", "sell", "strong_sell", "screen_only"]
    out = []
    for h in _HORIZON_ORDER:
        gh = df[df["horizon"] == h]
        if gh.empty:
            continue
        cuts = [("(all)", gh)] + [(b, gh[gh["bucket"] == b]) for b in order]
        for label, g in cuts:
            if g.empty:
                continue
            e = g[col].dropna()
            days = g["horizon_days"].dropna()
            row: dict[str, Any] = {
                "horizon": h,
                "horizon_days": int(days.iloc[0]) if len(days) else None,
                "rating": label, "picks": len(g), "n": len(e),
                "avg_excess": round(float(e.mean()), 2) if len(e) else None,
                "hit_pct": round(float((e > 0).mean()) * 100) if len(e) else None}
            if basis == "uni":
                ne = (g["net_excess_uni_h"].dropna() if "net_excess_uni_h" in g
                      else pd.Series(dtype=float))
                row["net_avg_excess"] = round(float(ne.mean()), 2) if len(ne) else None
            out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------- calibration w/ CIs
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n (in %). Well-behaved at small n and
    at 0%/100%, which is exactly where a naive ±1.96·sqrt(p(1-p)/n) lies."""
    if n <= 0:
        return (0.0, 100.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (round(max(0.0, centre - half) * 100, 1), round(min(1.0, centre + half) * 100, 1))


def _conv_bucket(c) -> str:
    if not isinstance(c, (int, float)):
        return "n/a"
    return "<=4" if c <= 4 else ("5-6" if c < 7 else ">=7")


def calibration_table(window: str = "4w", min_n: int | None = None,
                      clock: str = "fixed") -> pd.DataFrame:
    """Precision of past calls by (rating × conviction bucket × regime at pick time),
    with Wilson 95% CIs. This is the honest form of 'how much to trust this kind of
    call' — the same numbers a meta-labeling model would learn from, shown raw until
    there is enough history for a model (plan Track E gate: ≥1,000 graded picks).

    'hit' = net excess vs the equal-weight universe > 0 (costs charged to buy calls).

    clock='horizon' grades each pick on its own stated clock and reports the marginal
    cuts first (horizon; horizon x rating); the full horizon x rating x conviction x
    regime cut is emitted ONLY for cells that already reach min_n, because publishing
    81 mostly-empty cells is how a noise pattern becomes a house rule."""
    rows = pick_outcomes()
    if not rows:
        return pd.DataFrame()
    min_n = int(config.get("evaluation.min_n_for_ci", 8)) if min_n is None else min_n
    if clock == "horizon":
        return _horizon_calibration_table(rows, min_n)
    df = pd.DataFrame(rows)
    col = f"net_excess_uni_{window}"
    if col not in df:
        return pd.DataFrame()
    df = df.dropna(subset=[col])
    if df.empty:
        return pd.DataFrame()
    df["bucket"] = df.apply(_bucket, axis=1)
    df["conv"] = df["conviction"].apply(_conv_bucket)
    df["regime"] = df["regime"].fillna("unknown")
    out = []
    for (b, cv, rg), g in df.groupby(["bucket", "conv", "regime"], sort=False):
        n = len(g)
        k = int((g[col] > 0).sum())
        lo, hi = wilson_ci(k, n)
        out.append({"rating": b, "conviction": cv, "regime": rg, "n": n,
                    "precision_pct": round(k / n * 100, 1), "ci_lo": lo, "ci_hi": hi,
                    "avg_net_excess": round(float(g[col].mean()), 2),
                    "reliable": n >= min_n})
    order = {"strong_buy": 0, "buy": 1, "hold": 2, "sell": 3, "strong_sell": 4, "screen_only": 5}
    res = pd.DataFrame(out)
    res["_o"] = res["rating"].map(order).fillna(9)
    return res.sort_values(["_o", "conviction", "regime"]).drop(columns="_o").reset_index(drop=True)


def _horizon_calibration_table(rows: list[dict], min_n: int) -> pd.DataFrame:
    col = "net_excess_uni_h"
    df = pd.DataFrame(rows)
    if col not in df or "horizon" not in df:
        return pd.DataFrame()
    df = df.dropna(subset=[col])
    df = df[df["horizon"].notna()]
    if df.empty:
        return pd.DataFrame()
    df["bucket"] = df.apply(_bucket, axis=1)
    df["conv"] = df["conviction"].apply(_conv_bucket)
    df["regime"] = df["regime"].fillna("unknown")

    def _cell(cut, h, rating, conv, reg, g) -> dict[str, Any]:
        n = len(g)
        k = int((g[col] > 0).sum())
        lo, hi = wilson_ci(k, n)
        return {"cut": cut, "horizon": h, "rating": rating, "conviction": conv,
                "regime": reg, "n": n, "precision_pct": round(k / n * 100, 1),
                "ci_lo": lo, "ci_hi": hi,
                "avg_net_excess": round(float(g[col].mean()), 2), "reliable": n >= min_n}

    out = []
    for h in _HORIZON_ORDER:
        gh = df[df["horizon"] == h]
        if gh.empty:
            continue
        out.append(_cell("horizon", h, "(all)", "(all)", "(all)", gh))
        for b, g in gh.groupby("bucket", sort=False):
            out.append(_cell("horizon x rating", h, b, "(all)", "(all)", g))
        # The full cut is reported only where it is already readable.
        for (b, cv, rg), g in gh.groupby(["bucket", "conv", "regime"], sort=False):
            if len(g) >= min_n:
                out.append(_cell("full", h, b, cv, rg, g))
    return pd.DataFrame(out)


def claude_vs_screen(window: str = "4w", n_boot: int = 2000, seed: int = 7,
                     clock: str = "fixed") -> dict[str, Any]:
    """Does the Claude layer add value over the quant screen?

    Treatment = picks Claude rated buy/strong_buy; control = picks that reached the
    finalist slots but got no research (screen_only). Both are top-of-board names
    surfaced the same way, so the difference in NET excess vs the universe is the
    marginal value of the research + red-team layer (after paying to act on it).
    Bootstrap CI on the difference of means; sign and CI are the story, not the point.

    clock='horizon' compares each cohort on its OWN stated clock. Note what that makes
    the control: screen-only picks carry the QUANT default horizon, so the horizon-clock
    comparison also carries whatever difference the two horizon assignments make."""
    import numpy as np
    rows = pick_outcomes()
    col = "net_excess_uni_h" if clock == "horizon" else f"net_excess_uni_{window}"
    treat = np.array([r[col] for r in rows if r.get(col) is not None
                      and _bucket(r) in ("buy", "strong_buy")], dtype=float)
    ctrl = np.array([r[col] for r in rows if r.get(col) is not None
                     and _bucket(r) == "screen_only"], dtype=float)
    out: dict[str, Any] = {"window": "own horizon" if clock == "horizon" else window,
                           "clock": clock, "n_claude": int(len(treat)),
                           "n_screen": int(len(ctrl))}
    if len(treat) < 5 or len(ctrl) < 5:
        out["note"] = "not enough graded picks in both cohorts yet"
        return out
    diff = float(treat.mean() - ctrl.mean())
    rng = np.random.default_rng(seed)
    boots = [rng.choice(treat, len(treat)).mean() - rng.choice(ctrl, len(ctrl)).mean()
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out.update({"claude_avg": round(float(treat.mean()), 2),
                "screen_avg": round(float(ctrl.mean()), 2),
                "edge_pp": round(diff, 2), "ci_lo": round(float(lo), 2),
                "ci_hi": round(float(hi), 2),
                "significant": bool(lo > 0 or hi < 0)})
    return out


def red_team_stats() -> pd.DataFrame:
    """Red-team behaviour per prompt version: how often the devil's advocate killed a
    buy, and the haircut distribution. The July-2026 'overkill' (11/13 buys killed with
    a uniform haircut=2) would show up here within a week instead of after a run review."""
    with storage.get_conn() as cn:
        rows = [dict(r) for r in cn.execute(
            """SELECT prompt_version, red_team_json FROM research_reports
               WHERE red_team_json IS NOT NULL AND red_team_json != '{}' AND red_team_json != 'null'""")]
    recs = []
    for r in rows:
        try:
            rt = json.loads(r["red_team_json"]) if isinstance(r["red_team_json"], str) else r["red_team_json"]
        except json.JSONDecodeError:
            continue
        if not isinstance(rt, dict) or "survives" not in rt:
            continue
        pv = r.get("prompt_version")
        recs.append({"prompt": f"v{int(pv)}" if isinstance(pv, (int, float)) and pv == pv else "pre-v4",
                     "survives": bool(rt.get("survives")),
                     "haircut": rt.get("confidence_haircut")})
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    out = []
    for pv, g in df.groupby("prompt"):
        hc = pd.to_numeric(g["haircut"], errors="coerce").dropna()
        out.append({"prompt": pv, "n_red_teamed": len(g),
                    "kill_rate_pct": round(float((~g["survives"]).mean() * 100), 1),
                    "avg_haircut": round(float(hc.mean()), 2) if len(hc) else None,
                    "haircut_0": int((hc == 0).sum()), "haircut_1": int((hc == 1).sum()),
                    "haircut_2plus": int((hc >= 2).sum())})
    return pd.DataFrame(out)


def calibration_by_prompt_version(window: str = "4w") -> pd.DataFrame:
    """Buy-call performance grouped by the prompt version that produced the rating.
    Reports before versioning show as 'pre-v4'. Lets a prompt change be judged on
    its own outcomes instead of blending analysts."""
    rows = pick_outcomes()
    col = f"net_excess_uni_{window}"
    recs = [r for r in rows if r.get(col) is not None
            and _bucket(r) in ("buy", "strong_buy", "hold")]
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    df["prompt"] = df["prompt_version"].apply(
        lambda v: f"v{int(v)}" if isinstance(v, (int, float)) and v == v else "pre-v4")
    df["bucket"] = df.apply(_bucket, axis=1)
    out = []
    for (pv, b), g in df.groupby(["prompt", "bucket"]):
        n = len(g)
        k = int((g[col] > 0).sum())
        lo, hi = wilson_ci(k, n)
        out.append({"prompt": pv, "rating": b, "n": n, "hit_pct": round(k / n * 100, 1),
                    "ci_lo": lo, "ci_hi": hi, "avg_net_excess": round(float(g[col].mean()), 2)})
    return pd.DataFrame(out)


# ------------------------------------------------------------- shadow portfolio
def shadow_vs_actual() -> list[dict[str, Any]]:
    """For each completed portfolio deep-dive with targets: the PM's target-weight
    portfolio vs the actual weights, from dive date to now. Weight not deployed by
    the targets (trims/exits) is treated as CASH earning 0 — the target return is
    NOT renormalized over the survivors."""
    with storage.get_conn() as cn:
        runs = [dict(r) for r in cn.execute(
            "SELECT id, finished_at FROM portfolio_runs WHERE status='ok' ORDER BY id")]
        pos = [dict(r) for r in cn.execute(
            """SELECT run_id, symbol, weight_pct, target_weight_pct, last
               FROM portfolio_positions WHERE weight_pct IS NOT NULL""")]
    by_run: dict[int, list[dict]] = {}
    for p in pos:
        by_run.setdefault(p["run_id"], []).append(p)
    syms = sorted({p["symbol"] for p in pos})
    if not syms:
        return []
    t_runs = []
    for r in runs:
        try:
            t_runs.append(pd.Timestamp(datetime.fromisoformat(r["finished_at"])
                                       .astimezone(timezone.utc).date()))
        except (ValueError, TypeError):
            t_runs.append(None)
    oldest = min((t for t in t_runs if t is not None), default=None)
    px = _price_history(syms, period=_period_for(oldest))
    if px is None:
        return []
    out = []
    for r, t0 in zip(runs, t_runs):
        plist = by_run.get(r["id"])
        if not plist or t0 is None or all(p.get("target_weight_pct") is None
                                          for p in plist):
            continue

        def _now_ret(sym: str) -> float | None:
            if sym not in px.columns:
                return None
            s = px[sym].dropna()
            idx = s.index.searchsorted(t0)
            if idx >= len(s) or abs((s.index[idx] - t0).days) > 7:
                return None
            base = float(s.iloc[idx])
            return (float(s.iloc[-1]) / base - 1) if base else None

        rets = {p["symbol"]: _now_ret(p["symbol"]) for p in plist}
        covered = [p for p in plist if rets.get(p["symbol"]) is not None]
        if not covered:
            continue
        actual_w = sum(p["weight_pct"] or 0 for p in covered)
        if not actual_w:
            continue
        actual = sum((p["weight_pct"] or 0) * rets[p["symbol"]] for p in covered) / actual_w
        # Target: undeployed weight (vs the same actual base) sits in cash at 0%.
        target = sum((p.get("target_weight_pct") or 0) * rets[p["symbol"]]
                     for p in covered) / actual_w
        # Getting to the targets is not free: every point of weight moved is one side
        # of a trade (a sell leg here, a buy leg there), so charge ONE-WAY cost on the
        # total weight moved. Σ|Δw| already counts both legs of a rotation.
        moved = sum(abs((p.get("target_weight_pct") or 0) - (p["weight_pct"] or 0))
                    for p in covered) / actual_w          # fraction of the covered book
        one_way = float(config.get("evaluation.cost_bps", 30)) / 10000.0 / 2.0
        cost = one_way * moved
        out.append({"run_id": r["id"], "date": str(t0.date()),
                    "actual_ret_pct": round(actual * 100, 2),
                    "target_ret_pct": round(target * 100, 2),
                    "edge_pct": round((target - actual) * 100, 2),
                    "turnover_pct": round(moved * 100, 1),
                    "cost_pct": round(cost * 100, 3),
                    "edge_net_pct": round((target - actual - cost) * 100, 2)})
    return out


if __name__ == "__main__":
    for basis in ("uni", "spy"):
        cal = rating_calibration(basis)
        print(f"=== rating calibration (excess vs {basis.upper()}) ===")
        print(cal.to_string(index=False) if not cal.empty else "(no graded picks yet)")
        print()
    print("=== shadow (PM targets, cash-aware) vs actual ===")
    for r in shadow_vs_actual():
        print(r)
