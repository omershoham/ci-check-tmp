"""Rank-and-hold walk-forward engine.

Inputs
  scores      date × symbol frame, rows = rebalance dates (signal known at that close)
  closes      daily date × symbol ADJUSTED closes incl. benchmark columns
  spec        BacktestSpec (bucket size, costs, horizon, benchmarks)
  eligible    optional {date -> set(symbols)} restricting the cross-section per date
              (point-in-time universe); default = every symbol with a score AND ≥200
              prior price days (so a name enters only once it has traded long enough)

Per rebalance date t:
  * fill at the next available close f(t) > t  (one-day lag, conservative)
  * forward return  r = close[f(t+1)] / close[f(t)] - 1  where t+1 is the next rebalance
  * rank IC  = Spearman(score_t, r) over the eligible cross-section
  * deciles  = mean r per score decile (1 low … 10 high)
  * top bucket portfolio: equal-weight the top `top_frac` (or `top_n`), hold, pay
    `cost_bps_side` on Σ|Δw| at each rebalance
  * benchmarks: equal-weight eligible universe, and each column in spec.benchmarks

Everything is graded with the SAME forward-return convention as src/performance.py
(prior close base, complete windows only).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import metrics as M


@dataclass
class BacktestSpec:
    name: str
    top_frac: float = 0.10            # fraction of the eligible cross-section held
    top_n: int | None = None          # overrides top_frac when set
    cost_bps_side: float = 15.0       # commission+spread+slippage, one way
    min_history_days: int = 200       # a symbol needs this many prior closes to be eligible
    benchmarks: tuple[str, ...] = ("SPY", "SMH")
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    spec: BacktestSpec
    per_date: pd.DataFrame             # one row per rebalance date
    deciles: pd.DataFrame              # date × decile mean forward return
    summary: dict[str, Any]
    universe_note: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # -- persistence -------------------------------------------------------
    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.per_date.to_parquet(out_dir / "per_date.parquet")
        self.deciles.to_parquet(out_dir / "deciles.parquet")
        (out_dir / "summary.json").write_text(json.dumps(
            {"spec": asdict(self.spec), "summary": self.summary, "created_at": self.created_at,
             "universe_note": self.universe_note}, indent=2, default=str), encoding="utf-8")
        (out_dir / "report.md").write_text(self.to_markdown(), encoding="utf-8")
        return out_dir

    def to_markdown(self) -> str:
        s = self.summary
        lines = [f"# Backtest — {self.spec.name}", "",
                 f"_{self.created_at}_ · {s.get('n_dates')} rebalances "
                 f"{s.get('first_date')} → {s.get('last_date')} · "
                 f"top {self.spec.top_n or f'{self.spec.top_frac:.0%}'} · "
                 f"{self.spec.cost_bps_side:g} bp/side", "",
                 f"> **Universe:** {self.universe_note}", "",
                 "## Ranking quality (does the score order the cross-section?)", "",
                 f"- rank IC mean **{s.get('ic_mean')}**, t-stat **{s.get('ic_t')}** over "
                 f"{s.get('n_dates')} non-overlapping dates; positive on {s.get('ic_pos_share')} of dates",
                 f"- top-decile minus bottom-decile forward return: **{s.get('d10_minus_d1_pct')}%** per period",
                 f"- top-decile minus equal-weight universe: **{s.get('top_minus_ew_pct')}%** per period",
                 "", "## Portfolio (top bucket, equal weight, rank-and-hold)", "",
                 f"- annualized return gross **{s.get('ann_return_gross_pct')}%** / net **{s.get('ann_return_net_pct')}%**"
                 f" · EW universe {s.get('ann_return_ew_pct')}%"
                 + "".join(f" · {b} {s.get(f'ann_return_{b.lower()}_pct')}%" for b in self.spec.benchmarks),
                 f"- excess vs EW (net): **{s.get('ann_excess_ew_net_pct')}%** · hit rate vs EW {s.get('hit_vs_ew_pct')}%",
                 f"- Sharpe (net, ann.) **{s.get('sharpe_net')}** · vs EW long/short Sharpe {s.get('sharpe_ls_ew')}",
                 f"- **Deflated Sharpe {s.get('dsr')}** (trials counted: {s.get('n_trials')}; expected max "
                 f"skill-less SR {s.get('sr0_ann')}) — bar is 0.95",
                 f"- max drawdown (net) {s.get('max_dd_pct')}% · avg turnover/rebalance {s.get('avg_turnover_pct')}%"
                 f" · cost drag {s.get('cost_drag_ann_pct')}%/yr",
                 "", "## Concentration of the edge by year (kill rule: >70% from one year)", ""]
        by = s.get("excess_by_year") or {}
        for y, v in by.items():
            lines.append(f"- {y}: {v:+.2f}%")
        lines.append(f"- share of total edge from the best year: **{s.get('best_year_share')}**")
        return "\n".join(lines)


def _next_close_index(index: pd.DatetimeIndex, t: pd.Timestamp) -> int | None:
    """Position of the first trading day strictly after t."""
    i = index.searchsorted(t, side="right")
    return int(i) if i < len(index) else None


def run_rank_backtest(scores: pd.DataFrame, closes: pd.DataFrame, spec: BacktestSpec,
                      eligible: dict[pd.Timestamp, set[str]] | None = None,
                      n_trials: int = 1, universe_note: str = "") -> BacktestResult:
    closes = closes.sort_index()
    idx = closes.index
    dates = [d for d in scores.index if d in idx or idx.searchsorted(d, side="right") < len(idx)]
    dates = sorted(pd.Timestamp(d) for d in dates)
    bench_cols = [b for b in spec.benchmarks if b in closes.columns]
    rows, dec_rows = [], []
    prev_w: pd.Series | None = None
    counts = closes.notna().cumsum()             # prior price days per symbol

    for k, t in enumerate(dates):
        fi = _next_close_index(idx, t)           # fill day
        if fi is None:
            break
        nt = dates[k + 1] if k + 1 < len(dates) else None
        if nt is None:
            break
        ei = _next_close_index(idx, nt)          # exit/next fill day
        if ei is None:
            break
        f_day, e_day = idx[fi], idx[ei]
        s = scores.loc[t].dropna()
        # Eligible: scored, has a fill price, enough history; PIT universe if supplied.
        names = [c for c in s.index if c in closes.columns and c not in bench_cols]
        if eligible is not None and t in eligible:
            names = [c for c in names if c in eligible[t]]
        hist_ok = counts.loc[idx[fi - 1] if fi > 0 else idx[0]] if fi > 0 else None
        names = [c for c in names if hist_ok is not None and hist_ok.get(c, 0) >= spec.min_history_days
                 and pd.notna(closes.at[f_day, c]) and pd.notna(closes.at[e_day, c])]
        if len(names) < 10:
            continue
        s = s.loc[names]
        fwd = (closes.loc[e_day, names] / closes.loc[f_day, names] - 1) * 100
        ic = M.rank_ic(s, fwd)
        dec = M.decile_returns(s, fwd)
        if dec is not None:
            dec_rows.append(dec.rename(t))
        # top bucket
        n_top = spec.top_n or max(1, int(round(len(names) * spec.top_frac)))
        top = s.sort_values(ascending=False).index[:n_top]
        w = pd.Series(1.0 / n_top, index=top)
        to = M.turnover(prev_w, w)
        gross = float(fwd.loc[top].mean())
        cost = to * spec.cost_bps_side / 100.0        # bps -> % on weight moved (one way per leg)
        net = gross - cost
        ew = float(fwd.mean())
        row = {"date": t, "fill": f_day, "exit": e_day, "n": len(names), "n_top": n_top,
               "ic": ic, "ret_top_gross_pct": round(gross, 4), "cost_pct": round(cost, 4),
               "ret_top_net_pct": round(net, 4), "ret_ew_pct": round(ew, 4),
               "turnover": round(to, 4)}
        for b in bench_cols:
            if pd.notna(closes.at[f_day, b]) and pd.notna(closes.at[e_day, b]):
                row[f"ret_{b.lower()}_pct"] = round(float(closes.at[e_day, b] / closes.at[f_day, b] - 1) * 100, 4)
        rows.append(row)
        prev_w = w

    per = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    deciles = pd.DataFrame(dec_rows) if dec_rows else pd.DataFrame()
    summary = _summarize(per, deciles, spec, n_trials, bench_cols)
    return BacktestResult(spec=spec, per_date=per, deciles=deciles, summary=summary,
                          universe_note=universe_note)


def _ann(per_period_pct: pd.Series, ppy: float) -> float | None:
    r = per_period_pct.dropna() / 100
    if r.empty:
        return None
    return round((float((1 + r).prod()) ** (ppy / len(r)) - 1) * 100, 2)


def _summarize(per: pd.DataFrame, deciles: pd.DataFrame, spec: BacktestSpec,
               n_trials: int, bench_cols: list[str]) -> dict[str, Any]:
    if per.empty:
        return {"n_dates": 0, "note": "no evaluable rebalance dates"}
    # periods per year from the median spacing of rebalance dates
    gaps = pd.Series(per.index).diff().dt.days.dropna()
    ppy = 365.25 / float(gaps.median()) if not gaps.empty else 12.0
    fm = M.fama_macbeth(per["ic"])
    out: dict[str, Any] = {"n_dates": int(len(per)), "first_date": str(per.index.min().date()),
                           "last_date": str(per.index.max().date()),
                           "periods_per_year": round(ppy, 2), **fm}
    if not deciles.empty:
        dm = deciles.mean()
        out["d10_minus_d1_pct"] = round(float(dm.iloc[-1] - dm.iloc[0]), 3)
        out["decile_means_pct"] = [round(float(x), 3) for x in dm.values]
    excess_top_ew = per["ret_top_gross_pct"] - per["ret_ew_pct"]
    out["top_minus_ew_pct"] = round(float(excess_top_ew.mean()), 3)
    out["ann_return_gross_pct"] = _ann(per["ret_top_gross_pct"], ppy)
    out["ann_return_net_pct"] = _ann(per["ret_top_net_pct"], ppy)
    out["ann_return_ew_pct"] = _ann(per["ret_ew_pct"], ppy)
    for b in bench_cols:
        c = f"ret_{b.lower()}_pct"
        if c in per:
            out[f"ann_return_{b.lower()}_pct"] = _ann(per[c], ppy)
    net_excess = per["ret_top_net_pct"] - per["ret_ew_pct"]
    out["ann_excess_ew_net_pct"] = round(float(net_excess.mean()) * ppy, 2)
    out["hit_vs_ew_pct"] = round(float((net_excess > 0).mean()) * 100, 1)
    sh = M.sharpe(per["ret_top_net_pct"] / 100, ppy)
    out["sharpe_net"] = round(sh, 2) if sh is not None else None
    ls = M.sharpe(net_excess / 100, ppy)
    out["sharpe_ls_ew"] = round(ls, 2) if ls is not None else None
    dsr = M.deflated_sharpe(net_excess / 100, n_trials, ppy)
    out.update({"dsr": dsr.get("dsr"), "sr0_ann": dsr.get("sr0_ann"), "n_trials": int(n_trials)})
    mdd = M.max_drawdown(per["ret_top_net_pct"] / 100)
    out["max_dd_pct"] = round(mdd * 100, 2) if mdd is not None else None
    out["avg_turnover_pct"] = round(float(per["turnover"].mean()) * 100, 1)
    out["cost_drag_ann_pct"] = round(float(per["cost_pct"].mean()) * ppy, 2)
    by_year = net_excess.groupby(per.index.year).sum()
    out["excess_by_year"] = {str(int(y)): round(float(v), 2) for y, v in by_year.items()}
    pos_total = float(by_year[by_year > 0].sum())
    out["best_year_share"] = (round(float(by_year.max()) / pos_total, 2)
                              if pos_total > 0 and by_year.max() > 0 else None)
    return out


# ------------------------------------------------------------------ trials log
# Fields that make a run a DIFFERENT hypothesis. Anything not listed here (the wall
# clock, how many symbols happened to have prices that day, which report directory it
# landed in) is bookkeeping, not a new bet on the data.
_HASH_FIELDS = ("start", "rebalance", "top_frac", "top_n", "cost_bps")

# Fields that join the identity ONLY when they are set. A run at the shipped pillar
# weights is the same hypothesis as every run logged before `--pillar-weights`
# existed, so the key must be ABSENT from the payload then, not present-and-null:
# adding a null would re-hash all 34 configurations in `trials.jsonl` and make the
# next default run look like a 35th lottery ticket. A run at any OTHER weighting is
# a genuinely different bet on the data and gets its own hash, exactly once.
_OPTIONAL_HASH_FIELDS = ("pillars",)


def config_hash(name: str, params: dict[str, Any], universe: str | None = None) -> str:
    """Stable short hash of the strategy configuration a trial tested.

    The multiple-testing correction exists to charge you for SEARCHING: every distinct
    configuration you try is a lottery ticket, and Deflated Sharpe deflates the winner
    by how many you bought. Re-running a configuration you already ran buys no new
    ticket — it is verification, not search — so counting it inflates the penalty for
    doing the careful thing.

    Concretely: `trials.jsonl` reached 109 lines in this project but held only 13
    distinct configurations. The other 96 were identical re-runs proving a refactor was
    byte-identical, plus the horizon grid re-run on a corrected exchange calendar.
    Every Deflated Sharpe here was being penalised for regression testing.

    `pillars` (see `_OPTIONAL_HASH_FIELDS`) is part of the identity when, and only
    when, `--pillar-weights` asked for something other than the shipped weights: a
    re-weighted composite is a different signal, but the shipped weighting must keep
    the hash it has always had or every prior trial re-counts.

    `universe` is part of the identity because the same signal on 105 curated names and
    on 1,500 registrant-derived ones are two different hypotheses about two different
    populations. It is None on lines written before the field existed, and stays None
    when those are re-hashed, so history keeps exactly the identity it can honestly
    claim and no more.
    """
    payload = {"name": name, "universe": universe,
               **{k: params.get(k) for k in _HASH_FIELDS}}
    payload.update({k: params[k] for k in _OPTIONAL_HASH_FIELDS
                    if params.get(k) is not None})
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def log_trial(root: Path, name: str, params: dict[str, Any], summary: dict[str, Any],
              universe: str | None = None) -> int:
    """Append one run to trials.jsonl; return the count of DISTINCT configurations.

    Every run is still written down — the file is the audit trail of what was actually
    executed — but only distinct `config_hash` values count as trials.
    """
    root.mkdir(parents=True, exist_ok=True)
    p = root / "trials.jsonl"
    rec = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "name": name,
           "params": params, "universe": universe,
           "config_hash": config_hash(name, params, universe),
           "ic_mean": summary.get("ic_mean"), "ic_t": summary.get("ic_t"),
           "sharpe_ls_ew": summary.get("sharpe_ls_ew"), "n_dates": summary.get("n_dates")}
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return n_trials(root)


def trial_hashes(root: Path) -> set[str]:
    """Every distinct configuration ever logged.

    Lines written before `config_hash` existed are hashed from the fields they DID
    record, so pre-existing history is not thrown away.
    """
    p = root / "trials.jsonl"
    if not p.exists():
        return set()
    out: set[str] = set()
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = rec.get("config_hash")
            out.add(h or config_hash(rec.get("name", ""), rec.get("params") or {},
                                     rec.get("universe")))
    return out


def n_trials(root: Path) -> int:
    """How many DISTINCT configurations have been tried — the multiple-testing count."""
    return len(trial_hashes(root))


def n_trial_runs(root: Path) -> int:
    """How many runs are recorded, distinct or not. The audit number, not the penalty."""
    p = root / "trials.jsonl"
    if not p.exists():
        return 0
    with p.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())
