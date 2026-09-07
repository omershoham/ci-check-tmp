"""Train + validate the learned ranking pillar, walk-forward, on the history store.

    python -m scripts.train_ranker                     # LightGBM lambdarank, 2016->today
    python -m scripts.train_ranker --model xgb_reg
    python -m scripts.train_ranker --rebuild-features  # recompute the feature frame

Steps: feature frame (cached at models/rank_v1/features.parquet) -> walk-forward folds
-> out-of-sample scores through the SAME harness as the baselines -> composite baseline
from the same frame -> kill criteria -> report + latest.json (verdict recorded either way;
the pipeline only weights the pillar when the verdict passed).
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone

import pandas as pd

from src import universe
from src.backtest.engine import BacktestSpec, log_trial, n_trials, run_rank_backtest
from src.backtest.signals import composite_3pillar
from src.data.history_store import HistoryStore
from src.data.yfinance_live import month_ends
from src.ml import features as F
from src.ml import train as T
from src.ml.infer import models_root


def _log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm_rank", choices=["lgbm_rank", "lgbm_reg", "xgb_reg"])
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--cost-bps", type=float, default=15.0)
    args = ap.parse_args()

    store = HistoryStore()
    root = models_root()
    root.mkdir(parents=True, exist_ok=True)
    uni = universe.build_universe()
    syms = [u["symbol"] for u in uni]
    subs = {u["symbol"]: u["subsector"] for u in uni}
    closes = store.close_matrix(None, adjusted=True)
    if closes.empty:
        raise SystemExit("no prices in the history store — run scripts.data_refresh first")
    syms = [s for s in syms if s in closes.columns]
    dates = month_ends(date.fromisoformat(args.start), date.today())

    fpath = root / "features.parquet"
    if fpath.exists() and not args.rebuild_features:
        frame = pd.read_parquet(fpath)
        frame["as_of"] = pd.to_datetime(frame["as_of"])
        _log(f"features: loaded {len(frame)} rows from cache")
    else:
        _log(f"features: building for {len(syms)} symbols x {len(dates)} month-ends")
        frame = F.build_feature_frame(store, syms, subs, dates, log=_log)
        frame.to_parquet(fpath, index=False)
        _log(f"features: {len(frame)} rows, {frame['as_of'].nunique()} dates, "
             f"fundamentals coverage {frame['fcf_yield'].notna().mean():.0%}")
    if frame.empty:
        raise SystemExit("empty feature frame")

    spec = T.TrainSpec(model=args.model)
    _log(f"walk-forward: {spec.model}, first train {spec.first_train_months} months, "
         f"purge {spec.horizon_days}+{spec.embargo_days} trading days")
    scores, folds = T.walk_forward(frame, spec, root / "folds", log=_log)
    if scores.empty:
        raise SystemExit("no folds produced")

    # Out-of-sample scores through the harness (same fill/cost conventions as baselines).
    eligible = {d: set(store.universe_asof(d.date())) for d in scores.index}
    trials_root = store.root
    n_prior = n_trials(trials_root)
    bt = run_rank_backtest(scores, closes, BacktestSpec(name=f"learned_{spec.model}",
                                                        top_frac=args.top_frac, cost_bps_side=args.cost_bps),
                           eligible=eligible, n_trials=n_prior + 1,
                           universe_note="curated (see backtest reports for the survivorship caveat)")
    log_trial(trials_root, f"learned_{spec.model}", asdict_safe(spec), bt.summary)

    # Composite baseline from the SAME frame and dates (technical_score + fundamental pillar).
    tech = frame.pivot(index="as_of", columns="symbol", values="technical_score")
    from src import scoring
    fund_rows = {}
    for d, g in frame.groupby("as_of"):
        snap = g.set_index("symbol")
        fund_rows[d] = scoring.fundamental_pillar(snap, subs)
    fund = pd.DataFrame(fund_rows).T
    comp = composite_3pillar(tech, fund)
    comp = comp.loc[comp.index.isin(scores.index)]
    base = run_rank_backtest(comp, closes, BacktestSpec(name="composite_same_dates",
                                                        top_frac=args.top_frac, cost_bps_side=args.cost_bps),
                             eligible=eligible, n_trials=n_prior + 1)
    verdict = T.evaluate(bt.summary, base.summary.get("ic_mean"), folds)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = root / f"report_{ts}"
    bt.save(out)
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    (out / "folds.json").write_text(json.dumps([f.__dict__ for f in folds], indent=2, default=str),
                                    encoding="utf-8")
    (out / "baseline_summary.json").write_text(json.dumps(base.summary, indent=2, default=str), encoding="utf-8")
    last = folds[-1]
    latest = {"test_date": last.test_date, "model_path": last.model_path, "features":
              [c for c in F.FEATURES if c in frame.columns], "feature_version": F.FEATURE_VERSION,
              "spec": asdict_safe(spec), "summary": bt.summary, "baseline_ic": base.summary.get("ic_mean"),
              "verdict": verdict, "report_dir": str(out), "trained_at": ts}
    (root / "latest.json").write_text(json.dumps(latest, indent=2, default=str), encoding="utf-8")

    print(bt.to_markdown())
    print("\n## Composite baseline (same dates)")
    print(f"IC mean {base.summary.get('ic_mean')} t {base.summary.get('ic_t')} · "
          f"net excess vs EW {base.summary.get('ann_excess_ew_net_pct')}%/yr")
    print("\n## Kill criteria")
    for name, c in verdict["checks"].items():
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {name}: {c['detail']}")
    print(f"\nVERDICT: {'SHIP (weight may be raised)' if verdict['pass'] else 'DO NOT SHIP (stays at weight 0)'}")
    _log(f"saved -> {out}")


def asdict_safe(spec) -> dict:
    from dataclasses import asdict
    return asdict(spec)


if __name__ == "__main__":
    main()
