"""Walk-forward training of the pooled cross-sectional ranker + FIXED kill criteria.

Protocol (expanding window, monthly test dates):
  * test date d: rows with as_of == d
  * train rows: as_of <= d - (horizon + embargo) calendar days, so no training label
    window overlaps or touches the test period (purge + embargo, López de Prado)
  * early stopping on the LAST 12 MONTHS of the training window (time-ordered), never
    on the test date
  * features rank-transformed per date; label = per-date rank of 21d excess return
  * model: LightGBM lambdarank (group = date) — `model="lgbm_reg"` or `"xgb_reg"`
    regress the rank label instead (near-equivalent, allowed alternative)

Kill criteria are written here, BEFORE any result was seen, and evaluated by
`evaluate()`. Any one failing keeps the pillar at weight 0.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import features as F

KILL_CRITERIA = {
    "ic_t_min": 2.0,            # pooled IC t-stat on non-overlapping test dates
    "min_dates": 8,
    "ic_pos_share_min": 0.60,   # share of folds with positive IC
    "dsr_min": 0.50,            # deflated Sharpe (probability) of the long-top-vs-EW series
    "best_year_share_max": 0.70,
    "beat_baseline_ic_by": 0.01,  # must beat the replayed 3-pillar composite IC by this
    "importance_jaccard_min": 0.33,  # mean Jaccard of top-3 features across consecutive folds
}


@dataclass
class TrainSpec:
    model: str = "lgbm_rank"         # lgbm_rank | lgbm_reg | xgb_reg
    horizon_days: int = 21           # trading days of the label
    embargo_days: int = 21           # trading days after a label window before a test date
    first_train_months: int = 48
    params: dict[str, Any] = field(default_factory=lambda: {
        "num_leaves": 15, "min_child_samples": 200, "colsample_bytree": 0.7,
        "reg_lambda": 1.0, "learning_rate": 0.05, "n_estimators": 400, "subsample": 0.8,
        "subsample_freq": 1, "verbose": -1})
    label_bins: int = 10             # lambdarank relevance grades


@dataclass
class Fold:
    test_date: str
    n_train: int
    n_test: int
    best_iteration: int | None
    ic: float | None
    top_features: list[str]
    model_path: str


def _purge_cutoff(d: pd.Timestamp, spec: TrainSpec) -> pd.Timestamp:
    # trading days -> calendar days with slack (7/5) so purge is never too short
    cal = int((spec.horizon_days + spec.embargo_days) * 7 / 5) + 3
    return d - pd.Timedelta(days=cal)


# Native booster APIs (no scikit-learn dependency): LightGBM's sklearn wrappers need
# scikit-learn installed, which a fresh install won't have; lgb.train / xgb.train don't.
_SK_TO_LGB = {"num_leaves": "num_leaves", "min_child_samples": "min_data_in_leaf",
              "colsample_bytree": "feature_fraction", "reg_lambda": "lambda_l2",
              "learning_rate": "learning_rate", "subsample": "bagging_fraction",
              "subsample_freq": "bagging_freq", "verbose": "verbose"}


def _lgb_params(spec: TrainSpec) -> tuple[dict, int]:
    p = {_SK_TO_LGB[k]: v for k, v in spec.params.items() if k in _SK_TO_LGB}
    rounds = int(spec.params.get("n_estimators", 400))
    p.setdefault("verbose", -1)
    return p, rounds


def _fit(X_tr, y_tr, grp_tr, X_va, y_va, grp_va, spec: TrainSpec):
    """Returns (booster, best_iteration, gain_importance Series)."""
    if spec.model in ("lgbm_rank", "lgbm_reg"):
        import lightgbm as lgb
        params, rounds = _lgb_params(spec)
        if spec.model == "lgbm_rank":
            y_tr_i = np.clip((y_tr * spec.label_bins).astype(int), 0, spec.label_bins - 1)
            y_va_i = np.clip((y_va * spec.label_bins).astype(int), 0, spec.label_bins - 1)
            params.update({"objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
                           "label_gain": list(range(spec.label_bins))})
            dtr = lgb.Dataset(X_tr, label=y_tr_i, group=grp_tr, free_raw_data=False)
            dva = lgb.Dataset(X_va, label=y_va_i, group=grp_va, reference=dtr, free_raw_data=False)
        else:
            params.update({"objective": "regression", "metric": "l2"})
            dtr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            dva = lgb.Dataset(X_va, label=y_va, reference=dtr, free_raw_data=False)
        booster = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(30, verbose=False)])
        best = booster.best_iteration or None
        imp = pd.Series(booster.feature_importance("gain"), index=list(X_tr.columns))
        return booster, best, imp
    if spec.model == "xgb_reg":
        import xgboost as xgb
        p = {"max_depth": 4, "min_child_weight": 50, "colsample_bytree": 0.7, "lambda": 1.0,
             "eta": 0.05, "subsample": 0.8, "objective": "reg:squarederror", "verbosity": 0}
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_va, label=y_va)
        booster = xgb.train(p, dtr, num_boost_round=400, evals=[(dva, "val")],
                            early_stopping_rounds=30, verbose_eval=False)
        gain = booster.get_score(importance_type="gain")
        imp = pd.Series({c: gain.get(c, 0.0) for c in X_tr.columns})
        return booster, getattr(booster, "best_iteration", None), imp
    raise ValueError(f"unknown model {spec.model}")


def _predict(m, X) -> np.ndarray:
    try:
        import xgboost as xgb
        if isinstance(m, xgb.Booster):
            return np.asarray(m.predict(xgb.DMatrix(X)))
    except ImportError:
        pass
    it = getattr(m, "best_iteration", None) or None
    return np.asarray(m.predict(X, num_iteration=it) if it else m.predict(X))


def _save_model(m, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m.save_model(str(path))


def walk_forward(frame: pd.DataFrame, spec: TrainSpec, out_root: Path,
                 log=print) -> tuple[pd.DataFrame, list[Fold]]:
    """Returns (scores: date × symbol out-of-sample predictions, folds)."""
    df = frame.dropna(subset=["label_rank"]).copy()
    df["as_of"] = pd.to_datetime(df["as_of"])
    feats = [c for c in F.FEATURES if c in df.columns]
    df = F.rank_transform(df, feats)
    dates = sorted(df["as_of"].unique())
    if len(dates) < spec.first_train_months + 6:
        raise ValueError(f"need >= {spec.first_train_months + 6} dated cross-sections, have {len(dates)}")
    scores: dict[pd.Timestamp, pd.Series] = {}
    folds: list[Fold] = []
    for d in dates[spec.first_train_months:]:
        cutoff = _purge_cutoff(d, spec)
        tr = df[df["as_of"] <= cutoff]
        te = df[df["as_of"] == d]
        if len(te) < 10 or tr["as_of"].nunique() < 24:
            continue
        # time-ordered validation: last 12 cross-sections of the training window
        tr_dates = sorted(tr["as_of"].unique())
        va_dates = set(tr_dates[-12:])
        va = tr[tr["as_of"].isin(va_dates)]
        tr = tr[~tr["as_of"].isin(va_dates)]
        grp_tr = tr.groupby("as_of").size().tolist()
        grp_va = va.groupby("as_of").size().tolist()
        X_tr, y_tr = tr[feats], tr["label_rank"].values
        X_va, y_va = va[feats], va["label_rank"].values
        try:
            m, best, imp = _fit(X_tr, y_tr, grp_tr, X_va, y_va, grp_va, spec)
        except Exception as exc:  # noqa: BLE001
            log(f"  fold {d.date()} failed: {exc}")
            continue
        pred = pd.Series(_predict(m, te[feats]), index=te["symbol"].values)
        scores[d] = pred
        ic = pred.rank().corr(te.set_index("symbol")["label_rank"].rank())
        top = imp.sort_values(ascending=False).head(3).index.tolist()
        mp = out_root / f"{d.date()}" / "model.txt"
        _save_model(m, mp)
        (mp.parent / "meta.json").write_text(json.dumps({
            "test_date": str(d.date()), "features": feats, "spec": asdict(spec),
            "feature_version": F.FEATURE_VERSION, "n_train": int(len(tr)), "n_val": int(len(va)),
            "best_iteration": int(best) if best is not None else None,
            "importance_gain": {k: float(v) for k, v in imp.sort_values(ascending=False).items()},
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            indent=2), encoding="utf-8")
        folds.append(Fold(str(d.date()), int(len(tr)), int(len(te)),
                          int(best) if best is not None else None,
                          None if ic != ic else float(ic), top, str(mp)))
        log(f"  fold {d.date()}: train {len(tr)} rows, IC {ic:+.3f}, best_it {best}, top {top}")
    sc = pd.DataFrame(scores).T.sort_index()
    return sc, folds


def importance_stability(folds: list[Fold]) -> float | None:
    """Mean Jaccard similarity of consecutive folds' top-3 feature sets."""
    if len(folds) < 2:
        return None
    js = []
    for a, b in zip(folds, folds[1:]):
        sa, sb = set(a.top_features), set(b.top_features)
        js.append(len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0)
    return round(float(np.mean(js)), 3)


def evaluate(bt_summary: dict[str, Any], baseline_ic: float | None,
             folds: list[Fold]) -> dict[str, Any]:
    """Apply KILL_CRITERIA to the harness summary of the out-of-sample scores."""
    k = KILL_CRITERIA
    stab = importance_stability(folds)
    checks = {
        "ic_t": (bt_summary.get("ic_t") is not None and bt_summary["ic_t"] >= k["ic_t_min"]
                 and (bt_summary.get("n_dates") or 0) >= k["min_dates"],
                 f"IC t={bt_summary.get('ic_t')} on {bt_summary.get('n_dates')} dates (need >= {k['ic_t_min']} on >= {k['min_dates']})"),
        "ic_pos_share": ((bt_summary.get("ic_pos_share") or 0) >= k["ic_pos_share_min"],
                         f"positive-IC share {bt_summary.get('ic_pos_share')} (need >= {k['ic_pos_share_min']})"),
        "dsr": (bt_summary.get("dsr") is not None and bt_summary["dsr"] >= k["dsr_min"],
                f"deflated Sharpe {bt_summary.get('dsr')} (need >= {k['dsr_min']})"),
        "year_concentration": (bt_summary.get("best_year_share") is None
                               or bt_summary["best_year_share"] <= k["best_year_share_max"],
                               f"best-year share {bt_summary.get('best_year_share')} (need <= {k['best_year_share_max']})"),
        "beats_baseline": (baseline_ic is not None and bt_summary.get("ic_mean") is not None
                           and bt_summary["ic_mean"] - baseline_ic >= k["beat_baseline_ic_by"],
                           f"IC {bt_summary.get('ic_mean')} vs composite baseline {baseline_ic} (need +{k['beat_baseline_ic_by']})"),
        "importance_stability": (stab is not None and stab >= k["importance_jaccard_min"],
                                 f"top-3 feature Jaccard {stab} (need >= {k['importance_jaccard_min']})"),
    }
    # bool(): comparisons against numpy floats yield numpy.bool_, which json can't encode
    # and which would otherwise be written as the string "False" (truthy) by default=str.
    passed = all(bool(v[0]) for v in checks.values())
    return {"pass": bool(passed),
            "checks": {n: {"ok": bool(ok), "detail": det} for n, (ok, det) in checks.items()},
            "criteria": k, "importance_stability": stab}
