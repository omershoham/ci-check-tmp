import numpy as np
import pandas as pd

from src import scoring
from src.ml import features as F
from src.ml import train as T


def test_rank_transform_is_per_date_and_skips_regime_columns():
    df = pd.DataFrame({
        "as_of": pd.to_datetime(["2024-01-31"] * 3 + ["2024-02-29"] * 3),
        "symbol": list("ABC") * 2,
        "rsi": [10, 20, 30, 30, 20, 10],
        "spy_vs_200": [5, 5, 5, -2, -2, -2],
    })
    out = F.rank_transform(df, ["rsi", "spy_vs_200"])
    assert list(out["rsi"].round(3)) == [0.333, 0.667, 1.0, 1.0, 0.667, 0.333]
    assert list(out["spy_vs_200"]) == [5, 5, 5, -2, -2, -2]       # passed through


def test_purge_cutoff_covers_label_and_embargo():
    spec = T.TrainSpec(horizon_days=21, embargo_days=21)
    d = pd.Timestamp("2024-06-28")
    cut = T._purge_cutoff(d, spec)
    assert (d - cut).days >= 42 * 7 / 5          # at least 42 trading days in calendar terms


def test_evaluate_applies_every_kill_criterion():
    folds = [T.Fold("2024-01-31", 100, 50, 10, 0.05, ["a", "b", "c"], "x"),
             T.Fold("2024-02-29", 110, 50, 10, 0.04, ["a", "b", "d"], "x"),
             T.Fold("2024-03-31", 120, 50, 10, 0.06, ["a", "c", "b"], "x")]
    good = {"ic_t": 3.1, "n_dates": 40, "ic_pos_share": 0.7, "dsr": 0.9, "best_year_share": 0.4,
            "ic_mean": 0.05}
    v = T.evaluate(good, baseline_ic=0.01, folds=folds)
    assert v["pass"] is True and all(c["ok"] for c in v["checks"].values())
    bad = dict(good, ic_t=1.2)
    v = T.evaluate(bad, baseline_ic=0.01, folds=folds)
    assert v["pass"] is False and not v["checks"]["ic_t"]["ok"]
    v = T.evaluate(good, baseline_ic=0.045, folds=folds)          # only +0.005 over baseline
    assert not v["checks"]["beats_baseline"]["ok"]
    v = T.evaluate(dict(good, best_year_share=0.8), baseline_ic=0.0, folds=folds)
    assert not v["checks"]["year_concentration"]["ok"]
    unstable = [T.Fold("d1", 1, 1, 1, 0.1, ["a", "b", "c"], "x"),
                T.Fold("d2", 1, 1, 1, 0.1, ["x", "y", "z"], "x")]
    assert T.importance_stability(unstable) == 0.0
    assert not T.evaluate(good, 0.0, unstable)["checks"]["importance_stability"]["ok"]


def test_walk_forward_trains_and_scores_out_of_sample(tmp_path):
    """Smoke: the native-API training loop runs end to end on a synthetic frame and a
    planted signal is recovered out of sample (no scikit-learn required)."""
    import lightgbm  # noqa: F401  (skip cleanly if the optional dep is missing)
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-31", periods=40, freq="BME")
    syms = [f"S{i:02d}" for i in range(30)]
    rows = []
    for d in dates:
        q = rng.normal(size=len(syms))                    # true latent quality per date
        for s, qi in zip(syms, q):
            rec = {"as_of": d, "symbol": s, "label_rank": None}
            for f in F.FEATURES:
                rec[f] = rng.normal()
            rec["rsi"] = qi + rng.normal(0, 0.3)          # one informative feature
            rec["ret_63"] = qi * 0.5 + rng.normal(0, 0.5)
            rec["_q"] = qi
            rows.append(rec)
    df = pd.DataFrame(rows)
    df["label_rank"] = df.groupby("as_of")["_q"].rank(pct=True)
    spec = T.TrainSpec(model="lgbm_rank", first_train_months=24)
    spec.params.update({"min_child_samples": 20, "n_estimators": 60})
    scores, folds = T.walk_forward(df, spec, tmp_path / "folds", log=lambda *_: None)
    assert len(folds) >= 5 and not scores.empty
    ics = [f.ic for f in folds if f.ic is not None]
    assert np.mean(ics) > 0.3                              # planted signal recovered OOS
    assert all("rsi" in f.top_features or "ret_63" in f.top_features for f in folds)
    assert (tmp_path / "folds" / folds[-1].test_date / "model.txt").exists()
    # the saved booster reloads and scores the same rows the same way
    import lightgbm as lgb
    b = lgb.Booster(model_file=folds[-1].model_path)
    last = F.rank_transform(df[df["as_of"] == pd.Timestamp(folds[-1].test_date)], F.FEATURES)
    pred = b.predict(last[[c for c in F.FEATURES if c in last.columns]])
    assert np.allclose(np.sort(pred), np.sort(scores.loc[pd.Timestamp(folds[-1].test_date)].dropna().values), atol=1e-6)


def test_labels_are_forward_and_rank_transformed():
    idx = pd.bdate_range("2024-01-01", periods=60)
    closes = pd.DataFrame({"A": np.linspace(100, 130, 60), "B": np.linspace(100, 100, 60),
                           "C": np.linspace(100, 80, 60)} | {f"F{i}": 100 + np.random.default_rng(i).normal(0, 1, 60).cumsum() for i in range(8)},
                          index=idx)
    lab = F.labels(closes, [idx[5]], horizon=21)
    a = lab.set_index("symbol")
    assert a.loc["A", "fwd_21"] > a.loc["B", "fwd_21"] > a.loc["C", "fwd_21"]
    assert a.loc["A", "label_rank"] == 1.0 and abs(a["fwd_excess_21"].mean()) < 1e-9
    # the label window uses closes strictly AFTER as_of
    exp = (closes.loc[idx[6 + 21], "A"] / closes.loc[idx[6], "A"] - 1) * 100
    assert abs(a.loc["A", "fwd_21"] - exp) < 1e-9


def test_scoring_learned_pillar_is_dark_by_default_but_persisted(monkeypatch):
    m = {s: {"symbol": s, "name": s, "marketCap": 1e10, "price": 10.0, "freeCashflow": 1.0,
             "profitable": True, "trailingPE": 20 + i} for i, s in enumerate("ABCDEFGHIJ")}
    subs = {s: "semis_equipment" for s in m}
    ml = {s: float(i) for i, s in enumerate("ABCDEFGHIJ")}
    base = {r["symbol"]: r["blended"] for r in scoring.score_universe(m, subs)}
    with_ml = {r["symbol"]: r for r in scoring.score_universe(m, subs, ml_scores=ml)}
    assert all(abs(base[s] - with_ml[s]["blended"]) < 1e-12 for s in m)       # weight 0: no effect
    assert with_ml["J"]["metrics"]["pillar_learned_rank"] == 1.0                # but persisted
    assert scoring.effective_score_version() == 3
    monkeypatch.setattr(scoring, "learned_weight", lambda: 0.2)
    assert scoring.effective_score_version() == 4
    lit_rows = scoring.score_universe(m, subs, ml_scores=ml)
    lit = {r["symbol"]: r["blended"] for r in lit_rows}
    assert lit["J"] > base["J"] and lit["A"] < base["A"]                       # now it moves scores
    assert all(0.0 <= v <= 1.0 for v in lit.values())                          # still on the 0..1 scale
    rank_lit = {r["symbol"]: r["rank"] for r in lit_rows}
    rank_base = {r["symbol"]: r["rank"] for r in scoring.score_universe(m, subs)}
    assert rank_lit["J"] <= rank_base["J"] and rank_lit["A"] >= rank_base["A"]  # and ranks
