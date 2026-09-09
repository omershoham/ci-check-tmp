"""Harness sanity: a signal that truly predicts must score, noise must not, and the
multiple-testing guard must bite as trials grow."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.backtest import metrics as M
from src.backtest.engine import (BacktestSpec, config_hash, log_trial, n_trials,
                                 n_trial_runs, run_rank_backtest, trial_hashes)


def _market(seed=0, n_sym=40, n_days=1500, signal_strength=0.0):
    """Synthetic adjusted closes. Each symbol has a latent 'quality' q; when
    signal_strength>0 its daily drift is q*strength so a score = q predicts returns."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=n_days)
    q = rng.normal(size=n_sym)
    drift = 0.0002 + signal_strength * q * 0.0005
    rets = rng.normal(drift, 0.02, size=(n_days, n_sym))
    px = 100 * np.exp(np.cumsum(rets, axis=0))
    syms = [f"S{i:02d}" for i in range(n_sym)]
    closes = pd.DataFrame(px, index=idx, columns=syms)
    closes["SPY"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n_days)))
    closes["SMH"] = closes[syms].mean(axis=1)
    return closes, pd.Series(q, index=syms)


def _scores(closes, q, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(closes.index[0], closes.index[-1], freq="BME")
    dates = [d for d in dates if d >= closes.index[260]]
    rows = {d: q + rng.normal(0, noise, len(q)) for d in dates}
    return pd.DataFrame(rows).T


def test_true_signal_has_positive_ic_and_beats_ew():
    closes, q = _market(signal_strength=1.0)
    res = run_rank_backtest(_scores(closes, q), closes, BacktestSpec("true", top_frac=0.2))
    s = res.summary
    assert s["n_dates"] > 40
    assert s["ic_mean"] > 0.05 and s["ic_t"] > 3
    assert s["d10_minus_d1_pct"] > 0
    assert s["ann_excess_ew_net_pct"] > 0
    assert res.per_date["turnover"].iloc[1:].max() <= 0.05   # constant ranking -> ~no churn


def test_noise_signal_has_no_ic():
    closes, q = _market(signal_strength=0.0)
    noise = pd.Series(np.random.default_rng(5).normal(size=len(q)), index=q.index)
    res = run_rank_backtest(_scores(closes, noise, noise=1.0), closes, BacktestSpec("noise"))
    assert abs(res.summary["ic_mean"]) < 0.05
    assert abs(res.summary["ic_t"]) < 2.5


def test_costs_and_turnover_are_charged():
    closes, q = _market(signal_strength=1.0)
    # alternate two disjoint rankings each month -> 100% turnover every rebalance
    sc = _scores(closes, q)
    flip = sc.copy()
    flip.iloc[1::2] = -flip.iloc[1::2]
    cheap = run_rank_backtest(flip, closes, BacktestSpec("c", cost_bps_side=0.0)).summary
    dear = run_rank_backtest(flip, closes, BacktestSpec("d", cost_bps_side=50.0)).summary
    assert dear["avg_turnover_pct"] > 150
    assert dear["ann_return_net_pct"] < cheap["ann_return_net_pct"] - 5


def test_min_history_excludes_young_names():
    closes, q = _market()
    closes.iloc[:1000, 0] = np.nan          # S00 starts trading late
    res = run_rank_backtest(_scores(closes, q), closes, BacktestSpec("h"))
    assert res.per_date["n"].iloc[0] == 39 and res.per_date["n"].iloc[-1] == 40


def test_deflated_sharpe_shrinks_with_trials():
    r = pd.Series(np.random.default_rng(2).normal(0.004, 0.02, 120))
    one = M.deflated_sharpe(r, n_trials=1, periods_per_year=12)["dsr"]
    many = M.deflated_sharpe(r, n_trials=50, periods_per_year=12)["dsr"]
    assert one > many
    assert M.expected_max_sharpe(1, 0.01) == 0.0
    assert M.expected_max_sharpe(100, 0.01) > M.expected_max_sharpe(10, 0.01)


def test_wilson_free_metrics_edge_cases():
    assert M.rank_ic(pd.Series([1, 2, 3]), pd.Series([1, 2, 3])) is None   # < 10 names
    fm = M.fama_macbeth(pd.Series([], dtype=float))
    assert fm["n_dates"] == 0 and fm["ic_t"] is None
    assert M.max_drawdown(pd.Series([0.1, -0.5, 0.2])) < -0.4
    assert M.turnover(None, pd.Series({"A": 0.5, "B": 0.5})) == 1.0
    assert M.turnover(pd.Series({"A": 1.0}), pd.Series({"B": 1.0})) == 2.0


def test_trials_log_counts_distinct_configs_not_runs(tmp_path):
    """Deflated Sharpe charges you for SEARCHING. Re-running a config you already ran
    is verification, not another lottery ticket — 109 of this project's trial lines
    were 13 configurations."""
    assert n_trials(tmp_path) == 0
    p = {"start": "2016-01-01", "rebalance": "BME", "top_frac": 0.1, "top_n": None,
         "cost_bps": 15.0}
    assert log_trial(tmp_path, "composite", p, {"ic_mean": 0.1}, universe="curated") == 1
    # The SAME configuration again: recorded, but not a new trial.
    assert log_trial(tmp_path, "composite", p, {"ic_mean": 0.1}, universe="curated") == 1
    assert n_trial_runs(tmp_path) == 2
    # A changed cost assumption IS a new hypothesis.
    assert log_trial(tmp_path, "composite", {**p, "cost_bps": 30.0}, {},
                     universe="curated") == 2
    # So is the same signal on a different universe...
    assert log_trial(tmp_path, "composite", p, {}, universe="broad") == 3
    # ...and a different rebalance/label horizon.
    assert log_trial(tmp_path, "composite", {**p, "rebalance": "10d"}, {},
                     universe="broad") == 4
    # ...and a different signal.
    assert log_trial(tmp_path, "technical", p, {}, universe="broad") == 5


def test_legacy_trial_lines_are_hashed_from_the_fields_they_did_record(tmp_path):
    """Lines predate config_hash and universe. They must still count — as one trial
    each per distinct config, carrying universe=None, which is all that was knowable."""
    old = {"at": "2026-09-08T07:34:00+00:00", "name": "momentum",
           "params": {"start": "2016-01-01", "rebalance": "BME", "top_frac": 0.1,
                      "top_n": None, "cost_bps": 15.0},
           "ic_mean": 0.0154, "ic_t": 0.77}
    lines = [json.dumps(old), json.dumps(old), json.dumps({**old, "name": "technical"})]
    (tmp_path / "trials.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert n_trial_runs(tmp_path) == 3
    assert n_trials(tmp_path) == 2
    assert config_hash("momentum", old["params"], None) in trial_hashes(tmp_path)
    # A new run of the same config on the CURATED universe is a distinct trial: the
    # legacy line never said which universe it was, so it cannot claim to be the same one.
    assert log_trial(tmp_path, "momentum", old["params"], {}, universe="curated") == 3
