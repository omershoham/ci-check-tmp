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


# ---------------------------------------------------------------- pillar weights
# `--pillar-weights` (scripts/backtest.py) overrides scoring.pillars for ONE process.
# Three things have to hold or the override is worse than not having it: it must reach
# BOTH the replay and the live screen, it must make each weighting a distinct trial
# EXACTLY once, and asking for the shipped weighting must not count as a new one.

def test_pillar_weights_parse_and_reject_nonsense():
    from scripts.backtest import parse_pillar_weights
    assert parse_pillar_weights("technical=0.2,fundamental=0.45,sentiment=0.35") == {
        "technical": 0.2, "fundamental": 0.45, "sentiment": 0.35}
    assert parse_pillar_weights(" technical=0 , fundamental=0.65 ") == {
        "technical": 0.0, "fundamental": 0.65}
    for bad in ("technical", "tecnical=0.2", "technical=x", "technical=-0.1",
                "", "technical=0,fundamental=0,sentiment=0"):
        try:
            parse_pillar_weights(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should have been rejected")


def test_pillar_weights_override_reaches_replay_and_live_screen(monkeypatch):
    """One override on the exact dotted path, two consumers. `assert_replayable` is the
    replay's resolver and `scoring.pillar_set`/`learned_weight` the live screen's, so
    if both see it the composite cannot be replayed at weights the screen isn't using."""
    from scripts.backtest import apply_pillar_weights
    from src import config, scoring
    from src.backtest import signals
    monkeypatch.setattr(config, "_OVERRIDES", {}, raising=False)
    try:
        canon = apply_pillar_weights("technical=0.1,fundamental=0.55,sentiment=0.35")
        w = signals.assert_replayable()                       # the replay's path
        assert (w["technical"], w["fundamental"], w["sentiment"]) == (0.1, 0.55, 0.35)
        assert config.get("scoring.pillars")["technical"] == 0.1   # the live screen's
        assert scoring.pillar_set() == "technical,fundamental,sentiment"
        assert scoring.learned_weight() == 0.0
        assert canon and canon["technical"] == 0.1 and canon["catalyst"] == 0.0
        # technical=0 leaves a two-pillar composite, and pillar_set must say so:
        # a differently-shaped composite is a different partition for pillar_ic.
        apply_pillar_weights("technical=0.0,fundamental=0.65,sentiment=0.35")
        assert scoring.pillar_set() == "fundamental,sentiment"
    finally:
        config.clear_overrides()


def test_pillar_weights_merge_keeps_an_unreplayable_pillar_visible(monkeypatch):
    """Named weights MERGE over the configured dict. If they replaced it, a config with
    catalyst > 0 would silently lose it and the replay guard would stop firing."""
    from scripts.backtest import apply_pillar_weights
    from src import config
    from src.backtest import signals
    monkeypatch.setattr(config, "_OVERRIDES", {}, raising=False)
    monkeypatch.setattr(config, "load_config", lambda: {
        "scoring": {"pillars": {"technical": 0.35, "fundamental": 0.30,
                                "sentiment": 0.35, "catalyst": 0.2}}})
    try:
        apply_pillar_weights("technical=0.1,fundamental=0.55")
        try:
            signals.assert_replayable()
        except ValueError as exc:
            assert "catalyst" in str(exc)
        else:
            raise AssertionError("a weighted catalyst pillar must still refuse to replay")
    finally:
        config.clear_overrides()


def test_shipped_weighting_asked_for_explicitly_is_not_a_new_trial(monkeypatch):
    """`--pillar-weights` set to what config.yaml already says is the SAME hypothesis.
    It must return None so no `pillars` key enters the hash — otherwise the baseline
    stops deduping and every default run after it re-counts as a fresh trial."""
    from scripts.backtest import apply_pillar_weights
    from src import config
    monkeypatch.setattr(config, "_OVERRIDES", {}, raising=False)
    monkeypatch.setattr(config, "load_config", lambda: {
        "scoring": {"pillars": {"technical": 0.35, "fundamental": 0.30,
                                "sentiment": 0.35, "learned": 0.0, "catalyst": 0.0}}})
    try:
        assert apply_pillar_weights("technical=0.35,fundamental=0.30,sentiment=0.35") is None
        assert apply_pillar_weights("technical=0.325,fundamental=0.325") is not None
    finally:
        config.clear_overrides()


def test_a_weighting_is_one_trial_and_the_default_hash_is_unchanged(tmp_path):
    """The identity rule. Legacy lines carry no `pillars` key; a default run must hash
    to exactly what it hashed before the field existed, or all 34 recorded
    configurations re-count the first time anyone runs the baseline again."""
    base = {"start": "2016-01-01", "rebalance": "BME", "top_frac": 0.1, "top_n": None,
            "cost_bps": 15.0}
    # Absent, and explicitly-None, and the shipped weighting: all the same hash.
    h = config_hash("composite", base, "broad:1500")
    assert config_hash("composite", {**base, "pillars": None}, "broad:1500") == h
    c = {"technical": 0.2, "fundamental": 0.45, "sentiment": 0.35,
         "learned": 0.0, "catalyst": 0.0}
    assert config_hash("composite", {**base, "pillars": c}, "broad:1500") != h
    # ...and two different weightings are two different trials, each counted once.
    d = {**c, "technical": 0.1, "fundamental": 0.55}
    assert log_trial(tmp_path, "composite", base, {}, universe="broad:1500") == 1
    assert log_trial(tmp_path, "composite", {**base, "pillars": c}, {},
                     universe="broad:1500") == 2
    assert log_trial(tmp_path, "composite", {**base, "pillars": c}, {},
                     universe="broad:1500") == 2      # a re-run is verification
    assert log_trial(tmp_path, "composite", {**base, "pillars": d}, {},
                     universe="broad:1500") == 3
    assert n_trial_runs(tmp_path) == 4


def test_sentiment_weight_is_not_rank_neutral_when_a_pillar_is_missing():
    """The structural fact this whole exercise turns on. `composite_3pillar` divides by
    a PER-ROW weight sum, so a name whose fundamental pillar is NaN is normalized by
    (wT + wS) while a complete name is normalized by (wT + wF + wS). Changing wS
    therefore shrinks the two groups toward 0.5 at DIFFERENT rates and reorders them
    against each other. With no NaN present the ratio rule does hold and the ranking is
    invariant — both halves are asserted here because only the contrast explains it."""
    from src.backtest.signals import composite_3pillar
    d = [pd.Timestamp("2020-01-31")]
    rng = np.random.default_rng(7)
    syms = [f"S{i:02d}" for i in range(30)]
    tech = pd.DataFrame([rng.random(30)], index=d, columns=syms)
    fund = pd.DataFrame([rng.random(30)], index=d, columns=syms)
    scored = pd.DataFrame([[True] * 30], index=d, columns=syms)
    hi = {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.35}
    lo = {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.10}

    def rk(w, f):   # ranks, rounded first: two weightings that differ only by a
        s = composite_3pillar(tech, f, scored, w).loc[d[0]]   # rescaling can still land
        return s.round(12).rank()                             # 1e-17 apart in float64

    assert rk(hi, fund).equals(rk(lo, fund))   # every pillar present -> wS cannot matter
    thin = fund.copy()
    thin.loc[d[0], syms[:8]] = np.nan          # 8 of 30 names with no fundamental pillar
    thin_hi, thin_lo = rk(hi, thin), rk(lo, thin)
    # ...and once one is missing, it does — by far more than a rounding artefact: the
    # thin names climb several places against the complete ones.
    assert (thin_hi - thin_lo).abs().max() >= 2
    # The reordering is strictly BETWEEN the two groups, never inside either: within a
    # group the weights are identical, so wS only moves complete names past thin ones.
    for grp in (syms[:8], syms[8:]):
        assert thin_hi[grp].rank().equals(thin_lo[grp].rank())
