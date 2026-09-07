"""Ranking and portfolio metrics, with the multiple-testing guard built in.

Everything here is pure numpy/pandas + stdlib (NormalDist), so the harness adds no
dependency. Deflated Sharpe follows Bailey & López de Prado (2014): the observed
Sharpe is compared against the expected MAXIMUM Sharpe among N independent trials
with the same variance — which is what "I tried a dozen variants and kept the best"
actually produces.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()
_EULER_GAMMA = 0.5772156649015329


def rank_ic(scores: pd.Series, fwd: pd.Series) -> float | None:
    """Spearman rank correlation between a cross-section of scores and forward returns."""
    j = pd.concat([scores, fwd], axis=1, keys=["s", "f"]).dropna()
    if len(j) < 10:
        return None
    v = j["s"].rank().corr(j["f"].rank())
    return None if v != v else float(v)


def fama_macbeth(ics: pd.Series) -> dict:
    """Mean and t-stat of a per-date IC series (dates are the independent samples)."""
    v = ics.dropna()
    n = len(v)
    if n == 0:
        return {"n_dates": 0, "ic_mean": None, "ic_t": None, "ic_pos_share": None}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    t = mean / (sd / np.sqrt(n)) if sd > 0 else None
    return {"n_dates": n, "ic_mean": round(mean, 4), "ic_t": round(t, 2) if t is not None else None,
            "ic_pos_share": round(float((v > 0).mean()), 3)}


def decile_returns(scores: pd.Series, fwd: pd.Series, n_bins: int = 10) -> pd.Series | None:
    """Mean forward return per score decile (1 = lowest score ... n_bins = highest)."""
    j = pd.concat([scores, fwd], axis=1, keys=["s", "f"]).dropna()
    if len(j) < n_bins * 2:
        return None
    bins = pd.qcut(j["s"].rank(method="first"), n_bins, labels=False) + 1
    return j.groupby(bins)["f"].mean()


def sharpe(returns: pd.Series, periods_per_year: float) -> float | None:
    r = returns.dropna()
    if len(r) < 3 or r.std(ddof=1) == 0:
        return None
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float | None:
    r = returns.dropna()
    if r.empty:
        return None
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    return float(dd.min())


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int,
                         skew: float = 0.0, kurt: float = 3.0) -> float | None:
    """P(true SR > sr_benchmark) given the observed SR over n_obs periods (non-normal
    returns widen the SR's standard error via skew/kurtosis)."""
    if n_obs < 3:
        return None
    var = (1 - skew * sr + (kurt - 1) / 4 * sr * sr) / (n_obs - 1)
    if var <= 0:
        return None
    return float(_N.cdf((sr - sr_benchmark) / np.sqrt(var)))


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """Expected max Sharpe among n_trials zero-skill trials with SR variance sr_var."""
    if n_trials <= 1:
        return 0.0
    return float(np.sqrt(sr_var) * ((1 - _EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials)
                                    + _EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * np.e))))


def deflated_sharpe(returns: pd.Series, n_trials: int, periods_per_year: float,
                    sr_var_across_trials: float | None = None) -> dict:
    """Deflated Sharpe Ratio: PSR of the observed (per-period) SR against the expected
    max SR of `n_trials` skill-less trials. `sr_var_across_trials` defaults to the
    SR's own estimation variance, a conservative stand-in when the trial SRs are not
    all recorded. Returns {'sr_ann', 'sr0', 'dsr'}; dsr > 0.95 is the usual bar."""
    r = returns.dropna()
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return {"sr_ann": None, "sr0": None, "dsr": None}
    sr = float(r.mean() / r.std(ddof=1))           # per period
    skew = float(r.skew()) if n > 3 else 0.0
    kurt = float(r.kurt()) + 3 if n > 3 else 3.0
    est_var = (1 - skew * sr + (kurt - 1) / 4 * sr * sr) / (n - 1)
    var = sr_var_across_trials if sr_var_across_trials is not None else max(est_var, 1e-12)
    sr0 = expected_max_sharpe(max(int(n_trials), 1), var)
    dsr = probabilistic_sharpe(sr, sr0, n, skew, kurt)
    return {"sr_ann": round(sr * np.sqrt(periods_per_year), 3),
            "sr0_ann": round(sr0 * np.sqrt(periods_per_year), 3),
            "dsr": round(dsr, 3) if dsr is not None else None}


def turnover(prev_w: pd.Series | None, new_w: pd.Series) -> float:
    """Σ|Δw| over the union of names (both legs of every rotation)."""
    if prev_w is None or prev_w.empty:
        return float(new_w.abs().sum())
    idx = prev_w.index.union(new_w.index)
    return float((new_w.reindex(idx).fillna(0) - prev_w.reindex(idx).fillna(0)).abs().sum())
