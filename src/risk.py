"""Portfolio risk analytics: real correlation math instead of adjectives.

Computes from free yfinance daily history:
  - pairwise 90-day return correlations of the holdings
  - average intra-portfolio correlation (weighted)
  - portfolio beta vs a benchmark (QQQ) and vs the AI-theme proxy (SMH)
  - effective number of independent bets (correlation- and weight-adjusted)
  - a theme-shock scenario: estimated portfolio drawdown if SMH falls X%

Feeds numbers into the portfolio synthesis prompt and the dashboard, so "very high
correlation" becomes "avg pairwise 0.72, 2.3 effective bets, -14% in an SMH -20% shock".
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from . import config

_LOOKBACK_DAYS = 90


def _bench() -> str:
    return config.get("risk.benchmark", "QQQ")


def _theme() -> str:
    # One source of truth for the theme proxy: the relative-strength benchmark.
    return config.get("risk.theme_benchmark") or config.get("technical.rs_benchmark", "SMH")


def _returns(symbols: list[str], period: str = "9mo") -> pd.DataFrame | None:
    try:
        px = yf.download(list(dict.fromkeys(symbols)), period=period, interval="1d",
                         auto_adjust=True, progress=False, threads=True)["Close"]
        if isinstance(px, pd.Series):
            px = px.to_frame(symbols[0])
        px = px.dropna(how="all")
        rets = px.pct_change().dropna(how="all").tail(_LOOKBACK_DAYS)
        return rets if not rets.empty else None
    except Exception:  # noqa: BLE001
        return None


def portfolio_risk(weights: dict[str, float], shock_pct: float = -20.0) -> dict[str, Any]:
    """weights: symbol -> % of portfolio (0-100). Returns computed risk metrics.

    All values best-effort; missing tickers are dropped and weights renormalized.
    """
    syms = [s for s, w in weights.items() if w and w > 0]
    if len(syms) < 2:
        return {}
    _BENCH, _THEME = _bench(), _theme()
    rets = _returns(syms + [_BENCH, _THEME])
    if rets is None:
        return {}
    have = [s for s in syms if s in rets.columns and rets[s].notna().sum() >= 40]
    if len(have) < 2:
        return {}
    w = np.array([weights[s] for s in have], dtype=float)
    w = w / w.sum()  # renormalize over the names we actually have data for

    r = rets[have].dropna()
    corr = r.corr()
    # Weighted average pairwise correlation (off-diagonal)
    cm = corr.values
    ww = np.outer(w, w)
    off = ~np.eye(len(have), dtype=bool)
    avg_corr = float((cm[off] * ww[off]).sum() / ww[off].sum()) if off.any() else 0.0

    # Effective number of independent bets: 1 / (w' C w) with unit-variance assets.
    # N names, all uncorrelated, equal weight -> N; all perfectly correlated -> 1.
    denom = float(w @ cm @ w)
    eff_bets = round(1.0 / denom, 1) if denom > 0 else None

    # Diversification ratio (Choueifaty-Coignard) uses the COVARIANCE matrix, so a
    # high-vol name at 22% weight counts for its real risk — the correlation-based
    # eff_bets treats all names as equally volatile and understates concentration
    # exactly when a big position is also the riskiest one. DR^2 ≈ vol-aware bets.
    cov = r.cov().values
    sig = np.sqrt(np.diag(cov))
    port_var = float(w @ cov @ w)
    dr = float((w @ sig) / np.sqrt(port_var)) if port_var > 0 else None
    vol_aware_bets = round(dr * dr, 1) if dr else None

    port_ret = (r * w).sum(axis=1)
    # Empirical worst 20-day portfolio return in the lookback — a floor-free reality
    # check next to the model-based beta scenario.
    roll20 = (1 + port_ret).rolling(20).apply(lambda x: x.prod(), raw=True) - 1
    worst_20d = round(float(roll20.min()) * 100, 1) if roll20.notna().any() else None

    def _beta(bench: str) -> float | None:
        if bench not in rets.columns:
            return None
        b = rets[bench].reindex(port_ret.index).dropna()
        p = port_ret.reindex(b.index)
        var = float(b.var())
        return round(float(p.cov(b) / var), 2) if var else None

    beta_bench = _beta(_BENCH)
    beta_theme = _beta(_THEME)
    # Linear beta scenario is a FLOOR — correlations and betas rise in selloffs.
    # The stressed variant (beta x1.3) approximates that crisis behavior.
    scenario = (round(beta_theme * shock_pct, 1)
                if isinstance(beta_theme, (int, float)) else None)
    scenario_stressed = (round(beta_theme * 1.3 * shock_pct, 1)
                         if isinstance(beta_theme, (int, float)) else None)

    # Highest-correlated pair (the most redundant two holdings)
    top_pair = None
    if len(have) >= 2:
        best = -2.0
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                if cm[i, j] > best:
                    best = cm[i, j]
                    top_pair = (have[i], have[j], round(float(cm[i, j]), 2))

    dropped = [s for s in syms if s not in have]

    # Volatility targeting (policy.vol_target_pct): the scale that brings the book's
    # realized vol to target. <1 = de-risk gross exposure; capped at 1 (no leverage —
    # advisory tool). Fractional-Kelly in spirit: size to risk, not to conviction alone.
    ann_vol = float(port_ret.std() * np.sqrt(252) * 100)
    vol_target = float(config.get("policy.vol_target_pct", 12.0) or 0)
    vol_scale = round(min(1.0, vol_target / ann_vol), 2) if (ann_vol > 0 and vol_target > 0) else None

    # Drawdown from the lookback peak of the (current-weights) book — the trigger for
    # the drawdown-control rule in sizing (policy.drawdown_tighten_pct).
    wealth = (1 + port_ret).cumprod()
    dd_now = round(float(wealth.iloc[-1] / wealth.max() - 1) * 100, 1) if len(wealth) else None

    # Marginal correlation: each holding's weighted average correlation to the REST of
    # the book. High = adding to it buys more of the same risk; low = real diversification.
    marginal: dict[str, float] = {}
    for i, s in enumerate(have):
        others = [j for j in range(len(have)) if j != i]
        if others:
            wo = w[others]
            marginal[s] = round(float((cm[i, others] * wo).sum() / wo.sum()), 2)

    return {
        "avg_pairwise_corr": round(avg_corr, 2),
        "effective_bets": eff_bets,
        "vol_aware_bets": vol_aware_bets,
        "n_holdings": len(have),
        "dropped_symbols": dropped,  # no price data — metrics exclude them
        f"beta_{_BENCH.lower()}": beta_bench,
        f"beta_{_THEME.lower()}": beta_theme,
        "benchmark": _BENCH, "theme_benchmark": _THEME,
        "scenario_shock_pct": shock_pct,
        "scenario_drawdown_pct": scenario,
        "scenario_stressed_pct": scenario_stressed,
        "worst_20d_pct": worst_20d,
        "most_correlated_pair": top_pair,
        "ann_vol_pct": round(ann_vol, 1),
        "vol_target_pct": vol_target or None,
        "vol_scale": vol_scale,
        "drawdown_from_peak_pct": dd_now,
        "marginal_corr": marginal,
        "lookback_days": int(len(r)),
    }


def risk_context_lines(rm: dict[str, Any]) -> str:
    """Render risk metrics as prompt-ready lines for the synthesis stage."""
    if not rm:
        return "(risk metrics unavailable)"
    pair = rm.get("most_correlated_pair")
    pair_s = f"{pair[0]}/{pair[1]} at {pair[2]}" if pair else "n/a"
    dropped = rm.get("dropped_symbols") or []
    return (
        f"COMPUTED RISK METRICS ({rm.get('lookback_days')}d daily returns):\n"
        f"  - average weighted pairwise correlation: {rm.get('avg_pairwise_corr')}\n"
        f"  - effective independent bets: {rm.get('effective_bets')} "
        f"(vol-aware: {rm.get('vol_aware_bets')}) out of {rm.get('n_holdings')} holdings\n"
        f"  - portfolio beta vs QQQ: {rm.get('beta_qqq')}; vs SMH (AI-theme proxy): "
        f"{rm.get('beta_smh')}\n"
        f"  - scenario: SMH {rm.get('scenario_shock_pct', -20):.0f}% -> portfolio "
        f"~{rm.get('scenario_drawdown_pct')}% (stressed, beta x1.3: "
        f"~{rm.get('scenario_stressed_pct')}%); worst actual 20d in lookback: "
        f"{rm.get('worst_20d_pct')}%\n"
        f"  - annualized portfolio volatility: {rm.get('ann_vol_pct')}%"
        + (f" vs target {rm['vol_target_pct']:g}% -> scale gross exposure by "
           f"{rm['vol_scale']} (1.0 = no change; below 1 = de-risk)"
           if rm.get("vol_scale") is not None else "") + "\n"
        f"  - drawdown from the lookback peak: {rm.get('drawdown_from_peak_pct')}%\n"
        f"  - most redundant pair: {pair_s}"
        + (f"\n  - EXCLUDED (no price data): {', '.join(dropped)}" if dropped else "")
    )


if __name__ == "__main__":
    # Synthetic example weights (round numbers summing to 100) — a smoke test only,
    # deliberately NOT a real portfolio. Pass your own dict to try other shapes.
    demo = {"NVDA": 15.0, "AVGO": 15.0, "TSM": 10.0, "VRT": 10.0,
            "ANET": 10.0, "GOOGL": 15.0, "AMZN": 15.0, "MSFT": 10.0}
    rm = portfolio_risk(demo)
    print(risk_context_lines(rm))
