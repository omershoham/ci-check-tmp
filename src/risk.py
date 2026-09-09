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
        # Stable keys first: the ticker-derived ones below change name whenever the
        # configured benchmark changes, which silently turned every consumer reading
        # them by literal name into None. Prefer beta_bench / beta_theme.
        "beta_bench": beta_bench,
        "beta_theme": beta_theme,
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


def max_sector_weight() -> float:
    """The concentration cap, in percent. `policy.max_sector_weight` with the older
    `policy.max_subsector_weight` honoured as an alias so an existing user config
    keeps binding instead of silently falling back to the built-in default."""
    v = config.get("policy.max_sector_weight")
    if v is None:
        v = config.get("policy.max_subsector_weight", 40)
    return float(v or 0)


def default_add_size_pct(atr_pct: float | None = None) -> float:
    """The weight a NEW position would be opened at, in percent of the book.

    Stated assumption, because there is no such thing as a size-free "what if I buy
    it": the owner's own two policy knobs. `policy.risk_per_position_pct` risked to a
    2-ATR stop gives weight = risk% / (2 * ATR%); the LOWEST conviction cap in
    `policy.conviction_weight_caps` bounds it, because an unresearched candidate has
    no conviction yet and must not be sized as though it did. With no ATR the cap
    alone is the answer.
    """
    caps = config.get("policy.conviction_weight_caps", {3: 10, 5: 15}) or {}
    try:
        cap = min(float(v) for v in caps.values())
    except (TypeError, ValueError):
        cap = float(config.get("policy.max_position_weight", 20) or 20)
    if atr_pct and atr_pct > 0:
        # atr_pct is a FRACTION of price (technical.py's convention), so the 2-ATR stop
        # is a 2*atr_pct move and the position that risks risk_pct of the book to it is
        # risk_pct / (2*atr_pct) percent of the book.
        risk_pct = float(config.get("policy.risk_per_position_pct", 1.0) or 0)
        if risk_pct > 0:
            return round(min(cap, risk_pct / (2.0 * float(atr_pct))), 1)
    return round(cap, 1)


def _fit_from_returns(symbol: str, rets: pd.DataFrame | None,
                      weights: dict[str, float], sectors: dict[str, str] | None,
                      size_pct: float) -> dict[str, Any]:
    """The correlation half of candidate_fit, given a returns frame already downloaded."""
    out: dict[str, Any] = {"symbol": symbol, "size_pct": size_pct}
    held = [s for s, w in weights.items() if w and w > 0 and s != symbol]
    if rets is None or symbol not in rets.columns:
        return out
    have = [s for s in held if s in rets.columns and rets[s].notna().sum() >= 40]
    if not have or rets[symbol].notna().sum() < 40:
        return out
    r = rets[[symbol] + have].dropna()
    if len(r) < 40:
        return out
    cm = r.corr()
    w = np.array([weights[s] for s in have], dtype=float)
    w = w / w.sum()
    row = np.array([float(cm.loc[symbol, s]) for s in have])
    # Same definition portfolio_risk uses for a HOLDING's marginal_corr — a
    # weight-averaged correlation to the rest of the book — so a candidate's number is
    # comparable to the ones already on the portfolio page instead of being a second,
    # subtly different statistic with the same name.
    out["marginal_corr"] = round(float((row * w).sum()), 2)
    j = int(np.argmax(row))
    out["closest"] = have[j]
    out["closest_corr"] = round(float(row[j]), 2)
    out["lookback_days"] = int(len(r))
    return out


def candidate_fits(symbols: list[str], weights: dict[str, float],
                   sectors: dict[str, str] | None = None,
                   sizes: dict[str, float] | None = None) -> dict[str, dict[str, Any]]:
    """Book fit for several prospective adds at once — ONE price download for all.

    `weights` is symbol -> % of the current book (the same shape portfolio_risk takes).
    Per candidate: its marginal correlation to the book, its closest existing position,
    and what buying it at `sizes[symbol]` (default: default_add_size_pct) would do to
    its sector's weight against `policy.max_sector_weight`.

    Why this exists: `portfolio_risk.marginal_corr` answers the question only for names
    you ALREADY hold, so a screen that looks diversified can quietly rebuild the
    concentration you started with — the fifth semiconductor idea is scored exactly
    like the first. Batched because one yfinance download for 10 candidates plus the
    book is one network round trip; ten separate candidate_fit calls are ten.
    """
    syms = [s for s in dict.fromkeys(symbols) if s]
    held = [s for s, w in (weights or {}).items() if w and w > 0]
    sectors = sectors or {}
    cap = max_sector_weight()
    sector_now: dict[str, float] = {}
    for s in held:
        sector_now[sectors.get(s) or "Unknown"] = (
            sector_now.get(sectors.get(s) or "Unknown", 0.0) + float(weights[s]))
    rets = _returns(list(dict.fromkeys(syms + held))) if (syms and held) else None
    out: dict[str, dict[str, Any]] = {}
    for s in syms:
        size = float((sizes or {}).get(s) or default_add_size_pct())
        fit = _fit_from_returns(s, rets, weights or {}, sectors, size)
        sec = sectors.get(s) or "Unknown"
        # The book is rescaled by the new position: existing weights are diluted to
        # (100 - size)% of themselves. Adding `size` to the sector total without that
        # dilution overstates the resulting weight for every large add.
        now = float(sector_now.get(sec, 0.0))
        scale = (100.0 - size) / 100.0 if 0 <= size < 100 else 1.0
        after = now * scale + size
        fit.update({
            "sector": sec,
            "sector_positions": sum(1 for s2 in held if (sectors.get(s2) or "Unknown") == sec),
            "sector_weight_pct": round(now, 1),
            "sector_weight_after_pct": round(after, 1),
            "max_sector_weight_pct": cap or None,
            "over_cap": bool(cap and after > cap),
        })
        out[s] = fit
    return out


def candidate_fit(symbol: str, weights: dict[str, float],
                  sectors: dict[str, str] | None = None,
                  size_pct: float | None = None) -> dict[str, Any]:
    """Book fit for ONE prospective add. See candidate_fits."""
    sizes = {symbol: size_pct} if size_pct is not None else None
    return candidate_fits([symbol], weights, sectors, sizes).get(symbol, {"symbol": symbol})


def candidate_fit_line(fit: dict[str, Any] | None) -> str:
    """One prompt/board-ready sentence, or '' when there is nothing measured to say."""
    if not fit:
        return ""
    parts = []
    n = fit.get("sector_positions")
    sec = fit.get("sector")
    if sec and n is not None:
        parts.append(f"would be your {_ordinal(int(n) + 1)} {sec} position")
    if fit.get("closest") and fit.get("marginal_corr") is not None:
        parts.append(f"corr {fit['marginal_corr']:.2f} to the book, "
                     f"{fit['closest_corr']:.2f} with {fit['closest']}")
    if fit.get("sector_weight_after_pct") is not None and fit.get("sector") :
        cap = fit.get("max_sector_weight_pct")
        tail = ""
        if cap:
            tail = (f", ABOVE the {cap:g}% cap" if fit.get("over_cap")
                    else f", under the {cap:g}% cap")
        parts.append(f"sector weight {fit.get('sector_weight_pct'):g}% -> "
                     f"{fit['sector_weight_after_pct']:g}% at a "
                     f"{fit.get('size_pct'):g}% add{tail}")
    if not parts:
        return ""
    return "Book fit: " + "; ".join(parts) + "."


def candidate_fit_badge(fit: dict[str, Any] | None) -> str:
    """The compact board-column form: `0.81↔NVDA · IT 41%`."""
    if not fit:
        return ""
    bits = []
    if fit.get("closest") and fit.get("closest_corr") is not None:
        bits.append(f"{fit['closest_corr']:.2f}↔{fit['closest']}")
    sec, after = fit.get("sector"), fit.get("sector_weight_after_pct")
    if sec and after is not None:
        bits.append(f"{_sector_abbr(sec)} {after:.0f}%"
                    + ("⚠" if fit.get("over_cap") else ""))
    return " · ".join(bits)


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _sector_abbr(sector: str) -> str:
    """Initials of a GICS-like sector name, so a board column stays a column."""
    words = [w for w in str(sector).replace("-", " ").split() if w[:1].isalpha()]
    return "".join(w[0].upper() for w in words)[:3] or "?"


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
        f"  - portfolio beta vs {rm.get('benchmark')}: {rm.get('beta_bench')}; "
        f"vs {rm.get('theme_benchmark')} (theme proxy): {rm.get('beta_theme')}\n"
        f"  - scenario: {rm.get('theme_benchmark')} {rm.get('scenario_shock_pct', -20):.0f}% -> portfolio "
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
