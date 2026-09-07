import numpy as np
import pandas as pd

from src import deep_research, regime, risk


def _fake_returns(symbols, period="9mo"):
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2026-01-01", periods=90)
    common = rng.normal(0, 0.02, 90)
    data = {}
    for i, s in enumerate(symbols):
        if s in ("QQQ", "SMH"):
            data[s] = common * 0.8 + rng.normal(0, 0.005, 90)
        elif s == "DIV":                       # the uncorrelated diversifier
            data[s] = rng.normal(0, 0.01, 90)
        else:                                  # the AI cluster: mostly common factor
            data[s] = common + rng.normal(0, 0.008, 90)
    return pd.DataFrame(data, index=idx)


def test_portfolio_risk_reports_vol_scale_drawdown_and_marginal_corr(monkeypatch):
    monkeypatch.setattr(risk, "_returns", _fake_returns)
    monkeypatch.setattr(risk.config, "get", lambda k, d=None: {"policy.vol_target_pct": 12.0,
                                                                "risk.benchmark": "QQQ",
                                                                "risk.theme_benchmark": "SMH"}.get(k, d))
    rm = risk.portfolio_risk({"AAA": 40, "BBB": 40, "DIV": 20})
    assert rm["benchmark"] == "QQQ" and rm["theme_benchmark"] == "SMH"
    assert rm["ann_vol_pct"] > 12 and 0 < rm["vol_scale"] < 1        # too hot -> de-risk
    assert rm["drawdown_from_peak_pct"] <= 0
    mc = rm["marginal_corr"]
    # AAA's marginal corr = weighted avg of corr(AAA,BBB)~0.85 (w 40) and corr(AAA,DIV)~0
    # (w 20) ~ 0.57; the diversifier's is ~0.
    assert mc["DIV"] < 0.2 < 0.4 < mc["AAA"]
    lines = risk.risk_context_lines(rm)
    assert "scale gross exposure by" in lines and "drawdown from the lookback peak" in lines


def test_sizing_rules_tighten_on_drawdown_and_vol(monkeypatch):
    monkeypatch.setattr(regime, "detect", lambda *a, **k: {"regime": "risk_on"})
    base = deep_research.sizing_rules_text()
    assert "TIGHTENED" not in base and "DRAWDOWN CONTROL" not in base
    hot = deep_research.sizing_rules_text({"drawdown_from_peak_pct": -14.0, "vol_scale": 0.7,
                                           "ann_vol_pct": 17.0, "vol_target_pct": 12.0})
    assert "DRAWDOWN CONTROL" in hot and "VOLATILITY TARGET" in hot and "70%" in hot
    # caps in the drawdown case are 25% lower than the base caps
    import re
    def cap(txt):
        return float(re.search(r"conviction <=3 must not remain above ~([\d.]+)%", txt).group(1))
    assert abs(cap(hot) - cap(base) * 0.75) < 0.11
    calm = deep_research.sizing_rules_text({"drawdown_from_peak_pct": -3.0, "vol_scale": 1.0})
    assert "DRAWDOWN CONTROL" not in calm and "VOLATILITY TARGET" not in calm
