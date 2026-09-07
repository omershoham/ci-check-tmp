import numpy as np
import pandas as pd

from src import technical


def _ind(**kw):
    base = {"price": 100, "sma20": 98, "sma50": 95, "sma200": 90, "rsi": 60, "macd_hist": 1.0,
            "pos_52w": 0.8, "vol_trend": 1.2, "ret_20": 5.0, "mom_va": 1.0}
    base.update(kw)
    return base


def test_uptrend_scores_above_downtrend():
    up, up_state = technical._score_and_state(_ind())
    dn, dn_state = technical._score_and_state(_ind(price=80, sma50=90, sma200=95, rsi=30,
                                                   macd_hist=-1.0, pos_52w=0.1, mom_va=-1.0,
                                                   ret_20=-8.0))
    assert up > dn
    assert "Uptrend" in up_state and "Downtrend" in dn_state


def test_rising_volume_into_falling_price_lowers_score():
    confirm, _ = technical._score_and_state(_ind(vol_trend=1.5, ret_20=5.0))
    distribute, _ = technical._score_and_state(_ind(vol_trend=1.5, ret_20=-5.0))
    assert confirm > distribute


def test_missing_indicators_are_neutral_not_crash():
    s, state = technical._score_and_state({})
    assert 0.0 <= s <= 1.0 and state == "n/a"


def test_rsi_all_gains_is_100_and_flat_is_none_free():
    close = pd.Series(np.arange(1, 40, dtype=float))
    assert technical._rsi(close) == 100.0
    close = pd.Series(np.linspace(100, 90, 40)) + pd.Series(np.tile([0, 1], 20))
    v = technical._rsi(close)
    assert v is not None and 0 <= v <= 100


def test_detect_cross_reports_golden_and_days_ago():
    fast = pd.Series([1, 1, 1, 1, 3, 3, 3], dtype=float)
    slow = pd.Series([2, 2, 2, 2, 2, 2, 2], dtype=float)
    kind, days = technical._detect_cross(fast, slow, lookback=10)
    assert kind == "golden" and days == 2
    assert technical._detect_cross(slow, slow) == (None, None)
