import pandas as pd

from src import config, performance


def test_wilson_ci_is_sane():
    lo, hi = performance.wilson_ci(0, 0)
    assert (lo, hi) == (0.0, 100.0)
    lo, hi = performance.wilson_ci(10, 10)
    assert hi == 100.0 and lo > 60          # small-n certainty is not certainty
    lo, hi = performance.wilson_ci(50, 100)
    assert 39 < lo < 45 and 55 < hi < 61
    lo, hi = performance.wilson_ci(0, 8)
    assert lo == 0.0 and hi > 30


def test_memo_evicts_other_days():
    performance._MEMO.clear()
    performance._MEMO[("px", "yesterday")] = 1
    key = performance._memo_key("px")
    assert ("px", "yesterday") not in performance._MEMO
    assert key[-1] != "yesterday"


def _rows():
    # buys: mostly positive; screen-only: around zero; one hold.
    rows = []
    for i in range(12):
        rows.append({"run_id": i, "symbol": f"B{i}", "rating": "buy", "conviction": 6,
                     "prompt_version": 4, "regime": "neutral",
                     "net_excess_uni_4w": 2.0 + (i % 3) - 0.3})
    for i in range(12):
        rows.append({"run_id": i, "symbol": f"S{i}", "rating": None, "conviction": None,
                     "prompt_version": None, "regime": "neutral",
                     "net_excess_uni_4w": -0.5 + (i % 2)})
    rows.append({"run_id": 99, "symbol": "H", "rating": "hold", "conviction": 3,
                 "prompt_version": None, "regime": None, "net_excess_uni_4w": 0.4})
    return rows


def test_calibration_table_groups_and_flags_small_cells(monkeypatch):
    monkeypatch.setattr(performance, "pick_outcomes", _rows)
    tbl = performance.calibration_table("4w", min_n=8)
    assert set(tbl["rating"]) == {"buy", "screen_only", "hold"}
    buy = tbl[tbl["rating"] == "buy"].iloc[0]
    assert buy["n"] == 12 and buy["precision_pct"] == 100.0 and buy["reliable"]
    assert buy["ci_lo"] < 100.0                      # Wilson never claims certainty at n=12
    hold = tbl[tbl["rating"] == "hold"].iloc[0]
    assert hold["regime"] == "unknown" and not hold["reliable"]
    assert list(tbl["rating"])[0] == "buy"           # ordered bullish -> screen_only


def test_claude_vs_screen_sign_and_ci(monkeypatch):
    monkeypatch.setattr(performance, "pick_outcomes", _rows)
    a = performance.claude_vs_screen("4w", n_boot=500)
    assert a["n_claude"] == 12 and a["n_screen"] == 12
    assert a["edge_pp"] > 0 and a["ci_lo"] > 0 and a["significant"]


def test_claude_vs_screen_needs_both_cohorts(monkeypatch):
    monkeypatch.setattr(performance, "pick_outcomes", lambda: _rows()[:12])
    a = performance.claude_vs_screen("4w")
    assert "note" in a and "edge_pp" not in a


def test_cost_only_charged_to_buys(monkeypatch):
    monkeypatch.setattr(config, "get", lambda k, d=None: 50 if k == "evaluation.cost_bps" else d)
    assert performance._cost_pct("buy") == 0.5
    assert performance._cost_pct("Strong_Buy") == 0.5
    assert performance._cost_pct("hold") == 0.0
    assert performance._cost_pct(None) == 0.0


def test_by_prompt_version_labels_legacy(monkeypatch):
    monkeypatch.setattr(performance, "pick_outcomes", _rows)
    df = performance.calibration_by_prompt_version("4w")
    assert set(df["prompt"]) == {"v4", "pre-v4"}
    assert isinstance(df, pd.DataFrame) and "ci_lo" in df
