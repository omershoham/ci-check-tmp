"""The dashboard renders without raising, against a seeded temp DB.

The repo had no dashboard test at all, so every board-facing change (a renamed tag, a
column that no longer exists, a signal shape) was only ever discovered by opening the
page. `streamlit.testing.v1.AppTest` runs the script exactly as Streamlit does, in
this process, with no browser and no server.
"""
from __future__ import annotations

import json

import pytest

from src import local_llm

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(pytest.importorskip("pathlib").Path(__file__).resolve().parents[1]
          / "app" / "dashboard.py")


def _vec(**kw):
    return {k: kw.get(k, 0) for k in local_llm.CATALYST_KEYS}


@pytest.fixture
def seeded(storage, monkeypatch):
    """One finished run with a two-name board, one of which carries a catalyst
    vector — enough for every board/detail branch to execute."""
    rid = storage.create_run("deep")
    storage.upsert_tickers([
        {"symbol": "VRT", "name": "Vertiv", "subsector": "datacenter_power"},
        {"symbol": "NVDA", "name": "NVIDIA", "subsector": "semis_equipment"}])
    storage.upsert_screen_scores(rid, [
        {"symbol": "VRT", "quality": 0.72, "blended": 0.81, "rank": 1,
         "tags": ["quality"],
         "metrics": {"pillar_technical": 0.8, "pillar_fundamental": 0.7,
                     "pillar_sentiment": 0.6, "pillar_catalyst": 0.11,
                     "catalyst": _vec(customer_deal=8, product_cycle=6),
                     "signal_version": 2, "price": 120.0, "trailingPE": 30.0,
                     "sector": "Industrials", "fund_coverage": 0.9,
                     "tech_state": "Uptrend", "rsi": 55.0, "sentiment_label": "bullish"}},
        {"symbol": "NVDA", "quality": 0.6, "blended": 0.7, "rank": 2,
         "tags": ["higher-risk", "earnings-soon"],
         "metrics": {"pillar_technical": 0.7, "pillar_fundamental": 0.5,
                     "pillar_sentiment": 0.5, "pillar_catalyst": None,
                     "catalyst": None, "price": 200.0, "sector": "Information Technology",
                     "fund_coverage": 0.8}}])
    storage.upsert_signals(rid, [
        {"symbol": "VRT", "catalyst": _vec(customer_deal=8, product_cycle=6),
         "signal_version": 2, "hyperscaler_signal": 8,
         "catalysts": ["hyperscaler order"], "summary": "strong demand"}])
    storage.upsert_research(rid, {
        "symbol": "VRT", "rating": "buy", "conviction": 7.0, "thesis": "t",
        "chart_pattern": "up", "metrics_read": "m", "catalysts": ["c"], "risks": ["r"],
        "bull": "b", "bear": "b", "data_as_of": "2026-09-08", "prompt_version": 6,
        "model": "m", "horizon": "swing", "horizon_rationale": "print on the 21st",
        "horizon_source": "claude", "quant_horizon": "position",
        "expected_move_pct": 9.0, "invalidation_level": 101.0})
    storage.finish_run(rid, "ok")
    return rid


def test_dashboard_renders_without_exception(seeded):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    # The board actually rendered (title + at least one dataframe).
    assert at.title and at.title[0].value
    assert len(at.dataframe) >= 1


def test_board_table_shows_the_catalyst_column_and_no_deal_tag(seeded):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    board = at.dataframe[0].value
    assert "⚡ catalyst" in board.columns
    row = board[board["symbol"].str.startswith("VRT")].iloc[0]
    assert row["⚡ catalyst"] == "deal 8 · product 6"   # top 2 non-zero keys
    # The deleted tags must not come back through a stale badge map.
    assert "deal" not in " ".join(board["tags"].tolist())


def test_board_shows_the_horizon_column(seeded):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    board = at.dataframe[0].value
    assert "horizon" in board.columns
    vrt = board[board["symbol"].str.startswith("VRT")].iloc[0]
    assert vrt["horizon"] == "swing"              # Claude's own clock, no marker
    nvda = board[board["symbol"].str.startswith("NVDA")].iloc[0]
    # Never researched: the quant default, flagged as such.
    assert nvda["horizon"].endswith(" *")
    assert nvda["horizon"].split()[0] in ("swing", "position", "long")


def test_board_has_a_horizon_filter(seeded):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    radios = {r.label: r for r in at.radio}
    assert "Horizon" in radios
    assert radios["Horizon"].options == ["All", "swing", "position", "long"]
    at.radio(key="horizon_filter").set_value("long").run()
    assert not at.exception, at.exception
    # VRT claimed 'swing', so filtering to 'long' must drop it from the board.
    board = at.dataframe[0].value
    assert board.empty or not board["symbol"].str.startswith("VRT").any()


def test_accuracy_tab_has_a_clock_toggle(seeded):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    clock = at.radio(key="cal_clock")
    assert clock.options == ["Fixed windows (1w/4w/13w)", "Each idea's own horizon"]
    at.radio(key="cal_clock").set_value("Each idea's own horizon").run()
    assert not at.exception, at.exception


def test_tag_filter_offers_no_deal_options(seeded):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    tag_ms = [ms for ms in at.multiselect if "tag" in (ms.label or "").lower()]
    assert tag_ms and not ({"deal", "both"} & set(tag_ms[0].options))
