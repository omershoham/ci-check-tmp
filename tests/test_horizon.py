"""Phase 6: every idea carries its own clock, and is graded on THAT clock.

The load-bearing property tested here is not that the new numbers appear — it is that
adding them changes NOTHING about the fixed 1w/4w/13w series that six statistics and
every historical pick already depend on.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import deep_research, learning, performance, technical

# ------------------------------------------------------------------ prompt validation


def test_coerce_report_enum_checks_the_horizon():
    ok = deep_research._coerce_report({"rating": "buy", "conviction": 7, "horizon": " Swing "})
    assert ok["horizon"] == "swing"
    for bad in ("weeks", "medium", "", None, 3, "swing trade"):
        assert deep_research._coerce_report({"horizon": bad})["horizon"] is None


def test_coerce_report_clamps_expected_move_and_bounds_invalidation():
    r = deep_research._coerce_report({"expected_move_pct": 5000, "invalidation_level": 41.5})
    assert r["expected_move_pct"] == 300.0 and r["invalidation_level"] == 41.5
    r = deep_research._coerce_report({"expected_move_pct": -999, "invalidation_level": 0})
    assert r["expected_move_pct"] == -90.0 and r["invalidation_level"] is None
    r = deep_research._coerce_report({"expected_move_pct": "n/a", "invalidation_level": "$40"})
    assert r["expected_move_pct"] is None and r["invalidation_level"] is None


def test_prompt_is_horizon_conditional_and_versioned():
    assert deep_research.PROMPT_VERSION == 6
    t = deep_research._PROMPT_TMPL
    assert "{horizon_menu}" in t
    for key in ('"horizon"', '"horizon_rationale"', '"expected_move_pct"',
                '"invalidation_level"'):
        assert key in t
    # The coupling is the point: the clock has to constrain the evidence.
    assert "swing" in t and "RESOLVES inside" in t
    assert "must NOT be justified by a near-term catalyst" in t
    # The menu is rendered from config, so the spans can't drift from the day counts.
    menu = deep_research.horizon_menu_text()
    assert "10 trading days" in menu and "126 trading days" in menu


def test_quant_horizon_never_reaches_the_prompt(monkeypatch):
    """Claude's clock must be an independent choice or agreement measures anchoring."""
    seen = {}

    def _fake(prompt, allowed_tools=None, timeout=None, extra_args=None):
        seen["prompt"] = prompt
        return '{"rating":"buy","conviction":6,"horizon":"long"}', "raw"

    monkeypatch.setattr(deep_research, "_run_claude", _fake)
    out = deep_research.research_one("VRT", "Vertiv", "x", context="Quant blended 0.7")
    assert out["horizon"] == "long"
    assert "quant_horizon" not in seen["prompt"] and "default_horizon" not in seen["prompt"]


# ------------------------------------------------------------------ the quant default


def _tech(**kw):
    base = {"atr_pct": 0.025, "pos_52w": 0.7, "setup": "bullish", "tags": []}
    base.update(kw)
    return base


def test_default_horizon_rule():
    # 1. fast tape or an extended/breakout position -> resolves soon
    assert technical.default_horizon(_tech(atr_pct=0.06)) == "swing"
    assert technical.default_horizon(_tech(pos_52w=0.99)) == "swing"
    # 2. low-vol quality on an intact trend -> compounding, not an event
    assert technical.default_horizon(_tech(atr_pct=0.015, tags=["quality"])) == "long"
    # ...but a quality name breaking out is still a swing (rule 1 wins)
    assert technical.default_horizon(
        _tech(atr_pct=0.015, pos_52w=0.99, tags=["quality"])) == "swing"
    # 3. an ordinary established uptrend
    assert technical.default_horizon(_tech()) == "position"
    # 4. anything else falls back to the configured default
    assert technical.default_horizon(_tech(setup="bearish")) == "position"
    assert technical.default_horizon({}) == "position"


def test_default_horizon_is_deterministic():
    t = _tech(atr_pct=0.05)
    assert {technical.default_horizon(t) for _ in range(5)} == {"swing"}


# ------------------------------------------------------------------ pick grading


UNI = [f"U{i}" for i in range(12)]


@pytest.fixture
def graded_db(storage, monkeypatch):
    """One run, a 12-name scored universe, and two picks: one that claimed a swing
    horizon and one legacy pick that claimed nothing."""
    rid = storage.create_run("deep")
    storage.upsert_screen_scores(rid, [
        {"symbol": s, "blended": 0.5, "rank": i + 1, "tags": [], "metrics": {}}
        for i, s in enumerate(UNI)])
    storage.finish_run(rid, "ok")
    t0 = pd.Timestamp("2026-01-05")
    with storage.get_conn() as cn:
        cn.execute("INSERT INTO picks(run_id,symbol,surfaced_at,price_at_surface,"
                   "horizon,horizon_source) VALUES (?,?,?,?,?,?)",
                   (rid, "U0", t0.isoformat() + "+00:00", 100.0, "swing", "claude"))
        cn.execute("INSERT INTO picks(run_id,symbol,surfaced_at,price_at_surface) "
                   "VALUES (?,?,?,?)", (rid, "U1", t0.isoformat() + "+00:00", 100.0))

    # Synthetic prices: U0 compounds fastest, the rest of the universe drifts.
    idx = pd.bdate_range("2025-12-01", periods=400)
    cols = {}
    for i, s in enumerate(UNI):
        cols[s] = [100.0 * (1 + 0.001 * (i % 3) ) ** k for k in range(len(idx))]
    cols["SPY"] = [100.0 * 1.0005 ** k for k in range(len(idx))]
    cols["SMH"] = [100.0 * 1.0007 ** k for k in range(len(idx))]
    px = pd.DataFrame(cols, index=idx)
    monkeypatch.setattr(performance, "_price_history", lambda syms, period="1y": px)
    performance._MEMO.clear()
    return rid


def _fixed_keys(rec: dict) -> dict:
    return {k: v for k, v in rec.items()
            if any(k.endswith(f"_{w}") for w in performance._WINDOWS)}


def test_windows_are_frozen():
    assert performance._WINDOWS == {"1w": 5, "4w": 21, "13w": 65}


def test_pick_outcomes_adds_the_own_clock_and_leaves_the_fixed_windows_alone(graded_db, storage):
    rows = {r["symbol"]: r for r in performance.pick_outcomes()}
    assert set(rows) == {"U0", "U1"}
    swing, legacy = rows["U0"], rows["U1"]

    assert swing["horizon"] == "swing" and swing["horizon_source"] == "claude"
    assert swing["horizon_days"] == 10
    assert swing["ret_h"] is not None and swing["excess_uni_h"] is not None
    assert swing["net_excess_uni_h"] is not None
    assert swing["excess_bench_h"] is not None and swing["excess_theme_h"] is not None

    # Legacy pick: nothing invented.
    assert legacy["horizon"] is None
    assert legacy["horizon_source"] == "legacy"
    assert legacy["horizon_days"] is None
    for k in ("ret_h", "excess_uni_h", "excess_bench_h", "excess_theme_h",
              "net_excess_uni_h"):
        assert legacy[k] is None
    # ...but every fixed window is still graded.
    assert legacy["ret_4w"] is not None and legacy["net_excess_uni_4w"] is not None

    # Now strip the horizons and recompute: the fixed-window half must be identical.
    before = {s: _fixed_keys(r) for s, r in rows.items()}
    with storage.get_conn() as cn:
        cn.execute("UPDATE picks SET horizon=NULL, horizon_source=NULL")
    performance._MEMO.clear()
    after = {r["symbol"]: _fixed_keys(r) for r in performance.pick_outcomes()}
    assert before == after


def test_horizon_days_reads_config_and_refuses_to_guess():
    assert performance._horizon_days({"horizon": "swing"}) == 10
    assert performance._horizon_days({"horizon": "long"}) == 126
    assert performance._horizon_days({"horizon": None}) is None
    assert performance._horizon_days({"horizon": "quarterly"}) is None
    assert performance._horizon_days({}) is None


# ------------------------------------------------------------------ calibration clocks


def _rows_with_horizons():
    out = []
    for i in range(10):
        out.append({"run_id": i, "symbol": f"B{i}", "rating": "buy", "conviction": 7,
                    "regime": "neutral", "horizon": "swing", "horizon_days": 10,
                    "horizon_source": "claude", "prompt_version": 6,
                    "excess_uni_h": 1.5 + i * 0.1, "excess_theme_h": 1.0,
                    "excess_bench_h": 1.2, "net_excess_uni_h": 1.2 + i * 0.1,
                    "net_excess_uni_4w": 0.9})
    for i in range(10):
        out.append({"run_id": i, "symbol": f"S{i}", "rating": None, "conviction": None,
                    "regime": "neutral", "horizon": "position", "horizon_days": 42,
                    "horizon_source": "quant", "prompt_version": None,
                    "excess_uni_h": -0.5 + (i % 2), "excess_theme_h": 0.0,
                    "excess_bench_h": 0.1, "net_excess_uni_h": -0.5 + (i % 2),
                    "net_excess_uni_4w": -0.2})
    # A legacy pick with no clock at all must simply not appear in horizon cuts.
    out.append({"run_id": 99, "symbol": "L", "rating": "hold", "conviction": 3,
                "regime": None, "horizon": None, "horizon_days": None,
                "horizon_source": "legacy", "prompt_version": None,
                "excess_uni_h": None, "excess_theme_h": None, "excess_bench_h": None,
                "net_excess_uni_h": None, "net_excess_uni_4w": 0.4})
    return out


def test_rating_calibration_takes_a_clock(monkeypatch):
    monkeypatch.setattr(performance, "pick_outcomes", _rows_with_horizons)
    fixed = performance.rating_calibration()
    assert "avg_excess_4w" in fixed.columns and "horizon" not in fixed.columns

    hz = performance.rating_calibration(clock="horizon")
    assert set(hz["horizon"]) == {"swing", "position"}      # the legacy pick is absent
    # Marginal first: the '(all)' row for a horizon precedes its per-rating rows.
    swing = hz[hz["horizon"] == "swing"]
    assert list(swing["rating"])[0] == "(all)"
    assert int(swing[swing["rating"] == "(all)"]["n"].iloc[0]) == 10
    assert swing[swing["rating"] == "(all)"]["horizon_days"].iloc[0] == 10
    assert "net_avg_excess" in hz.columns


def test_calibration_table_horizon_clock_reports_marginals_first(monkeypatch):
    monkeypatch.setattr(performance, "pick_outcomes", _rows_with_horizons)
    tbl = performance.calibration_table(clock="horizon", min_n=8)
    cuts = list(tbl["cut"])
    assert cuts[0] == "horizon"
    assert "horizon x rating" in cuts
    # The 81-cell cut only appears where n already reaches min_n.
    full = tbl[tbl["cut"] == "full"]
    assert full.empty or (full["n"] >= 8).all()
    assert set(tbl["horizon"]) == {"swing", "position"}


def test_claude_vs_screen_takes_a_clock(monkeypatch):
    monkeypatch.setattr(performance, "pick_outcomes", _rows_with_horizons)
    fixed = performance.claude_vs_screen("4w", n_boot=200)
    assert fixed["clock"] == "fixed" and fixed["window"] == "4w"
    hz = performance.claude_vs_screen(n_boot=200, clock="horizon")
    assert hz["clock"] == "horizon" and hz["window"] == "own horizon"
    assert hz["n_claude"] == 10 and hz["n_screen"] == 10
    assert hz["edge_pp"] > 0


# ------------------------------------------------------------------ learning loop


def test_graded_calls_prefer_the_calls_own_clock(monkeypatch):
    monkeypatch.setattr(performance, "pick_outcomes", lambda: [
        {"run_id": 1, "symbol": "AAA", "rating": "buy", "conviction": 7,
         "surfaced_at": "2026-01-05", "price_at_surface": 100.0,
         "horizon": "swing", "horizon_days": 10, "net_excess_uni_h": 3.3,
         "excess_uni_4w": 9.9, "excess_uni_1w": 8.8},
        {"run_id": 2, "symbol": "BBB", "rating": "buy", "conviction": 5,
         "surfaced_at": "2026-01-05", "price_at_surface": 50.0,
         "horizon": None, "net_excess_uni_h": None, "excess_uni_4w": 1.1},
    ])
    calls = {c["symbol"]: c for c in learning.graded_calls()}
    own = calls["AAA"]
    assert own["grade_clock"] == "horizon"
    assert own["grade_excess"] == 3.3 and own["grade_window"] == "swing"
    legacy = calls["BBB"]
    assert legacy["grade_clock"] == "fixed"
    assert legacy["grade_excess"] == 1.1 and legacy["grade_window"] == "4w"


def test_symbol_track_record_names_the_clock(monkeypatch):
    monkeypatch.setattr(performance, "pick_outcomes", lambda: [
        {"run_id": 1, "symbol": "AAA", "rating": "buy", "conviction": 7,
         "surfaced_at": "2026-01-05", "price_at_surface": 100.0,
         "horizon": "swing", "net_excess_uni_h": -2.0},
    ])
    txt = learning.symbol_track_record("AAA")
    assert "over your stated swing horizon" in txt


def test_pillar_ic_accepts_a_horizon(monkeypatch):
    # No deep-run rows in a fresh DB, so this asserts the parameter exists and the
    # default is preserved rather than the study itself (which needs months of runs).
    import inspect
    sig = inspect.signature(learning.pillar_ic)
    assert sig.parameters["horizon_days"].default == 21


# ------------------------------------------------------------------ storage + harness


def test_horizon_columns_exist(storage):
    with storage.get_conn() as cn:
        picks = {r["name"] for r in cn.execute("PRAGMA table_info(picks)")}
        rr = {r["name"] for r in cn.execute("PRAGMA table_info(research_reports)")}
        ver = cn.execute("SELECT MAX(version) v FROM schema_version").fetchone()["v"]
    assert {"horizon", "horizon_source"} <= picks
    assert {"horizon", "horizon_rationale", "horizon_source", "quant_horizon",
            "expected_move_pct", "invalidation_level"} <= rr
    assert ver == 4      # 4: screen_scores.blended_pct


def test_upsert_research_persists_the_horizon_fields(storage):
    rid = storage.create_run("deep")
    storage.upsert_research(rid, {
        "symbol": "VRT", "rating": "buy", "conviction": 7, "horizon": "swing",
        "horizon_rationale": "orders land at the Oct 21 print",
        "horizon_source": "claude", "quant_horizon": "position",
        "expected_move_pct": 8.5, "invalidation_level": 118.0,
        "catalysts": [], "risks": []})
    with storage.get_conn() as cn:
        r = dict(cn.execute("SELECT * FROM research_reports WHERE run_id=?", (rid,)).fetchone())
    assert r["horizon"] == "swing" and r["horizon_source"] == "claude"
    assert r["quant_horizon"] == "position"          # both clocks, side by side
    assert r["expected_move_pct"] == 8.5 and r["invalidation_level"] == 118.0
    assert r["horizon_rationale"].startswith("orders land")


def test_record_picks_writes_the_clock_and_legacy_rows_stay_null(storage):
    rid = storage.create_run("deep")
    storage.record_picks(rid, [
        {"symbol": "VRT", "price": 120.0, "horizon": "swing", "horizon_source": "claude"},
        {"symbol": "NVDA", "price": 200.0, "horizon": "position", "horizon_source": "quant"},
        {"symbol": "SPY", "price": 500.0}])          # the benchmark snapshot: no clock
    with storage.get_conn() as cn:
        rows = {r["symbol"]: dict(r) for r in
                cn.execute("SELECT * FROM picks WHERE run_id=?", (rid,))}
    assert rows["VRT"]["horizon"] == "swing" and rows["VRT"]["horizon_source"] == "claude"
    assert rows["NVDA"]["horizon_source"] == "quant"
    assert rows["SPY"]["horizon"] is None and rows["SPY"]["horizon_source"] is None


def test_every_n_trading_days():
    from datetime import date

    from src.data.yfinance_live import every_n_trading_days, month_ends
    days = list(every_n_trading_days(date(2026, 1, 1), date(2026, 6, 30), 10))
    assert len(days) > 5
    # Spacing is 10 SESSIONS, so the calendar gap is 14 days plus any holiday in
    # between — that variance is the point, not a defect.
    gaps = {(b - a).days for a, b in zip(days, days[1:])}
    assert gaps and min(gaps) == 14 and max(gaps) <= 18
    assert all(d.weekday() < 5 for d in days)
    # The default grid is untouched.
    assert month_ends(date(2026, 1, 1), date(2026, 6, 30))[0] == pd.Timestamp("2026-01-30")
    with pytest.raises(ValueError):
        list(every_n_trading_days(date(2026, 1, 1), date(2026, 6, 30), 0))


# ------------------------------------------------------------------ held positions


def test_position_horizon_reads_the_latest_pick(storage):
    from src import deep_research as dr
    assert dr.position_horizon("NVDA") == "position"      # nothing claimed -> config default
    rid = storage.create_run("deep")
    storage.record_picks(rid, [{"symbol": "NVDA", "price": 1.0, "horizon": "swing",
                                "horizon_source": "claude"}])
    rid2 = storage.create_run("deep")
    storage.record_picks(rid2, [{"symbol": "NVDA", "price": 1.0, "horizon": "long",
                                 "horizon_source": "claude"}])
    assert dr.position_horizon("nvda") == "long"          # the MOST RECENT pick wins


def test_deep_dive_is_horizon_aware(monkeypatch, storage):
    """The holding is judged on the clock it is already on, and returns the same three
    fields the board research does — otherwise the portfolio path grades a long
    position on a four-week wobble."""
    from src import deep_research as dr
    seen = {}

    def _fake(prompt):
        seen["prompt"] = prompt
        return ({"action": "hold", "conviction": 6, "target_weight_pct": 10,
                 "thesis_health": "intact", "horizon": "LONG ",
                 "horizon_rationale": "compounding, no dated catalyst",
                 "expected_move_pct": 12.0, "invalidation_level": 95.0}, "raw")

    monkeypatch.setattr(dr, "_run_json", _fake)
    out = dr.deep_dive_holding("NVDA", "NVIDIA", "semis",
                               {"shares": 10, "avg_cost": 100, "last": 120, "pnl_pct": 20},
                               horizon="long")
    assert 'CURRENT CLOCK IS "long"' in seen["prompt"]
    assert "126 trading days" in seen["prompt"]          # the same menu as the board prompt
    assert "ON THAT CLOCK" in seen["prompt"]
    assert out["horizon"] == "long"                       # coerced through the shared enum
    assert out["expected_move_pct"] == 12.0 and out["invalidation_level"] == 95.0
    assert out["prompt_version"] == dr.PROMPT_VERSION


def test_deep_dive_horizon_persists(storage):
    rid = storage.create_portfolio_run()
    storage.upsert_portfolio_position(rid, {
        "symbol": "NVDA", "action": "hold", "conviction": 6, "horizon": "long",
        "invalidation": "loses the hyperscaler order book",   # prose column, unchanged
        "expected_move_pct": 12.0, "invalidation_level": 95.0})
    rows = storage.get_portfolio(rid)["positions"]
    assert rows[0]["horizon"] == "long"
    assert rows[0]["expected_move_pct"] == 12.0
    assert rows[0]["invalidation_level"] == 95.0
    assert rows[0]["invalidation"].startswith("loses")     # numeric level did not replace it


def test_every_n_trading_days_steps_over_real_sessions():
    from datetime import date

    from src.data.yfinance_live import every_n_trading_days
    from src.market_calendar import trading_days
    sessions = trading_days(date(2026, 1, 1), date(2026, 12, 31))
    # NYSE closes ~9 days a year, so a session grid is strictly shorter than a
    # business-day one — that difference IS the clock error this fixes.
    bdays = pd.bdate_range("2026-01-01", "2026-12-31")
    assert 240 <= len(sessions) < len(bdays)
    assert pd.Timestamp("2026-01-01") not in sessions      # New Year's Day
    grid = list(every_n_trading_days(date(2026, 1, 1), date(2026, 12, 31), 10))
    # Every grid point is a real session, exactly 10 sessions apart.
    pos = [sessions.get_loc(g) for g in grid]
    assert pos == list(range(0, len(sessions), 10))
