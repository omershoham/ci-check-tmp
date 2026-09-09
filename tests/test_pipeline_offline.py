"""End-to-end `pipeline.run("deep")` with every network/LLM/broker edge faked.

This is the FakeProvider run from the plan: proves the orchestration, storage and
the health record work together in seconds, with no Yahoo, Ollama, IBKR or Claude.
Deliberately runs with Ollama 'down' and IBKR 'offline' so the health record must
come out DEGRADED and name both stages — the failure mode that went unseen for five
weeks in Aug-Sep 2026.
"""
from __future__ import annotations

import json

import pytest

from src import (deep_research, fundamentals, health, ibkr, learning, local_llm, news,
                 pipeline, regime, technical)

SYMS = ["NVDA", "AMD", "AVGO", "TSM", "VRT", "ANET", "GOOGL", "AMZN", "MSFT", "CEG",
        "IREN", "META"]


def _fake_metrics(store, symbols, as_of=None, extra=None):
    """Stands in for the SEC-store read that stage 1 now does. Names outside the test
    set come back with `error`, i.e. "no snapshot" — which is what drives the
    fundamentals stage to DEGRADED on coverage."""
    out = {}
    for i, s in enumerate(symbols):
        if s not in SYMS:
            out[s] = {"symbol": s, "error": "faked: no SEC snapshot"}
            continue
        out[s] = {"symbol": s, "name": s, "marketCap": 1e11, "price": 100.0 + i,
                  "avg_dollar_volume": 5e8, "freeCashflow": 1e9, "profitable": True,
                  "trailingPE": 20 + i, "revenueGrowth": 0.1 + i / 100,
                  "grossMargins": 0.5, "returnOnEquity": 0.2, "returnOnAssets": 0.1,
                  "operatingMargins": 0.3, "debtToEquity": 40.0,
                  "priceToSalesTrailing12Months": 5 + i, "enterpriseToEbitda": 15 + i,
                  "earningsGrowth": 0.1, "fcf_yield": 0.02,
                  "fcf_margin": 0.2, "mom_50": 1.0 + i / 100, "mom_200": 1.05}
    return out


def _fake_techs(symbols, use_cache=True):
    return {s: {"technical_score": 0.4 + 0.04 * i, "state": "Uptrend", "rsi": 55.0,
                "macd_hist": 0.5, "pos_52w": 0.7, "atr_pct": 0.02, "price": 100.0 + i,
                "sma50": 95.0, "ret_63": 5.0, "ret_126": 10.0, "benchmark": "SMH"}
            for i, s in enumerate(symbols)}


def _fake_claude(prompt, allowed_tools=None, timeout=None, extra_args=None):
    if "DEVIL'S ADVOCATE" in prompt:
        body = {"bear_case": "x", "new_evidence": [], "survives": True,
                "suggested_rating": "buy", "confidence_haircut": 1, "note": "n"}
    else:
        body = {"ticker": "X", "rating": "buy", "conviction": 7, "deal_likelihood": "high",
                "thesis": "faked", "chart_pattern": "up", "metrics_read": "m",
                "horizon": "swing", "horizon_rationale": "orders land at the next print",
                "expected_move_pct": 9.0, "invalidation_level": 88.0,
                "catalysts": ["c"], "risks": ["r"], "bull": "b", "bear": "b",
                "data_as_of": "2026-09-07", "latest_price_seen": 100}
        # One name comes back with an unusable horizon, so the run also exercises the
        # quant fallback: a pick must ALWAYS end up with a clock and an honest source.
        if "(TSM)" in prompt:
            body["horizon"] = "a few weeks"
    text = json.dumps(body)
    return text, json.dumps({"result": text, "session_id": "fake"})


@pytest.fixture
def offline(monkeypatch, workspace, curated):
    # `curated` pins BOTH universe.source and the theme pack: the fake metrics below
    # are the curated names, and the shipped default (broad + general) would otherwise
    # pull in whatever registrant table this machine has — or, with only the source
    # pinned, an empty universe, because the general pack ships no ticker list.
    monkeypatch.setattr(fundamentals, "from_store", _fake_metrics)
    # `.info` survives only as enrich() on the shortlist; the offline run must make no
    # network call at all, so it returns nothing and forwardPE simply stays absent.
    monkeypatch.setattr(fundamentals, "enrich", lambda syms, **k: {s: {} for s in syms})
    monkeypatch.setattr(technical, "compute_many", _fake_techs)
    monkeypatch.setattr(technical, "yf_last_prices", lambda syms: {})
    monkeypatch.setattr(technical, "benchmark_last_price", lambda *a, **k: None)
    monkeypatch.setattr(regime, "detect", lambda *a, **k: {"regime": "neutral", "breadth_pct": 50,
                                                          "benchmark": "SPY"})
    monkeypatch.setattr(local_llm, "available", lambda: False)          # Ollama DOWN
    monkeypatch.setattr(news, "get_news", lambda *a, **k: [])
    monkeypatch.setattr(ibkr, "fetch_holdings",
                        lambda: {"connected": False, "holdings": [], "note": "offline"})
    monkeypatch.setattr(ibkr, "fetch_portfolio_and_quotes",
                        lambda syms: {"connected": False, "holdings": [], "quotes": [],
                                      "note": "IB Gateway/TWS offline"})   # IBKR OFFLINE
    monkeypatch.setattr(deep_research, "_run_claude", _fake_claude)
    monkeypatch.setattr(deep_research, "auth_status",
                        lambda: {"claude_on_path": True, "oauth_token_set": True,
                                 "api_key_set": False})
    monkeypatch.setattr(learning, "learning_block", lambda sym=None: "")
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    return workspace


def test_deep_run_offline_records_degraded_health(offline, storage):
    rid = pipeline.run("deep")
    run = storage.list_runs(1)[0]
    assert run["id"] == rid and run["status"] == "ok"
    assert run["health"] == health.DEGRADED
    hz = health.RunHealth.from_json(run["health_json"])
    stages = {n: st for n, st, _ in hz.problems()}
    assert stages.get("sentiment") == health.DEGRADED
    assert stages.get("broker") == health.DEGRADED
    assert hz.stage("research").status == health.OK
    assert hz.stage("fundamentals").status == health.DEGRADED   # SEC coverage 12/105

    board = storage.get_board(rid)
    assert len(board) == len(SYMS)
    rated = [r for r in board if r.get("rating")]
    assert len(rated) == 10                                        # finalist cap
    with storage.get_conn() as cn:
        pv = cn.execute("SELECT DISTINCT prompt_version, model FROM research_reports "
                        "WHERE run_id=?", (rid,)).fetchall()
    assert len(pv) == 1 and pv[0]["prompt_version"] == deep_research.PROMPT_VERSION
    # Red team ran and haircut conviction 7 -> 6 on every buy.
    assert all(r["conviction"] == 6 for r in rated)

    # Ollama is DOWN in this fixture, so every catalyst key must be UNKNOWN (null),
    # never a fabricated 0 - and the row must still say which contract it speaks.
    from src import local_llm as _llm
    with storage.get_conn() as cn:
        sig = cn.execute("SELECT catalyst_json, signal_version, hyperscaler_signal "
                         "FROM signals WHERE run_id=?", (rid,)).fetchall()
    assert sig
    for row in sig:
        cat = json.loads(row["catalyst_json"])
        assert tuple(cat) == _llm.CATALYST_KEYS
        assert set(cat.values()) == {None}
        assert row["signal_version"] == 2
        assert row["hyperscaler_signal"] == cat["customer_deal"]
    # The catalyst pillar is persisted from day one even with nothing to measure...
    with storage.get_conn() as cn:
        m = json.loads(cn.execute("SELECT metrics_json FROM screen_scores WHERE run_id=? "
                                  "LIMIT 1", (rid,)).fetchone()["metrics_json"])
    assert "pillar_catalyst" in m
    # ...and at weight 0 it must not appear in the partition key.
    with storage.get_conn() as cn:
        psets = {r["pillar_set"] for r in cn.execute(
            "SELECT DISTINCT pillar_set FROM screen_scores WHERE run_id=?", (rid,))}
        deals = {r["deal"] for r in cn.execute(
            "SELECT DISTINCT deal FROM screen_scores WHERE run_id=?", (rid,))}
    assert psets == {"technical,fundamental,sentiment"}
    assert deals == {None}
    assert not any({"deal", "both"} & set(r.get("tags") or []) for r in board)

    # Every pick carries the clock it will be graded on, and says who chose it. TSM's
    # report named an unusable horizon, so it falls back to the quant default.
    with storage.get_conn() as cn:
        picks = {r["symbol"]: dict(r) for r in
                 cn.execute("SELECT * FROM picks WHERE run_id=?", (rid,))}
        reports = {r["symbol"]: dict(r) for r in
                   cn.execute("SELECT * FROM research_reports WHERE run_id=?", (rid,))}
    assert len(picks) == 10
    assert all(p["horizon"] in ("swing", "position", "long") for p in picks.values())
    assert picks["NVDA"]["horizon"] == "swing" and picks["NVDA"]["horizon_source"] == "claude"
    assert picks["TSM"]["horizon_source"] == "quant"
    assert picks["TSM"]["horizon"] == "position"          # technical.default_horizon
    # The report keeps Claude's own claim (NULL when it made none) plus the quant label
    # alongside it, so agreement(claude, quant) stays a query.
    assert reports["NVDA"]["horizon"] == "swing"
    assert reports["NVDA"]["expected_move_pct"] == 9.0
    assert reports["NVDA"]["invalidation_level"] == 88.0
    assert reports["NVDA"]["horizon_rationale"].startswith("orders land")
    assert reports["TSM"]["horizon"] is None
    assert all(r["quant_horizon"] == "position" for r in reports.values())


def test_light_run_skips_research(offline, storage):
    rid = pipeline.run("light")
    run = storage.list_runs(1)[0]
    assert run["health"] == health.DEGRADED
    assert not storage.runs_with_research()
    assert len(storage.get_board(rid)) == len(SYMS)


def test_deep_run_arms_candidate_triggers_and_a_later_run_fires_them(offline, storage,
                                                                     monkeypatch):
    """Phase 8's gate: a researched name you do NOT own gets an armed level, is labelled
    `watch` rather than `hold`, and a later run at a lower price flips it onto the board.

    The fake Claude returns invalidation_level 88 for every name, and the fake metrics
    price names at 100+i, so run 1 arms 88 and run 2 (prices 60+i) trips it.
    """
    from src import recommend, risk, triggers
    rid = pipeline.run("deep")

    armed = [t for t in storage.active_triggers() if t["source"] == "candidate"]
    assert armed, "no candidate triggers armed"
    assert all(t["kind"] == "price_below" and t["threshold"] == 88.0 for t in armed)
    # Nothing is held in this fixture, so every researched name is a candidate.
    assert len(armed) == 10

    board = {r["symbol"]: r for r in storage.get_board(rid)}
    watched = [s for s, r in board.items() if recommend.recommendation(r)[0] == "watch"]
    # The fake reports rate `buy`; a `hold` on an armed name is what `watch` is for, so
    # assert the LABELLING RULE directly on a row carrying this run's real trigger count.
    nvda = dict(board["NVDA"])
    assert nvda["candidate_triggers"] == 1
    assert recommend.recommendation({**nvda, "rating": "hold"})[0] == "watch"
    assert recommend.recommendation({**nvda, "rating": "hold",
                                     "held_position": 5})[0] == "hold"
    assert watched == []          # ...and nothing is mislabelled while ratings are buy

    # A second run with the tape 40 points lower: the level trips and the board says so.
    monkeypatch.setattr(fundamentals, "from_store",
                        lambda store, symbols, as_of=None, extra=None: {
                            s: {**v, "price": (v.get("price") or 100.0) - 40.0}
                            for s, v in _fake_metrics(store, symbols).items()})
    rid2 = pipeline.run("light")
    fired = storage.recent_fired_triggers()
    assert {t["symbol"] for t in fired} >= {"NVDA"}
    assert any("< 88" in t["trigger_detail"] for t in fired)
    row2 = {r["symbol"]: r for r in storage.get_board(rid2)}["NVDA"]
    assert row2["fired_triggers"] >= 1        # the board row carries the fired marker
    assert row2["candidate_triggers"] == 0    # a fired trigger is no longer armed


def test_rank_delta_and_catalyst_freshness_are_persisted(offline, storage):
    rid1 = pipeline.run("light")
    with storage.get_conn() as cn:
        first = cn.execute("SELECT rank_prev FROM screen_scores WHERE run_id=?",
                           (rid1,)).fetchall()
    assert all(r["rank_prev"] is None for r in first)      # no comparable earlier run
    rid2 = pipeline.run("light")
    with storage.get_conn() as cn:
        rows = {r["symbol"]: dict(r) for r in cn.execute(
            "SELECT symbol, rank, rank_prev FROM screen_scores WHERE run_id=?", (rid2,))}
    assert rows and all(v["rank_prev"] == v["rank"] for v in rows.values())
    # `news.get_news` is faked to return nothing, so freshness is honestly NULL rather
    # than a fabricated 0 — the same genuine-0-vs-unknown discipline as the catalysts.
    with storage.get_conn() as cn:
        fresh = {r["catalyst_freshness_days"] for r in cn.execute(
            "SELECT catalyst_freshness_days FROM signals WHERE run_id=?", (rid2,))}
    assert fresh == {None}
    assert pipeline._freshness_days([]) is None
    import datetime as _dt
    recent = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=3)
    got = pipeline._freshness_days([{"published": recent.strftime("%Y-%m-%d")},
                                    {"published": "2020-01-01"}])
    assert 2.0 <= got <= 4.0                               # newest item wins


def test_book_fit_reaches_a_finalists_research_context(offline, storage, monkeypatch):
    """8c: the analyst is told how the name fits the book that already exists."""
    from src import deep_research, ibkr, risk
    monkeypatch.setattr(ibkr, "fetch_holdings", lambda: {
        "connected": True, "note": "ok",
        "holdings": [{"symbol": "MSFT", "position": 100.0, "avg_cost": 100.0},
                     {"symbol": "GOOGL", "position": 100.0, "avg_cost": 100.0}]})
    monkeypatch.setattr(risk, "candidate_fits", lambda syms, w, secs=None, sizes=None: {
        s: {"symbol": s, "marginal_corr": 0.81, "closest": "MSFT", "closest_corr": 0.81,
            "sector": "Information Technology", "sector_positions": 3,
            "sector_weight_pct": 34.0, "sector_weight_after_pct": 41.0,
            "max_sector_weight_pct": 40.0, "over_cap": True, "size_pct": 10.0}
        for s in syms})
    seen: list[str] = []
    real = deep_research.build_context
    monkeypatch.setattr(deep_research, "build_context",
                        lambda *a, **k: seen.append(real(*a, **k)) or seen[-1])
    rid = pipeline.run("deep")
    fit_ctx = [c for c in seen if "Book fit:" in c]
    assert fit_ctx, "no finalist context carried a book-fit line"
    assert "4th Information Technology position" in fit_ctx[0]
    assert "0.81 with MSFT" in fit_ctx[0] and "ABOVE the 40% cap" in fit_ctx[0]
    # ...and it is persisted on the board row, so the column can render it.
    board = {r["symbol"]: r for r in storage.get_board(rid)}
    fits = [r for r in board.values() if (r.get("metrics") or {}).get("book_fit")]
    assert fits and risk.candidate_fit_badge(fits[0]["metrics"]["book_fit"]) \
        .endswith("IT 41%⚠")
