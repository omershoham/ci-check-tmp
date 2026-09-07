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


def _fake_metrics(symbols, **_):
    out = {}
    for i, s in enumerate(symbols):
        if s not in SYMS:
            out[s] = {"symbol": s, "error": "faked: not in test set"}
            continue
        out[s] = {"symbol": s, "name": s, "marketCap": 1e11, "price": 100.0 + i,
                  "avg_dollar_volume": 5e8, "freeCashflow": 1e9, "profitable": True,
                  "trailingPE": 20 + i, "forwardPE": 18 + i, "revenueGrowth": 0.1 + i / 100,
                  "grossMargins": 0.5, "returnOnEquity": 0.2, "fcf_yield": 0.02,
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
                "catalysts": ["c"], "risks": ["r"], "bull": "b", "bear": "b",
                "data_as_of": "2026-09-07", "latest_price_seen": 100}
    text = json.dumps(body)
    return text, json.dumps({"result": text, "session_id": "fake"})


@pytest.fixture
def offline(monkeypatch, workspace):
    monkeypatch.setattr(fundamentals, "fetch_many", _fake_metrics)
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
    assert hz.stage("fundamentals").status == health.DEGRADED   # 95 of 107 faked as errors

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


def test_light_run_skips_research(offline, storage):
    rid = pipeline.run("light")
    run = storage.list_runs(1)[0]
    assert run["health"] == health.DEGRADED
    assert not storage.runs_with_research()
    assert len(storage.get_board(rid)) == len(SYMS)
