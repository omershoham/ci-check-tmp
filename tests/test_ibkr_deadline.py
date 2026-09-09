"""A half-dead IB Gateway must degrade the run, not hang it.

The live bug (reproduced twice in Phase 7): port 4001 open but the session lapsed —
the Gateway's state every morning until someone answers the 2FA prompt — and
`ib.connect()` never returns. `get_positions` was capped, the connect/handshake was
not, and the pipeline put no wall-clock around either, so the run blocked forever and
wrote NO health record at all.
"""
from __future__ import annotations

import threading
import time

from src import ibkr


def _block_forever(*a, **k):
    threading.Event().wait()          # never returns, exactly like the real hang


def test_fetch_holdings_returns_inside_the_deadline(monkeypatch):
    monkeypatch.setattr(ibkr, "_IB_AVAILABLE", True)
    monkeypatch.setattr(ibkr, "_connect", _block_forever)
    monkeypatch.setattr(ibkr, "_total_timeout", lambda: 1.0)
    t0 = time.monotonic()
    res = ibkr.fetch_holdings()
    elapsed = time.monotonic() - t0
    assert res["connected"] is False and res["holdings"] == []
    assert res["note"] == ibkr.TIMEOUT_NOTE
    assert elapsed < 5.0                       # 1s bound + generous scheduling slack


def test_fetch_portfolio_and_quotes_returns_inside_the_deadline(monkeypatch):
    monkeypatch.setattr(ibkr, "_IB_AVAILABLE", True)
    monkeypatch.setattr(ibkr, "_connect", _block_forever)
    monkeypatch.setattr(ibkr, "_total_timeout", lambda: 1.0)
    t0 = time.monotonic()
    res = ibkr.fetch_portfolio_and_quotes(["NVDA"])
    assert res["connected"] is False and res["quotes"] == [] and res["holdings"] == []
    assert "timed out" in res["note"]
    assert time.monotonic() - t0 < 5.0


def test_a_healthy_call_is_untouched(monkeypatch):
    """The deadline must be invisible when the Gateway answers."""
    monkeypatch.setattr(ibkr, "_IB_AVAILABLE", True)
    monkeypatch.setattr(ibkr, "_connect", lambda *a, **k: object())
    monkeypatch.setattr(ibkr, "get_positions",
                        lambda ib: [{"symbol": "NVDA", "position": 10.0,
                                     "avg_cost": 100.0, "market_price": None}])
    res = ibkr.fetch_holdings()
    assert res["connected"] is True and res["note"] == "ok"
    assert res["holdings"][0]["symbol"] == "NVDA"


def test_an_exception_inside_the_call_degrades_rather_than_raising(monkeypatch):
    monkeypatch.setattr(ibkr, "_IB_AVAILABLE", True)
    def _boom(*a, **k):
        raise RuntimeError("gateway said no")
    monkeypatch.setattr(ibkr, "_connect", _boom)
    res = ibkr.fetch_holdings()
    assert res["connected"] is False and "gateway said no" in res["note"]
