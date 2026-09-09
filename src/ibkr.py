"""IBKR integration via ib_async (holdings + live quotes), fully defensive.

Connects to a running IB Gateway / TWS with the API enabled. Every entry point
degrades gracefully: if the gateway is offline or ib_async is missing, callers
get an "offline" result and the pipeline/dashboard continue without IBKR.
"""
from __future__ import annotations

import threading
from typing import Any, Callable

from . import config

try:
    from ib_async import IB, Stock  # type: ignore
    _IB_AVAILABLE = True
except Exception:  # noqa: BLE001
    _IB_AVAILABLE = False

TIMEOUT_NOTE = "timed out (half-dead Gateway?)"


def _total_timeout() -> float:
    return float(config.get("ibkr.total_timeout", 45.0) or 45.0)


def _bounded(fn: Callable[[], dict[str, Any]], fallback: dict[str, Any],
             deadline_s: float | None = None) -> dict[str, Any]:
    """Run `fn` with a HARD wall-clock; on expiry return `fallback` and keep going.

    The failure this prevents (reproduced twice in Phase 7): with port 4001 open but
    the Gateway's session lapsed — its state every morning until someone answers the
    2FA prompt — `ib.connect()` never returns. `get_positions` is capped at
    positions_max_wait but the connect/handshake is not, and the pipeline put no clock
    around the call at all, so the whole run blocked forever and wrote NO health
    record: the one failure mode that is invisible in the dashboard.

    Same shape as technical._download_with_deadline, with one deliberate difference:
    the worker is a raw DAEMON thread, not a ThreadPoolExecutor. A pool's threads are
    non-daemon and joined at interpreter exit, so an abandoned thread stuck inside
    ib.connect() would move the hang from the middle of the run to the end of it.
    A daemon thread can be abandoned and the process still exits.

    The abandoned thread keeps holding `ibkr.client_id`, so a retry inside the same
    process can be refused by the Gateway; that degrades to `connected=False` again,
    which is the same answer, so it is not worth a client-id rotation.
    """
    box: dict[str, Any] = {}

    def _work() -> None:
        try:
            box["result"] = fn()
        except Exception as exc:  # noqa: BLE001
            box["result"] = {**fallback, "note": f"error: {str(exc)[:120]}"}

    t = threading.Thread(target=_work, name="ibkr-bounded", daemon=True)
    t.start()
    t.join(deadline_s if deadline_s is not None else _total_timeout())
    if t.is_alive():
        return {**fallback, "note": TIMEOUT_NOTE}
    return box.get("result", fallback)


def _connect(client_id: int | None = None):
    """Return a connected IB instance, or None if unavailable/offline.

    Retries a few times: the Gateway can transiently refuse a fresh connection
    right after another one closed.
    """
    if not _IB_AVAILABLE or not config.get("ibkr.enabled", True):
        return None
    cid = client_id if client_id is not None else int(config.get("ibkr.client_id", 17))
    timeout = int(config.get("ibkr.connect_timeout", 8))
    for attempt in range(3):
        ib = IB()
        try:
            ib.connect(config.get("ibkr.host", "127.0.0.1"),
                       int(config.get("ibkr.port", 4001)),
                       clientId=cid, timeout=timeout, readonly=True)
            return ib
        except Exception:  # noqa: BLE001
            try:
                ib.disconnect()
            except Exception:  # noqa: BLE001
                pass
            if attempt < 2:
                try:
                    ib.sleep(1.5)
                except Exception:  # noqa: BLE001
                    import time as _t
                    _t.sleep(1.5)
    return None


def get_positions(ib) -> list[dict[str, Any]]:
    # Aggregate per symbol: ib.positions() returns one row PER ACCOUNT per contract, so
    # the same ticker appears twice across two accounts. Summing shares and computing a
    # share-weighted average cost yields one row per symbol — required because the
    # holdings table has `symbol` as PRIMARY KEY (duplicates would crash the upsert).
    agg: dict[str, dict[str, float]] = {}
    try:
        ib.reqPositions()
        # Positions arrive in per-account bursts with gaps, so a plain plateau is
        # NOT the end of the stream (early-stopping on one once returned 3 of 7).
        # Wait for a full quiet window with no new rows instead, floored at
        # positions_wait and capped at positions_max_wait. The cap matters since
        # the on-demand launcher starts the Gateway cold: right after login the
        # first rows can take ~5-8s, where the old flat 3s wait returned nothing.
        min_wait = float(config.get("ibkr.positions_wait", 3.0))
        max_wait = float(config.get("ibkr.positions_max_wait", 25.0))
        quiet    = float(config.get("ibkr.positions_quiet", 3.0))
        step, elapsed, last_count, unchanged_for = 0.5, 0.0, -1, 0.0
        while elapsed < max_wait:
            ib.sleep(step)
            elapsed += step
            count = len(ib.positions())
            unchanged_for = unchanged_for + step if count == last_count else 0.0
            last_count = count
            if elapsed >= min_wait and count > 0 and unchanged_for >= quiet:
                break
        for p in ib.positions():
            sym = (p.contract.symbol or "").upper()
            if not sym:
                continue
            pos = float(p.position)
            cost = float(p.avgCost)
            a = agg.setdefault(sym, {"position": 0.0, "cost_x_pos": 0.0})
            a["position"] += pos
            a["cost_x_pos"] += cost * pos
    except Exception:  # noqa: BLE001
        return []
    out = []
    for sym, a in agg.items():
        pos = a["position"]
        if not pos:  # long in one account, short in the other nets to zero — not a holding
            continue
        avg_cost = a["cost_x_pos"] / pos
        out.append({"symbol": sym, "position": pos,
                    "avg_cost": avg_cost, "market_price": None})
    return out


def _num(x):
    """Return x if it's a real positive number, else None.

    IBKR returns -1.0 as a 'no data available' sentinel for some quotes; treat
    any non-positive value as missing.
    """
    try:
        v = float(x) if x == x and x is not None else None
    except (TypeError, ValueError):
        return None
    return v if (v is not None and v > 0) else None


def get_quotes(ib, symbols: list[str]) -> list[dict[str, Any]]:
    out = []
    if not symbols:
        return out
    try:
        # 1=realtime (needs subscription), 3=delayed (free), 4=delayed-frozen.
        ib.reqMarketDataType(int(config.get("ibkr.market_data_type", 3)))
        contracts = [Stock(s, "SMART", "USD") for s in symbols]
        ib.qualifyContracts(*contracts)
        # Only quote contracts that actually qualified (have a conId). One bad ticker
        # (delisted, dotted class share, foreign listing) otherwise raises and wipes the
        # entire batch — including holdings marks.
        qualified = [c for c in contracts if getattr(c, "conId", 0)]
        if not qualified:
            return out
        tickers = ib.reqTickers(*qualified)
        for t in tickers:
            sym = (t.contract.symbol or "").upper()
            # Try real-time then delayed fields, then close, then computed mid.
            last = (_num(t.last) or _num(getattr(t, "delayedLast", None))
                    or _num(t.close) or _num(getattr(t, "delayedClose", None)))
            if last is None:
                last = _num(t.marketPrice())
            # Skip names with no usable price: writing last=None would overwrite a
            # previously-good quote with NULL and blank out P&L for that holding.
            if last is not None:
                out.append({"symbol": sym, "last": last})
    except Exception:  # noqa: BLE001
        return out
    return out


def _fetch_holdings() -> dict[str, Any]:
    if not _IB_AVAILABLE:
        return {"connected": False, "holdings": [], "note": "ib_async not installed"}
    ib = _connect()  # same client id as the main fetch; calls are sequential
    if ib is None:
        return {"connected": False, "holdings": [], "note": "offline"}
    try:
        return {"connected": True, "holdings": get_positions(ib), "note": "ok"}
    finally:
        try:
            ib.disconnect()
        except Exception:  # noqa: BLE001
            pass


def fetch_holdings() -> dict[str, Any]:
    """Positions-only fetch (fast), used early so the news scan can cover holdings.

    Uses the same client id as the later quotes fetch; the two calls are sequential
    (this one disconnects before the quotes fetch connects), so they never collide.
    Bounded by `ibkr.total_timeout` — connect + handshake + positions, one clock.
    """
    return _bounded(_fetch_holdings, {"connected": False, "holdings": [], "note": "offline"})


def _fetch_portfolio_and_quotes(symbols: list[str]) -> dict[str, Any]:
    if not _IB_AVAILABLE:
        return {"connected": False, "holdings": [], "quotes": [],
                "note": "ib_async not installed"}
    ib = _connect()
    if ib is None:
        return {"connected": False, "holdings": [], "quotes": [],
                "note": "IB Gateway/TWS offline or refused connection"}
    try:
        holdings = get_positions(ib)
        # Always quote held names too, even if they're outside the visible board ranks.
        held_syms = [h["symbol"] for h in holdings]
        quote_syms = list(dict.fromkeys(list(symbols) + held_syms))
        quotes = get_quotes(ib, quote_syms)
        return {"connected": True, "holdings": holdings, "quotes": quotes, "note": "ok"}
    finally:
        try:
            ib.disconnect()
        except Exception:  # noqa: BLE001
            pass


def fetch_portfolio_and_quotes(symbols: list[str]) -> dict[str, Any]:
    """High-level: connect once, pull positions + quotes, disconnect.

    Returns {'connected': bool, 'holdings': [...], 'quotes': [...], 'note': str}.
    Bounded by `ibkr.total_timeout`: connect, positions AND quotes share one hard
    wall-clock, because the run needs an answer, not a complete answer.
    """
    return _bounded(lambda: _fetch_portfolio_and_quotes(symbols),
                    {"connected": False, "holdings": [], "quotes": [],
                     "note": "IB Gateway/TWS offline or refused connection"})


if __name__ == "__main__":
    print("ib_async importable:", _IB_AVAILABLE)
    res = fetch_portfolio_and_quotes(["NVDA", "VRT"])
    print("connected:", res["connected"], "| note:", res["note"])
    print("holdings:", res["holdings"][:3])
    print("quotes:", res["quotes"][:3])
