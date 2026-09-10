"""The data-quality gate, and a synthetic reproduction of every data bug this project
has actually shipped.

WHY THESE FOUR CASES. Each one reached the live board, was found by a human reading
numbers, and cost real analysis before anyone noticed. A gate is only worth having if
it would have caught them, so each is rebuilt here from the recorded symptom and the
check that must fire is asserted by name:

  * MSFT's revenue pinned to a tag retired in 2010 -> gross margin 338%
  * TSM at 9 fact rows because the ingest read only the `us-gaap` namespace
  * ASML discarded whole because it reports in EUR against a hardcoded USD unit
  * yfinance `.info["freeCashflow"]` being ONE QUARTER, ~4x too small on every row

Plus the two bugs fixed alongside the gate: Dominion's missing capex tag (FCF silently
becomes CFO) and Berkshire's per-share-class share count (no count -> no market cap ->
silent floor failure).
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.data import quality
from src.data.history_store import HistoryStore

AS_OF = date(2026, 6, 30)
PREV = pd.Timestamp("2026-05-31")
NOW = pd.Timestamp("2026-06-30")


def _row(symbol, as_of=NOW, **kw):
    base = {
        "as_of": as_of, "symbol": symbol, "price": 100.0, "sharesOutstanding": 1e9,
        "marketCap": 1e11, "totalRevenue": 5e10, "freeCashflow": 5e9,
        "grossMargins": 0.5, "operatingMargins": 0.3, "fcf_margin": 0.1,
        "trailingPE": 20.0, "priceToBook": 5.0, "ev_sales": 2.0,
        "priceToSalesTrailing12Months": 2.0, "enterpriseToEbitda": 12.0,
        "p_ffo": None, "financial_currency": "USD", "returnOnEquity": 0.2,
        "returnOnAssets": 0.1, "debtToEquity": 40.0, "revenueGrowth": 0.1,
        "earningsGrowth": 0.1, "fcf_yield": 0.05, "netIncome": 5e9,
        "totalAssets": 1e11, "totalDebt": 1e10, "totalCash": 2e10,
        "ffo_yield": None, "ffo_margin": None, "forwardPE": None, "pegRatio": None,
        "net_cash": 1e10, "p_fcf": 20.0, "enterpriseValue": 1e11, "ffo": None,
        "cash_runway_quarters": None, "rule_of_40": 20.0, "mom_50": 1.0,
        "mom_200": 1.0, "profitable": True,
    }
    base.update(kw)
    return base


def _facts(symbol, concepts=("totalRevenue", "netIncome", "cfo", "capex"),
           n=30, filed=pd.Timestamp("2026-05-01"), end=pd.Timestamp("2026-03-31")):
    rows = []
    for c in concepts:
        for i in range(n):
            rows.append({"symbol": symbol, "cik": "0000000001", "concept": c,
                         "tag": f"{c}Tag", "unit": "USD",
                         "start": end - pd.Timedelta(days=90),
                         "end": end - pd.Timedelta(days=91 * i), "val": 1e9,
                         "filed": filed - pd.Timedelta(days=91 * i),
                         "form": "10-Q", "frame": None, "fy": 2026.0, "fp": "Q1"})
    return rows


@pytest.fixture
def store(tmp_path):
    """A store holding one healthy name, so a check that fires is a real finding."""
    s = HistoryStore(tmp_path / "hist")
    s.write_snapshots(pd.DataFrame([_row("GOOD", PREV), _row("GOOD", NOW)]))
    s.upsert_facts(pd.DataFrame(_facts("GOOD")))
    px = pd.DataFrame([{"date": NOW - pd.Timedelta(days=i), "symbol": "GOOD",
                        "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                        "adj_close": 100.0, "volume": 1e7} for i in range(40)])
    s.upsert_prices(px)
    return s


def _run(store, **kw):
    kw.setdefault("symbols", sorted(store.snapshots()["symbol"].unique()))
    kw.setdefault("sectors", {})
    kw.setdefault("baseline", {})
    return quality.run(store, as_of=AS_OF, **kw)


def _checks(rep):
    return {f.check for f in rep.findings}


def _sev(rep, check):
    return {f.severity for f in rep.findings if f.check == check}


# ------------------------------------------------------------------ clean baseline
def test_healthy_store_is_clean(store):
    rep = _run(store)
    assert rep.counts()["critical"] == 0, [f.line() for f in rep.findings]
    assert rep.stats["symbols"] == 1


# ------------------------------------- bug 1: MSFT revenue pinned to a retired tag
def test_catches_stale_revenue_tag_gross_margin_over_one(store):
    """The recorded symptom was a 338% gross margin: current gross profit over a
    revenue figure last reported in 2010."""
    store.write_snapshots(pd.DataFrame([
        _row("MSFTLIKE", PREV, totalRevenue=2.4e11, grossMargins=0.69),
        _row("MSFTLIKE", NOW, totalRevenue=6.0e10, grossMargins=3.38)]))
    rep = _run(store)
    assert "gross_margin_impossible" in _checks(rep)
    assert _sev(rep, "gross_margin_impossible") == {quality.CRITICAL}
    # ...and the second, independent signature of the same bug: a TTM that became a
    # quarter. Two checks fire on one cause on purpose — either alone can be evaded.
    assert "ttm_revenue_collapse" in _checks(rep)


def test_catches_a_concept_whose_tag_was_retired(store):
    """The general form, at the fact level: one concept's newest period is years behind
    the rest of the same filer's statements. This is what a retired tag looks like
    BEFORE it produces an impossible ratio — and it is the Dominion capex case too."""
    rows = _facts("STALECAP", concepts=("totalRevenue", "netIncome", "cfo"))
    rows += _facts("STALECAP", concepts=("capex",), n=4,
                   filed=pd.Timestamp("2020-05-01"), end=pd.Timestamp("2020-03-31"))
    store.upsert_facts(pd.DataFrame(rows))
    store.write_snapshots(pd.DataFrame([_row("STALECAP", NOW)]))
    rep = _run(store)
    assert "concept_tag_retired" in _checks(rep)
    assert any(f.symbol == "STALECAP" and "capex" in str(f.observed)
               for f in rep.findings if f.check == "concept_tag_retired")


# ------------------------------------------ bug 2: TSM at 9 rows (namespace filter)
def test_catches_a_filer_reduced_to_a_handful_of_facts(store):
    """TSM had 9 fact rows against NVDA's 2,668 because the ingest read only us-gaap.
    Nothing about the surviving numbers was wrong, so only a count could see it."""
    store.upsert_facts(pd.DataFrame(_facts("TSMLIKE", concepts=("totalRevenue",), n=9)))
    store.write_snapshots(pd.DataFrame([_row("TSMLIKE", NOW)]))
    rep = _run(store)
    assert "thin_facts" in _checks(rep)
    assert any(f.symbol == "TSMLIKE" for f in rep.findings if f.check == "thin_facts")


def test_catches_a_coverage_regression_against_the_baseline(store):
    """The half of the TSM/ASML failure a bound can never see: names simply stop being
    scorable. Only a remembered previous state catches it."""
    baseline = {"coverage": {"Information Technology": {"n": 100, "scored": 100}}}
    rep = _run(store, sectors={"GOOD": "Information Technology"}, baseline=baseline)
    assert "sector_coverage_regression" in _checks(rep)
    assert _sev(rep, "sector_coverage_regression") == {quality.CRITICAL}


def test_catches_a_store_that_shrank(store):
    rep = _run(store, baseline={"totals": {"facts": 10_000_000}})
    assert "store_shrank" in _checks(rep)
    assert _sev(rep, "store_shrank") == {quality.CRITICAL}


# ------------------------------------------ bug 3: ASML discarded (hardcoded USD unit)
def test_catches_a_usd_multiple_on_a_non_usd_reporter(store):
    """The inverse of the ASML/TSM currency bug: the row survived, but its market cap
    is USD and its income statement is not, so the multiple is not a ratio."""
    store.write_snapshots(pd.DataFrame([
        _row("ADRLIKE", NOW, financial_currency="EUR", trailingPE=206.0)]))
    rep = _run(store)
    assert "ccy_sensitive_leak" in _checks(rep)
    assert _sev(rep, "ccy_sensitive_leak") == {quality.CRITICAL}


def test_catches_the_disappearance_of_every_foreign_filer(store):
    """ASML's actual symptom was an ABSENCE — the EUR filer produced no rows at all.
    A store with not one non-USD reporter left in it means the currency path is dead.
    Needs a realistic cross-section: the check is deliberately silent on a small store,
    where all-domestic is a legitimate outcome rather than evidence of anything."""
    store.write_snapshots(pd.DataFrame(
        [_row(f"US{i:04d}", NOW) for i in range(250)]))
    rep = _run(store)
    assert _sev(rep, "nonusd_filers_absent") == {quality.CRITICAL}


def test_a_store_with_foreign_filers_does_not_trip_that_check(store):
    store.write_snapshots(pd.DataFrame([
        _row("EURCO", NOW, financial_currency="EUR", trailingPE=None,
             priceToBook=None, ev_sales=None, priceToSalesTrailing12Months=None,
             enterpriseToEbitda=None, net_cash=None, fcf_yield=None, p_fcf=None)]))
    rep = _run(store)
    assert "nonusd_filers_absent" not in _checks(rep)
    assert "ccy_sensitive_leak" not in _checks(rep)


# ---------------------------------- bug 4: yfinance freeCashflow is a single QUARTER
def test_catches_info_freecashflow_being_quarterly(store):
    """`.info` is treated as the SUSPECT side: the finding names `.info`, not the store.
    Verified ratio at the time: MSFT SEC TTM 67.0B vs `.info` 16.5B = 4.06x."""
    rep = _run(store)
    quality.check_cross_source(
        store.snapshots(end=NOW.date()).drop_duplicates("symbol", keep="last").set_index("symbol"), ["GOOD"], rep,
        fetch=lambda syms: {"GOOD": {"freeCashflow": 5e9 / 4.06, "marketCap": 1e11}})
    assert "info_fcf_is_quarterly" in _checks(rep)
    assert any("ONE QUARTER" in f.detail for f in rep.findings
               if f.check == "info_fcf_is_quarterly")


def test_cross_source_flags_a_real_market_cap_disagreement(store):
    rep = _run(store)
    quality.check_cross_source(
        store.snapshots(end=NOW.date()).drop_duplicates("symbol", keep="last").set_index("symbol"), ["GOOD"], rep,
        fetch=lambda syms: {"GOOD": {"marketCap": 4e11}})
    assert "cross_source_disagreement" in _checks(rep)


# ------------------------------------------------ bug 5: Dominion / DLR missing capex
def test_catches_a_filer_with_cash_flow_but_no_capex_tag(store):
    """No capex tag means `freeCashflow` silently BECOMES cash flow from operations, so
    a capex-heavy utility or REIT reads strongly FCF-positive. Nothing is out of
    bounds; the number is simply the wrong quantity."""
    store.upsert_facts(pd.DataFrame(
        _facts("UTILITY", concepts=("totalRevenue", "netIncome", "cfo"))))
    store.write_snapshots(pd.DataFrame([_row("UTILITY", NOW)]))
    rep = _run(store)
    assert "capex_tag_missing" in _checks(rep)
    assert any(f.symbol == "UTILITY" for f in rep.findings
               if f.check == "capex_tag_missing")


# --------------------------------------------- bug 6: BRK-B has no share count at all
def test_catches_a_liquid_name_with_no_share_count(store, monkeypatch):
    """The failure leaves NO trace: null market cap -> fails the liquidity floor ->
    the name is simply absent, with nothing anywhere saying why. META, WMT and BRK-B
    all vanished this way."""
    px = pd.DataFrame([{"date": NOW - pd.Timedelta(days=i), "symbol": "BIGCO",
                        "open": 500.0, "high": 500.0, "low": 500.0, "close": 500.0,
                        "adj_close": 500.0, "volume": 1e7} for i in range(30)])
    store.upsert_prices(px)
    quality.check_floor_blockers(store, ["BIGCO"], AS_OF, rep := quality.Report())
    assert "no_share_count" in _checks(rep)
    assert rep.stats["floor_blocked"] == 1


def test_a_name_with_a_share_count_is_not_flagged_as_blocked(store):
    px = pd.DataFrame([{"date": NOW - pd.Timedelta(days=i), "symbol": "GOOD",
                        "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                        "adj_close": 100.0, "volume": 1e7} for i in range(30)])
    store.upsert_prices(px)
    store.upsert_facts(pd.DataFrame(
        _facts("GOOD", concepts=("totalRevenue", "netIncome", "cfo", "capex"))
        + [{"symbol": "GOOD", "cik": "0000000001", "concept": "sharesOutstanding",
            "tag": "CommonStockSharesOutstanding", "unit": "shares", "start": None,
            "end": pd.Timestamp("2026-03-31"), "val": 1e9,
            "filed": pd.Timestamp("2026-05-01"), "form": "10-Q", "frame": None,
            "fy": 2026.0, "fp": "Q1"}]))
    quality.check_floor_blockers(store, ["GOOD"], AS_OF, rep := quality.Report())
    assert "no_share_count" not in _checks(rep)


# ------------------------------------------------------- remaining invariant checks
def test_catches_a_market_cap_that_is_not_price_times_shares(store):
    store.write_snapshots(pd.DataFrame([
        _row("BADCAP", NOW, marketCap=3e11, price=100.0, sharesOutstanding=1e9)]))
    rep = _run(store)
    assert _sev(rep, "marketcap_inconsistent") == {quality.CRITICAL}


def test_catches_a_negative_pe_living_in_the_store(store):
    store.write_snapshots(pd.DataFrame([_row("LOSSCO", NOW, trailingPE=-14.0)]))
    rep = _run(store)
    assert _sev(rep, "pe_nonpositive") == {quality.CRITICAL}


def test_catches_a_share_count_that_jumped_by_orders_of_magnitude(store):
    store.write_snapshots(pd.DataFrame([
        _row("SPLITCO", PREV, sharesOutstanding=1e9),
        _row("SPLITCO", NOW, sharesOutstanding=1e13, marketCap=1e15)]))
    rep = _run(store)
    assert "share_count_jump" in _checks(rep)


def test_catches_a_screened_name_with_no_snapshot_at_all(store):
    rep = _run(store, symbols=["GOOD", "VANISHED"])
    assert "no_snapshot" in _checks(rep)
    assert any(f.symbol == "VANISHED" for f in rep.findings if f.check == "no_snapshot")


def test_catches_a_store_whose_snapshots_are_past_the_freshness_cut(store):
    rep = _run(store, symbols=["GOOD"])
    assert "snapshots_stale" not in _checks(rep)
    rep = quality.run(store, symbols=["GOOD"], sectors={}, baseline={},
                      as_of=AS_OF + timedelta(days=200))
    assert _sev(rep, "snapshots_stale") == {quality.CRITICAL}


# ------------------------------------------------------------------------ plumbing
def test_baseline_round_trips_through_the_store(store):
    rep = _run(store)
    assert quality.load_baseline(store) == {}
    written = quality.write_baseline(store, rep)
    assert quality.load_baseline(store)["coverage"] == written["coverage"]
    assert quality.baseline_path(store).parent == store.root


def test_findings_carry_a_symbol_a_value_and_a_reason(store):
    store.write_snapshots(pd.DataFrame([_row("BADGM", NOW, grossMargins=2.0)]))
    rep = _run(store)
    f = next(x for x in rep.findings if x.check == "gross_margin_impossible")
    assert f.symbol == "BADGM" and f.observed == 2.0 and "gross profit" in f.detail
    assert rep.worst() == quality.CRITICAL
    assert set(rep.to_dict()) == {"counts", "stats", "seconds", "findings"}


def test_catches_a_market_cap_far_below_the_name_s_own_daily_volume(store):
    """A WRONG share count is worse than a missing one and looks like nothing: the
    market cap is simply small. `marketCap / ADV` is scale-free, so it catches a count
    off by orders of magnitude without an opinion about valuation. Measured cases:
    Amcor at 13,001 shares against $100m/day, Genius Sports at 1.7m."""
    px = pd.DataFrame([{"date": NOW - pd.Timedelta(days=i), "symbol": "GOOD",
                        "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                        "adj_close": 100.0, "volume": 1e7} for i in range(30)])
    store.upsert_prices(px)
    store.upsert_facts(pd.DataFrame(
        _facts("GOOD")
        + [{"symbol": "GOOD", "cik": "0000000001", "concept": "sharesOutstanding",
            "tag": "CommonStockSharesOutstanding", "unit": "shares", "start": None,
            "end": pd.Timestamp("2026-03-31"), "val": 13_001.0,
            "filed": pd.Timestamp("2026-05-01"), "form": "10-Q", "frame": None,
            "fy": 2026.0, "fp": "Q1"}]))
    snap = pd.DataFrame([_row("GOOD", NOW, sharesOutstanding=13_001.0,
                              marketCap=1.3e6)]).set_index("symbol")
    quality.check_floor_blockers(store, ["GOOD"], AS_OF, rep := quality.Report(),
                                 snap=snap)
    assert "market_cap_implausible" in _checks(rep)
    assert "no_share_count" not in _checks(rep)      # it HAS a count; it is wrong


def test_a_share_count_move_matched_by_the_price_is_not_reported(store):
    """A count and a price that move together leave the market cap continuous — a
    re-tagging, or a split both sides have absorbed. Only a count that moves while the
    price does not is evidence of anything, so the market cap has to move too."""
    store.write_snapshots(pd.DataFrame([
        _row("SPLITCO", PREV, sharesOutstanding=2.5e8, price=400.0, marketCap=1e11),
        _row("SPLITCO", NOW, sharesOutstanding=1e9, price=100.0, marketCap=1e11)]))
    rep = _run(store)
    assert "share_count_jump" not in _checks(rep)
