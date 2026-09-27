"""Gaps a hand-picked universe hits that the broad one does not: sectors and sizes
for names the bulk SEC path never sees (found by the first `scripts/demo` run)."""
from __future__ import annotations

from src import fundamentals, universe
from src.data import registrants
from src.data.history_store import HistoryStore


def test_curated_names_get_sectors_from_the_cached_sic_lookup(workspace):
    store = HistoryStore()
    calls: list[str] = []

    def fake_fetch(cik):
        calls.append(cik)
        return {"1": "3674", "2": "6022"}.get(cik)

    cikmap = {"NVDA": "1", "JPM": "2", "ZZZZ": "3"}
    res = registrants.refresh_symbol_sics(store, ["NVDA", "JPM", "ZZZZ"],
                                          fetch=fake_fetch, cikmap=cikmap)
    assert res == {"fetched": 2, "cached": 0, "unresolved": 1}
    secs = universe.sector_map([{"symbol": s} for s in ("NVDA", "JPM", "ZZZZ")],
                               store=store)
    assert secs["NVDA"] == "Information Technology"
    assert secs["JPM"] == "Financials"
    assert secs["ZZZZ"] == "Unknown"
    # SIC is close to constant: a second refresh only asks about names still unknown.
    calls.clear()
    registrants.refresh_symbol_sics(store, ["NVDA", "JPM", "ZZZZ"], fetch=fake_fetch,
                                    cikmap=cikmap)
    assert calls == ["3"]


def test_names_without_an_sec_share_count_are_sized_from_yahoo():
    """Visa tags its share count per class, so the store cannot size it and a name the
    user picked on purpose would silently fail the floor."""
    metrics = {"V": {"price": 340.0}, "AAPL": {"price": 230.0, "marketCap": 3.4e12},
               "NOPX": {}}
    seen: list[list[str]] = []

    def fake_fetch(syms, **kw):
        seen.append(list(syms))
        return {"V": {"marketCap": 6.6e11}}

    assert fundamentals.fill_missing_market_caps(metrics, fetch=fake_fetch) == ["V"]
    assert seen == [["V"]]                          # only the unsized, priced name
    assert metrics["V"]["marketCap"] == 6.6e11 and metrics["V"]["marketCap_source"] == "yahoo"
    assert fundamentals.passes_floor({**metrics["V"], "avg_dollar_volume": 1e9})


def test_the_yahoo_fallback_refuses_to_paper_over_a_thin_store():
    metrics = {f"S{i}": {"price": 10.0} for i in range(30)}
    called = []
    assert fundamentals.fill_missing_market_caps(
        metrics, max_names=25, fetch=lambda *a, **k: called.append(1) or {}) == []
    assert not called


def _cf(filed_by_end):
    """A minimal companyfacts payload: quarterly Revenues, one row per (end, filed)."""
    rows = [{"start": f"{e[:4]}-{int(e[5:7]) - 2:02d}-01", "end": e, "val": 100.0,
             "filed": f, "form": "10-Q", "fy": int(e[:4]), "fp": "Q", "accn": f"a{i}"}
            for i, (e, f) in enumerate(filed_by_end)]
    return {"facts": {"us-gaap": {"Revenues": {"units": {"USD": rows}}}}}


def test_a_reorganized_registrant_keeps_its_predecessors_history(monkeypatch):
    """Exxon's 2026 holding-company reorganization moved XOM to a new CIK with one
    filing; the configured predecessor's earlier filings are merged in under the
    successor's CIK, and only those filed BEFORE the successor's first."""
    from src import config
    from src.data import sec_facts
    monkeypatch.setitem(config._OVERRIDES, "universe.cik_predecessors",
                        {"XOM": ["34088"]})
    payloads = {
        "0000034088": _cf([("2025-09-30", "2025-11-03"), ("2026-03-31", "2026-05-04"),
                           ("2026-06-30", "2026-08-03")]),
    }
    monkeypatch.setattr(sec_facts, "fetch_companyfacts",
                        lambda cik, pause=0.15: payloads.get(cik))
    new = sec_facts.facts_frame("XOM", "0002115436", _cf([("2026-06-30", "2026-08-03")]))
    got = sec_facts._with_predecessors("XOM", "0002115436", new, pause=0)
    assert sorted(got["filed"].dt.strftime("%Y-%m-%d")) == \
        ["2025-11-03", "2026-05-04", "2026-08-03"]          # no double 2026-08-03
    assert set(got["cik"]) == {"0002115436"}
    assert sec_facts.predecessor_ciks(["XOM"]) == {"0000034088": "XOM"}
    assert sec_facts.predecessor_ciks(["AAPL"]) == {}
