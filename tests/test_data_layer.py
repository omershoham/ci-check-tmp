"""Point-in-time guarantees of the history store and the SEC facts assembly — all
synthetic, no network. The PIT assertion here is the guard against trap #1 in the
plan: nothing filed after `as_of` may ever influence a value 'as of' that date."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.analytics.metrics import derive
from src.data import sec_facts
from src.data.history_store import HistoryStore


def _flow(concept, tag, rows):
    """rows: (start, end, val, filed)"""
    return [{"symbol": "TST", "cik": "0000000001", "concept": concept, "tag": tag, "unit": "USD",
             "start": s, "end": e, "val": v, "filed": f, "form": "10-Q", "frame": None,
             "fy": None, "fp": None} for s, e, v, f in rows]


def _inst(concept, tag, rows):
    """rows: (end, val, filed)"""
    return [{"symbol": "TST", "cik": "0000000001", "concept": concept, "tag": tag, "unit": "USD",
             "start": None, "end": e, "val": v, "filed": f, "form": "10-Q", "frame": None,
             "fy": None, "fp": None} for e, v, f in rows]


@pytest.fixture
def facts():
    # FY2023 quarters Q1-Q3 reported quarterly, Q4 only inside the 10-K (FY total),
    # then FY2024 Q1..Q3. Filing lags of ~40 days.
    rev = _flow("totalRevenue", "Revenues", [
        ("2023-01-01", "2023-03-31", 100, "2023-05-10"),
        ("2023-04-01", "2023-06-30", 110, "2023-08-09"),
        ("2023-07-01", "2023-09-30", 120, "2023-11-08"),
        ("2023-01-01", "2023-12-31", 460, "2024-02-25"),   # FY -> Q4 = 130
        ("2024-01-01", "2024-03-31", 140, "2024-05-09"),
        ("2024-04-01", "2024-06-30", 150, "2024-08-08"),
        ("2024-07-01", "2024-09-30", 160, "2024-11-07"),
    ])
    ni = _flow("netIncome", "NetIncomeLoss", [
        ("2023-01-01", "2023-03-31", 10, "2023-05-10"),
        ("2023-04-01", "2023-06-30", 11, "2023-08-09"),
        ("2023-07-01", "2023-09-30", 12, "2023-11-08"),
        ("2023-01-01", "2023-12-31", 46, "2024-02-25"),
        ("2024-01-01", "2024-03-31", 14, "2024-05-09"),
        ("2024-04-01", "2024-06-30", 15, "2024-08-08"),
        ("2024-07-01", "2024-09-30", 16, "2024-11-07"),
    ])
    assets = _inst("totalAssets", "Assets", [
        ("2023-12-31", 1000, "2024-02-25"), ("2024-09-30", 1200, "2024-11-07")])
    shares = [dict(r, unit="shares") for r in _inst("sharesOutstanding",
              "EntityCommonStockSharesOutstanding",
              [("2024-02-15", 50, "2024-02-25"), ("2024-10-31", 52, "2024-11-07")])]
    df = pd.DataFrame(rev + ni + assets + shares)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    return df


def test_q4_is_derived_from_fy_and_ttm_chains(facts):
    ts = pd.Timestamp("2024-12-15")
    rev = sec_facts._pit(facts[facts["concept"] == "totalRevenue"], ts)
    q = sec_facts._quarters(rev)
    assert 130 in q["val"].values                      # Q4 2023 = 460 - (100+110+120)
    assert sec_facts._ttm(rev, ts) == 130 + 140 + 150 + 160
    assert sec_facts._ttm_prev(rev, ts) == 100 + 110 + 120 + 130   # the four quarters before
    # A fixed-lag cutoff that lands before Q4'23 was filed has no chain and no FY -> None
    assert sec_facts._ttm(rev, ts, lag_days=365) is None


def test_point_in_time_never_uses_unfiled_facts(facts):
    # On 2024-02-10 the FY2023 10-K (filed 02-25) does NOT exist yet: no Q4, only 3
    # quarters -> chain of 4 impossible -> no annual fallback available either -> None.
    early = pd.Timestamp("2024-02-10")
    rev_early = sec_facts._pit(facts[facts["concept"] == "totalRevenue"], early)
    assert (rev_early["filed"] <= early).all()
    assert sec_facts._ttm(rev_early, early) is None
    # One day after the filing, the FY figure is knowable and TTM = FY total.
    after = pd.Timestamp("2024-02-26")
    rev_after = sec_facts._pit(facts[facts["concept"] == "totalRevenue"], after)
    assert sec_facts._ttm(rev_after, after) == 460


def test_raw_asof_and_derive_end_to_end(facts):
    raw = sec_facts.raw_asof(facts, date(2024, 12, 15), price=20.0, ma50=19.0, ma200=18.0)
    assert raw["totalRevenue"] == 580 and raw["netIncome"] == 58
    assert raw["sharesOutstanding"] == 52 and raw["marketCap"] == 20.0 * 52
    assert raw["totalAssets"] == 1200
    assert abs(raw["revenueGrowth"] - (580 / 460 - 1)) < 1e-9
    m = derive(raw, as_of=date(2024, 12, 15))
    assert abs(m["trailingPE"] - (20.0 * 52 / 58)) < 1e-9
    assert abs(m["priceToSalesTrailing12Months"] - (20.0 * 52 / 580)) < 1e-9
    assert abs(m["returnOnAssets"] - 58 / 1200) < 1e-9
    assert abs(m["mom_50"] - 20 / 19) < 1e-9
    assert m["profitable"] is True
    assert "days_to_earnings" not in m                  # historical raw has no earnings stamp


def test_fast_path_matches_reference_path(facts):
    """The prepared/numpy path must reproduce the reference pandas path exactly,
    including the PIT cut-off behaviour around the 10-K filing date."""
    pf = sec_facts.prepare_symbol(facts)
    for d in (date(2024, 2, 10), date(2024, 2, 26), date(2024, 12, 15), date(2023, 6, 1)):
        ref = sec_facts.raw_asof(facts, d, price=20.0, ma50=19.0, ma200=18.0)
        fast = sec_facts.raw_asof_prepared(pf, d, price=20.0, ma50=19.0, ma200=18.0)
        if ref is None or fast is None:
            assert ref is None and fast is None, d
            continue
        for k in ("totalRevenue", "netIncome", "totalAssets", "sharesOutstanding", "marketCap",
                  "revenueGrowth", "earningsGrowth"):
            a, b = ref.get(k), fast.get(k)
            assert (a is None and b is None) or abs(a - b) < 1e-9, (d, k, a, b)


def test_derive_is_shared_definition():
    """Same raw figures -> same metrics whether they came from .info or from facts."""
    raw = {"price": 50.0, "marketCap": 5e9, "freeCashflow": 2e8, "totalRevenue": 1e9,
           "totalCash": 3e8, "totalDebt": 1e8, "revenueGrowth": 0.2, "grossMargins": 0.6,
           "financialCurrency": "USD"}
    a = derive(raw)
    b = derive(dict(raw))
    assert a == b
    assert abs(a["fcf_yield"] - 0.04) < 1e-12 and a["p_fcf"] == 25.0
    assert a["rule_of_40"] == 20 + 20 and a["net_cash"] == 2e8
    adr = derive(dict(raw, financialCurrency="TWD"))
    assert adr["fcf_yield"] is None and adr["grossMargins"] == 0.6   # masked vs kept


def test_history_store_roundtrip_and_pit_snapshot_lookup(tmp_path):
    store = HistoryStore(tmp_path / "hist")
    px = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02"]),
                       "symbol": ["AAA", "AAA", "BBB"], "open": [1, 1, 2], "high": [1, 1, 2],
                       "low": [1, 1, 2], "close": [10.0, 11.0, 20.0],
                       "adj_close": [9.0, 10.0, 20.0], "volume": [1, 1, 1]})
    assert store.upsert_prices(px) == 3
    # newer row for the same (symbol, date) wins
    store.upsert_prices(px.iloc[[1]].assign(close=12.0, adj_close=11.0))
    mat = store.close_matrix(["AAA", "BBB"], adjusted=False)
    assert mat.loc["2024-01-03", "AAA"] == 12.0 and pd.isna(mat.loc["2024-01-03", "BBB"])

    snaps = pd.DataFrame([
        {"as_of": "2024-01-31", "symbol": "AAA", "fcf_yield": 0.01},
        {"as_of": "2024-02-29", "symbol": "AAA", "fcf_yield": 0.02},
        {"as_of": "2023-06-30", "symbol": "BBB", "fcf_yield": 0.09},   # stale (>120d)
    ])
    store.write_snapshots(snaps)
    got = store.fundamentals_asof(["AAA", "BBB"], date(2024, 2, 15))
    assert got.loc["AAA", "fcf_yield"] == 0.01                        # Feb-29 not yet knowable
    assert "BBB" not in got.index                                     # too stale to serve
    got2 = store.fundamentals_asof(["AAA"], date(2024, 3, 15))
    assert got2.loc["AAA", "fcf_yield"] == 0.02

    store.record_universe(date(2024, 1, 31), [{"symbol": "AAA", "subsector": "x"},
                                              {"symbol": "BBB", "subsector": "y"}])
    store.record_universe(date(2024, 2, 29), [{"symbol": "AAA", "subsector": "x"}])
    assert store.universe_asof(date(2024, 2, 10)) == ["AAA", "BBB"]
    assert store.universe_asof(date(2024, 3, 1)) == ["AAA"]
    assert "recorded snapshot" in store.universe_source(date(2024, 3, 1))
    assert "hindsight" in store.universe_source(date(2023, 1, 1))
    assert store.manifest()["prices"]["rows"] == 3
