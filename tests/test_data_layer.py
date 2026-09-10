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


# ---------------------------------------------------- partitioned store (Phase 2)
def _px(sym, dates, close):
    return pd.DataFrame({"date": pd.to_datetime(dates), "symbol": sym,
                         "open": close, "high": close, "low": close, "close": close,
                         "adj_close": close, "volume": [1_000.0] * len(dates)})


def test_append_plus_compact_equals_upsert(tmp_path):
    """`upsert_prices` is append + compact; the two must land the same bytes, or the
    daily fast path and the batch path have quietly forked."""
    a = _px("AAA", ["2023-12-28", "2023-12-29", "2024-01-02"], [10.0, 11.0, 12.0])
    b = _px("AAA", ["2024-01-02", "2024-01-03"], [99.0, 13.0])       # 01-02 restated
    ref = HistoryStore(tmp_path / "ref")
    ref.upsert_prices(a)
    ref.upsert_prices(b)

    inc = HistoryStore(tmp_path / "inc")
    inc.append_prices(a)
    inc.append_prices(b)
    # before compaction the duplicate is visible — that is the documented contract
    assert len(inc.prices()) == 5
    inc.compact_prices([2024])
    assert ref.prices().equals(inc.prices())
    assert inc.prices().set_index(["symbol", "date"]).loc[("AAA", pd.Timestamp("2024-01-02")), "close"] == 99.0
    assert ref.manifest()["prices"]["rows"] == inc.manifest()["prices"]["rows"] == 4
    # one directory per year, and 2023 was never rewritten by the second write
    assert sorted(p.name for p in (inc.prices_dir).glob("year=*")) == ["year=2023", "year=2024"]


def test_partition_pruning_matches_a_full_read(tmp_path):
    store = HistoryStore(tmp_path / "hist")
    store.upsert_prices(pd.concat([
        _px("AAA", ["2022-06-01", "2023-06-01", "2024-06-01"], [1.0, 2.0, 3.0]),
        _px("BBB", ["2022-06-01", "2023-06-01", "2024-06-01"], [4.0, 5.0, 6.0])]))
    full = store.prices()
    win = store.prices(["AAA"], start=date(2023, 1, 1), end=date(2023, 12, 31))
    expect = full[(full["symbol"] == "AAA") & (full["date"].dt.year == 2023)].reset_index(drop=True)
    assert win.equals(expect)
    assert store.close_matrix(start=date(2023, 1, 1), end=date(2023, 12, 31)).shape == (1, 2)
    assert store.symbols_with_prices() == ["AAA", "BBB"]
    assert store.last_date() == pd.Timestamp("2024-06-01")


def test_shares_asof_never_sees_the_future(tmp_path, facts):
    store = HistoryStore(tmp_path / "hist")
    store.upsert_facts(facts)
    # the 2024-10-31 count was filed 2024-11-07: invisible the day before, served after
    assert store.shares_asof(["TST"], date(2024, 11, 6)).loc["TST"] == 50
    assert store.shares_asof(["TST"], date(2024, 11, 8)).loc["TST"] == 52
    assert store.shares_asof(["TST"], date(2024, 2, 1)).empty
    assert store.facts_quarters() == ["2023q2", "2023q3", "2023q4", "2024q1", "2024q2",
                                      "2024q3", "2024q4"]
    # A quarter directory exists because companyfacts filed into it — the bulk path
    # must not read that as "the DERA ZIP for 2024q1 is already ingested".
    assert store.facts_quarters(source="fsds") == []
    store.append_facts_quarter("2024q1", facts[facts["filed"] == "2024-02-25"],
                               source="fsds")
    assert store.facts_quarters(source="fsds") == ["2024q1"]


def test_legacy_single_file_store_is_migrated_in_place(tmp_path):
    """The layout an existing checkout has on disk must survive the upgrade untouched."""
    root = tmp_path / "hist"
    px = _px("AAA", ["2023-12-29", "2024-01-02"], [10.0, 11.0])
    (root / "prices").mkdir(parents=True)
    px.to_parquet(root / "prices" / "daily.parquet", index=False)
    store = HistoryStore(root)
    got = store.prices()                                   # lazy migration on first read
    assert got["close"].tolist() == [10.0, 11.0]
    assert not (root / "prices" / "daily.parquet").exists()
    assert (root / "prices" / "daily.parquet.migrated").exists()   # original kept
    assert sorted(p.name for p in (root / "prices" / "daily").glob("year=*")) == \
        ["year=2023", "year=2024"]


def test_snapshot_replay_is_identical_serial_and_parallel(tmp_path, facts):
    """The process pool exists to be faster, not different: same rows, same order."""
    from src.data import sec_facts as SF
    store = HistoryStore(tmp_path / "hist")
    store.upsert_facts(pd.concat([facts, facts.assign(symbol="TST2")], ignore_index=True))
    days = pd.bdate_range("2023-01-02", "2024-12-31")
    store.upsert_prices(pd.concat([
        _px("TST", days, [20.0] * len(days)), _px("TST2", days, [30.0] * len(days))]))
    dates = [pd.Timestamp("2024-06-28"), pd.Timestamp("2024-12-31")]
    one = SF.build_snapshots(store, ["TST", "TST2"], dates, log=lambda m: None, workers=1)
    many = SF.build_snapshots(store, ["TST", "TST2"], dates, log=lambda m: None, workers=2)
    assert not one.empty and one.equals(many)


# ---------------------------------------------------------------- test isolation
def test_the_default_history_store_lives_inside_pytest_tmp(tmp_path_factory):
    """`HistoryStore()` with no root must never resolve to the user's real data.

    It used to: the constructor read `config.HOME` directly, so `record_universe` in
    the offline pipeline test appended a row to `$CLAUDE_TRADER_HOME/data/history`.
    Cloning this repo and running the suite mutated your own store.
    """
    base = tmp_path_factory.getbasetemp().resolve()
    assert base in HistoryStore().root.resolve().parents


def test_a_store_outside_the_tmp_tree_fails_loudly(tmp_path):
    """The redirect above is a default; this is the enforcement.

    A helper that builds its own path from `config.HOME`, or a store constructed
    before the fixture ran, would slip past the monkeypatch — and the damage is
    invisible, because the write succeeds and the suite still passes.
    """
    import pathlib
    with pytest.raises(AssertionError, match="OUTSIDE pytest's tmp tree"):
        HistoryStore(pathlib.Path.home() / "definitely-not-a-test-store")


# ------------------------------------------------------------------------- capex
def _quarters(concept, tag, vals, year=2024):
    """Four quarters of one tag, filed ~40 days after each period end."""
    ends = [(f"{year}-01-01", f"{year}-03-31", f"{year}-05-09"),
            (f"{year}-04-01", f"{year}-06-30", f"{year}-08-08"),
            (f"{year}-07-01", f"{year}-09-30", f"{year}-11-07"),
            (f"{year}-10-01", f"{year}-12-31", f"{year + 1}-02-20")]
    return _flow(concept, tag, [(s, e, v, f) for (s, e, f), v in zip(ends, vals)])


def _prepared(rows):
    """Prepared facts for a synthetic filer. Revenue is always present because
    `raw_asof_prepared` returns None outright when nothing core is knowable."""
    rows = rows + _quarters("totalRevenue", "Revenues", [1000, 1000, 1000, 1000])
    df = pd.DataFrame(rows)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    return sec_facts.prepare_symbol(df)


def test_a_utility_capex_tag_is_subtracted_from_operating_cash_flow(tmp_path):
    """Dominion reports capex as `...IncludingNuclearFuel`, Digital Realty as
    `PaymentsToDevelopRealEstateAssets`. With neither in FLOW_TAGS, `freeCashflow`
    silently BECAME cash flow from operations, so a capex-heavy utility and a REIT both
    read strongly FCF-positive — a wrong number, not a missing one."""
    for tag in ("PaymentsToAcquirePropertyPlantAndEquipmentIncludingNuclearFuel",
                "PaymentsToDevelopRealEstateAssets", "PaymentsForCapitalImprovements"):
        pf = _prepared(_quarters("cfo", "NetCashProvidedByUsedInOperatingActivities",
                                 [100, 100, 100, 100])
                       + _quarters("capex", tag, [30, 30, 30, 30]))
        raw = sec_facts.raw_asof_prepared(pf, date(2025, 3, 31), price=10.0)
        assert raw["cfo"] == 400 and raw["freeCashflow"] == 400 - 120, tag


def test_capex_falls_back_to_cash_flow_only_when_no_tag_resolves(tmp_path):
    """The behaviour the fallbacks exist to stop, pinned so it cannot come back
    silently: with no capex tag at all, FCF is CFO and nothing says so."""
    pf = _prepared(_quarters("cfo", "NetCashProvidedByUsedInOperatingActivities",
                             [100, 100, 100, 100]))
    raw = sec_facts.raw_asof_prepared(pf, date(2025, 3, 31), price=10.0)
    assert raw["freeCashflow"] == raw["cfo"] == 400


def test_the_freshest_capex_tag_wins_and_ties_go_to_list_order(tmp_path):
    """`_best()` picks the tag whose TTM ends LATEST — so a filer that switched tags is
    read correctly, and one that reports two tags in the SAME filing produces an exact
    tie that `FLOW_TAGS` order must break toward the broader measure. Getting that
    backwards would make a narrow sub-component (capital improvements) out-rank total
    PP&E and UNDERSTATE capex for names that were previously right."""
    old = _quarters("capex", "PaymentsToAcquirePropertyPlantAndEquipment",
                    [30, 30, 30, 30], year=2023)
    new = _quarters("capex", "PaymentsForCapitalImprovements", [50, 50, 50, 50],
                    year=2024)
    pf = _prepared(_quarters("cfo", "NetCashProvidedByUsedInOperatingActivities",
                             [100, 100, 100, 100]) + old + new)
    raw = sec_facts.raw_asof_prepared(pf, date(2025, 3, 31), price=10.0)
    assert raw["freeCashflow"] == 400 - 200                       # freshest tag wins

    tie = (_quarters("cfo", "NetCashProvidedByUsedInOperatingActivities",
                     [100, 100, 100, 100])
           + _quarters("capex", "PaymentsForCapitalImprovements", [10, 10, 10, 10])
           + _quarters("capex", "PaymentsToAcquirePropertyPlantAndEquipment",
                       [30, 30, 30, 30]))
    raw = sec_facts.raw_asof_prepared(_prepared(tie), date(2025, 3, 31), price=10.0)
    assert raw["freeCashflow"] == 400 - 120                       # broadest tag wins


def test_companyfacts_emits_the_us_gaap_share_fallbacks(tmp_path):
    """Both ingest paths must be able to serve the same concepts. This one used to emit
    ONLY the dei cover-page count, so a filer that omits it arrived with no share count
    at all -> null market cap -> silent liquidity-floor failure."""
    cf = {"facts": {"us-gaap": {
        "CommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2024-09-30", "val": 1000, "filed": "2024-11-07", "form": "10-Q"}]}},
        "WeightedAverageNumberOfSharesOutstandingBasic": {"units": {"shares": [
            {"start": "2024-07-01", "end": "2024-09-30", "val": 990,
             "filed": "2024-11-07", "form": "10-Q"}]}}}}}
    df = sec_facts.facts_frame("TST", "1", cf)
    sh = df[df["concept"] == "sharesOutstanding"]
    assert set(sh["tag"]) == {"CommonStockSharesOutstanding",
                              "WeightedAverageNumberOfSharesOutstandingBasic"}
    # A weighted average is tagged as a duration; `prepare_symbol` reads the share
    # count from the INSTANT table only, so it has to be re-stamped or it is dropped.
    assert sh["start"].isna().all()
    assert sec_facts.prepare_symbol(df).instants["sharesOutstanding"]


def _shares_row(tag, end, val, filed):
    return {"symbol": "TST", "cik": "0000000001", "concept": "sharesOutstanding",
            "tag": tag, "unit": "shares", "start": None, "end": end, "val": val,
            "filed": filed, "form": "10-Q", "frame": None, "fy": None, "fp": None}


def _shares(rows):
    df = pd.DataFrame(rows + _quarters("totalRevenue", "Revenues", [1, 1, 1, 1]))
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    pf = sec_facts.prepare_symbol(df)
    return sec_facts.raw_asof_prepared(pf, date(2026, 9, 10), price=1.0
                                       )["sharesOutstanding"]


def test_an_abandoned_share_tag_cannot_outrank_a_current_one():
    """Pure list order is the MSFT stale-tag bug one level down. Amcor kept a single
    2019 `CommonStockSharesOutstanding` row of 13,001 ahead of a current
    `CommonStockSharesIssued` of 462m — a market cap four orders of magnitude out, and
    the name silently below the liquidity floor."""
    assert _shares([
        _shares_row("CommonStockSharesOutstanding", "2019-03-31", 13_001, "2019-05-01"),
        _shares_row("CommonStockSharesIssued", "2026-03-31", 462e6, "2026-05-07"),
    ]) == 462e6


def test_a_zero_share_count_is_not_a_share_count():
    """Moody's tags `CommonStockSharesOutstanding` as 0 on every filing. Zero is not
    None, so it was returned as the answer and the market cap became undefined."""
    assert _shares([
        _shares_row("CommonStockSharesOutstanding", "2026-03-31", 0.0, "2026-04-23"),
        _shares_row("CommonStockSharesIssued", "2026-03-31", 342.9e6, "2026-04-23"),
    ]) == 342.9e6


def test_a_share_count_far_below_the_weighted_average_is_rejected():
    """Accenture's consolidated `CommonStockSharesOutstanding` is really its Class X
    count (300,673) against 616m weighted-average shares. A count cannot be a
    thousandth of the average OF that count, so a $180bn company was priced at $56m.

    The fallthrough lands on the weighted average, NOT on `CommonStockSharesIssued`:
    issued includes treasury stock and is a different quantity from outstanding, so a
    period average of the right number beats a point reading of the wrong one. For
    Accenture the two are 616.2m and 680.5m against ~625m actually outstanding."""
    assert _shares([
        _shares_row("CommonStockSharesOutstanding", "2026-05-31", 300_673, "2026-06-18"),
        _shares_row("CommonStockSharesIssued", "2026-05-31", 680.5e6, "2026-06-18"),
        _shares_row("WeightedAverageNumberOfSharesOutstandingBasic", "2026-05-31",
                    616.2e6, "2026-06-18"),
    ]) == 616.2e6


def test_issued_shares_are_the_last_resort_not_a_peer_of_outstanding():
    """Issued includes treasury stock. Ranked above the weighted averages it overstated
    market cap 2-3.5x on 216 of ~1,500 screened names -- AIG 1,906.7m issued vs 538.1m
    outstanding, Kroger 1,918.0m vs 613.0m, Aflac 1,358.8m vs 513.1m. It is still a
    valid last resort: a filer holding no treasury stock has issued == outstanding."""
    rows = [
        _shares_row("CommonStockSharesIssued", "2026-03-31", 1906.7e6, "2026-05-01"),
        _shares_row("WeightedAverageNumberOfSharesOutstandingBasic", "2026-03-31",
                    538.1e6, "2026-05-01"),
    ]
    assert _shares(rows) == 538.1e6
    # ...but with nothing better available, issued is used rather than nothing.
    assert _shares(rows[:1]) == 1906.7e6


def test_a_stale_weighted_average_is_not_used_as_the_plausibility_reference():
    """NextEra's weighted-average tags stop in 2019, BEFORE its 4-for-1 split. Using
    an abandoned 482m reference rejected a current, correct 2,085m count and cut its
    market cap from $174bn to $40bn — the plausibility check turning into the bug it
    exists to catch."""
    assert _shares([
        _shares_row("CommonStockSharesOutstanding", "2026-03-31", 2.085e9, "2026-04-25"),
        _shares_row("WeightedAverageNumberOfSharesOutstandingBasic", "2019-09-30",
                    481.9e6, "2019-10-25"),
    ]) == 2.085e9


def test_shares_asof_and_the_snapshot_path_agree_on_the_same_facts(tmp_path):
    """These two are the halves of one decision: `shares_asof` decides whether a name
    passes the market-cap floor at all, `raw_asof_prepared` prices every multiple once
    it has. `shares_asof` used to take whichever row was filed LAST regardless of tag,
    so Accenture's Class X count (300,673, the most recent filing) dropped a $180bn
    company from the universe while its snapshot had 680m."""
    store = HistoryStore(tmp_path / "hist")
    rows = [
        _shares_row("CommonStockSharesOutstanding", "2026-05-31", 300_673, "2026-06-18"),
        _shares_row("CommonStockSharesIssued", "2026-05-31", 680.5e6, "2026-06-18"),
        _shares_row("WeightedAverageNumberOfSharesOutstandingBasic", "2026-05-31",
                    616.2e6, "2026-06-18"),
    ]
    df = pd.DataFrame(rows + _quarters("totalRevenue", "Revenues", [1, 1, 1, 1]))
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    store.upsert_facts(df)
    floor = store.shares_asof(["TST"], date(2026, 9, 10))["TST"]
    snap = sec_facts.raw_asof_prepared(sec_facts.prepare_symbol(df), date(2026, 9, 10),
                                       price=1.0)["sharesOutstanding"]
    assert floor == snap == 616.2e6
