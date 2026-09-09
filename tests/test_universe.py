"""Broad universe: the filtering ladder, the point-in-time floor, and the promise that
the curated path did not move.

All synthetic — no SEC request, no Yahoo request. The floor and the ladder are where a
survivorship or look-ahead bug would hide, so both are asserted against hand-built data
whose right answer is obvious by inspection.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src import config, fundamentals, theme, universe
from src.data import registrants
from src.data.history_store import HistoryStore


# ------------------------------------------------------------------ the floor
def test_passes_floor_pit_matches_the_info_shim():
    """`passes_floor(dict)` must be `passes_floor_pit(numbers)` with different field
    names, or the live screen and the broad universe are floored by two different rules."""
    cases = [
        {"price": 50.0, "marketCap": 1e10, "avg_dollar_volume": 5e7},   # comfortably in
        {"price": 4.99, "marketCap": 1e10, "avg_dollar_volume": 5e7},   # price floor
        {"price": 50.0, "marketCap": 2.99e8, "avg_dollar_volume": 5e7},  # cap floor
        {"price": 50.0, "marketCap": 1e10, "avg_dollar_volume": 4.9e6},  # ADV floor
        {"price": 50.0, "marketCap": 1e10, "avg_dollar_volume": None},  # unknown ADV passes
        {"price": None, "marketCap": 1e10, "avg_dollar_volume": 5e7},   # unverifiable
        {"price": 50.0, "marketCap": None, "avg_dollar_volume": 5e7},
        {"price": 5.0, "marketCap": 3e8, "avg_dollar_volume": 5e6},     # exactly on it
    ]
    for m in cases:
        assert fundamentals.passes_floor(m) is fundamentals.passes_floor_pit(
            m["price"], m["marketCap"], m["avg_dollar_volume"]), m
    verdicts = [fundamentals.passes_floor(m) for m in cases]
    assert verdicts == [True, False, False, False, True, False, False, True]


def test_floor_reads_config_thresholds():
    f = fundamentals.floor_thresholds()
    cfg = config.get("universe.liquidity_floor", {})
    assert f["min_market_cap"] == cfg["min_market_cap"]
    assert f["min_price"] == cfg["min_price"]
    assert f["min_avg_dollar_volume"] == cfg["min_avg_dollar_volume"]


# ----------------------------------------------------------------- the ladder
def _exchange_rows():
    return pd.DataFrame([
        # cik,        name,          ticker,  exchange
        ("0000000001", "Operating Co", "AAA", "Nasdaq"),      # keeps
        ("0000000002", "Big Board Co", "BBB", "NYSE"),        # keeps
        ("0000000003", "Pink Sheets",  "CCC", "OTC"),         # off-exchange
        ("0000000004", "No Venue",     "DDD", ""),            # blank exchange
        ("0000000005", "SPAC Warrant", "EEEW", "Nasdaq"),     # ticker shape (5 letters, ok)
        ("0000000006", "Unit",         "FFF-U", "Nasdaq"),    # unit
        ("0000000007", "Closed End",   "GGG", "NYSE"),        # SIC 6726
        ("0000000008", "Blank Check",  "HHH", "Nasdaq"),      # SIC 6770
        ("0000000009", "A REIT",       "III", "NYSE"),        # SIC 6798 — MUST survive
        ("0000000010", "Never Filed",  "JJJ", "Nasdaq"),      # no XBRL statements
        ("0000000011", "Fund Reports", "KKK", "NYSE"),        # N-CSR only
    ], columns=["cik", "name", "ticker", "exchange"])


def _sub_rows():
    return pd.DataFrame([
        ("0000000001", "Operating Co", "7372", "10-K|10-Q", "2016-01-05", "2026-05-05"),
        ("0000000002", "Big Board Co", "2911", "10-K", "2016-01-05", "2026-05-05"),
        ("0000000003", "Pink Sheets", "7372", "10-K", "2016-01-05", "2026-05-05"),
        ("0000000004", "No Venue", "7372", "10-K", "2016-01-05", "2026-05-05"),
        ("0000000005", "SPAC Warrant", "7372", "10-K", "2016-01-05", "2026-05-05"),
        ("0000000006", "Unit", "7372", "10-K", "2016-01-05", "2026-05-05"),
        ("0000000007", "Closed End", "6726", "N-2", "2016-01-05", "2026-05-05"),
        ("0000000008", "Blank Check", "6770", "10-K", "2016-01-05", "2026-05-05"),
        ("0000000009", "A REIT", "6798", "10-K", "2016-01-05", "2026-05-05"),
        ("0000000011", "Fund Reports", "6199", "24F-2NT|N-CSR", "2016-01-05", "2026-05-05"),
    ], columns=["cik", "sec_name", "sic", "forms", "first_filed", "last_filed"])


@pytest.fixture
def ladder(monkeypatch, tmp_path):
    monkeypatch.setattr(registrants, "load_exchange", lambda force=False: _exchange_rows())
    monkeypatch.setattr(registrants, "build_sub_index",
                        lambda store, force=False, log=None: _sub_rows())
    return HistoryStore(tmp_path / "history")


@pytest.mark.parametrize("ticker,keep,why", [
    ("AAPL", True, "plain common stock"),
    ("BRK-B", True, "share class — the case a bare ^[A-Z]{1,5}$ silently deleted"),
    ("BF-B", True, "Brown-Forman class B"),
    ("MOG-A", True, "Moog class A"),
    ("MKC-V", True, "McCormick non-voting"),
    ("ABCDE", True, "five letters is still a ticker"),
    ("XYZ-W", False, "warrant"),
    ("XYZ-WT", False, "warrant, two-letter form"),
    ("XYZ-U", False, "unit"),
    ("XYZ-UN", False, "unit, two-letter form"),
    ("XYZ-R", False, "right"),
    ("XYZ-RI", False, "right, two-letter form"),
    ("XYZ-P", False, "preferred, bare — the one single-letter suffix that is not a class"),
    ("XYZ-PA", False, "preferred series A"),
    ("ABCDEF", False, "six letters is not a US common-stock ticker"),
    ("BRK.B", False, "dotted form is not what the SEC file uses"),
])
def test_ticker_shape_rule(ticker, keep, why):
    ok = (bool(registrants._TICKER_RE.match(ticker))
          and not ticker.endswith(registrants._DERIVATIVE_SUFFIXES))
    assert ok is keep, why


def test_ladder_counts_and_survivors(ladder):
    df, funnel = registrants.build_candidates(ladder)
    assert funnel == {"sec_listed": 11, "on_exchange": 9, "common_stock_ticker": 8,
                      "files_financials": 7, "operating_company": 5,
                      "not_fund_reporter": 4}
    assert sorted(df["symbol"]) == ["AAA", "BBB", "EEEW", "III"]
    # The REIT survives the same filter that removes the closed-end fund next to it.
    assert df.loc[df["symbol"] == "III", "sector"].iloc[0] == "Real Estate"
    assert df.loc[df["symbol"] == "AAA", "industry_group"].iloc[0] == "Software & Services"


def test_refresh_registrants_round_trips(ladder):
    res = registrants.refresh_registrants(ladder)
    assert res["rows"] == 4
    back = registrants.load_registrants(ladder)
    assert set(back.columns) >= {"symbol", "cik", "name", "exchange", "sic", "sector",
                                 "industry_group", "first_filed", "last_filed",
                                 "first_price_date", "last_price_date"}


# ----------------------------------------------------------- investable(as_of)
def _prices(sym, start, n, close, volume):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"date": idx, "symbol": sym, "open": close, "high": close,
                         "low": close, "close": float(close),
                         "adj_close": float(close), "volume": float(volume)})


def _shares(sym, cik, rows):
    """rows: (end, val, filed)"""
    return pd.DataFrame([{"symbol": sym, "cik": cik, "concept": "sharesOutstanding",
                          "tag": "EntityCommonStockSharesOutstanding", "unit": "shares",
                          "start": None, "end": pd.Timestamp(e), "val": float(v),
                          "filed": pd.Timestamp(f), "form": "10-Q", "frame": None,
                          "fy": None, "fp": None} for e, v, f in rows])


@pytest.fixture
def pit_store(tmp_path):
    """Three names, 300 business days to 2024-06-28:

      BIG   $100, 1M shares/day traded, 40M shares -> $4.0bn cap, $100M ADV : passes
      THIN  $100, 1k shares/day traded, 40M shares -> $4.0bn cap, $100k ADV : ADV floor
      GONE  $100 but stops trading in January                               : delisted
      YOUNG $100, only 50 sessions of history                               : too new
    """
    store = HistoryStore(tmp_path / "history")
    px = pd.concat([
        _prices("BIG", "2023-05-01", 300, 100, 1_000_000),
        _prices("THIN", "2023-05-01", 300, 100, 1_000),
        _prices("GONE", "2023-05-01", 180, 100, 1_000_000),
        _prices("YOUNG", "2024-04-15", 50, 100, 1_000_000),
    ], ignore_index=True)
    store.upsert_prices(px)
    facts = pd.concat([
        _shares("BIG", "0000000001", [("2023-12-31", 40e6, "2024-02-01")]),
        _shares("THIN", "0000000002", [("2023-12-31", 40e6, "2024-02-01")]),
        _shares("GONE", "0000000003", [("2023-12-31", 40e6, "2024-02-01")]),
        _shares("YOUNG", "0000000004", [("2023-12-31", 40e6, "2024-02-01")]),
    ], ignore_index=True)
    store.upsert_facts(facts)
    reg = pd.DataFrame([
        ("BIG", "0000000001", "Big Co", "NYSE", "7372"),
        ("THIN", "0000000002", "Thin Co", "NYSE", "7372"),
        ("GONE", "0000000003", "Gone Co", "NYSE", "7372"),
        ("YOUNG", "0000000004", "Young Co", "NYSE", "7372"),
    ], columns=["symbol", "cik", "name", "exchange", "sic"])
    reg["sector"], reg["industry_group"] = "Information Technology", "Software & Services"
    path = registrants.registrants_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    reg.to_parquet(path, index=False)
    return store


def test_investable_applies_the_floor_and_the_recency_rule(pit_store):
    inv = universe.investable(date(2024, 6, 28), store=pit_store)
    assert [r["symbol"] for r in inv] == ["BIG"]
    row = inv[0]
    assert row["price"] == 100.0
    assert row["marketCap"] == pytest.approx(4e9)
    assert row["adv"] == pytest.approx(1e8)
    # `subsector` is the industry group, so every existing consumer keeps working.
    assert row["subsector"] == row["industry_group"] == "Software & Services"
    assert row["sector"] == "Information Technology"


def test_investable_is_point_in_time_on_share_counts(tmp_path):
    """A share count FILED after `as_of` may not move the market cap. Here the later
    filing triples the count; on the earlier date the floor must still see the old one."""
    store = HistoryStore(tmp_path / "history")
    store.upsert_prices(_prices("AAA", "2023-05-01", 300, 100, 1_000_000))
    store.upsert_facts(_shares("AAA", "0000000001", [
        ("2023-12-31", 2.0e6, "2024-02-01"),     # $200m cap — below the $300m floor
        ("2024-03-31", 9.0e6, "2024-07-01"),     # $900m cap — but not knowable until July
    ]))
    reg = pd.DataFrame([{"symbol": "AAA", "cik": "0000000001", "name": "A", "exchange": "NYSE",
                         "sic": "7372", "sector": "Information Technology",
                         "industry_group": "Software & Services"}])
    p = registrants.registrants_path(store)
    p.parent.mkdir(parents=True, exist_ok=True)
    reg.to_parquet(p, index=False)

    assert universe.investable(date(2024, 6, 28), store=store) == []
    later = universe.investable(date(2024, 8, 30), store=store)
    assert [r["symbol"] for r in later] == ["AAA"]
    assert later[0]["marketCap"] == pytest.approx(9e8)


def test_multi_class_cik_is_summed_and_deduped(tmp_path):
    """Two tickers, one CIK. The company must appear ONCE, at the size of both classes
    together — attributed to whichever class actually trades."""
    store = HistoryStore(tmp_path / "history")
    store.upsert_prices(pd.concat([
        _prices("XYZ", "2023-05-01", 300, 100, 1_000_000),      # the liquid class
        _prices("XYZA", "2023-05-01", 300, 100, 200_000),       # the other one
    ], ignore_index=True))
    # Only one class carries the SEC filing (one CIK files, one ticker is mapped to it).
    store.upsert_facts(pd.concat([
        _shares("XYZ", "0000000009", [("2023-12-31", 3.0e6, "2024-02-01")]),
        _shares("XYZA", "0000000009", [("2023-12-31", 2.0e6, "2024-02-01")]),
    ], ignore_index=True))
    reg = pd.DataFrame([
        {"symbol": "XYZ", "cik": "0000000009", "name": "X", "exchange": "NYSE",
         "sic": "7372", "sector": "Information Technology",
         "industry_group": "Software & Services"},
        {"symbol": "XYZA", "cik": "0000000009", "name": "X", "exchange": "NYSE",
         "sic": "7372", "sector": "Information Technology",
         "industry_group": "Software & Services"},
    ])
    p = registrants.registrants_path(store)
    p.parent.mkdir(parents=True, exist_ok=True)
    reg.to_parquet(p, index=False)

    inv = universe.investable(date(2024, 6, 28), store=store)
    assert [r["symbol"] for r in inv] == ["XYZ"]           # the liquid class represents it
    assert inv[0]["marketCap"] == pytest.approx(5e8)       # 3m + 2m shares at $100


def test_max_symbols_caps_by_dollar_volume(pit_store):
    assert universe.investable(date(2024, 6, 28), store=pit_store, max_symbols=0) != []
    assert len(universe.investable(date(2024, 6, 28), store=pit_store, max_symbols=1)) == 1


# ------------------------------------------------------------- curated is intact
def test_curated_build_universe_is_the_theme_pack_list(curated):
    """The shipped default must be byte-identical to what it always was: the four
    sub-sector lists, deduped, first sub-sector winning on overlap. They moved from
    config.yaml into themes/ai_infra.yaml; the RESULT may not have moved with them."""
    expected = []
    seen = set()
    for sub, syms in theme.curated_sectors("ai_infra").items():
        for s in syms or []:
            s = str(s).strip().upper()
            if s and s not in seen:
                seen.add(s)
                expected.append({"symbol": s, "subsector": sub})
    assert universe.build_universe(source="curated") == expected
    assert len(expected) == 105
    assert expected[0]["symbol"] == "NVDA"
    # GOOG and GOOGL are BOTH curated members — the multi-class de-dup is a
    # broad-universe rule, and must not reach back into the curated list.
    assert {"GOOG", "GOOGL"} <= seen


def test_configured_source_defaults_to_curated(monkeypatch):
    """A config with no `universe.source` key must behave exactly as it did before
    this key existed."""
    monkeypatch.setattr(config, "get", lambda path, default=None: default)
    assert universe.configured_source() == "curated"


# ------------------------------------------- a backtest must not read the whole store
def _synth_prices(sym, index, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    px = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(index))))
    return pd.DataFrame({"date": index, "symbol": sym, "open": px, "high": px, "low": px,
                         "close": px, "adj_close": px, "volume": 1e6})


def test_backtest_ignores_symbols_outside_the_universe(tmp_path, monkeypatch, capsys):
    """A download nobody asked about must not move a curated backtest.

    `close_matrix(None)` handed the engine every symbol in the store, and the engine
    takes its rebalance CALENDAR from that frame's index (`_next_close_index`). A
    symbol that trades on a day the universe does not therefore becomes the fill day —
    on which every scored name is NaN — and the rebalance is silently dropped. Here the
    intruder trades every calendar day, which is the extreme version of the same thing.
    """
    from scripts import backtest as B
    from src import universe as U

    store = HistoryStore(tmp_path / "history")
    members = [f"M{i:02d}" for i in range(14)]
    bdays = pd.bdate_range("2018-01-01", periods=900)
    store.upsert_prices(pd.concat(
        [_synth_prices(s, bdays, i) for i, s in enumerate(members)]
        + [_synth_prices(b, bdays, 90 + i) for i, b in enumerate(("SPY", "SMH"))],
        ignore_index=True))
    monkeypatch.setattr(U, "configured_source", lambda: "curated")
    monkeypatch.setattr(U, "current_universe",
                        lambda **kw: [{"symbol": s, "subsector": "x"} for s in members])
    monkeypatch.setattr(U, "symbols", lambda: members)

    def _run():
        B.run_one("momentum", store, date(2019, 6, 1), 0.2, None, 15.0)
        # The rebalance calendar and every metric line — minus the two that legitimately
        # move between two runs: the report's own timestamp, and the deflated Sharpe
        # (which counts trials, and running the backtest twice IS two trials).
        lines = capsys.readouterr().out.splitlines()
        head = [ln.split("· ", 1)[-1] for ln in lines if "rebalances" in ln]
        rows = [ln for ln in lines if ln.startswith("- ") and "Deflated Sharpe" not in ln]
        return head + rows

    before = _run()
    assert any("rebalances" in ln for ln in before) and len(before) > 8
    store.upsert_prices(_synth_prices(
        "ZZZZ", pd.date_range("2016-01-01", periods=1600, freq="D"), 7))
    assert _run() == before
