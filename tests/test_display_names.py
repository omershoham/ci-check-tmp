"""Company names on the board: SEC's ticker file is the fallback for every name the
bounded `.info` enrichment never reaches (found by the first `scripts/demo` run, whose
board showed "NVDA" in the name column). No test here touches the network."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from src import fundamentals, news
from src.data.history_store import HistoryStore

_SEC = {"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
        "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "2": {"cik_str": 909832, "ticker": "COST", "title": "COSTCO WHOLESALE CORP /NEW"},
        "3": {"cik_str": 1045810, "ticker": "NVDA.X", "title": "NVIDIA CORP"}}


@pytest.fixture
def sec_file(workspace, monkeypatch):
    """SEC's ticker file served from memory; the per-process memos reset around it."""
    calls: list[int] = []
    monkeypatch.setattr(news, "_company_tickers", lambda: calls.append(1) or _SEC)
    news._ticker_name_map.cache_clear()
    news._ticker_cik_map.cache_clear()
    yield calls
    news._ticker_name_map.cache_clear()
    news._ticker_cik_map.cache_clear()


@pytest.mark.parametrize("title, shown", [
    ("NVIDIA CORP", "Nvidia Corp"),
    ("Apple Inc.", "Apple Inc."),                          # filer's own casing kept
    ("COSTCO WHOLESALE CORP /NEW", "Costco Wholesale Corp"),
    ("BANK OF AMERICA CORP /DE/", "Bank of America Corp"),
    ("MERCK & CO., INC.", "Merck & Co., Inc."),
    ("AT&T INC.", "AT&T Inc."),
    ("GE AEROSPACE", "GE Aerospace"),
    ("CVS HEALTH Corp", "CVS Health Corp"),
    ("PROCTER & GAMBLE Co", "Procter & Gamble Co"),       # half-cased by the filer
    ("ELI LILLY & Co", "Eli Lilly & Co"),
    ("THE COCA-COLA CO", "The Coca-Cola Co"),
])
def test_sec_titles_are_made_presentable(title, shown):
    assert news.display_name(title) == shown


def test_name_map_is_cached_daily_and_the_cik_map_keeps_its_contract(sec_file, workspace):
    names = news._ticker_name_map()
    assert names == {"NVDA": "Nvidia Corp", "AAPL": "Apple Inc.",
                     "COST": "Costco Wholesale Corp", "NVDA.X": "Nvidia Corp"}
    assert (workspace / "cache" / f"sec_ticker_names_{date.today()}.json").exists()
    assert news._ticker_cik_map()["NVDA"] == "0001045810"      # still ticker -> CIK
    assert sec_file == [1, 1]                  # each map fetched once, then memoised


def test_offline_the_newest_cached_names_are_used(workspace, monkeypatch):
    old = workspace / "cache" / f"sec_ticker_names_{date.today() - timedelta(days=3)}.json"
    old.write_text('{"NVDA": "Nvidia Corp"}', encoding="utf-8")
    monkeypatch.setattr(news, "_company_tickers", lambda: {})     # network down
    news._ticker_name_map.cache_clear()
    try:
        assert news._ticker_name_map() == {"NVDA": "Nvidia Corp"}
    finally:
        news._ticker_name_map.cache_clear()


def test_from_store_names_come_from_the_universe_then_sec_then_the_ticker(sec_file):
    got = fundamentals.from_store(HistoryStore(), ["NVDA", "COST", "ZZZZ"],
                                  names={"COST": "COSTCO WHOLESALE CORP /NEW"},
                                  extra={"NVDA": {"name": "NVIDIA Corporation"}})
    assert got["NVDA"]["name"] == "NVIDIA Corporation"       # Yahoo's wins when present
    assert got["COST"]["name"] == "Costco Wholesale Corp"    # universe row (registrants)
    assert got["ZZZZ"]["name"] == "ZZZZ"                     # unknown: the ticker
    assert fundamentals.display_names(["NVDA"]) == {"NVDA": "Nvidia Corp"}
