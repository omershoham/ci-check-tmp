"""Sector-stratified finalist selection (pipeline.select_finalists).

The cohort this function picks is what Claude researches AND what
`performance.claude_vs_screen` treats as its control group. Left as a plain top-N of a
1,500-name board it becomes whichever sector the composite currently favours, and the
"analyst vs screen" comparison quietly becomes "one sector vs the market".
"""
from __future__ import annotations

from src import pipeline


def _board(spec):
    """spec = [(symbol, sector)] in RANK ORDER (best first), the way score_universe
    hands the board over."""
    return [{"symbol": s, "sector": sec, "blended": 1.0 - i / 100.0}
            for i, (s, sec) in enumerate(spec)]


IT = "Information Technology"


def _all_it_top_ten():
    spec = [(f"IT{i}", IT) for i in range(12)]
    spec += [("HC0", "Health Care"), ("HC1", "Health Care"), ("HC2", "Health Care"),
             ("HC3", "Health Care"), ("FI0", "Financials"), ("FI1", "Financials"),
             ("EN0", "Energy"), ("UT0", "Utilities")]
    return _board(spec)


def test_cap_binds_when_the_top_of_the_board_is_one_sector():
    board = _all_it_top_ten()
    chosen = pipeline.select_finalists(board, 10, 3)
    assert len(chosen) == 10
    assert sum(1 for r in chosen if r["sector"] == IT) == 3
    # The three kept are the BEST three of that sector, not an arbitrary three.
    assert [r["symbol"] for r in chosen][:3] == ["IT0", "IT1", "IT2"]
    # ...and no other sector is over the cap either.
    for sec in {r["sector"] for r in chosen}:
        assert sum(1 for r in chosen if r["sector"] == sec) <= 3


def test_selection_is_deterministic_and_ranked():
    board = _all_it_top_ten()
    a = pipeline.select_finalists(board, 10, 3)
    b = pipeline.select_finalists(list(board), 10, 3)
    assert [r["symbol"] for r in a] == [r["symbol"] for r in b]
    # Output preserves board order (the greedy pass walks the ranking once).
    order = [r["symbol"] for r in board]
    assert [order.index(r["symbol"]) for r in a] == sorted(
        order.index(r["symbol"]) for r in a)


def test_cap_yields_rather_than_shrinking_the_research_budget():
    """A single-sector universe (the curated ai_infra list) has no sector comparison
    left to protect, so the cap must not silently cut 10 finalists down to 3."""
    board = _board([(f"IT{i}", IT) for i in range(20)])
    chosen = pipeline.select_finalists(board, 10, 3)
    assert [r["symbol"] for r in chosen] == [f"IT{i}" for i in range(10)]


def test_cap_of_zero_is_the_old_plain_top_n():
    board = _all_it_top_ten()
    assert pipeline.select_finalists(board, 10, 0) == board[:10]
    assert pipeline.select_finalists(board, 10, None) == board[:10]


def test_missing_sector_is_its_own_bucket_not_a_free_pass():
    board = _board([("A", None), ("B", ""), ("C", None), ("D", None), ("E", "Energy")])
    chosen = pipeline.select_finalists(board, 4, 3)
    assert len(chosen) == 4
    # A, B, C fill the Unknown bucket; D is over the cap, so Energy comes first.
    assert [r["symbol"] for r in chosen] == ["A", "B", "C", "E"]


def test_sectors_override_wins_over_the_row_field():
    board = _board([("A", IT), ("B", IT), ("C", IT), ("D", IT)])
    chosen = pipeline.select_finalists(board, 2, 1, sectors={"A": IT, "B": "Energy"})
    assert [r["symbol"] for r in chosen] == ["A", "B"]


def test_sector_mix_reports_the_cohort():
    chosen = pipeline.select_finalists(_all_it_top_ten(), 10, 3)
    mix = pipeline.sector_mix(chosen)
    assert mix.startswith("Health Care 3") or mix.startswith("Information Technology 3")
    assert "Information Technology 3" in mix
