"""SIC -> sector/industry_group.

Every case here is one where a 2-digit `sic // 100` mapping gets it wrong. If these
pass, the peer sets the broad universe ranks inside are the ones a human would draw.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data import sic_sectors as S


@pytest.mark.parametrize("sic,sector,group,why", [
    # The refinements. Each is a company filed under a major group that means
    # something else entirely.
    (2834, S.HEALTH, "Pharmaceuticals, Biotechnology & Life Sciences", "PFE, in 'Chemicals'"),
    (2836, S.HEALTH, "Pharmaceuticals, Biotechnology & Life Sciences", "biologicals"),
    (3571, S.TECH, "Technology Hardware & Equipment", "AAPL, in 'Industrial machinery'"),
    (3576, S.TECH, "Technology Hardware & Equipment", "ANET, networking"),
    (3559, S.TECH, "Semiconductors & Semiconductor Equipment", "AMAT, semi equipment"),
    (3674, S.TECH, "Semiconductors & Semiconductor Equipment", "NVDA"),
    (3841, S.HEALTH, "Health Care Equipment & Services", "surgical instruments"),
    (3851, S.HEALTH, "Health Care Equipment & Services", "ophthalmic goods"),
    (3812, S.TECH, "Technology Hardware & Equipment", "search/detection electronics"),
    (3826, S.TECH, "Technology Hardware & Equipment", "lab analytical instruments"),
    (3827, S.TECH, "Technology Hardware & Equipment", "optical instruments"),
    (6021, S.FINANCIALS, "Banks", "JPM"),
    (6199, S.FINANCIALS, "Financial Services", "top of the 6020-6199 finance block"),
    (6798, S.REAL_ESTATE, "Equity REITs", "AMT — NOT a fund, unlike its 6726 neighbour"),
    (6512, S.REAL_ESTATE, "Real Estate Management & Development", "65xx operators"),
    (7372, S.TECH, "Software & Services", "MSFT"),
    (7370, S.TECH, "Software & Services", "GOOGL/META"),
    (7310, S.COMMS, "Media & Entertainment", "advertising"),
    (7375, S.COMMS, "Media & Entertainment", "information retrieval"),
    (4911, S.UTILITIES, "Utilities", "NEE"),
    (4931, S.UTILITIES, "Utilities", "combination utilities"),
    (1311, S.ENERGY, "Energy", "crude petroleum & natural gas"),
    (1040, S.MATERIALS, "Materials", "10xx metal mining"),
    (3711, S.DISCRETIONARY, "Automobiles & Components", "TSLA/F/GM"),
    (5331, S.STAPLES, "Consumer Staples Distribution & Retail",
     "variety stores — WMT/TGT/COST, Staples since GICS 2023"),
    (5411, S.STAPLES, "Consumer Staples Distribution & Retail", "grocery stores"),
    (5912, S.STAPLES, "Consumer Staples Distribution & Retail", "591x drug stores"),
    (5211, S.DISCRETIONARY, "Consumer Discretionary Distribution & Retail", "rest of 52-59"),
    (5961, S.DISCRETIONARY, "Consumer Discretionary Distribution & Retail", "AMZN"),
    (6324, S.HEALTH, "Health Care Equipment & Services", "UNH — managed care, not insurance"),
    (9995, S.UNKNOWN, "Unknown", "99xx non-classifiable"),
])
def test_known_codes(sic, sector, group, why):
    assert S.classify(sic) == (sector, group), why


def test_neighbours_do_not_leak():
    """The refinements must carve out EXACTLY their range, not the major group."""
    assert S.sector(2832) == S.MATERIALS      # just below the pharma window
    assert S.sector(2837) == S.MATERIALS      # just above it
    assert S.sector(3569) == S.INDUSTRIALS    # just below computer equipment
    assert S.sector(3580) == S.INDUSTRIALS    # just above it
    assert S.sector(6726) == S.FINANCIALS     # mapped, then EXCLUDED by the universe
    assert S.sector(6770) == S.FINANCIALS


def test_missing_and_junk_are_unknown():
    for bad in (None, "", "   ", float("nan"), "n/a"):
        assert S.classify(bad) == (S.UNKNOWN, "Unknown")


def test_string_and_float_codes_agree():
    """sub.txt gives strings, parquet round-trips floats; both must land in one bucket."""
    assert S.classify("3674") == S.classify(3674) == S.classify(3674.0)


def test_every_group_belongs_to_its_sector():
    """No industry group may appear under two sectors — the two levels are one table."""
    owner: dict[str, str] = {}
    for pair in list(S._EXACT.values()) + [g for _, _, g in S._RANGES]:
        sector, group = pair
        assert owner.setdefault(group, sector) == sector, group
    assert set(S.SECTORS) | {S.UNKNOWN} >= set(owner.values())


def test_classify_frame_matches_scalar():
    codes = pd.Series(["3674", "2834", "6798", None, "9999"])
    out = S.classify_frame(codes)
    assert list(out["sector"]) == [S.classify(c)[0] for c in codes]
    assert list(out["industry_group"]) == [S.classify(c)[1] for c in codes]


def test_sector_overrides_are_name_level_and_opt_in():
    """SIC 7370 cannot separate infrastructure software from the ad-funded platforms
    GICS moved to Communication Services in 2018, so those names are corrected by
    ticker — and ONLY when the caller passes symbols, so a SIC-pure classification
    stays available."""
    assert S.classify(7370) == (S.TECH, "Software & Services")     # the code alone
    sic = pd.Series(["7370", "7370", "7372"], index=[0, 1, 2])
    syms = pd.Series(["GOOGL", "META", "MSFT"], index=[0, 1, 2])
    plain = S.classify_frame(sic)
    assert list(plain["sector"]) == [S.TECH] * 3                   # no symbols -> SIC-pure
    fixed = S.classify_frame(sic, symbols=syms)
    assert list(fixed["sector"]) == [S.COMMS, S.COMMS, S.TECH]
    assert list(fixed["industry_group"]) == ["Media & Entertainment",
                                             "Media & Entertainment", "Software & Services"]
    # An override may only move a name to a real sector/group pair.
    for sector, group in S.SECTOR_OVERRIDES.values():
        assert sector in S.SECTORS and group in S.INDUSTRY_GROUPS
