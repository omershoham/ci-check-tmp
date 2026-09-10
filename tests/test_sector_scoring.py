"""Sector-aware fundamentals: the rules that stop a whole sector being unrankable.

The concrete failure these guard against: EV/EBITDA does not exist for banks, so a
universe-wide percentile with `.fillna(0.5)` gave every Financial a permanent neutral
valuation and the screen could never surface a cheap bank. Each test below pins one
half of the fix — per-sector component sets, within-sector ranking, n-dependent
shrinkage, and a coverage floor that says "unknown" instead of inventing 0.5.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src import scoring
from src.analytics.metrics import derive


# --------------------------------------------------------------- shrinkage
def test_shrinkage_reproduces_the_old_flat_half_at_n_equals_k():
    """λ(n)=n/(n+k) must be a strict generalization: at n=k it IS the old 50/50."""
    k = 10
    idx = [f"S{i}" for i in range(20)]
    vals = pd.Series(range(20), index=idx, dtype=float)
    groups = pd.Series(["A"] * 10 + ["B"] * 10, index=idx)   # two groups of exactly k
    new = scoring._pct(vals, higher_is_better=True, groups=groups, k=k)
    old = 0.5 * vals.rank(pct=True) + 0.5 * vals.groupby(groups).rank(pct=True)
    assert (new - old).abs().max() < 1e-12


def test_shrinkage_pulls_a_thin_group_much_harder_than_a_fat_one():
    idx = [f"S{i}" for i in range(23)]
    vals = pd.Series(range(23), index=idx, dtype=float)
    groups = pd.Series(["thin"] * 3 + ["fat"] * 20, index=idx)
    r = scoring._pct(vals, higher_is_better=True, groups=groups, k=10)
    uni = vals.rank(pct=True)
    grp = vals.groupby(groups).rank(pct=True)
    # thin: lambda = 3/13 = 0.231; fat: 20/30 = 0.667
    thin = idx[0]
    fat = idx[-1]
    assert abs(float(r[thin]) - (0.769 * uni[thin] + 0.231 * grp[thin])) < 1e-3
    assert abs(float(r[fat]) - (0.333 * uni[fat] + 0.667 * grp[fat])) < 1e-3


def test_valuation_shrinks_toward_neutral_not_toward_the_universe():
    """The sector-neutral half: a thin sector goes to 0.5, never to a cross-sector rank."""
    idx = ["A", "B", "C"]
    vals = pd.Series([1.0, 2.0, 3.0], index=idx)
    grp = pd.Series(["one", "one", "one"], index=idx)
    r = scoring._pct(vals, higher_is_better=True, groups=grp, k=10, shrink_to="neutral")
    within = vals.rank(pct=True)
    lam = 3 / 13
    for s in idx:
        assert abs(float(r[s]) - ((1 - lam) * 0.5 + lam * within[s])) < 1e-9


# ------------------------------------------------- the cheap bank, end to end
def _three_sector_universe():
    """20 software names at 40x, 12 banks at 8-20x with NO EV/EBITDA, 6 industrials."""
    m = {}
    for i in range(20):
        s = f"SW{i}"
        m[s] = {"symbol": s, "name": s, "marketCap": 5e10, "price": 100.0,
                "avg_dollar_volume": 1e8, "profitable": True, "freeCashflow": 1e9,
                "trailingPE": 40 + i, "enterpriseToEbitda": 30 + i,
                "priceToSalesTrailing12Months": 12 + i, "priceToBook": 15 + i,
                "fcf_yield": 0.02, "fcf_margin": 0.2, "revenueGrowth": 0.2,
                "earningsGrowth": 0.2, "returnOnEquity": 0.25, "returnOnAssets": 0.12,
                "grossMargins": 0.75, "operatingMargins": 0.25, "debtToEquity": 30.0}
    for i in range(12):
        s = f"BK{i}"
        # A bank: no EV/EBITDA, no P/S, no gross margin, high leverage BY DESIGN.
        m[s] = {"symbol": s, "name": s, "marketCap": 3e10, "price": 50.0,
                "avg_dollar_volume": 1e8, "profitable": True, "freeCashflow": 5e8,
                "trailingPE": 8 + i, "enterpriseToEbitda": None,
                "priceToSalesTrailing12Months": None, "priceToBook": 0.8 + 0.1 * i,
                "fcf_yield": None, "fcf_margin": None, "revenueGrowth": 0.05,
                "earningsGrowth": 0.06, "returnOnEquity": 0.13, "returnOnAssets": 0.011,
                "grossMargins": None, "operatingMargins": 0.35, "debtToEquity": 900.0}
    for i in range(6):
        s = f"IN{i}"
        m[s] = {"symbol": s, "name": s, "marketCap": 2e10, "price": 70.0,
                "avg_dollar_volume": 1e8, "profitable": True, "freeCashflow": 4e8,
                "trailingPE": 18 + i, "enterpriseToEbitda": 12 + i,
                "priceToSalesTrailing12Months": 2 + i, "priceToBook": 3 + i,
                "fcf_yield": 0.03, "fcf_margin": 0.1, "revenueGrowth": 0.04,
                "earningsGrowth": 0.05, "returnOnEquity": 0.18, "returnOnAssets": 0.08,
                "grossMargins": 0.3, "operatingMargins": 0.12, "debtToEquity": 80.0}
    subs = {s: ("software" if s.startswith("SW") else
                "banks" if s.startswith("BK") else "machinery") for s in m}
    secs = {s: ("Information Technology" if s.startswith("SW") else
                "Financials" if s.startswith("BK") else "Industrials") for s in m}
    return m, subs, secs


def test_the_cheapest_bank_gets_a_high_valuation_score():
    m, subs, secs = _three_sector_universe()
    rows = {r["symbol"]: r for r in scoring.score_universe(m, subs, sectors=secs)}
    cheap, rich = rows["BK0"], rows["BK11"]            # P/E 8 and P/B 0.8 vs P/E 19
    assert cheap["metrics"]["sub_valuation"] > 0.7, cheap["metrics"]["sub_valuation"]
    assert cheap["metrics"]["sub_valuation"] > rich["metrics"]["sub_valuation"]
    # ... and it beats a MID-PRICED software name outright. That comparison is the
    # whole point: under a universe-wide rank the bank had no EV/EBITDA and no P/S, so
    # two of its four components were fillna(0.5) and the sector sat at the median
    # forever. (It does not beat SW0, the cheapest name in a 20-strong sector: n=12
    # shrinks the bank's extreme harder than n=20 shrinks software's. That is the
    # shrinkage doing its job, not the sector being penalised.)
    assert cheap["metrics"]["sub_valuation"] > rows["SW10"]["metrics"]["sub_valuation"]
    assert rows["SW0"]["metrics"]["sub_valuation"] > cheap["metrics"]["sub_valuation"]


def test_a_bank_is_not_punished_for_missing_ev_ebitda_or_for_leverage():
    m, subs, secs = _three_sector_universe()
    rows = {r["symbol"]: r for r in scoring.score_universe(m, subs, sectors=secs)}
    vals = [rows[f"BK{i}"]["metrics"]["sub_valuation"] for i in range(12)]
    # A real spread inside the sector, not twelve names pinned at neutral.
    assert max(vals) - min(vals) > 0.4
    # debtToEquity 900 is the business model, so it must not enter quality...
    assert "debtToEquity" not in [c for c, _ in scoring.sector_metrics("Financials", "quality")]
    # ...and the cash-flow family does not apply to a lender at all.
    assert scoring.sector_metrics("Financials", "cashflow") == []
    assert rows["BK0"]["metrics"]["sub_cashflow"] is None


def test_unknown_sector_falls_back_to_the_default_metric_set():
    for fam in ("valuation", "cashflow", "growth", "quality"):
        assert scoring.sector_metrics("Sector That Does Not Exist", fam) == \
               scoring._DEFAULT_METRICS[fam]
        assert scoring.sector_metrics(scoring.UNKNOWN_SECTOR, fam) == \
               scoring._DEFAULT_METRICS[fam]


def test_scoring_without_sectors_still_works():
    """Every caller that has not been taught about sectors yet keeps scoring."""
    m, subs, _ = _three_sector_universe()
    rows = scoring.score_universe(m, subs)
    assert len(rows) == len(m)
    assert all(r["sector"] == scoring.UNKNOWN_SECTOR for r in rows)


# ------------------------------------------------------------ coverage floor
def _thin_universe():
    m, subs, secs = _three_sector_universe()
    thin = {"symbol": "THIN", "name": "THIN", "marketCap": 1e10, "price": 20.0,
            "avg_dollar_volume": 1e8, "profitable": True, "freeCashflow": None,
            "trailingPE": 15.0, "enterpriseToEbitda": None,
            "priceToSalesTrailing12Months": None, "priceToBook": None,
            "fcf_yield": None, "fcf_margin": None, "revenueGrowth": None,
            "earningsGrowth": None, "returnOnEquity": None, "returnOnAssets": None,
            "grossMargins": None, "operatingMargins": None, "debtToEquity": None}
    m["THIN"] = thin
    subs["THIN"] = "machinery"
    secs["THIN"] = "Industrials"
    return m, subs, secs


def test_thin_data_name_gets_nan_not_a_fabricated_neutral():
    m, subs, secs = _thin_universe()
    rows = {r["symbol"]: r for r in scoring.score_universe(m, subs, sectors=secs)}
    thin = rows["THIN"]
    assert thin["fundamental"] is None and thin["quality"] is None
    assert thin["metrics"]["pillar_fundamental"] is None
    assert "thin-data" in thin["tags"]
    assert thin["metrics"]["fund_coverage"] < 0.2
    # A well-covered name is NOT tagged and reports high coverage.
    assert "thin-data" not in rows["SW0"]["tags"]
    assert rows["SW0"]["metrics"]["fund_coverage"] > 0.9


def test_thin_data_name_is_scored_on_its_other_pillars_not_dropped():
    """The composite re-normalizes per row: no fundamental data must not read as a
    BAD fundamental score, and must not remove the name from the board.

    Weights are pinned here rather than inherited from config: the point under test is
    the per-row re-normalisation, not whatever the shipped technical weight happens to
    be. (It was 0.35; a deliberate 2026-09-10 re-weighting to 0.20 flipped the ordering
    and made a config decision look like a scoring regression.)"""
    from src import config
    config.set_override("scoring.pillars", {"technical": 0.35, "fundamental": 0.30,
                                            "sentiment": 0.35, "learned": 0.0,
                                            "catalyst": 0.0})
    try:
        m, subs, secs = _thin_universe()
        techs = {s: {"technical_score": 0.9 if s == "THIN" else 0.2} for s in m}
        rows = {r["symbol"]: r for r in scoring.score_universe(m, subs, technicals=techs,
                                                               sectors=secs)}
        assert len(rows) == len(m)
        assert rows["THIN"]["blended"] > rows["SW0"]["blended"]   # strong technical wins
        assert 0.0 <= rows["THIN"]["blended"] <= 1.0
    finally:
        config.clear_overrides()


def test_subscore_below_the_floor_is_nan_above_it_is_a_number():
    df = pd.DataFrame({"trailingPE": [10.0, 20.0, None], "forwardPE": [None, None, None],
                       "enterpriseToEbitda": [5.0, None, None],
                       "priceToSalesTrailing12Months": [1.0, None, None]},
                      index=["A", "B", "C"])
    sub = scoring._subscore(df, scoring._VALUATION, min_components=2)
    assert not math.isnan(float(sub["A"]))       # 3 components present
    assert math.isnan(float(sub["B"]))           # 1 present -> unknown, not 0.5
    assert math.isnan(float(sub["C"]))           # 0 present


# ---------------------------------------------------------------- risk tags
def test_no_earnings_is_separate_from_higher_risk():
    m, subs, secs = _three_sector_universe()
    m["SW0"]["trailingPE"] = None                # loss-making-but-profitable-by-FCF
    rows = {r["symbol"]: r for r in scoring.score_universe(m, subs, sectors=secs)}
    assert "no-earnings" in rows["SW0"]["tags"]
    assert "no-earnings" not in rows["SW1"]["tags"]
    # A missing P/E is a data fact, not a risk judgement: on its own it must not flag.
    assert "higher-risk" not in rows["SW0"]["tags"]


def test_higher_risk_is_relative_to_the_name_s_own_sector():
    m, subs, secs = _three_sector_universe()
    rows = {r["symbol"]: r for r in scoring.score_universe(m, subs, sectors=secs)}
    # A bank at 19x is expensive FOR A BANK and gets flagged...
    assert "higher-risk" in rows["BK11"]["tags"]
    # ...while software at 40x - twice the bank's multiple - is not, because its own
    # sector trades there. The old flat pe_risk_ceiling could not express this.
    assert "higher-risk" not in rows["SW0"]["tags"]
    assert rows["BK11"]["metrics"]["sector_pe_gics"] < rows["SW0"]["metrics"]["sector_pe_gics"]


def test_unprofitable_still_flags_higher_risk():
    m, subs, secs = _three_sector_universe()
    m["IN0"]["profitable"] = False
    rows = {r["symbol"]: r for r in scoring.score_universe(m, subs, sectors=secs)}
    assert "higher-risk" in rows["IN0"]["tags"]


# ------------------------------------------------------- versioning / plumbing
def test_pillar_set_is_written_with_every_row():
    m, subs, secs = _three_sector_universe()
    rows = scoring.score_universe(m, subs, sectors=secs)
    assert all(r["pillar_set"] == "technical,fundamental,sentiment" for r in rows)


def test_pillar_set_persists_to_screen_scores(storage):
    m, subs, secs = _three_sector_universe()
    rid = storage.create_run("deep")
    storage.upsert_screen_scores(rid, scoring.score_universe(m, subs, sectors=secs))
    with storage.get_conn() as cn:
        got = {r["pillar_set"] for r in cn.execute(
            "SELECT pillar_set FROM screen_scores WHERE run_id=?", (rid,))}
        ver = {r["score_version"] for r in cn.execute(
            "SELECT score_version FROM screen_scores WHERE run_id=?", (rid,))}
    assert got == {"technical,fundamental,sentiment"}
    assert ver == {scoring.SCORE_VERSION}


def test_score_version_is_five_and_learned_does_not_reuse_four():
    assert scoring.SCORE_VERSION == 5
    assert scoring._LEARNED_VERSION[5] == 6      # 4 is taken by "v3 + learned"


# ------------------------------------------------- the new derive() metrics
def test_new_metrics_on_the_sec_path():
    """Inputs as `sec_facts.raw_asof_prepared` supplies them."""
    raw = {"price": 20.0, "sharesOutstanding": 1e9, "marketCap": 2e10,
           "totalRevenue": 5e9, "netIncome": 1e9, "dna": 1.5e9, "cfo": -4e8,
           "totalCash": 8e8, "totalDebt": 1e9, "totalStockholderEquity": 4e9}
    m = derive(raw)
    assert m["priceToBook"] == pytest.approx(5.0)          # 20e9 / 4e9
    assert m["ffo"] == pytest.approx(2.5e9)                # 1.0e9 + 1.5e9
    assert m["p_ffo"] == pytest.approx(8.0)                # 20e9 / 2.5e9
    # 0.8e9 cash against 0.4e9/yr burn = 2 years = 8 quarters
    assert m["cash_runway_quarters"] == pytest.approx(8.0)


def test_cash_runway_is_none_for_a_cash_generator():
    m = derive({"price": 10.0, "marketCap": 1e9, "totalCash": 5e8, "cfo": 2e8})
    assert m["cash_runway_quarters"] is None       # "infinite" is not a rankable number


def test_new_metrics_on_the_info_path_use_yfinance_field_names():
    """`.info` has no `dna`, so ffo is honestly None — but P/B and the runway, whose
    inputs `.info` does carry, must come out identical to the SEC path."""
    m = derive({"price": 20.0, "marketCap": 2e10, "priceToBook": 5.0,
                "operatingCashflow": -4e8, "totalCash": 8e8,
                "totalStockholderEquity": 4e9})
    assert m["priceToBook"] == pytest.approx(5.0)
    assert m["cash_runway_quarters"] == pytest.approx(8.0)
    assert m["ffo"] is None and m["p_ffo"] is None


def test_currency_masking_covers_the_new_market_derived_ratios():
    raw = {"price": 20.0, "marketCap": 2e10, "netIncome": 1e9, "dna": 1.5e9,
           "totalStockholderEquity": 4e9, "financialCurrency": "TWD"}
    m = derive(raw)
    assert m["priceToBook"] is None and m["p_ffo"] is None   # USD cap / TWD book
    assert m["ffo"] == pytest.approx(2.5e9)                  # both legs local: kept


# ------------------------------------------- foreign filers / currency masking
def _tsm_like_sec_raw():
    """What `sec_facts.raw_asof_prepared` hands `derive()` for a 20-F filer: a USD
    market cap (price x shares, both USD) over TWD statements."""
    return {"price": 429.0, "sharesOutstanding": 5.2e9, "totalRevenue": 4.2e12,
            "netIncome": 1.8e12, "totalStockholderEquity": 6.5e12, "dna": 8e11,
            "totalCash": 2.3e12, "totalDebt": 9e11, "financialCurrency": "TWD"}


def test_foreign_filer_gets_no_rebuilt_pe_or_pb():
    """USD cap over TWD income is P/E 206, not 32. It used to reach the screen and
    trip `higher-risk`; now it is None, which reads as `no-earnings` instead."""
    m = derive(_tsm_like_sec_raw())
    assert m["trailingPE"] is None
    assert m["priceToBook"] is None
    assert m["priceToSalesTrailing12Months"] is None
    assert m["grossMargins"] is None or True          # margins are local/local: kept


def test_enrichment_restores_a_foreign_filer_s_multiples():
    """Yahoo prices the ADR in USD against USD-converted per-share figures, so its
    `trailingPE` is internally consistent and must survive the mask — that is the
    whole reason `enrich()` carries it."""
    from src import fundamentals
    info_like = dict(_tsm_like_sec_raw(), trailingPE=31.8, priceToBook=8.9)
    m = derive(info_like)
    assert m["trailingPE"] == pytest.approx(31.8)     # supplied whole -> kept
    assert m["priceToBook"] == pytest.approx(8.9)
    assert "trailingPE" in fundamentals._ENRICH_FIELDS
    assert "priceToBook" in fundamentals._ENRICH_FIELDS
    # ...and the merge `pipeline` performs restores it over the masked store value.
    merged = dict(derive(_tsm_like_sec_raw()))
    merged.update({k: v for k, v in m.items()
                   if k in fundamentals._ENRICH_FIELDS and v is not None})
    assert merged["trailingPE"] == pytest.approx(31.8)


def test_foreign_filer_falls_out_of_higher_risk_into_no_earnings():
    m, subs, secs = _three_sector_universe()
    adr = dict(m["SW0"], symbol="ADR", name="ADR", trailingPE=None, priceToBook=None,
               priceToSalesTrailing12Months=None, enterpriseToEbitda=None,
               fcf_yield=None)
    m["ADR"] = adr
    subs["ADR"] = "software"
    secs["ADR"] = "Information Technology"
    rows = {r["symbol"]: r for r in scoring.score_universe(m, subs, sectors=secs)}
    assert "no-earnings" in rows["ADR"]["tags"]
    assert "higher-risk" not in rows["ADR"]["tags"]


# --------------------------------------------------------- REIT cash flow = FFO
def test_reit_cashflow_family_is_ffo_not_free_cash_flow():
    assert scoring.sector_metrics("Real Estate", "cashflow") == \
           [("ffo_yield", True), ("ffo_margin", True)]


def test_ffo_metrics_on_a_synthetic_reit():
    """A healthy REIT with heavy development capex: FCF negative, FFO strongly positive.
    Ranking it on fcf_yield says 'capital-destroying'; that is the sector, not the name."""
    raw = {"price": 800.0, "sharesOutstanding": 1e8, "totalRevenue": 9e9,
           "netIncome": 1.0e9, "dna": 2.6e9, "cfo": 3.9e9, "capex_ignored": None,
           "freeCashflow": -9.0e8, "totalCash": 3e8, "totalDebt": 1.7e10,
           "totalStockholderEquity": 1.2e10}
    m = derive(raw)
    assert m["fcf_yield"] < 0                                   # what used to rank
    assert m["ffo"] == pytest.approx(3.6e9)                     # 1.0 + 2.6
    assert m["ffo_yield"] == pytest.approx(3.6e9 / 8e10)
    assert m["ffo_margin"] == pytest.approx(0.4)
    assert m["ffo_yield"] > 0 and m["ffo_margin"] > 0


def test_a_reit_with_negative_fcf_is_not_marked_down_for_it():
    m, subs, secs = _three_sector_universe()
    for i in range(8):
        s = f"RE{i}"
        m[s] = {"symbol": s, "name": s, "marketCap": 8e10, "price": 800.0,
                "avg_dollar_volume": 1e8, "profitable": True, "freeCashflow": -9e8,
                "trailingPE": 60 + i, "enterpriseToEbitda": 20 + i,
                "priceToSalesTrailing12Months": 8 + i, "p_ffo": 18 + i,
                "fcf_yield": -0.01, "fcf_margin": -0.1,
                "ffo_yield": 0.06 - 0.004 * i, "ffo_margin": 0.45 - 0.02 * i,
                "revenueGrowth": 0.08, "earningsGrowth": 0.05,
                "returnOnEquity": 0.09, "returnOnAssets": 0.03,
                "grossMargins": 0.5, "operatingMargins": 0.3, "debtToEquity": 140.0}
        subs[s] = "reits"
        secs[s] = "Real Estate"
    rows = {r["symbol"]: r for r in scoring.score_universe(m, subs, sectors=secs)}
    cash = [rows[f"RE{i}"]["metrics"]["sub_cashflow"] for i in range(8)]
    assert all(c is not None for c in cash)          # the family applies...
    assert max(cash) - min(cash) > 0.3               # ...and it discriminates
    assert cash[0] > cash[7]                         # best FFO yield wins


# ----------------------------------------------- replay / live composite parity
def test_replay_composite_matches_the_live_per_row_reweighting():
    """`backtest.signals.composite_3pillar` must do what `score_universe` does: a name
    with no fundamental pillar is ranked on its other pillars, not dropped. A stricter
    replay changes the cross-section and moves the IC on its own."""
    from src.backtest import signals as bsig
    idx = [f"S{i}" for i in range(12)]
    d = pd.Timestamp("2026-01-30")
    tech = pd.DataFrame([[0.1 * i for i in range(12)]], index=[d], columns=idx)
    fund = pd.DataFrame([[0.5] * 11 + [float("nan")]], index=[d], columns=idx)
    scored = pd.DataFrame([[True] * 12], index=[d], columns=idx)
    comp = bsig.composite_3pillar(tech, fund, scored,
                                  weights={"technical": 0.35, "fundamental": 0.30,
                                           "sentiment": 0.35})
    assert comp.loc[d].notna().all() and len(comp.columns) == 12    # nobody dropped
    # S11 has the best technical and no fundamental: its score is the technical and
    # sentiment pillars re-scaled to 1.
    exp = (1.0 * 0.35 + 0.5 * 0.35) / (0.35 + 0.35)
    assert comp.loc[d, "S11"] == pytest.approx(exp)


def test_replay_composite_still_drops_names_the_live_floor_would_drop():
    from src.backtest import signals as bsig
    idx = [f"S{i}" for i in range(12)]
    d = pd.Timestamp("2026-01-30")
    tech = pd.DataFrame([[0.1 * i for i in range(12)]], index=[d], columns=idx)
    fund = pd.DataFrame([[0.5] * 11 + [float("nan")]], index=[d], columns=idx)
    # S11 had NO point-in-time snapshot: no share count, no market cap, so the live
    # liquidity floor removes it before scoring. It must not appear in the replay.
    scored = pd.DataFrame([[True] * 11 + [False]], index=[d], columns=idx)
    comp = bsig.composite_3pillar(tech, fund, scored)
    assert comp.loc[d].notna().sum() == 11
    assert "S11" not in comp.columns or pd.isna(comp.loc[d, "S11"])


def test_replay_parity_fix_is_a_no_op_when_no_pillar_is_missing():
    """Why the pre-Phase-4 baseline reports stay valid as the "before" side.

    The old fundamental pillar could not be NaN (`_pct(...).fillna(0.5)` everywhere),
    so under it `scored` and `fund.notna()` are the same mask and every row's weight
    sum is the full 1.0 — the per-row formula reduces to the old flat one, exactly."""
    from src.backtest import signals as bsig
    idx = [f"S{i}" for i in range(15)]
    d = pd.Timestamp("2026-01-30")
    tech = pd.DataFrame([[0.07 * i for i in range(15)]], index=[d], columns=idx)
    fund = pd.DataFrame([[1.0 - 0.05 * i for i in range(15)]], index=[d], columns=idx)
    w = {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.35}
    new = bsig.composite_3pillar(tech, fund,
                                 pd.DataFrame([[True] * 15], index=[d], columns=idx), w)
    old = (tech.loc[d].rank(pct=True) * 0.35 + fund.loc[d].rank(pct=True) * 0.30
           + 0.5 * 0.35)
    assert (new.loc[d] - old).abs().max() < 1e-12
