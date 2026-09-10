"""The metric-name contract between `analytics.metrics.derive()` and `scoring`.

`scoring`'s component lists are plain strings and `_component_percentiles` skips a
column the frame does not carry. That is right for a company that lacks a value and
silently fatal for a TYPO: a misspelt component would leave the fundamental pillar on
every name, in every run, forever, while `coverage` — measured against the surviving
list — still read healthy. Nothing would go red.

These tests are what goes red. They pin, in order:
  * the vocabulary declaration (`DERIVED_FIELDS`) to what `derive` actually emits,
  * the probe raw dict to every field `derive` actually reads, so the contract cannot
    pass by feeding `derive` nothing,
  * every `scoring` component list, `_MASK_NONPOS` and `_CCY_SENSITIVE` to that
    vocabulary,
  * that each designed metric is genuinely COMPUTABLE from the SEC path's raw inputs
    (a name in the vocabulary that nothing can ever produce is the same bug wearing a
    different hat), and
  * that a typo raises rather than being skipped.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src import scoring
from src.analytics import metrics as M
from src.analytics.metrics import DERIVED_FIELDS, RAW_FIELDS, derive

# `.info` supplies the price under three names; only `price` is in RAW_FIELDS because
# that is what the SEC builder writes. The other two are live-path aliases.
_LIVE_ALIASES = ("currentPrice", "regularMarketPrice")

# Exactly the keys `sec_facts.raw_asof_prepared` returns. Values are a plausible large
# profitable filer; this is the historical path's whole input surface, so any designed
# metric that cannot be built from THIS dict is not really live on the SEC path.
SEC_RAW: dict[str, object] = {
    "price": 100.0,
    "sharesOutstanding": 1.0e9,
    "marketCap": 1.0e11,
    "totalRevenue": 5.0e10,
    "netIncome": 8.0e9,
    "grossProfits": 3.0e10,
    "operatingIncome": 1.0e10,
    "freeCashflow": 7.0e9,
    "ebitda": 1.2e10,
    "totalAssets": 1.5e11,
    "totalStockholderEquity": 6.0e10,
    "totalCash": 2.0e10,
    "totalDebt": 1.0e10,
    "cfo": 9.0e9,
    "dna": 3.0e9,
    "revenueGrowth": 0.15,
    "earningsGrowth": 0.22,
    "fiftyDayAverage": 95.0,
    "twoHundredDayAverage": 90.0,
    "financialCurrency": "USD",
}

# The live path's surface: every raw field `derive` is documented to read.
INFO_RAW: dict[str, object] = {
    **{k: 1.0 for k in RAW_FIELDS},
    **{k: 100.0 for k in _LIVE_ALIASES},
    **SEC_RAW,
    "trailingEps": 8.0,
    "operatingCashflow": 9.0e9,
    "trailingPE": 12.5,
    "forwardPE": 11.0,
    "pegRatio": 1.4,
    "enterpriseToEbitda": 7.5,
    "enterpriseToRevenue": 1.8,
    "priceToSalesTrailing12Months": 2.0,
    "priceToBook": 1.7,
    "grossMargins": 0.6,
    "operatingMargins": 0.2,
    "returnOnEquity": 0.13,
    "returnOnAssets": 0.05,
    "debtToEquity": 16.7,
    "earningsTimestamp": 1_800_000_000,
    "earningsTimestampStart": 1_800_000_000,
}

# Every column any scoring component list can reference, deduped.
def _all_designed_columns() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {
        "_VALUATION": [c for c, _ in scoring._VALUATION],
        "_CASHFLOW": [c for c, _ in scoring._CASHFLOW],
        "_GROWTH": [c for c, _ in scoring._GROWTH],
        "_QUALITY": [c for c, _ in scoring._QUALITY],
    }
    for sector, fams in scoring.SECTOR_METRICS.items():
        for fam, comps in fams.items():
            out[f'SECTOR_METRICS["{sector}"]["{fam}"]'] = [c for c, _ in comps]
    return out


# ------------------------------------------- the probe covers everything derive reads
def test_the_probe_raw_dict_supplies_every_field_derive_reads():
    """Without this, the whole contract could pass on an empty dict.

    `derive` returns its full key set whatever it is fed (missing input -> None value),
    so a probe that omitted half the raw fields would still satisfy the key-set
    assertions below while proving nothing about computability.
    """
    src = Path(M.__file__).read_text(encoding="utf-8")
    read = set(re.findall(r'raw\.get\(\s*"([A-Za-z0-9_]+)"', src))
    assert read, "could not find any raw.get(...) reads — did derive() change shape?"
    missing = sorted(read - set(INFO_RAW))
    assert not missing, f"probe raw dict does not supply: {missing}"


def test_declared_vocabulary_is_exactly_what_derive_emits():
    """`DERIVED_FIELDS` is the contract everything else is checked against, so it must
    not be allowed to drift from the function it describes."""
    emitted = set(derive(INFO_RAW))
    declared = set(DERIVED_FIELDS)
    assert emitted - declared == set(), f"emitted but undeclared: {sorted(emitted - declared)}"
    assert declared - emitted == set(), f"declared but never emitted: {sorted(declared - emitted)}"
    assert len(DERIVED_FIELDS) == len(declared), "DERIVED_FIELDS has duplicates"


def test_sec_and_live_paths_emit_the_same_vocabulary():
    """Live/historical parity: one `derive`, one key set. Only the earnings calendar
    differs, because an SEC snapshot has no next-earnings timestamp."""
    live = set(derive(INFO_RAW))
    hist = set(derive(SEC_RAW))
    assert live - hist == {"earnings_date", "days_to_earnings"}
    assert hist - live == set()


# ----------------------------------------------------------------- the contract
@pytest.mark.parametrize("list_name", sorted(_all_designed_columns()))
def test_every_scoring_component_is_a_metric_derive_emits(list_name):
    cols = _all_designed_columns()[list_name]
    dead = [c for c in cols if c not in set(DERIVED_FIELDS)]
    assert not dead, (f"{list_name} references metrics derive() never emits: {dead} — "
                      f"these are silently dropped from the fundamental pillar")


def test_mask_and_currency_tuples_reference_emitted_names():
    vocab = set(DERIVED_FIELDS)
    assert not sorted(scoring._MASK_NONPOS - vocab)
    assert not sorted(set(M._CCY_SENSITIVE) - vocab)
    assert not sorted(set(M._CCY_FALLBACK_ONLY) - set(M._CCY_SENSITIVE))


def test_every_designed_metric_is_computable_from_sec_inputs():
    """A name in the vocabulary that no input can ever produce is the same failure as a
    typo. Only the two ANALYST estimates are exempt: SEC filings do not contain them,
    `enrich()` supplies `forwardPE` for the ~50 shortlist names, and coverage is what
    reports its absence on the rest."""
    live_only = {"forwardPE", "pegRatio"}
    m = derive(SEC_RAW)
    designed = {c for cols in _all_designed_columns().values() for c in cols}
    uncomputable = sorted(c for c in designed - live_only if m.get(c) is None)
    assert not uncomputable, (f"designed but not computable from the SEC path's raw "
                              f"inputs: {uncomputable}")
    for c in sorted(live_only & designed):
        assert c in m, f"{c} must still be an emitted key, even when None"


# ----------------------------------------------------------------- it has teeth
def test_a_typo_raises_instead_of_being_silently_skipped():
    """The regression this whole file exists for."""
    df = pd.DataFrame({"trailingPE": [10.0, 20.0]}, index=["A", "B"])
    with pytest.raises(KeyError, match="never emits"):
        scoring._subscore(df, [("trailingPE", False), ("enterpriseToEBITDA", False)])
    with pytest.raises(KeyError, match="never emits"):
        scoring._designed([("priceToSales", False)])


def test_import_time_check_names_the_offending_list():
    bad = dict(scoring.SECTOR_METRICS)
    bad["Utilities"] = {"cashflow": [("ffo_margln", True)]}
    orig = scoring.SECTOR_METRICS
    scoring.SECTOR_METRICS = bad
    try:
        with pytest.raises(ValueError, match=r'SECTOR_METRICS\["Utilities"\]\["cashflow"\]'):
            scoring._check_metric_vocabulary()
    finally:
        scoring.SECTOR_METRICS = orig


# ----------------------------------------------------------------- coverage honesty
def test_coverage_ignores_a_metric_no_name_in_the_cross_section_can_supply():
    """Coverage measures what THIS company supplied, not what the design wished for.

    `forwardPE` is an analyst estimate the SEC store cannot hold, so it is absent for
    every name except the ~50 `fundamentals.enrich()` reaches. Counting it against the
    other ~1,400 subtracts the same 1/13 from all of them: it moves the median (0.833
    -> 0.778) and trips the research prompt's low-coverage note on 68 more names while
    telling you nothing about any of them. A constant is not a diagnostic.

    The defence against a MIS-WIRED metric name is `_check_metric_vocabulary`, which
    raises at import (see test_scoring_metric_names_are_all_emitted_by_derive). Coverage
    is not that defence and does not need to be."""
    cols = {c: [1.0 + i, 2.0 + i] for i, c in enumerate(
        ["trailingPE", "enterpriseToEbitda", "priceToSalesTrailing12Months",
         "fcf_yield", "fcf_margin", "revenueGrowth", "earningsGrowth",
         "returnOnEquity", "returnOnAssets", "grossMargins", "operatingMargins",
         "debtToEquity"])}
    cols["forwardPE"] = [None, None]        # designed, present as a column, never fed
    df = pd.DataFrame(cols, index=["A", "B"])
    df["symbol"] = df.index
    subs = {s: "software" for s in df.index}
    secs = {s: "Information Technology" for s in df.index}
    cov = scoring.fundamental_detail(df, subs, secs, tag_inputs=False)["coverage"]
    # All 12 rankable components carry data -> full coverage; the unfeedable 13th is
    # not held against A. A name genuinely missing a metric still scores below 1.0.
    assert float(cov["A"]) == pytest.approx(1.0)

    df2 = df.copy()
    df2.loc["A", "fcf_margin"] = None
    cov2 = scoring.fundamental_detail(df2, subs, secs, tag_inputs=False)["coverage"]
    assert float(cov2["A"]) < 1.0


# ------------------------------------------- every OTHER reader of derive()'s keys
# The bug class is "a module reads a metric name nobody emits". `scoring` is the
# biggest such reader but not the only one, and the others fail even more quietly: an
# ML feature that names a key `derive` does not emit goes all-NaN, gets rank-
# transformed to a constant, and contributes nothing to the learned pillar forever
# without a single error. These pin the rest of the readers to the same vocabulary.

# Features `ml.features` builds ITSELF from emitted keys rather than reading straight
# off the snapshot row. Listed explicitly so a new local feature is a deliberate edit
# here, not a silent exemption.
_FEATURES_COMPUTED_LOCALLY = {"net_cash_mcap", "snapshot_age_days"}


def test_ml_feature_names_are_metrics_derive_emits():
    from src.ml import features as mlf
    vocab = set(DERIVED_FIELDS) | _FEATURES_COMPUTED_LOCALLY
    dead = [f for f in mlf.FUND_FEATURES if f not in vocab]
    assert not dead, (f"ml.features.FUND_FEATURES names metrics derive() never emits: "
                      f"{dead} — the feature would be all-NaN on every row, forever")


def test_ml_feature_builder_reads_only_emitted_keys():
    """`fundamentals_features` reads the snapshot row by key. Every key it asks for must
    be one `derive` produced (plus `as_of`, which the store stamps on)."""
    from src.ml import features as mlf
    src = Path(mlf.__file__).read_text(encoding="utf-8")
    body = src[src.index("def fundamentals_features"):src.index("def labels")]
    read = set(re.findall(r'r\.get\(\s*"([A-Za-z0-9_]+)"', body))
    assert read, "could not find the snapshot reads — did fundamentals_features change shape?"
    dead = sorted(read - set(DERIVED_FIELDS) - {"as_of"})
    assert not dead, f"ml.features reads keys derive() never emits: {dead}"


def test_backtest_replays_the_pillar_over_emitted_columns_only():
    """`backtest.signals.fundamental_frames` hands the raw snapshot frame to
    `scoring.fundamental_pillar`, so the replay sees exactly `derive`'s columns. A
    designed component outside that vocabulary would be dropped in the replay while the
    live screen (same code) dropped it too — silently agreeing on a half-wired pillar."""
    snap = pd.DataFrame([derive(SEC_RAW), derive(SEC_RAW)], index=["A", "B"])
    snap["symbol"] = snap.index
    designed = {c for cols in _all_designed_columns().values() for c in cols}
    assert designed - set(snap.columns) == set(), \
        f"a designed component is not a snapshot column: {sorted(designed - set(snap.columns))}"
    pillar = scoring.fundamental_pillar(snap, {"A": "sw", "B": "sw"},
                                        {"A": "Information Technology",
                                         "B": "Information Technology"})
    assert len(pillar) == 2


def test_dashboard_valuation_keys_are_written_into_metrics_json():
    """The dashboard reads `ev_ebitda` from `metrics_json`, and `derive` has no such
    key — `score_universe` maps `enterpriseToEbitda` onto that display name. Nothing
    else pins that mapping, so dropping it would blank the EV/EBITDA row on the board
    with no error anywhere."""
    m = {s: {"symbol": s, "name": s, "marketCap": 1e10, "price": 10.0, "profitable": True,
             "trailingPE": 20.0 + i, "enterpriseToEbitda": 12.0 + i, "ev_sales": 3.0 + i,
             "priceToSalesTrailing12Months": 4.0 + i, "fcf_yield": 0.05, "fcf_margin": 0.2,
             "revenueGrowth": 0.1, "earningsGrowth": 0.1, "returnOnEquity": 0.2,
             "returnOnAssets": 0.1, "grossMargins": 0.5, "operatingMargins": 0.2,
             "debtToEquity": 30.0}
         for i, s in enumerate(("A", "B"))}
    secs = {s: "Information Technology" for s in m}
    row = scoring.score_universe(m, {s: "sw" for s in m}, sectors=secs)[0]
    for k in ("ev_ebitda", "ev_sales", "sector_evebitda", "fund_coverage"):
        assert k in row["metrics"], f"dashboard reads metrics_json['{k}'] and it is absent"
    assert row["metrics"]["ev_ebitda"] is not None


def test_forward_pe_is_absent_on_the_sec_path_and_that_is_why_coverage_drops():
    """`forwardPE` is 1 of 4 default valuation components and 1 of 3 for Financials, and
    the SEC store structurally cannot supply it — only `fundamentals.enrich()` does, for
    the shortlist. Coverage counts it as missing (that is the point of the designed
    denominator); the pillar still ranks on the components that exist."""
    from src import fundamentals
    assert "forwardPE" in fundamentals._ENRICH_FIELDS
    assert derive(SEC_RAW)["forwardPE"] is None
    assert "forwardPE" in [c for c, _ in scoring._VALUATION]
    assert "forwardPE" in [c for c, _ in scoring.sector_metrics("Financials", "valuation")]


def test_family_score_empty_and_nonempty_branches_agree_on_the_denominator():
    """`_family_score` reports a RANKABLE component count. Both of its return paths must
    say so: the empty branch once returned the DESIGNED count, which would have given a
    family with no data anywhere a non-zero denominator and a zero numerator — reading
    as '0% covered' instead of dropping out of the coverage ratio."""
    idx = ["A", "B"]
    df = pd.DataFrame({"fcf_yield": [None, None], "fcf_margin": [None, None]}, index=idx)
    _, n_present, n_cols = scoring._family_score(df, scoring._CASHFLOW, None, None,
                                                 "universe", 2)
    assert list(n_cols) == [0.0, 0.0], "empty branch must report zero rankable components"
    assert list(n_present) == [0.0, 0.0]
    df2 = pd.DataFrame({"fcf_yield": [0.05, 0.03], "fcf_margin": [None, None]}, index=idx)
    _, _, n_cols2 = scoring._family_score(df2, scoring._CASHFLOW, None, None, "universe", 0)
    assert list(n_cols2) == [1.0, 1.0], "only fcf_yield is rankable here"


# ---------------------------------------------------- XBRL tag-list hygiene
# A missing comma between two adjacent string literals in a tag list is not a syntax
# error -- Python concatenates them. The result is one tag name that matches nothing in
# any filing, so BOTH intended tags go silently dead and every filer that used either
# one loses that concept. It happened once, in FLOW_TAGS["capex"], and the only symptom
# was a slightly larger `capex_tag_missing` count in the data-quality report.
def test_no_tag_name_is_two_names_glued_together():
    from src.data import sec_facts as SF

    lists = {f"FLOW_TAGS[{k!r}]": v for k, v in SF.FLOW_TAGS.items()}
    lists |= {f"INSTANT_TAGS[{k!r}]": v for k, v in SF.INSTANT_TAGS.items()}
    lists["SHARES_TAGS"] = SF.SHARES_TAGS

    # Real us-gaap/IFRS element names are long, but not THIS long: the longest tag the
    # SEC taxonomy actually publishes is comfortably under 80 characters.
    offenders = {name: [t for t in tags if len(t) > 80] for name, tags in lists.items()}
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"implausibly long tag names (missing comma?): {offenders}"

    for name, tags in lists.items():
        assert len(tags) == len(set(tags)), f"{name} has duplicate tags"
        for t in tags:
            assert t and t[0].isupper() and t.isalnum(), f"{name}: malformed tag {t!r}"
