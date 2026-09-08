"""FSDS bulk ingest: point-in-time correctness rules, and parity with companyfacts.

Two layers:

  * Synthetic-ZIP tests (always run, no network) pin the rules that are easy to get
    silently wrong — the segments/coreg filter, the derived `start`, keeping amended
    submissions, and the `ifrs` vs `ifrs-full` namespace difference between the two
    SEC datasets.
  * A real parity check that runs only when quarterly ZIPs are already cached. It is
    the Phase 1 gate: FSDS-derived metrics must match companyfacts-derived metrics,
    because everything downstream (prepare_symbol, raw_asof_prepared, the snapshots,
    the backtest) is shared and must not notice which path filled the facts table.
"""
from __future__ import annotations

import io
import zipfile
from datetime import date

import pandas as pd
import pytest

from src.data import sec_facts, sec_fsds

_SUB_COLS = ["adsh", "cik", "name", "sic", "countryba", "stprba", "cityba", "zipba",
             "bas1", "bas2", "baph", "countryma", "stprma", "cityma", "zipma", "mas1",
             "mas2", "countryinc", "stprinc", "ein", "former", "changed", "afs", "wksi",
             "fye", "form", "period", "fy", "fp", "filed", "accepted", "prevrpt",
             "detail", "instance", "nciks", "aciks"]
_NUM_COLS = ["adsh", "tag", "version", "ddate", "qtrs", "uom", "segments", "coreg",
             "value", "footnote"]


def _zip(subs: list[dict], nums: list[dict], tmp_path) -> "object":
    """Build a minimal FSDS quarter ZIP with the real column layout."""
    sub = pd.DataFrame([{**{c: "" for c in _SUB_COLS}, **s} for s in subs])[_SUB_COLS]
    num = pd.DataFrame([{**{c: "" for c in _NUM_COLS}, **n} for n in nums])[_NUM_COLS]
    p = tmp_path / "2024q1.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("sub.txt", sub.to_csv(sep="\t", index=False))
        z.writestr("num.txt", num.to_csv(sep="\t", index=False))
    return p


def _sub(adsh="0000000001-24-000001", cik="1045810", form="10-Q",
         filed="20240215", **kw) -> dict:
    return {"adsh": adsh, "cik": cik, "form": form, "filed": filed,
            "period": "20231231", "fy": "2023", "fp": "Q4", "prevrpt": "0",
            "detail": "1", "name": "TESTCO", "sic": "3674", **kw}


def _num(tag="Revenues", version="us-gaap/2023", ddate="20231231", qtrs="1",
         uom="USD", value="100", adsh="0000000001-24-000001", **kw) -> dict:
    return {"adsh": adsh, "tag": tag, "version": version, "ddate": ddate,
            "qtrs": qtrs, "uom": uom, "value": value, "segments": "", "coreg": "", **kw}


C2S = {"0001045810": "TSTA"}   # zfill(10) of the CIK used by _sub()


def _parse(tmp_path, subs, nums):
    return sec_fsds.parse_quarter(_zip(subs, nums, tmp_path), set(C2S), C2S)


# ------------------------------------------------------------------ PIT rules
def test_segment_and_coreg_rows_are_dropped(tmp_path):
    """The #1 FSDS trap: segment/subsidiary rows summed into a consolidated total
    silently corrupt every margin. Only fully-consolidated rows may survive."""
    df = _parse(tmp_path, [_sub()], [
        _num(value="100"),                                  # consolidated -> keep
        _num(value="60", segments="ProductOrService=Chips"),  # segment      -> drop
        _num(value="40", coreg="SubsidiaryCo"),               # subsidiary   -> drop
    ])
    assert len(df) == 1
    assert df["val"].iloc[0] == 100.0


def test_amended_submissions_are_kept(tmp_path):
    """prevrpt=1 marks a submission later amended. The ORIGINAL filing is still what
    was knowable at the time, so dropping it would destroy point-in-time truth;
    restatements are resolved downstream by filing date, not here."""
    df = _parse(tmp_path, [_sub(prevrpt="1")], [_num()])
    assert len(df) == 1


def test_start_is_derived_from_ddate_and_qtrs(tmp_path):
    """FSDS gives a period END plus a length in quarters; companyfacts gives an
    explicit start. The derived start must land in the window _derive_quarters
    expects, and instants (qtrs=0) must stay NaT."""
    df = _parse(tmp_path, [_sub()], [
        _num(tag="Revenues", qtrs="1", ddate="20231231"),
        _num(tag="Assets", qtrs="0", ddate="20231231", value="500"),
    ])
    rev = df[df["concept"] == "totalRevenue"].iloc[0]
    span = (rev["end"] - rev["start"]).days
    assert sec_facts._Q_DAYS[0] <= span <= sec_facts._Q_DAYS[1], span
    assert pd.isna(df[df["concept"] == "totalAssets"]["start"].iloc[0])


def test_annual_span_matches_the_annual_window(tmp_path):
    df = _parse(tmp_path, [_sub(form="10-K")], [_num(qtrs="4", ddate="20231231")])
    row = df.iloc[0]
    span = (row["end"] - row["start"]).days
    assert sec_facts._A_DAYS[0] <= span <= sec_facts._A_DAYS[1], span


def test_ifrs_namespace_is_accepted(tmp_path):
    """companyfacts calls the taxonomy 'ifrs-full'; FSDS versions it 'ifrs'. Accepting
    only one silently drops every 20-F filer (TSM and the foreign-issuer cohort)."""
    df = _parse(tmp_path, [_sub(form="20-F")],
                [_num(tag="Revenue", version="ifrs/2023", uom="TWD", value="900")])
    assert len(df) == 1
    assert df["unit"].iloc[0] == "TWD"
    assert df["concept"].iloc[0] == "totalRevenue"


def test_company_extension_namespace_is_ignored(tmp_path):
    """Custom company tags share names with standard ones but are not comparable.
    Paired with a valid row so the test cannot pass merely by parsing nothing."""
    df = _parse(tmp_path, [_sub()],
                [_num(version="0000000001-24-000001", value="123"),
                 _num(version="us-gaap/2023", value="456")])
    assert list(df["val"]) == [456.0]


def test_filed_date_comes_from_the_submission(tmp_path):
    """`filed` is the whole reason this dataset is usable — it must be the submission's
    filing date, never the period end."""
    df = _parse(tmp_path, [_sub(filed="20240215", period="20231231")], [_num()])
    assert df["filed"].iloc[0] == pd.Timestamp("2024-02-15")
    assert df["end"].iloc[0] == pd.Timestamp("2023-12-31")


def test_non_currency_units_are_rejected_for_monetary_concepts(tmp_path):
    df = _parse(tmp_path, [_sub()],
                [_num(uom="pure", value="0.5"), _num(uom="USD", value="7")])
    assert list(df["val"]) == [7.0]


def test_share_count_requires_share_units(tmp_path):
    df = _parse(tmp_path, [_sub()], [
        _num(tag="CommonStockSharesOutstanding", qtrs="0", uom="shares", value="1000"),
        _num(tag="CommonStockSharesIssued", qtrs="0", uom="USD", value="7"),
    ])
    assert list(df["concept"]) == ["sharesOutstanding"]
    assert df["val"].iloc[0] == 1000.0


def test_output_schema_matches_companyfacts(tmp_path):
    """Both ingest paths feed the same `facts` table and the same prepare_symbol."""
    df = _parse(tmp_path, [_sub()], [_num()])
    expected = {"symbol", "cik", "concept", "tag", "unit", "start", "end", "val",
                "filed", "form", "frame", "fy", "fp"}
    assert set(df.columns) == expected


def test_parsed_rows_feed_prepare_symbol(tmp_path):
    """End-to-end: FSDS rows must be consumable by the existing PIT assembly with no
    adaptation, which is what makes this a drop-in source swap."""
    nums, subs = [], []
    for i, (end, filed) in enumerate([("20230331", "20230415"), ("20230630", "20230715"),
                                      ("20230930", "20231015"), ("20231231", "20240215")]):
        adsh = f"0000000001-24-00000{i}"
        subs.append(_sub(adsh=adsh, filed=filed))
        nums.append(_num(adsh=adsh, tag="Revenues", qtrs="1", ddate=end, value="100"))
        nums.append(_num(adsh=adsh, tag="Assets", qtrs="0", ddate=end, value="900"))
    df = _parse(tmp_path, subs, nums)
    pf = sec_facts.prepare_symbol(df)
    raw = sec_facts.raw_asof_prepared(pf, date(2024, 3, 31), price=10.0)
    assert raw is not None
    assert raw["totalRevenue"] == pytest.approx(400.0)   # four quarters -> TTM
    assert raw["totalAssets"] == pytest.approx(900.0)


def test_nothing_filed_after_as_of_is_visible(tmp_path):
    """The PIT guarantee, restated for this source."""
    subs = [_sub(adsh="a1", filed="20240215"), _sub(adsh="a2", filed="20240515")]
    nums = [_num(adsh="a1", ddate="20231231", value="100"),
            _num(adsh="a2", ddate="20240331", value="999")]
    df = _parse(tmp_path, subs, nums)
    pf = sec_facts.prepare_symbol(df)
    raw = sec_facts.raw_asof_prepared(pf, date(2024, 3, 1), price=1.0)
    vals = [] if raw is None else [raw.get("totalRevenue")]
    assert 999.0 not in vals


# ------------------------------------------------- real parity (cached ZIPs only)
# Each ZIP holds a ~600 MB num.txt, so parsing is seconds per quarter. The gate is
# about agreement, not coverage: the most recent few quarters exercise every code path
# and keep the suite fast. Raise for a deeper audit via a one-off run, not in CI.
_PARITY_QUARTERS = 4


def _cached_quarters() -> list:
    d = sec_fsds.cache_dir()
    return sorted(d.glob("*.zip")) if d.exists() else []


def _parity_sample() -> list:
    return _cached_quarters()[-_PARITY_QUARTERS:]


@pytest.mark.skipif(len(_cached_quarters()) < 4,
                    reason="needs cached FSDS ZIPs (scripts.data_refresh --facts-bulk)")
def test_fsds_matches_companyfacts_on_cached_quarters():
    """Phase 1 gate: metrics derived from the bulk path must match the per-CIK path.

    Compared per (symbol, tag, end, filed) rather than per concept: several tags map
    to one concept and legitimately differ (LongTermDebt includes the current portion,
    LongTermDebtNoncurrent does not), so aggregating by concept would compare apples
    to pears. Comparing raw facts, not snapshots, keeps a failure pointed at ingest.
    """
    from src.data.history_store import HistoryStore
    store = HistoryStore()
    cf = store.facts()
    if cf.empty:
        pytest.skip("no companyfacts in the store to compare against")

    from src import news
    cikmap = news._ticker_cik_map()
    syms = ["NVDA", "MSFT", "AVGO"]
    c2s = {str(cikmap[s]).zfill(10): s for s in syms if s in cikmap}

    parts = [sec_fsds.parse_quarter(p, set(c2s), c2s) for p in _parity_sample()]
    parts = [p for p in parts if not p.empty]
    if not parts:
        pytest.skip("cached quarters carry no rows for the sample symbols")
    fs = pd.concat(parts, ignore_index=True)

    key = ["symbol", "tag", "end", "filed"]
    a = fs.groupby(key)["val"].max().rename("fsds")
    b = cf[cf["symbol"].isin(syms)].groupby(key)["val"].max().rename("cf")
    both = pd.concat([a, b], axis=1).dropna()
    if len(both) < 50:
        pytest.skip(f"only {len(both)} overlapping facts - not a meaningful gate")

    close = (both["fsds"] - both["cf"]).abs() <= both["cf"].abs() * 1e-6 + 1.0
    share = float(close.mean())
    assert share >= 0.95, (
        f"only {share:.1%} of {len(both)} overlapping facts agree; "
        f"worst:\n{both[~close].head(10)}")
