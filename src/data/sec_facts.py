"""Point-in-time fundamentals from SEC XBRL `companyfacts`.

One JSON per company (https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json)
carries every reported fact with the date it was FILED. Keying on `filed` — never on
the period end — is the whole point: a 10-K for FY2024 (period end Dec-31) that was
filed on Feb-25 did not exist for a January decision.

Pipeline:  fetch -> long facts frame -> `raw_asof(symbol, date)` (TTM flows, latest
instants, price × as-filed shares) -> `analytics.metrics.derive()` -> monthly snapshots.

Scope: the curated universe (one request per CIK). The market-wide Financial
Statement Data Sets can feed the same `facts` table later for a SIC-derived,
survivorship-aware universe.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import requests

from .. import news, parallel
from ..analytics.metrics import derive

_CF_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# concept -> ordered fallback list of tags. Flows are period totals; instants are
# balance-sheet points. Names on the left are what analytics.metrics.derive reads.
# IFRS (`ifrs-full`) tags are appended AFTER the us-gaap ones for the same concept:
# `prepare_symbol` walks each list in order and takes the first tag with data, so a
# domestic filer still resolves to its us-gaap tag and a 20-F filer falls through to
# IFRS. Without these, every foreign private issuer (TSM, ASML, ARM...) produced only
# its `dei` share count and was scored on an empty fundamental pillar.
_NAMESPACES = ("us-gaap", "ifrs-full")
FLOW_TAGS: dict[str, list[str]] = {
    "totalRevenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                     "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
                     "Revenue", "RevenueFromContractsWithCustomers"],
    "netIncome": ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic",
                  "ProfitLoss", "ProfitLossAttributableToOwnersOfParent"],
    "grossProfits": ["GrossProfit"],
    "operatingIncome": ["OperatingIncomeLoss", "ProfitLossFromOperatingActivities"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
            "CashFlowsFromUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets",
              "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"],
    "dna": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
            "DepreciationAmortizationAndAccretionNet", "Depreciation",
            "DepreciationAndAmortisationExpense"],
}
INSTANT_TAGS: dict[str, list[str]] = {
    "totalAssets": ["Assets"],
    "totalStockholderEquity": ["StockholdersEquity",
                               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                               "EquityAttributableToOwnersOfParent", "Equity"],
    "totalCash": ["CashAndCashEquivalentsAtCarryingValue",
                  "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                  "CashAndCashEquivalents"],
    "debt_lt": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations",
                "NoncurrentPortionOfNoncurrentBorrowings", "LongtermBorrowings"],
    "debt_st": ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings",
                "LongTermDebtAndCapitalLeaseObligationsCurrent",
                "CurrentPortionOfNoncurrentBorrowings", "ShorttermBorrowings"],
}
SHARES_TAG = ("dei", "EntityCommonStockSharesOutstanding")
# Ordered fallbacks for the share count. The dei cover-page tag is the cleanest and is
# what companyfacts serves, but the Financial Statement Data Sets carry it only as a
# segmented row, so the bulk path has to fall back to the balance sheet. Market cap
# depends on this, so an empty result here silently removes a name from any valuation
# screen. Order matters: cover page, then outstanding, then issued, then diluted avg.
SHARES_TAGS = ["EntityCommonStockSharesOutstanding",
               "CommonStockSharesOutstanding",
               "CommonStockSharesIssued",
               "WeightedAverageNumberOfDilutedSharesOutstanding"]

_Q_DAYS = (75, 105)      # a fiscal quarter
_A_DAYS = (340, 380)     # a fiscal year


# ---------------------------------------------------------------- fetch/parse
def fetch_companyfacts(cik: str, pause: float = 0.15) -> dict[str, Any] | None:
    url = _CF_URL.format(cik=str(cik).zfill(10))
    try:
        resp = requests.get(url, headers=news._sec_headers(), timeout=30)
        time.sleep(pause)                      # SEC fair access: stay well under 10 req/s
        if resp.status_code != 200:
            return None
        return resp.json()
    except (requests.RequestException, ValueError):
        return None


def _currency_unit(units: dict[str, Any]) -> str | None:
    """Pick the monetary unit a filer reported a tag in.

    companyfacts keys units by measure: "USD", "EUR", "TWD", "shares", "USD/shares".
    Prefer USD when present (a filer may report both), else the first plain 3-letter
    currency code. Per-share and share units are not monetary and are skipped.
    """
    if "USD" in units:
        return "USD"
    for u in units:
        if len(u) == 3 and u.isalpha() and u.isupper():
            return u
    return None


def facts_frame(symbol: str, cik: str, cf: dict[str, Any]) -> pd.DataFrame:
    """Flatten the tags we use into a long frame. `unit` carries the filer's own
    reporting currency (USD/EUR/TWD/...) or "shares" for the dei count; `start` is
    NaT for instants."""
    rows = []
    facts = cf.get("facts") or {}
    wanted: list[tuple[str, str]] = []                # (tag, concept)
    for concept, tags in FLOW_TAGS.items():
        wanted += [(t, concept) for t in tags]
    for concept, tags in INSTANT_TAGS.items():
        wanted += [(t, concept) for t in tags]
    for tag, concept in wanted:
        # First namespace that carries this tag wins, so a us-gaap filer never picks up
        # an identically-named IFRS tag (Assets, GrossProfit and ProfitLoss exist in both).
        units: dict[str, Any] = {}
        for ns in _NAMESPACES:
            u = ((facts.get(ns) or {}).get(tag) or {}).get("units") or {}
            if u:
                units = u
                break
        unit = _currency_unit(units)
        if unit is None:
            continue
        for it in units.get(unit, []):
            if it.get("val") is None or not it.get("end") or not it.get("filed"):
                continue
            rows.append({"symbol": symbol, "cik": str(cik).zfill(10), "concept": concept,
                         "tag": tag, "unit": unit, "start": it.get("start"),
                         "end": it["end"], "val": float(it["val"]), "filed": it["filed"],
                         "form": it.get("form"), "frame": it.get("frame"),
                         "fy": it.get("fy"), "fp": it.get("fp")})
    sh_units = ((facts.get(SHARES_TAG[0]) or {}).get(SHARES_TAG[1]) or {}).get("units") or {}
    for it in sh_units.get("shares", []):
        if it.get("val") is None or not it.get("end") or not it.get("filed"):
            continue
        rows.append({"symbol": symbol, "cik": str(cik).zfill(10), "concept": "sharesOutstanding",
                     "tag": SHARES_TAG[1], "unit": "shares", "start": it.get("start"),
                     "end": it["end"], "val": float(it["val"]), "filed": it["filed"],
                     "form": it.get("form"), "frame": it.get("frame"),
                     "fy": it.get("fy"), "fp": it.get("fp")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def refresh_facts(store, symbols: list[str], pause: float = 0.15,
                  log=print) -> dict[str, int]:
    """Fetch companyfacts for each symbol with a known CIK; replace its rows in the store."""
    cikmap = news._ticker_cik_map()
    frames, missing, empty = [], [], []
    for i, sym in enumerate(symbols, 1):
        cik = cikmap.get(sym.upper())
        if not cik:
            missing.append(sym)
            continue
        cf = fetch_companyfacts(cik, pause=pause)
        if not cf:
            missing.append(sym)
            continue
        df = facts_frame(sym.upper(), cik, cf)
        if df.empty:
            empty.append(sym)
            continue
        frames.append(df)
        if i % 10 == 0 or i == len(symbols):
            log(f"  facts {i}/{len(symbols)}")
    n = store.upsert_facts(pd.concat(frames, ignore_index=True)) if frames else 0
    if missing:
        log(f"  no CIK / fetch failed: {missing}")
    if empty:
        log(f"  no usable us-gaap facts: {empty}")
    return {"rows": n, "fetched": len(frames), "missing": len(missing), "empty": len(empty)}


# ------------------------------------------------------ PIT assembly (fast path)
# Quarter derivation depends only on WHICH filings exist, not on the as-of date, so it
# is done ONCE per symbol+tag with each period's `filed` date kept; a date then costs a
# numpy mask (filed<=d & end<=d) instead of a pandas groupby. 55s/symbol -> <1s.

@dataclass
class _Tab:
    """Sorted numpy views of one (concept, tag): periods with value + filing date."""
    start: np.ndarray
    end: np.ndarray
    val: np.ndarray
    filed: np.ndarray

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "_Tab":
        df = df.sort_values(["end", "filed"])
        return cls(df["start"].values.astype("datetime64[ns]") if "start" in df else np.array([], dtype="datetime64[ns]"),
                   df["end"].values.astype("datetime64[ns]"), df["val"].values.astype(float),
                   df["filed"].values.astype("datetime64[ns]"))

    def knowable(self, as_of: np.datetime64) -> "_Tab":
        m = (self.filed <= as_of) & (self.end <= as_of)
        return _Tab(self.start[m], self.end[m], self.val[m], self.filed[m])

    def dedupe_by_end(self) -> "_Tab":
        """Latest filing per period end (restatements count once they are public)."""
        if len(self.end) == 0:
            return self
        # sorted by (end, filed): the last row of each end-group is the latest filing
        keep = np.r_[self.end[1:] != self.end[:-1], True]
        return _Tab(self.start[keep], self.end[keep], self.val[keep], self.filed[keep])


def _derive_quarters(flow: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(quarters, annual) frames for one tag from ALL its facts. Missing Q4s are derived
    as FY minus the three quarters inside the FY that were known when the 10-K was
    filed, and carry the 10-K's filing date — so a derived Q4 is knowable exactly when
    the annual figure was."""
    f = flow.dropna(subset=["start"]).copy()
    if f.empty:
        e = pd.DataFrame(columns=["start", "end", "val", "filed"])
        return e, e
    f["days"] = (f["end"] - f["start"]).dt.days
    q = f[(f["days"] >= _Q_DAYS[0]) & (f["days"] <= _Q_DAYS[1])][["start", "end", "val", "filed"]]
    a = f[(f["days"] >= _A_DAYS[0]) & (f["days"] <= _A_DAYS[1])][["start", "end", "val", "filed"]]
    derived = []
    if not a.empty and not q.empty:
        qe, qs, qv, qf = q["end"].values, q["start"].values, q["val"].values, q["filed"].values
        for ys, ye, yv, yf in zip(a["start"].values, a["end"].values, a["val"].values, a["filed"].values):
            m = (qs >= ys - np.timedelta64(10, "D")) & (qe < ye) & (qf <= yf)
            if not m.any():
                continue
            # latest filing per quarter end among those knowable at the 10-K filing
            sub = pd.DataFrame({"end": qe[m], "val": qv[m], "filed": qf[m]}).sort_values(["end", "filed"])
            sub = sub.drop_duplicates("end", keep="last")
            already_q4 = np.any(np.abs((qe - ye).astype("timedelta64[D]").astype(int)) <= 10)
            if len(sub) == 3 and not already_q4:
                last_end = sub["end"].max()
                derived.append({"start": last_end + pd.Timedelta(days=1), "end": ye,
                                "val": float(yv - sub["val"].sum()), "filed": yf})
    if derived:
        q = pd.concat([q, pd.DataFrame(derived)], ignore_index=True)
    return q.sort_values(["end", "filed"]).reset_index(drop=True), a.sort_values(["end", "filed"]).reset_index(drop=True)


def _ttm_tab_dated(qt: _Tab, at: _Tab, as_of: np.datetime64):
    """(TTM value, period end) at as_of. The end lets a caller compare how CURRENT
    each candidate tag is, which is what distinguishes a live tag from a retired one
    that still returns a decade-old figure."""
    q = qt.knowable(as_of).dedupe_by_end()
    if len(q.end) >= 4:
        s, e, v = q.start[-4:], q.end[-4:], q.val[-4:]
        gaps = (s[1:] - e[:-1]).astype("timedelta64[D]").astype(int)
        if np.all(gaps <= 12):
            return float(v.sum()), e[-1]
    a = at.knowable(as_of).dedupe_by_end()
    if len(a.end):
        return float(a.val[-1]), a.end[-1]
    return None, None


def _ttm_tab(qt: _Tab, at: _Tab, as_of: np.datetime64) -> float | None:
    return _ttm_tab_dated(qt, at, as_of)[0]


def _ttm_prev_tab(qt: _Tab, at: _Tab, as_of: np.datetime64) -> float | None:
    q = qt.knowable(as_of).dedupe_by_end()
    if len(q.end) >= 8:
        s, e, v = q.start[-8:], q.end[-8:], q.val[-8:]
        gaps = (s[1:] - e[:-1]).astype("timedelta64[D]").astype(int)
        if np.all(gaps <= 12):
            return float(v[:4].sum())
    if len(q.end) == 0:
        return None
    latest_end = q.end[-1]
    a = at.knowable(as_of).dedupe_by_end()
    if len(a.end) == 0:
        return None
    lo, hi = latest_end - np.timedelta64(460, "D"), latest_end - np.timedelta64(240, "D")
    m = (a.end >= lo) & (a.end <= hi)
    # If the TTM itself had to fall back to the latest FY (no 4-quarter chain), the
    # year-earlier base must be strictly older than that FY — otherwise growth reads
    # exactly 0.0. With a real quarter chain the latest FY is a legitimate base.
    chain4 = len(q.end) >= 4 and np.all(
        (q.start[-4:][1:] - q.end[-4:][:-1]).astype("timedelta64[D]").astype(int) <= 12)
    if not chain4:
        m &= a.end < a.end[-1]
    if not m.any():
        return None
    target = latest_end - np.timedelta64(365, "D")
    idx = np.argmin(np.abs((a.end[m] - target).astype("timedelta64[D]").astype(int)))
    return float(a.val[m][idx])


def _instant_tab(it: _Tab, as_of: np.datetime64) -> float | None:
    k = it.knowable(as_of).dedupe_by_end()
    return float(k.val[-1]) if len(k.end) else None


@dataclass
class PreparedFacts:
    """One symbol's facts pre-split per concept/tag into period tables."""
    flows: dict[str, list[tuple[_Tab, _Tab]]]      # concept -> [(quarters, annual)] in tag order
    instants: dict[str, list[_Tab]]                # concept -> [tab] in tag order
    currency: str = "USD"                          # the filer's own statement currency


def _statement_currency(sym_facts: pd.DataFrame) -> str:
    """The currency the income statement is reported in.

    Taken from revenue (falling back to net income) rather than from any monetary tag,
    so a USD-denominated debt footnote in an otherwise-EUR filer can't flip it.
    """
    for concept in ("totalRevenue", "netIncome"):
        u = sym_facts.loc[sym_facts["concept"] == concept, "unit"]
        u = u[u != "shares"]
        if not u.empty:
            return str(u.mode().iloc[0])
    return "USD"


def prepare_symbol(sym_facts: pd.DataFrame) -> PreparedFacts:
    flows: dict[str, list[tuple[_Tab, _Tab]]] = {}
    insts: dict[str, list[_Tab]] = {}
    for concept, tags in FLOW_TAGS.items():
        g = sym_facts[sym_facts["concept"] == concept]
        lst = []
        for tag in tags:
            gt = g[g["tag"] == tag]
            if gt.empty:
                continue
            q, a = _derive_quarters(gt)
            lst.append((_Tab.from_df(q) if not q.empty else _Tab.from_df(pd.DataFrame(
                {"start": pd.Series(dtype="datetime64[ns]"), "end": pd.Series(dtype="datetime64[ns]"),
                 "val": pd.Series(dtype=float), "filed": pd.Series(dtype="datetime64[ns]")})),
                        _Tab.from_df(a) if not a.empty else _Tab.from_df(pd.DataFrame(
                {"start": pd.Series(dtype="datetime64[ns]"), "end": pd.Series(dtype="datetime64[ns]"),
                 "val": pd.Series(dtype=float), "filed": pd.Series(dtype="datetime64[ns]")}))))
        if lst:
            flows[concept] = lst
    for concept, tags in list(INSTANT_TAGS.items()) + [("sharesOutstanding", SHARES_TAGS)]:
        g = sym_facts[(sym_facts["concept"] == concept) & (sym_facts["start"].isna())]
        lst = []
        for tag in tags:
            gt = g[g["tag"] == tag]
            if not gt.empty:
                lst.append(_Tab.from_df(gt.assign(start=pd.NaT)))
        if lst:
            insts[concept] = lst
    return PreparedFacts(flows, insts, _statement_currency(sym_facts))


def raw_asof_prepared(pf: PreparedFacts, as_of: date, price: float | None,
                      ma50: float | None = None, ma200: float | None = None) -> dict[str, Any] | None:
    ts = np.datetime64(pd.Timestamp(as_of), "ns")

    _pick: dict[str, int | None] = {}

    def _best(concept: str) -> int | None:
        """Index of the tag whose TTM window ends LATEST.

        Companies switch revenue tags over time (MSFT: Revenues -> SalesRevenueNet ->
        RevenueFromContractWithCustomerExcludingAssessedTax) and the retired tag keeps
        returning its final value forever. Taking the first tag that yielded anything
        therefore pinned MSFT's revenue to a 2010 figure while gross profit stayed
        current, producing a 338% gross margin. Freshness decides; the FLOW_TAGS order
        only breaks ties.
        """
        if concept in _pick:
            return _pick[concept]
        best_i, best_end = None, None
        for i, (qt, at) in enumerate(pf.flows.get(concept, [])):
            v, e = _ttm_tab_dated(qt, at, ts)
            if v is None or e is None:
                continue
            if best_end is None or e > best_end:
                best_i, best_end = i, e
        _pick[concept] = best_i
        return best_i

    def flow(concept: str, prev: bool = False) -> float | None:
        # One tag serves both the current and year-earlier TTM, so a growth ratio is
        # never computed across two different tags.
        i = _best(concept)
        if i is None:
            return None
        qt, at = pf.flows[concept][i]
        return _ttm_prev_tab(qt, at, ts) if prev else _ttm_tab(qt, at, ts)

    def inst(concept: str) -> float | None:
        for it in pf.instants.get(concept, []):
            v = _instant_tab(it, ts)
            if v is not None:
                return v
        return None

    if not pf.flows and not pf.instants:
        return None
    rev, rev_prev = flow("totalRevenue"), flow("totalRevenue", prev=True)
    ni, ni_prev = flow("netIncome"), flow("netIncome", prev=True)
    cfo, capex = flow("cfo"), flow("capex")
    opinc, dna = flow("operatingIncome"), flow("dna")
    shares = inst("sharesOutstanding")
    d_lt, d_st = inst("debt_lt"), inst("debt_st")
    debt = None if (d_lt is None and d_st is None) else (d_lt or 0.0) + (d_st or 0.0)
    if rev is None and ni is None and shares is None and inst("totalAssets") is None:
        return None
    return {
        "price": price,
        "sharesOutstanding": shares,
        "marketCap": (price * shares) if (price and shares) else None,
        "totalRevenue": rev,
        "netIncome": ni,
        "grossProfits": flow("grossProfits"),
        "operatingIncome": opinc,
        "freeCashflow": (cfo - abs(capex)) if (cfo is not None and capex is not None) else cfo,
        "ebitda": (opinc + dna) if (opinc is not None and dna is not None) else None,
        "totalAssets": inst("totalAssets"),
        "totalStockholderEquity": inst("totalStockholderEquity"),
        "totalCash": inst("totalCash"),
        "totalDebt": debt,
        "revenueGrowth": (rev / rev_prev - 1) if (rev and rev_prev and rev_prev > 0) else None,
        "earningsGrowth": (ni / ni_prev - 1) if (ni is not None and ni_prev and ni_prev > 0) else None,
        "fiftyDayAverage": ma50,
        "twoHundredDayAverage": ma200,
        "financialCurrency": pf.currency,
    }


# -------------------------------------------------- PIT assembly (reference path)
# Kept for tests and single-date use; `build_snapshots` uses the prepared fast path,
# which implements the same rules.
def _pit(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Facts KNOWABLE on as_of: filed on/before it, period ended on/before it. Among
    duplicates of the same period keep the latest filing (restatements count only once
    they were public)."""
    d = df[(df["filed"] <= as_of) & (df["end"] <= as_of)]
    if d.empty:
        return d
    # Dedupe per concept+tag+period: two concepts reported for the same period are
    # different facts, not duplicates.
    return (d.sort_values("filed")
             .groupby(["concept", "tag", "start", "end"], dropna=False).tail(1))


def _quarters(flow: pd.DataFrame) -> pd.DataFrame:
    """Quarterly periods from a concept's PIT flow facts; derive missing Q4s from FY
    minus the three quarters inside that fiscal year."""
    f = flow.dropna(subset=["start"]).copy()
    if f.empty:
        return f
    f["days"] = (f["end"] - f["start"]).dt.days
    q = f[(f["days"] >= _Q_DAYS[0]) & (f["days"] <= _Q_DAYS[1])].copy()
    a = f[(f["days"] >= _A_DAYS[0]) & (f["days"] <= _A_DAYS[1])]
    derived = []
    for _, yr in a.iterrows():
        inside = q[(q["start"] >= yr["start"] - pd.Timedelta(days=10)) & (q["end"] < yr["end"])]
        if len(inside) == 3 and not ((q["end"] - yr["end"]).abs() <= pd.Timedelta(days=10)).any():
            last_end = inside["end"].max()
            derived.append({"start": last_end + pd.Timedelta(days=1), "end": yr["end"],
                            "val": float(yr["val"] - inside["val"].sum()),
                            "filed": yr["filed"], "days": (yr["end"] - last_end).days})
    if derived:
        q = pd.concat([q, pd.DataFrame(derived)], ignore_index=True)
    return q.sort_values("end").drop_duplicates("end", keep="last")


def _ttm(flow: pd.DataFrame, as_of: pd.Timestamp, lag_days: int = 0) -> float | None:
    """Sum of the 4 most recent consecutive quarters ending <= as_of - lag; falls back
    to the latest fiscal-year figure when quarters can't be chained."""
    cutoff = as_of - pd.Timedelta(days=lag_days)
    q = _quarters(flow)
    q = q[q["end"] <= cutoff]
    if len(q) >= 4:
        last4 = q.tail(4)
        gaps = last4["start"].values[1:] - last4["end"].values[:-1]
        if all(pd.Timedelta(g) <= pd.Timedelta(days=12) for g in gaps):
            return float(last4["val"].sum())
    f = flow.dropna(subset=["start"]).copy()
    f["days"] = (f["end"] - f["start"]).dt.days
    a = f[(f["days"] >= _A_DAYS[0]) & (f["days"] <= _A_DAYS[1]) & (f["end"] <= cutoff)]
    return float(a.sort_values("end")["val"].iloc[-1]) if not a.empty else None


def _ttm_prev(flow: pd.DataFrame, as_of: pd.Timestamp) -> float | None:
    """The TTM one year BEFORE the latest TTM: quarters 5-8 back when eight chain,
    else the fiscal year that ended at least ~300 days before the latest quarter end.
    Defined on quarter boundaries (not a fixed 365-day lag) so YoY growth compares
    like with like regardless of where as_of falls inside a quarter."""
    q = _quarters(flow)
    q = q[q["end"] <= as_of]
    if len(q) >= 8:
        last8 = q.tail(8)
        gaps = last8["start"].values[1:] - last8["end"].values[:-1]
        if all(pd.Timedelta(g) <= pd.Timedelta(days=12) for g in gaps):
            return float(last8.head(4)["val"].sum())
    if q.empty:
        return None
    latest_end = q["end"].max()
    # Fallback: the fiscal-year total whose end is closest to one year before the
    # latest quarter end (accept 8-15 months back — a FY ending one quarter "late"
    # is still the best available year-earlier base).
    f = flow.dropna(subset=["start"]).copy()
    f["days"] = (f["end"] - f["start"]).dt.days
    annual_all = f[(f["days"] >= _A_DAYS[0]) & (f["days"] <= _A_DAYS[1]) & (f["end"] <= as_of)]
    a = annual_all[(annual_all["end"] <= latest_end - pd.Timedelta(days=240))
                   & (annual_all["end"] >= latest_end - pd.Timedelta(days=460))]
    # Same rule as the fast path: only when the TTM fell back to the latest FY must the
    # year-earlier base be strictly older than it.
    last4 = q.tail(4)
    chain4 = len(last4) == 4 and all(
        pd.Timedelta(g) <= pd.Timedelta(days=12)
        for g in (last4["start"].values[1:] - last4["end"].values[:-1]))
    if not chain4 and not annual_all.empty:
        a = a[a["end"] < annual_all["end"].max()]
    if a.empty:
        return None
    target = latest_end - pd.Timedelta(days=365)
    best = a.iloc[(a["end"] - target).abs().argsort().iloc[0]]
    return float(best["val"])


def _instant(inst: pd.DataFrame, as_of: pd.Timestamp) -> float | None:
    d = inst[inst["start"].isna()] if "start" in inst else inst
    if d.empty:
        return None
    d = d[d["end"] <= as_of]
    if d.empty:
        return None
    last_end = d["end"].max()
    return float(d[d["end"] == last_end].sort_values("filed")["val"].iloc[-1])


def raw_asof(sym_facts: pd.DataFrame, as_of: date, price: float | None,
             ma50: float | None = None, ma200: float | None = None) -> dict[str, Any] | None:
    """Build the raw dict `derive()` expects, as knowable on `as_of`."""
    ts = pd.Timestamp(as_of)
    pit = _pit(sym_facts, ts)
    if pit.empty:
        return None
    by = {c: g for c, g in pit.groupby("concept")}

    def flow(concept: str, prev: bool = False) -> float | None:
        g = by.get(concept)
        if g is None:
            return None
        # Freshest tag wins, not the first with data — see _best() in the fast path.
        # Tie-break by FLOW_TAGS order. Same tag serves current and year-earlier TTM.
        best_tag, best_end = None, None
        for tag in FLOW_TAGS.get(concept, []):
            gt = g[g["tag"] == tag]
            if gt.empty or _ttm(gt, ts) is None:
                continue
            e = gt["end"].max()
            if best_end is None or e > best_end:
                best_tag, best_end = tag, e
        if best_tag is None:
            return None
        gt = g[g["tag"] == best_tag]
        return _ttm_prev(gt, ts) if prev else _ttm(gt, ts)

    def inst(concept: str) -> float | None:
        g = by.get(concept)
        if g is None:
            return None
        for tag in INSTANT_TAGS.get(concept, []) + SHARES_TAGS:
            gt = g[g["tag"] == tag]
            if not gt.empty:
                v = _instant(gt, ts)
                if v is not None:
                    return v
        return None

    rev, rev_prev = flow("totalRevenue"), flow("totalRevenue", prev=True)
    ni, ni_prev = flow("netIncome"), flow("netIncome", prev=True)
    cfo, capex = flow("cfo"), flow("capex")
    opinc, dna = flow("operatingIncome"), flow("dna")
    shares = inst("sharesOutstanding")
    debt = (inst("debt_lt") or 0.0) + (inst("debt_st") or 0.0)
    if inst("debt_lt") is None and inst("debt_st") is None:
        debt = None
    # Nothing core is knowable yet (e.g. before the first 10-K) -> no snapshot row,
    # rather than a row of Nones that would score as neutral everywhere.
    if rev is None and ni is None and shares is None and inst("totalAssets") is None:
        return None
    raw: dict[str, Any] = {
        "price": price,
        "sharesOutstanding": shares,
        "marketCap": (price * shares) if (price and shares) else None,
        "totalRevenue": rev,
        "netIncome": ni,
        "grossProfits": flow("grossProfits"),
        "operatingIncome": opinc,
        "freeCashflow": (cfo - abs(capex)) if (cfo is not None and capex is not None) else cfo,
        "ebitda": (opinc + dna) if (opinc is not None and dna is not None) else None,
        "totalAssets": inst("totalAssets"),
        "totalStockholderEquity": inst("totalStockholderEquity"),
        "totalCash": inst("totalCash"),
        "totalDebt": debt,
        "revenueGrowth": (rev / rev_prev - 1) if (rev and rev_prev and rev_prev > 0) else None,
        "earningsGrowth": (ni / ni_prev - 1) if (ni is not None and ni_prev and ni_prev > 0) else None,
        "fiftyDayAverage": ma50,
        "twoHundredDayAverage": ma200,
        "financialCurrency": _statement_currency(sym_facts),
    }
    return raw


# ---------------------------------------------------------------- snapshots
def _snapshot_symbol(payload: tuple) -> list[dict[str, Any]]:
    """One symbol's rows across every as-of date. Module-level and store-free so it can
    be shipped to a spawned worker (see src/parallel.py); it receives plain frames."""
    sym, sf, px, ma50, ma200, dates = payload
    pf = prepare_symbol(sf)                # once per symbol; each date is then a mask
    out = []
    for d in dates:
        pxd = px[px.index <= d]
        if pxd.empty:
            continue
        price = float(pxd.iloc[-1])
        m50 = ma50.get(pxd.index[-1])
        m200 = ma200.get(pxd.index[-1])
        raw = raw_asof_prepared(pf, d.date(), price,
                                None if pd.isna(m50) else float(m50),
                                None if pd.isna(m200) else float(m200))
        if raw is None:
            continue
        m = derive(raw, as_of=d.date())
        m.update({"as_of": d, "symbol": sym, "sharesOutstanding": raw.get("sharesOutstanding"),
                  "netIncome": raw.get("netIncome"), "totalAssets": raw.get("totalAssets"),
                  "totalDebt": raw.get("totalDebt"), "totalCash": raw.get("totalCash")})
        m.pop("earnings_date", None); m.pop("days_to_earnings", None)
        out.append(m)
    return out


def build_snapshots(store, symbols: list[str], dates: list[pd.Timestamp],
                    log=print, workers: int | None = None) -> pd.DataFrame:
    """Derived metrics per (as_of, symbol) using only facts filed by as_of and the
    raw close on/before as_of. Writes to the store and returns the frame.

    Symbols are independent, so the O(symbols x dates) double loop is farmed out to a
    process pool; `workers=1` is the old serial loop, byte for byte."""
    facts = store.facts(symbols)
    # Bound the pivot to the window the snapshot grid can actually read: the 200-day
    # moving average needs ~a year of lead-in, nothing needs the years before that.
    lo = (min(dates) - pd.Timedelta(days=600)).date() if dates else None
    hi = max(dates).date() if dates else None
    closes = store.close_matrix(symbols, adjusted=False, start=lo, end=hi)
    if facts.empty or closes.empty:
        log("  snapshots: facts or prices missing")
        return pd.DataFrame()
    ma50 = closes.rolling(50).mean()
    ma200 = closes.rolling(200).mean()
    by_sym = {s: g for s, g in facts.groupby("symbol", sort=False)}
    payloads = [(sym, by_sym[sym], closes[sym].dropna(), ma50[sym], ma200[sym], dates)
                for sym in symbols
                if sym in by_sym and not by_sym[sym].empty and sym in closes.columns]
    chunks = parallel.replay_symbols(_snapshot_symbol, payloads, workers=workers,
                                     log=log, label="snapshots", every=10,
                                     work=len(payloads) * len(dates))
    df = pd.DataFrame(parallel.flatten(chunks))
    if not df.empty:
        store.write_snapshots(df)
    return df
