"""Automated plausibility checks over the point-in-time store.

WHY THIS EXISTS. Every data bug this project has found was caught by a human reading
numbers off the board, never by a test:

  * MSFT's revenue pinned to a tag retired in 2010 (a 338% gross margin, 13/105 names);
  * TSM sitting at 9 fact rows because the ingest only read the `us-gaap` namespace;
  * ASML discarded entirely because it reports in EUR and the unit was hardcoded USD;
  * yfinance's `.info["freeCashflow"]` being one QUARTER, so every fcf ratio it fed the
    live board was ~4x too small.

All four are silent: nothing raises, nothing goes red, the pillar simply runs on a
wrong number. `tests/` guards the FUNCTIONS; this module guards the DATA those
functions produced, which is a different failure class — a parser can be correct and
the store still be wrong because the tags changed underneath it.

WHAT IT RETURNS. `run()` produces `Finding` records — symbol, check, severity, the
observed value, and why it is implausible — never a bare boolean. A gate that says
"failed" and nothing else is a gate nobody can act on.

SEVERITY. `CRITICAL` is "this number is impossible, or the store lost data it used to
have"; the CLI exits non-zero on it. `ERROR` is "almost certainly wrong, but a rare
filer could legitimately look like this". `WARN` is "worth an eye". Coverage
regressions are CRITICAL by design: that is the tier TSM and ASML would have tripped,
and a coverage loss is the one defect that gets quieter, not louder, the worse it gets.

THE BASELINE. Ratio bounds catch absurdity; they cannot catch "the ingest silently
halved coverage and every surviving number is fine". That needs a remembered previous
state, so per-sector coverage and store totals are written to
`<store>/quality_baseline.json` and compared on every later run.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

CRITICAL, ERROR, WARN = "critical", "error", "warn"
_ORDER = {CRITICAL: 0, ERROR: 1, WARN: 2}

BASELINE_NAME = "quality_baseline.json"

# The CRITICAL checks that make the store unusable RIGHT NOW, as opposed to the ones
# that say a handful of rows are wrong. `scripts.doctor` fails on these and warns on
# the rest, because doctor answers "can I run today?" — nineteen bad microcap rows out
# of 5,241 do not stop a run, a store that lost a third of its facts does. The
# distinction is here, next to the severities, rather than hidden in doctor: a new
# store-wide check must be added to this set deliberately.
BLOCKING = frozenset({
    "facts_empty", "snapshots_empty", "prices_empty", "snapshots_stale",
    "store_shrank", "sector_coverage_regression", "sector_disappeared",
    "nonusd_filers_absent", "no_snapshot_regression", "currency_column_missing",
    "cfo_concept_absent",
})

# Cross-currency multiples `analytics.metrics.derive` masks for a non-USD reporter.
# Imported rather than re-listed: a metric added to the masking tuple must be checked
# here automatically or the check silently stops covering it.
from ..analytics.metrics import _CCY_SENSITIVE  # noqa: E402

# Bounds. Deliberately wide — these are IMPOSSIBILITY tests, not opinions about
# valuation. A bound that fires on a merely unusual company teaches people to ignore
# the gate, which is worse than not having one.
_MAX_ABS = {
    "priceToBook": 200.0,
    "ev_sales": 500.0,
    "priceToSalesTrailing12Months": 500.0,
    "enterpriseToEbitda": 2000.0,
    "trailingPE": 5000.0,
    "p_ffo": 2000.0,
}
_MIN_FACT_ROWS = 40          # a genuine filer publishes hundreds of mapped facts
_FACTS_STALE_DAYS = 200      # a quarterly filer that has not filed in 200d is odd
_FACTS_DEAD_DAYS = 400       # ...and at 400d it is not an active filer at all
_CONCEPT_LAG_DAYS = 500      # one concept this far behind the rest is a retired tag
_SNAPSHOT_FRESH_DAYS = 120   # `HistoryStore.fundamentals_asof` serves nothing older
_COVERAGE_DROP = 0.10        # fraction of per-sector coverage a rebuild may lose
_TOTALS_DROP = 0.05
# A real company is worth many months of its own turnover. Below this, the share
# count — not the valuation — is what is wrong.
# A ratio of 5 means the entire company changes hands every trading week. No real
# company does that; a share count wrong by an order of magnitude does.
_MCAP_ADV_FLOOR = 5.0
# The month-over-month checks need a month between the two cross-sections.
_PREV_MIN_GAP_DAYS = 20


@dataclass
class Finding:
    check: str
    severity: str
    symbol: str | None
    observed: Any
    detail: str

    def line(self) -> str:
        who = self.symbol or "-"
        return f"[{self.severity.upper():8}] {self.check:26} {who:10} {self.detail}"


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    def add(self, check: str, severity: str, symbol: str | None,
            observed: Any, detail: str) -> None:
        self.findings.append(Finding(check, severity, symbol, observed, detail))

    def counts(self) -> dict[str, int]:
        out = {CRITICAL: 0, ERROR: 0, WARN: 0}
        for f in self.findings:
            out[f.severity] = out.get(f.severity, 0) + 1
        return out

    def by_check(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for f in sorted(self.findings, key=lambda x: (_ORDER.get(x.severity, 9), x.check)):
            e = out.setdefault(f.check, {"severity": f.severity, "n": 0, "examples": []})
            e["n"] += 1
            if len(e["examples"]) < 5:
                e["examples"].append(f"{f.symbol}={f.observed}" if f.symbol else str(f.observed))
        return out

    def blocking(self) -> list[Finding]:
        return [f for f in self.findings
                if f.severity == CRITICAL and f.check in BLOCKING]

    def worst(self) -> str | None:
        return min((f.severity for f in self.findings), key=lambda s: _ORDER.get(s, 9),
                   default=None)

    def to_dict(self) -> dict[str, Any]:
        return {"counts": self.counts(), "stats": self.stats, "seconds": self.seconds,
                "findings": [asdict(f) for f in self.findings]}


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _cap(report: Report, check: str, severity: str, rows: Iterable[tuple[str, Any]],
         detail: Callable[[str, Any], str], limit: int = 500) -> int:
    """Record findings, but never more than `limit` of one kind.

    A broken ingest can make every row fail one check; a report with 5,800 identical
    lines in it is unreadable and the count is the only part anybody needs."""
    n = 0
    for sym, val in rows:
        n += 1
        if n <= limit:
            report.add(check, severity, sym, val, detail(sym, val))
    if n > limit:
        report.add(check, severity, None, n,
                   f"...and {n - limit} more (only the first {limit} are listed)")
    return n


# ------------------------------------------------------------------ ratio bounds
def check_ratio_bounds(snap: pd.DataFrame, report: Report) -> None:
    """Ratios whose value is arithmetically impossible, not merely unusual.

    `grossMargins > 1` is the MSFT signature: gross profit read from a live tag over a
    revenue read from a tag retired in 2010. It cannot happen to a correctly-assembled
    filing, which is exactly what makes it a gate rather than a heuristic.
    """
    gm = _num(snap, "grossMargins")
    _cap(report, "gross_margin_impossible", CRITICAL,
         [(s, round(float(v), 3)) for s, v in gm[gm > 1.0].items()],
         lambda s, v: f"gross margin {v:.1%} — gross profit exceeds revenue "
                      f"(stale revenue tag?)")
    _cap(report, "gross_margin_absurd", ERROR,
         [(s, round(float(v), 3)) for s, v in gm[gm < -5.0].items()],
         lambda s, v: f"gross margin {v:.0%} — revenue is implausibly small vs cost")

    om = _num(snap, "operatingMargins")
    _cap(report, "operating_margin_impossible", ERROR,
         [(s, round(float(v), 3)) for s, v in om[om > 1.0].items()],
         lambda s, v: f"operating margin {v:.1%} > 100% of revenue")

    fm = _num(snap, "fcf_margin")
    _cap(report, "fcf_margin_impossible", ERROR,
         [(s, round(float(v), 3)) for s, v in fm[fm > 1.5].items()],
         lambda s, v: f"free cash flow is {v:.1f}x revenue")

    # A non-positive P/E must be NULL, not a number: `_MASK_NONPOS` masks it at ranking
    # time, but a negative value living in the store means `derive` produced one, and
    # anything reading the store outside `scoring` would rank it as the cheapest name
    # in the universe.
    pe = _num(snap, "trailingPE")
    _cap(report, "pe_nonpositive", CRITICAL,
         [(s, round(float(v), 2)) for s, v in pe[pe <= 0].items()],
         lambda s, v: f"trailingPE {v} — a loss-maker must carry NULL, not a number")

    rev = _num(snap, "totalRevenue")
    _cap(report, "revenue_negative", ERROR,
         [(s, float(v)) for s, v in rev[rev < 0].items()],
         lambda s, v: f"TTM revenue {v:,.0f} is negative")

    for col, lim in _MAX_ABS.items():
        v = _num(snap, col)
        _cap(report, f"{col}_out_of_bounds", WARN,
             [(s, round(float(x), 1)) for s, x in v[v.abs() > lim].items()],
             lambda s, x, c=col, l=lim: f"{c} {x} exceeds the plausible bound +/-{l:g}")


# ----------------------------------------------------------- internal consistency
def check_internal_consistency(snap: pd.DataFrame, prev: pd.DataFrame,
                               report: Report) -> None:
    """Invariants that hold between fields of the SAME row, and between a row and the
    same symbol one snapshot earlier.

    `marketCap == price x sharesOutstanding` is how `raw_asof_prepared` builds it, so a
    mismatch means something wrote a market cap from somewhere else — the exact way a
    USD cap ends up over a local-currency statement.

    The TTM-revenue break is the generic form of the MSFT bug: a trailing-twelve-month
    figure that suddenly reads about a QUARTER of the previous month's is a TTM that
    collapsed into a single period, which is what happens when the four-quarter chain
    breaks and the fallback picks up a lone annual or quarterly row.
    """
    mc, px, sh = _num(snap, "marketCap"), _num(snap, "price"), _num(snap, "sharesOutstanding")
    both = mc.notna() & px.notna() & sh.notna() & (mc != 0)
    rel = ((mc - px * sh).abs() / mc.abs()).where(both)
    _cap(report, "marketcap_inconsistent", CRITICAL,
         [(s, round(float(v), 4)) for s, v in rel[rel > 0.01].items()],
         lambda s, v: f"marketCap differs from price x shares by {v:.1%}")

    if prev is not None and not prev.empty:
        pr = _num(prev, "totalRevenue")
        cu = _num(snap, "totalRevenue")
        common = cu.index.intersection(pr.index)
        a, b = cu.loc[common], pr.loc[common]
        ok = a.notna() & b.notna() & (b > 0)
        ratio = (a / b).where(ok)
        quarterly = ratio[(ratio > 0.15) & (ratio < 0.40)]
        _cap(report, "ttm_revenue_collapse", CRITICAL,
             [(s, round(float(v), 3)) for s, v in quarterly.items()],
             lambda s, v: f"TTM revenue fell to {v:.0%} of the previous snapshot — "
                          f"a TTM that became one quarter (stale/switched tag)")
        other = ratio[((ratio <= 0.15) | ((ratio >= 0.40) & (ratio < 0.5)))
                      | (ratio > 2.5)]
        _cap(report, "ttm_revenue_break", ERROR,
             [(s, round(float(v), 3)) for s, v in other.items()],
             lambda s, v: f"TTM revenue moved to {v:.0%} of the previous snapshot")

        ps, cs = _num(prev, "sharesOutstanding"), _num(snap, "sharesOutstanding")
        pm, cm = _num(prev, "marketCap"), _num(snap, "marketCap")
        common = cs.index.intersection(ps.index)
        a, b = cs.loc[common], ps.loc[common]
        ok = a.notna() & b.notna() & (b > 0)
        r = (a / b).where(ok)
        # The share count moving is not by itself evidence: a count and a price that
        # move together leave the market cap continuous, which is what a re-tagging or
        # a correctly-handled split looks like. The finding is a count that moves while
        # the price does NOT, so the market cap jumps with it.
        #
        # That second case includes a REAL defect this check found: the price series is
        # retroactively split-adjusted (yfinance restates `close` back through history)
        # while the SEC share count is point-in-time and stays pre-split until the
        # filing that restates it. So for a splitting company every market cap between
        # the split and that filing is wrong by the split factor — CrowdStrike's
        # 4-for-1 and KLA's 10-for-1 are both in the store right now.
        mr = (cm.loc[common] / pm.loc[common]).where(ok & (pm.loc[common] > 0))
        jump = r[((r > 3.0) | (r < 0.34)) & ((mr > 2.0) | (mr < 0.5) | mr.isna())]
        _cap(report, "share_count_jump", ERROR,
             [(s, round(float(v), 2)) for s, v in jump.items()],
             lambda s, v: f"shares outstanding moved {v:.1f}x since the previous "
                          f"cross-section and the market cap moved with it — a units "
                          f"error, a class double-count, or a split the price series "
                          f"already restated and the share count has not")


# ---------------------------------------------------------------------- currency
def check_currency(snap: pd.DataFrame, report: Report) -> None:
    """Non-USD reporters must not carry a USD-denominated multiple.

    `derive` masks `_CCY_SENSITIVE` for them because a USD market cap over a TWD income
    statement is not a ratio — it is the bug that gave TSM a P/E of 206 and a false
    `higher-risk` tag. A surviving value here means the masking did not run, or ran
    against a currency the row does not actually report in.

    The mirror image is checked too: if the store holds NO non-USD reporter at all, the
    IFRS/currency path is dead and the whole 20-F cohort has quietly vanished. That is
    the ASML failure — it does not show up as a wrong number, it shows up as an absence.
    """
    if "financial_currency" not in snap.columns:
        report.add("currency_column_missing", CRITICAL, None, None,
                   "snapshots carry no financial_currency — ADR masking cannot work")
        return
    ccy = snap["financial_currency"].astype("string")
    foreign = snap[ccy.notna() & (ccy != "USD")]
    report.stats["nonusd_symbols"] = int(len(foreign))
    report.stats["currencies"] = (ccy.value_counts().head(12).to_dict())
    # Only meaningful on a real cross-section: a handful of rows can legitimately be
    # all-domestic, and a gate that cries wolf on a small store gets switched off.
    if foreign.empty and len(snap) >= 200:
        report.add("nonusd_filers_absent", CRITICAL, None, 0,
                   "not one non-USD reporter in the store — the IFRS/currency ingest "
                   "path is dead (this is the ASML/TSM signature)")
    leaked: list[tuple[str, Any]] = []
    for col in _CCY_SENSITIVE:
        if col not in foreign.columns:
            continue
        v = pd.to_numeric(foreign[col], errors="coerce")
        for s, x in v[v.notna()].items():
            leaked.append((s, f"{col}={float(x):.3g}"))
    _cap(report, "ccy_sensitive_leak", CRITICAL, leaked,
         lambda s, v: f"non-USD reporter carries a USD-denominated multiple {v}")

    rev = _num(snap, "totalRevenue")
    missing = snap.index[rev.notna() & ccy.isna()]
    _cap(report, "currency_unknown", WARN, [(s, None) for s in missing],
         lambda s, v: "revenue present but no reporting currency recorded")


# ---------------------------------------------------------------------- coverage
def check_coverage(snap: pd.DataFrame, sectors: dict[str, str], baseline: dict,
                   report: Report) -> None:
    """Per-sector share of names the fundamental pillar can actually score, against a
    stored baseline.

    This is the check TSM and ASML would have tripped. Both failures made a name's
    fundamentals VANISH; nothing was wrong with any number that remained, so no bound
    could see it. Only "there used to be more of this and now there is less" can.
    """
    from .. import scoring
    subs = {s: "" for s in snap.index}
    det = scoring.fundamental_detail(snap, subs, sectors, tag_inputs=False)
    pillar = det["pillar"]
    sec = pd.Series({s: sectors.get(s) or scoring.UNKNOWN_SECTOR for s in snap.index})
    have = pillar.notna()
    cov: dict[str, dict[str, int]] = {}
    for s in sorted(set(sec)):
        m = sec == s
        cov[s] = {"n": int(m.sum()), "scored": int((m & have).sum())}
    report.stats["coverage"] = cov
    report.stats["scored"] = int(have.sum())
    report.stats["symbols"] = int(len(snap))

    base = (baseline or {}).get("coverage") or {}
    for s, cur in cov.items():
        b = base.get(s)
        if not b or not b.get("scored"):
            continue
        drop = 1.0 - (cur["scored"] / b["scored"])
        if drop > _COVERAGE_DROP:
            report.add("sector_coverage_regression", CRITICAL, None,
                       f"{s}:{b['scored']}->{cur['scored']}",
                       f"{s}: names with a fundamental pillar fell {drop:.0%} "
                       f"({b['scored']} -> {cur['scored']}) since the baseline")
    for s, b in base.items():
        if s not in cov and b.get("scored"):
            report.add("sector_disappeared", CRITICAL, None, s,
                       f"sector {s} had {b['scored']} scored names in the baseline and "
                       f"has none now")


def check_totals(store, baseline: dict, report: Report) -> None:
    """Store-wide row counts against the baseline. A rebuild that halves the fact table
    is a data loss with no wrong number anywhere in it."""
    man = store.manifest() or {}
    cur = {k: int((man.get(k) or {}).get("rows") or 0)
           for k in ("prices", "facts", "snapshots")}
    cur["fact_symbols"] = int((man.get("facts") or {}).get("symbols") or 0)
    report.stats["totals"] = cur
    base = (baseline or {}).get("totals") or {}
    for k, v in cur.items():
        b = int(base.get(k) or 0)
        if b and v < b * (1 - _TOTALS_DROP):
            report.add("store_shrank", CRITICAL, None, f"{k}:{b}->{v}",
                       f"{k} fell {1 - v / b:.0%} ({b:,} -> {v:,}) since the baseline")


# ------------------------------------------------------------- facts-level checks
def check_facts(store, symbols: list[str], as_of: date, report: Report) -> None:
    """Checks that need the raw fact table rather than a derived snapshot.

    THIN FACTS is the TSM signature stated directly: a real filer publishes hundreds of
    mapped facts; nine means the parser found almost nothing and everything derived
    from it is an accident.

    MISSING CAPEX is the D/DLR signature. `raw_asof_prepared` computes
    `freeCashflow = cfo - |capex|` and falls back to `cfo` when no capex tag resolves,
    so a capex-heavy filer whose capex tag is not in `FLOW_TAGS` reads FCF-POSITIVE.
    Nothing is out of bounds; the number is simply the wrong quantity. Note that the
    `fcf <= cfo` invariant itself cannot be violated by that code path — subtraction of
    an absolute value guarantees it — which is why the DETECTABLE form of the bug is
    the missing tag, not the broken inequality.
    """
    facts = store.facts(symbols, columns=["symbol", "concept", "filed", "end"])
    if facts.empty:
        report.add("facts_empty", CRITICAL, None, 0, "the fact table is empty")
        return
    n_rows = facts.groupby("symbol", sort=False).size()
    last = facts.groupby("symbol", sort=False)["filed"].max()
    report.stats["fact_symbols_seen"] = int(len(n_rows))

    want = set(symbols)
    absent = sorted(want - set(n_rows.index))
    _cap(report, "no_facts_at_all", ERROR, [(s, 0) for s in absent],
         lambda s, v: "in the universe but the store holds no SEC facts for it")

    thin = n_rows[n_rows < _MIN_FACT_ROWS]
    _cap(report, "thin_facts", ERROR, [(s, int(v)) for s, v in thin.items()],
         lambda s, v: f"only {v} mapped fact rows — a real filer publishes hundreds "
                      f"(namespace/parse failure?)")

    ts = pd.Timestamp(as_of)
    age = (ts - last).dt.days
    _cap(report, "facts_dead", ERROR, [(s, int(v)) for s, v in age[age > _FACTS_DEAD_DAYS].items()],
         lambda s, v: f"latest filing is {v} days old — not an active filer")
    stale = age[(age > _FACTS_STALE_DAYS) & (age <= _FACTS_DEAD_DAYS)]
    _cap(report, "facts_stale", WARN, [(s, int(v)) for s, v in stale.items()],
         lambda s, v: f"latest filing is {v} days old (a quarterly filer should be <120)")

    # A CONCEPT whose freshest period is years behind the rest of the same filer's
    # statements is a retired tag still returning its last value. That is the MSFT bug
    # in its general form (revenue pinned to a tag abandoned in 2010 while gross profit
    # stayed current) and the Dominion capex case in its specific one (its capex tag
    # stops in 2020, so FCF = today's CFO minus 2019's capex). `_best()` picks the
    # freshest TAG within a concept but has no way to notice that every tag for one
    # concept is stale.
    core = ("totalRevenue", "netIncome", "cfo", "capex", "grossProfits",
            "operatingIncome", "dna")
    ends = (facts[facts["concept"].isin(core)]
            .groupby(["symbol", "concept"], sort=False)["end"].max().unstack())
    if not ends.empty:
        newest = ends.max(axis=1)
        lag = ends.apply(lambda c: (newest - c).dt.days)
        stale_pairs = [(sym, f"{con}:{int(v)}d")
                       for con in lag.columns
                       for sym, v in lag[con][lag[con] > _CONCEPT_LAG_DAYS].items()]
        _cap(report, "concept_tag_retired", ERROR, sorted(stale_pairs),
             lambda s, v: f"{v.split(':')[0]} last reported {v.split(':')[1]} before "
                          f"this filer's newest statement — a retired tag still "
                          f"returning its final value")

    # capex presence, over the window a TTM can reach back into
    recent = store.facts(symbols, start=(ts - pd.Timedelta(days=1100)).date(),
                         columns=["symbol", "concept"])
    if recent.empty:
        return
    by = recent.groupby(["symbol", "concept"]).size().unstack(fill_value=0)
    if "cfo" not in by.columns:
        report.add("cfo_concept_absent", CRITICAL, None, 0,
                   "no cash-flow-from-operations facts anywhere in the recent window")
        return
    capex = by["capex"] if "capex" in by.columns else pd.Series(0, index=by.index)
    gap = by.index[(by["cfo"] > 0) & (capex == 0)]
    report.stats["capex_missing"] = int(len(gap))
    _cap(report, "capex_tag_missing", ERROR, [(s, 0) for s in gap],
         lambda s, v: "reports operating cash flow but no capex tag — freeCashflow "
                      "silently falls back to CFO and reads too high")


# ----------------------------------------------------------------- floor blockers
def check_floor_blockers(store, candidates: list[str], as_of: date,
                         report: Report, snap: pd.DataFrame | None = None) -> None:
    """Names that are priced and liquid but carry no share count, or a wrong one.

    The point-in-time floor is `price x sharesOutstanding`, so a missing share count
    is a NULL market cap and the name fails the floor SILENTLY — it does not appear
    anywhere, and no ratio is wrong because no ratio exists. That has already happened
    twice in this project: META/WMT/BRK published only a weighted-average share count
    (a duration, dropped as flow-shaped), and Berkshire tags shares only per share
    class. Both were found by a human noticing a mega-cap was absent from a board.

    Liquidity is the filter that makes this actionable: thousands of warrants and
    preferreds legitimately have no share count, and one real name hidden among them
    is a finding nobody reads.
    """
    from .. import config
    if not candidates:
        return
    ts = pd.Timestamp(as_of)
    px = store.prices(candidates, start=(ts - pd.Timedelta(days=45)).date(),
                      end=as_of, columns=["close", "volume"])
    if px.empty:
        return
    g = px.assign(_dv=px["close"] * px["volume"]).groupby("symbol", sort=False)
    adv = g["_dv"].mean()
    min_adv = float(config.get("universe.liquidity_floor.min_avg_dollar_volume", 5e6) or 0)
    min_price = float(config.get("universe.liquidity_floor.min_price", 5.0) or 0)
    liquid = adv[(adv >= min_adv) & (g["close"].last() >= min_price)]
    if liquid.empty:
        return
    shares = store.shares_asof(sorted(liquid.index), as_of)
    blocked = sorted(set(liquid.index) - set(shares[shares > 0].index))
    report.stats["floor_blocked"] = len(blocked)
    _cap(report, "no_share_count", ERROR,
         [(s, round(float(liquid[s]) / 1e6, 1)) for s in blocked],
         lambda s, v: f"${v}M/day of volume but no share count knowable — null market "
                      f"cap, so it fails the liquidity floor with no trace")

    # A WRONG share count is worse than a missing one, and it looks like nothing: the
    # market cap is simply small. A company cannot be worth less than a few weeks of
    # its own trading, so `marketCap / ADV` is a scale-free way to catch a count that
    # is off by orders of magnitude. Measured cases: Amcor 13,001 shares (a 2019 row
    # outranking a current one) against $100m/day, Genius Sports 1.7m against $30m/day.
    if snap is None or snap.empty:
        return
    mc = _num(snap, "marketCap")
    common = mc.index.intersection(liquid.index)
    ratio = (mc.loc[common] / liquid.loc[common]).dropna()
    bad = ratio[ratio < _MCAP_ADV_FLOOR]
    _cap(report, "market_cap_implausible", ERROR,
         [(s, round(float(v), 1)) for s, v in bad.items()],
         lambda s, v: f"market cap is only {v:.0f}x its average DAILY dollar volume — "
                      f"the share count is wrong by orders of magnitude")


# --------------------------------------------------------------------- staleness
def check_staleness(store, snap_date: pd.Timestamp | None, as_of: date,
                    report: Report) -> None:
    ts = pd.Timestamp(as_of)
    if snap_date is None:
        report.add("snapshots_empty", CRITICAL, None, None, "no snapshots in the store")
    else:
        age = (ts - snap_date).days
        report.stats["snapshot_age_days"] = int(age)
        if age > _SNAPSHOT_FRESH_DAYS:
            report.add("snapshots_stale", CRITICAL, None, age,
                       f"newest snapshot is {age} days old — past the {_SNAPSHOT_FRESH_DAYS}d "
                       f"freshness cut, so the live screen would be served NOTHING")
        elif age > 35:
            report.add("snapshots_stale", WARN, age, age,
                       f"newest snapshot is {age} days old")
    last_px = store.last_date()
    if last_px is None:
        report.add("prices_empty", CRITICAL, None, None, "no prices in the store")
        return
    age = (ts - last_px).days
    report.stats["price_age_days"] = int(age)
    if age > 10:
        report.add("prices_stale", ERROR if age > 30 else WARN, None, age,
                   f"newest price bar is {age} days old ({last_px.date()})")


# ------------------------------------------------------------------ cross-source
# `.info` is the SUSPECT side by default and the messages say so. Its `freeCashflow` is
# a single quarter (verified: MSFT 16.5B vs SEC TTM 67.0B), and its growth fields are
# most-recent-quarter year-over-year where the SEC path is TTM vs TTM. So a
# disagreement is evidence about the store only for the fields where `.info` is sound:
# market cap, revenue, and the multiples Yahoo computes in USD for an ADR.
_CROSS_FIELDS = {"marketCap": 0.10, "totalRevenue": 0.15, "priceToBook": 0.35}
_CROSS_SUSPECT = ("freeCashflow", "revenueGrowth", "earningsGrowth")


def check_cross_source(snap: pd.DataFrame, symbols: list[str], report: Report,
                       fetch: Callable[[list[str]], dict[str, dict]] | None = None,
                       ) -> None:
    """Compare the SEC-derived snapshot with yfinance `.info` on a sample.

    Network-bound, so it is opt-in and runs on a sample, never the whole store.
    """
    if fetch is None:
        # `fetch_many`, not `enrich`: enrich returns only the handful of estimate
        # fields the live screen still needs from Yahoo (forwardPE, pegRatio, the
        # earnings timestamp) and carries neither market cap nor revenue nor free cash
        # flow — the three quantities there is anything to disagree about.
        from .. import fundamentals
        fetch = lambda syms: fundamentals.fetch_many(syms, progress=False)  # noqa: E731
    info = fetch(symbols) or {}
    report.stats["cross_source_sampled"] = len(info)
    for sym, row in info.items():
        if sym not in snap.index or not row:
            continue
        s = snap.loc[sym]
        for col, tol in _CROSS_FIELDS.items():
            a, b = pd.to_numeric(pd.Series([s.get(col)]), errors="coerce").iloc[0], row.get(col)
            try:
                b = float(b)
            except (TypeError, ValueError):
                continue
            if a != a or not b:
                continue
            rel = abs(a - b) / abs(b)
            if rel > tol:
                report.add("cross_source_disagreement", WARN, sym,
                           f"{col} sec={a:.4g} info={b:.4g}",
                           f"{col} differs from yfinance by {rel:.0%} "
                           f"(.info is the suspect side unless corroborated)")
        for col in _CROSS_SUSPECT:
            a = pd.to_numeric(pd.Series([s.get(col)]), errors="coerce").iloc[0]
            b = row.get(col)
            try:
                b = float(b)
            except (TypeError, ValueError):
                continue
            if a != a or not b:
                continue
            ratio = a / b
            if col == "freeCashflow" and 2.5 < ratio < 6.0:
                report.add("info_fcf_is_quarterly", WARN, sym,
                           round(float(ratio), 2),
                           f"SEC TTM freeCashflow is {ratio:.1f}x .info's — .info "
                           f"reports ONE QUARTER; do not treat .info as the reference")


# --------------------------------------------------------------------------- run
def baseline_path(store):
    return store.root / BASELINE_NAME


def load_baseline(store) -> dict:
    p = baseline_path(store)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_baseline(store, report: Report) -> dict:
    """Freeze the current coverage and totals as the thing future runs must not fall
    below. Written only on request: a gate that re-baselines itself every run cannot
    detect a regression, it can only ratify one."""
    data = {"written_at": date.today().isoformat(),
            "coverage": report.stats.get("coverage", {}),
            "totals": report.stats.get("totals", {}),
            "scored": report.stats.get("scored"),
            "symbols": report.stats.get("symbols"),
            "no_snapshot": report.stats.get("no_snapshot"),
            "nonusd_symbols": report.stats.get("nonusd_symbols")}
    baseline_path(store).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def run(store, symbols: list[str] | None = None, as_of: date | None = None,
        sectors: dict[str, str] | None = None, baseline: dict | None = None,
        cross_source: int = 0,
        cross_fetch: Callable[[list[str]], dict[str, dict]] | None = None) -> Report:
    """Every check, over the newest cross-section the store holds.

    `symbols` defaults to whatever the newest snapshot date covers, so the gate is
    honest about the store as it IS rather than as a config file says it should be.
    """
    import time
    t0 = time.time()
    d = as_of or date.today()
    report = Report()

    # The SCREENED universe is the denominator: a name that fell out of the store
    # entirely has no snapshot row to be checked, so judging the store only by the rows
    # it happens to contain is exactly the blind spot ASML fell through. Recorded
    # membership (the floored, investable list) is the right denominator rather than
    # the raw candidate table — the candidate table carries warrants and preferreds
    # that legitimately never file, and drowning the report in those teaches people to
    # ignore it.
    screened: list[str] = list(symbols or [])
    if not screened:
        try:
            from .. import universe
            screened = store.universe_asof(d, source=universe.membership_source())
            if not screened:
                screened = universe.symbols()
        except Exception:  # noqa: BLE001 - no registrant table is not a reason to fail
            screened = []
    symbols = None

    snaps = store.snapshots(symbols=symbols,
                            start=(pd.Timestamp(d) - pd.Timedelta(days=400)).date(),
                            end=d)
    if snaps.empty:
        report.add("snapshots_empty", CRITICAL, None, None,
                   "no snapshots in the last 400 days")
        report.seconds = time.time() - t0
        return report
    # The newest date can be a PARTIAL rebuild (one symbol re-run today), and so can
    # the one before it. Both the current cross-section and the one it is compared
    # against must cover most of the store, or the month-over-month checks silently
    # run on a few hundred names and report almost nothing — which reads exactly like
    # "clean".
    counts = snaps.groupby("as_of").size()
    full = counts[counts >= counts.max() * 0.8]
    latest = max(full.index)
    # ...and the comparison date has to be a MONTH back, not merely the previous full
    # cross-section. The store carries daily snapshots alongside the month-end grid, so
    # "the one before" is usually yesterday — and nothing detectable changes in a day,
    # so every month-over-month check would quietly find nothing and read as clean.
    prev_dates = [x for x in full.index
                  if x <= latest - pd.Timedelta(days=_PREV_MIN_GAP_DAYS)]
    prev = (snaps[snaps["as_of"] == max(prev_dates)].set_index("symbol")
            if prev_dates else pd.DataFrame())
    snap = snaps[snaps["as_of"] == latest].set_index("symbol")
    report.stats["as_of"] = str(pd.Timestamp(latest).date())
    report.stats["prev_as_of"] = str(pd.Timestamp(max(prev_dates)).date()) if prev_dates else None

    syms = sorted(snap.index)
    known = sorted(set(screened) | set(syms)) if screened else syms
    report.stats["universe_symbols"] = len(screened)
    if sectors is None:
        from .. import universe
        sectors = universe.sector_map(store=store, symbols=known)
    baseline = load_baseline(store) if baseline is None else baseline

    missing = sorted(set(screened) - set(syms))
    report.stats["no_snapshot"] = len(missing)
    _cap(report, "no_snapshot", ERROR, [(s, None) for s in missing],
         lambda s, v: "screened by the live pipeline but carries no snapshot row — it "
                      "can never be scored, and it fails silently")
    b_ns = (baseline or {}).get("no_snapshot")
    if b_ns is not None and len(missing) > max(b_ns * 1.1, b_ns + 25):
        report.add("no_snapshot_regression", CRITICAL, None,
                   f"{b_ns}->{len(missing)}",
                   f"names with no snapshot rose {b_ns} -> {len(missing)} since the "
                   f"baseline — an ingest change dropped a cohort")

    check_ratio_bounds(snap, report)
    check_internal_consistency(snap, prev, report)
    check_currency(snap, report)
    check_coverage(snap, sectors, baseline, report)
    check_totals(store, baseline, report)
    check_facts(store, sorted(screened) or syms, d, report)
    try:
        from .. import universe
        check_floor_blockers(store, universe.symbols(), d, report, snap=snap)
    except Exception as exc:  # noqa: BLE001 - no candidate table is not a failure
        report.add("candidates_unavailable", WARN, None, str(exc),
                   f"could not read the candidate table: {exc}")
    check_staleness(store, pd.Timestamp(latest), d, report)
    if cross_source:
        sample = syms[:: max(1, len(syms) // cross_source)][:cross_source]
        try:
            check_cross_source(snap, sample, report, fetch=cross_fetch)
        except Exception as exc:  # noqa: BLE001 - a network check never fails the gate
            report.add("cross_source_unavailable", WARN, None, str(exc),
                       f"cross-source comparison could not run: {exc}")
    report.seconds = time.time() - t0
    return report
