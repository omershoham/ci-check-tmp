"""Bulk point-in-time fundamentals from the SEC's Financial Statement Data Sets.

`sec_facts` fetches one companyfacts JSON per CIK. That is the right trade at 105
names and the wrong one at 1,500: payloads run 1-30 MB each, so a broad universe
means tens of GB and hours of parsing. DERA publishes the same underlying data as
one ~55 MB ZIP per quarter covering EVERY filer, which is O(1) in universe size.

    https://www.sec.gov/files/dera/data/financial-statement-data-sets/YYYYqQ.zip
      sub.txt   one row per submission: adsh, cik, name, sic, form, period, filed
      num.txt   one row per reported number: adsh, tag, version, ddate, qtrs, uom,
                segments, coreg, value          (~600 MB uncompressed)

Output schema is IDENTICAL to `sec_facts.facts_frame`, so `prepare_symbol`,
`raw_asof_prepared` and every downstream consumer are untouched.

Point-in-time correctness rules enforced here (each one is a way to get it wrong):

  * `filed` comes from sub.txt and is the real filing date. Keying on it - never on
    the period end - is the entire reason this dataset works.
  * Keep ONLY rows with empty `segments` AND empty `coreg`. Without this you sum
    segment-level and subsidiary-level revenue into consolidated revenue and every
    margin ratio is silently garbage.
  * Do NOT filter on `prevrpt`. prevrpt=1 means the submission was later amended;
    the original filing is still exactly what was knowable at the time. Restatement
    handling already happens downstream in `_Tab.dedupe_by_end`, which keeps the
    latest filing per period among those filed on or before the as-of date.
  * Do NOT filter on `detail`. detail=0 only means footnotes were not tagged; the
    primary-statement tags this reads are present either way.
  * A ZIP is keyed by the quarter of FILING, not of the period: a FY2023 10-K filed
    2024-02-25 lives in 2024q1.zip.
"""
from __future__ import annotations

import re
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import requests

from .. import config, news
from . import sec_facts as SF

_ZIP_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip"
_QUARTER_RE = re.compile(r"^(\d{4})q([1-4])$")

# Namespaces we read. `version` in num.txt looks like "us-gaap/2025" or "ifrs/2025".
# NOTE the naming difference between the two SEC datasets: companyfacts keys the IFRS
# taxonomy as "ifrs-full", FSDS versions it as "ifrs". Accept both or every foreign
# private issuer (TSM and the 20-F cohort) silently yields nothing.
_NS = ("us-gaap", "ifrs", "ifrs-full", "dei")

# num.txt is ~600 MB; read it in pieces and drop all but the ~30 tags we use before
# anything else. That is what keeps peak memory in the hundreds of MB.
_CHUNK = 500_000


def tag_to_concept() -> dict[str, str]:
    """Reverse of sec_facts' concept -> tag fallback lists. Built from those dicts so
    the two ingest paths can never drift apart on which tags mean what."""
    out: dict[str, str] = {}
    for concept, tags in SF.FLOW_TAGS.items():
        for t in tags:
            out.setdefault(t, concept)
    for concept, tags in SF.INSTANT_TAGS.items():
        for t in tags:
            out.setdefault(t, concept)
    for t in SF.SHARES_TAGS:
        out.setdefault(t, "sharesOutstanding")
    return out


def cache_dir() -> Path:
    d = Path(config.cache_dir()) / "fsds"
    d.mkdir(parents=True, exist_ok=True)
    return d


def quarters_between(start: str, end: str | None = None) -> list[str]:
    """Inclusive list of 'YYYYqQ' labels. `end` defaults to the current quarter."""
    def parse(q: str) -> tuple[int, int]:
        m = _QUARTER_RE.match(q.lower())
        if not m:
            raise ValueError(f"bad quarter label {q!r}, expected e.g. '2015q1'")
        return int(m.group(1)), int(m.group(2))

    y0, q0 = parse(start)
    if end:
        y1, q1 = parse(end)
    else:
        today = date.today()
        y1, q1 = today.year, (today.month - 1) // 3 + 1
    out = []
    y, q = y0, q0
    while (y, q) <= (y1, q1):
        out.append(f"{y}q{q}")
        q += 1
        if q > 4:
            y, q = y + 1, 1
    return out


def download_quarter(q: str, force: bool = False, timeout: int = 600) -> Path | None:
    """Fetch one quarterly ZIP into the cache. Returns None when SEC has no such file
    (the current quarter is published with a 1-2 month lag)."""
    path = cache_dir() / f"{q}.zip"
    if path.exists() and not force and path.stat().st_size > 0:
        return path
    try:
        resp = requests.get(_ZIP_URL.format(q=q), headers=news._sec_headers(),
                            timeout=timeout, stream=True)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    tmp = path.with_suffix(".part")
    with open(tmp, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
    tmp.replace(path)
    return path


def _is_class_only(segments: pd.Series) -> pd.Series:
    """True for a segments string whose ONLY axis is the share class.

    `segments` is a ";"-joined list of `Axis=Member` pairs. `ClassOfStock` is the one
    axis whose members partition the company's shares, so its parts sum to the total.
    Every other axis on a share tag (`EquityComponents`, `ConsolidatedEntities`,
    `Restatement`, `LegalEntity`...) slices the SAME shares a different way and summing
    it would multiply the count. Requiring EVERY axis to be `ClassOfStock` — not merely
    one of them — is what keeps "Class B, as previously reported" out of the sum.
    """
    s = segments.fillna("")
    parts = s.str.rstrip(";").str.split(";")
    return s.ne("") & parts.apply(
        lambda ps: all(p.split("=", 1)[0] == "ClassOfStock" for p in ps if p))


def _fold_share_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve share rows down to one consolidated count per filing + period.

    Berkshire is the case this exists for: it tags shares ONLY per class with a
    non-empty `segments`, so the consolidated-only PIT rule discarded every share row
    it publishes — no share count, no market cap, no floor pass, and BRK-B was absent
    from the universe with nothing anywhere saying why.

    Three rules, each of which a real filing breaks if it is missing:

      * A consolidated row WINS whenever one exists. Alphabet publishes both the class
        breakdown and the CIK total (12,211M = 5,835 + 861 + 5,515); summing on top of
        the total would double every Alphabet market cap ever computed.
      * A class breakdown is used ONLY when every segmented row for that filing and
        period sits on the `ClassOfStock` axis. Regeneron is why: its
        `CommonStockSharesOutstanding` is tagged `ClassOfStock=CommonClassA` (1.8m) and
        `EquityComponents=CommonStock` (137.9m). Those are two different axes over the
        same shares, so "the classes" is not a partition of anything — summing the one
        recognisable axis gave Regeneron 1.8m shares and a market cap 77x too small.
        When the axes are mixed, the honest answer is no share count at all.
      * Within a usable group, a member naming two classes at once
        ("CommonClassAAndClassB") is dropped when the individual classes are also
        present — 5% of class groups carry both, and adding them counts the same
        shares twice.

    CAVEAT, recorded because it is a real inaccuracy: the sum is of SHARE COUNTS, and
    classes with unequal economics do not add up to an economic share count. Berkshire
    is the extreme (an A share is 1,500 B shares); its filings happen to report both
    classes in "equivalent" units, so the sum is right to within 0.07%, but a filer
    with genuinely unequal classes would produce a market cap in the wrong proportion.
    That is still strictly better than the name being invisible.
    """
    if df.empty or "segments" not in df.columns:
        return df
    is_sh = df["concept"] == "sharesOutstanding"
    seg = df["segments"].fillna("")
    # Non-share rows must still be consolidated-only: relaxing that for anything else
    # re-opens the #1 FSDS trap (segment revenue summed into the total).
    rest = df[~is_sh & seg.eq("")]
    sh = df[is_sh]
    if sh.empty:
        return rest.reset_index(drop=True)
    seg = seg.loc[sh.index]
    key = ["adsh", "tag", "end"]
    cons = sh[seg.eq("")]
    cls = sh[seg.ne("")]
    if cls.empty:
        return pd.concat([rest, cons], ignore_index=True)

    ktup = cls[key].apply(tuple, axis=1)
    have_cons = set(map(tuple, cons[key].itertuples(index=False, name=None)))
    cls, ktup = cls[~ktup.isin(have_cons)], ktup[~ktup.isin(have_cons)]
    if cls.empty:
        return pd.concat([rest, cons], ignore_index=True)

    # A group is usable only if EVERY one of its segmented rows is class-only.
    class_only = _is_class_only(cls["segments"])
    usable = {k for k, ok in class_only.groupby(ktup).all().items() if ok}
    keep = ktup.isin(usable)
    cls, ktup = cls[keep], ktup[keep]
    if cls.empty:
        return pd.concat([rest, cons], ignore_index=True)

    combined = cls["segments"].fillna("").str.contains("And", case=True)
    singles = set(map(tuple, cls.loc[~combined, key].itertuples(index=False, name=None)))
    cls = cls[~(combined & ktup.isin(singles))]
    agg = {c: "first" for c in cls.columns if c not in key + ["val"]}
    folded = cls.groupby(key, as_index=False).agg({**agg, "val": "sum"})
    return pd.concat([rest, cons, folded[df.columns]], ignore_index=True)


def _submissions(z: zipfile.ZipFile) -> pd.DataFrame:
    with z.open("sub.txt") as fh:
        sub = pd.read_csv(fh, sep="\t", dtype=str, usecols=[
            "adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed"],
            encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")
    sub["cik"] = sub["cik"].astype(str).str.zfill(10)
    sub["filed"] = pd.to_datetime(sub["filed"], format="%Y%m%d", errors="coerce")
    return sub.dropna(subset=["filed"])


def _numbers(z: zipfile.ZipFile, keep_tags: set[str],
             keep_ciks: set[str] | None, sub: pd.DataFrame) -> pd.DataFrame:
    """Rows of num.txt for the tags (and optionally CIKs) we care about."""
    adsh_ok: set[str] | None = None
    if keep_ciks is not None:
        adsh_ok = set(sub.loc[sub["cik"].isin(keep_ciks), "adsh"])
        if not adsh_ok:
            return pd.DataFrame()
    parts = []
    with z.open("num.txt") as fh:
        for chunk in pd.read_csv(fh, sep="\t", dtype=str, usecols=[
                "adsh", "tag", "version", "ddate", "qtrs", "uom", "segments",
                "coreg", "value"], chunksize=_CHUNK, encoding="utf-8",
                encoding_errors="replace", on_bad_lines="skip"):
            c = chunk[chunk["tag"].isin(keep_tags)]
            if adsh_ok is not None:
                c = c[c["adsh"].isin(adsh_ok)]
            if c.empty:
                continue
            # Consolidated figures only: a non-empty segments/coreg row is a slice of
            # the company, and summing those into the total corrupts every ratio.
            # The ONE exception is the share count, where the "slices" are share
            # CLASSES and their sum IS the consolidated total — see `_fold_share_classes`.
            # Segmented SHARE rows are carried through and resolved later by
            # `_fold_share_classes`, which needs to see the rows on OTHER axes too:
            # whether a share-class breakdown is the whole company or a fragment can
            # only be decided by looking at its siblings.
            keep = c["coreg"].fillna("").eq("") & (
                c["segments"].fillna("").eq("") | c["tag"].isin(SF.SHARES_TAGS))
            c = c[keep]
            c = c[c["version"].fillna("").str.split("/").str[0].isin(_NS)]
            c = c[c["value"].notna()]
            if not c.empty:
                parts.append(c)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def parse_quarter(zip_path: Path, keep_ciks: set[str] | None = None,
                  cik_to_symbol: dict[str, str] | None = None,
                  keep_tags: set[str] | None = None) -> pd.DataFrame:
    """One quarter's ZIP -> the same long frame shape as sec_facts.facts_frame.

    `keep_tags` narrows the parse to a subset of the mapped tags (the broad-universe
    share-count ingest wants only the four SHARES_TAGS across every filer)."""
    t2c = tag_to_concept()
    with zipfile.ZipFile(zip_path) as z:
        sub = _submissions(z)
        num = _numbers(z, set(keep_tags or t2c), keep_ciks, sub)
    if num.empty:
        return pd.DataFrame()

    df = num.merge(sub[["adsh", "cik", "form", "fy", "fp", "filed"]], on="adsh", how="inner")
    if df.empty:
        return pd.DataFrame()

    df["concept"] = df["tag"].map(t2c)
    df["end"] = pd.to_datetime(df["ddate"], format="%Y%m%d", errors="coerce")
    df["val"] = pd.to_numeric(df["value"], errors="coerce")
    df["qtrs"] = pd.to_numeric(df["qtrs"], errors="coerce")
    df = df.dropna(subset=["end", "val", "qtrs", "concept"])

    # `start` is the one derived field: FSDS gives a period END plus a length in
    # quarters, companyfacts gives an explicit start. qtrs=0 is an instant (NaT).
    df["start"] = pd.NaT
    flow = df["qtrs"] > 0
    # qtrs takes a handful of small integer values, so offset by group rather than
    # building a DateOffset per row (which pandas cannot vectorise).
    for n in sorted(df.loc[flow, "qtrs"].unique()):
        m = flow & (df["qtrs"] == n)
        df.loc[m, "start"] = df.loc[m, "end"] - pd.DateOffset(months=3 * int(n))

    # sharesOutstanding must reach the store as an INSTANT — `prepare_symbol` reads it
    # from the instant table only. The preferred tags already are instants; the
    # last-resort fallback, WeightedAverageNumberOf*, is tagged as a duration and was
    # therefore being dropped here. That is not cosmetic: Walmart, Meta and Berkshire
    # publish NO instant share count in num.txt, so the whole class of filers that tags
    # only the weighted average came out of the broad universe with a null market cap
    # and silently failed the liquidity floor. Re-stamp those as instants at their
    # period end (`SHARES_TAGS` already ranks them last, so a real instant still wins).
    wavg = df["tag"].str.startswith("WeightedAverageNumber", na=False)
    df.loc[(df["concept"] == "sharesOutstanding") & flow & wavg, "start"] = pd.NaT
    df = df[~((df["concept"] == "sharesOutstanding") & flow & ~wavg)]
    # Monetary tags must carry a currency; the share count must carry "shares".
    is_shares = df["concept"] == "sharesOutstanding"
    df = df[(is_shares & df["uom"].eq("shares"))
            | (~is_shares & df["uom"].str.fullmatch(r"[A-Z]{3}", na=False))]
    df = _fold_share_classes(df)

    if cik_to_symbol:
        df["symbol"] = df["cik"].map(cik_to_symbol)
        df = df.dropna(subset=["symbol"])
    else:
        df["symbol"] = None

    out = df.rename(columns={"uom": "unit"})[
        ["symbol", "cik", "concept", "tag", "unit", "start", "end", "val",
         "filed", "form", "fy", "fp"]].copy()
    out["frame"] = None
    return out.reset_index(drop=True)


def refresh_facts_bulk(store, symbols: list[str] | None = None,
                       start_quarter: str = "2015q1", end_quarter: str | None = None,
                       force: bool = False, source: str = "fsds",
                       log: Callable[[str], None] = print) -> dict[str, Any]:
    """Ingest quarterly ZIPs into the store's `facts` table, one quarter at a time.

    `symbols` restricts to those tickers (None = every filer, which is the point of
    this path for a broad universe).

    A quarter is written straight into its own `filed_quarter=` partition and never
    read back, so the whole backfill costs one parse per ZIP instead of a table
    rewrite per ZIP. Quarters this path has already ingested are skipped by directory
    listing — the ZIP is immutable, so re-parsing it can only produce what is already
    there. The listing asks for `source="fsds"` specifically: a quarter directory also
    exists when the per-CIK companyfacts path happened to file a row into it, and
    treating that as "already ingested" would silently skip the whole backfill.

    `force=True` re-ingests them, and REPLACES each quarter's directory — use it when
    the PARSE changed (a new tag in FLOW_TAGS, a namespace fix), and re-run
    `--facts` afterwards if this store also carries companyfacts rows.
   
    `source` names the marker. It is what lets a SECOND symbol set be ingested into
    quarters an earlier run already covered: the marker records which pass has run, not
    which symbols it happened to want, so re-running with a wider `symbols` list under
    the default marker would skip every quarter and silently do nothing.
    """
    cikmap = news._ticker_cik_map()
    cik_to_symbol: dict[str, str] = {}
    for sym, cik in cikmap.items():
        if symbols is not None and sym not in set(symbols):
            continue
        cik_to_symbol.setdefault(str(cik).zfill(10), sym)
    keep_ciks = set(cik_to_symbol) if symbols is not None else None

    quarters = quarters_between(start_quarter, end_quarter)
    have = set(store.facts_quarters(source=source))
    rows, syms, got, missing, skipped = 0, set(), [], [], []
    for i, q in enumerate(quarters, 1):
        if q in have and not force:
            skipped.append(q)
            continue
        path = download_quarter(q)
        if path is None:
            missing.append(q)
            continue
        try:
            df = parse_quarter(path, keep_ciks, cik_to_symbol)
        except (zipfile.BadZipFile, ValueError) as exc:
            log(f"  {q}: unreadable ({exc}) - skipped")
            missing.append(q)
            continue
        if not df.empty:
            # Same (symbol, tag, period, filing) key the companyfacts path produces.
            df = df.drop_duplicates(["symbol", "tag", "end", "start", "filed"], keep="last")
            store.append_facts_quarter(q, df, replace=force, source=source,
                                       touch_manifest=False)
            rows += len(df)
            syms |= set(df["symbol"].dropna())
        got.append(q)
        log(f"  {q}: {len(df):7d} rows  ({i}/{len(quarters)})")

    # One distinct-symbol scan for the whole backfill, not one per quarter.
    store.refresh_facts_manifest()
    return {"rows": int(rows), "quarters": got, "missing": missing,
            "skipped": skipped, "symbols": len(syms)}


def refresh_shares_bulk(store, cik_to_symbol: dict[str, str],
                        start_quarter: str = "2015q1", end_quarter: str | None = None,
                        force: bool = False, source: str = "fsds-shares",
                        log: Callable[[str], None] = print) -> dict[str, Any]:
    """Ingest ONLY the share-count tags, for every CIK in `cik_to_symbol`.

    The broad liquidity floor is `price x shares outstanding`, point in time, for
    thousands of names on every rebalance date - so `shares_asof` has to cover the
    whole candidate list, not the 105 the full FSDS ingest was run for. A full ingest
    of every filer would parse ~30 tags out of 27 GB of `num.txt`; this parses the same
    files for four tags and takes ~5 seconds a quarter, because the cost is the CSV
    parse and the tag filter is what bounds memory, not time.

    Tracked under its own source marker (default `fsds-shares`) so it is idempotent per
    quarter independently of the full ingest; pass a new `source` to re-run it after a
    PARSE change without replacing — and so destroying — what other passes wrote. Symbols the full ingest already covered get a
    duplicate share row per period; the values are identical and every reader
    (`shares_asof`, `_Tab.dedupe_by_end`) keeps the last one, so it is wasted bytes
    rather than a wrong number.
    """
    quarters = quarters_between(start_quarter, end_quarter)
    have = set(store.facts_quarters(source=source))
    keep_ciks = set(cik_to_symbol)
    rows, syms, got, missing, skipped = 0, set(), [], [], []
    for i, q in enumerate(quarters, 1):
        if q in have and not force:
            skipped.append(q)
            continue
        path = download_quarter(q)
        if path is None:
            missing.append(q)
            continue
        try:
            df = parse_quarter(path, keep_ciks, cik_to_symbol, keep_tags=set(SF.SHARES_TAGS))
        except (zipfile.BadZipFile, ValueError) as exc:
            log(f"  {q}: unreadable ({exc}) - skipped")
            missing.append(q)
            continue
        if not df.empty:
            df = df.drop_duplicates(["symbol", "tag", "end", "start", "filed"], keep="last")
            store.append_facts_quarter(q, df, replace=False, source=source,
                                       touch_manifest=False)
            rows += len(df)
            syms |= set(df["symbol"].dropna())
        got.append(q)
        log(f"  {q}: {len(df):7d} share rows  ({i}/{len(quarters)})")
    store.refresh_facts_manifest()
    return {"rows": int(rows), "quarters": got, "missing": missing,
            "skipped": skipped, "symbols": len(syms)}
