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
            c = c[c["segments"].fillna("").eq("") & c["coreg"].fillna("").eq("")]
            c = c[c["version"].fillna("").str.split("/").str[0].isin(_NS)]
            c = c[c["value"].notna()]
            if not c.empty:
                parts.append(c)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def parse_quarter(zip_path: Path, keep_ciks: set[str] | None = None,
                  cik_to_symbol: dict[str, str] | None = None) -> pd.DataFrame:
    """One quarter's ZIP -> the same long frame shape as sec_facts.facts_frame."""
    t2c = tag_to_concept()
    with zipfile.ZipFile(zip_path) as z:
        sub = _submissions(z)
        num = _numbers(z, set(t2c), keep_ciks, sub)
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

    # sharesOutstanding is a dei instant; drop any flow-shaped rows for it.
    df = df[~((df["concept"] == "sharesOutstanding") & flow)]
    # Monetary tags must carry a currency; the share count must carry "shares".
    is_shares = df["concept"] == "sharesOutstanding"
    df = df[(is_shares & df["uom"].eq("shares"))
            | (~is_shares & df["uom"].str.fullmatch(r"[A-Z]{3}", na=False))]

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
                       log: Callable[[str], None] = print) -> dict[str, Any]:
    """Ingest quarterly ZIPs into the store's `facts` table.

    `symbols` restricts to those tickers (None = every filer, which is the point of
    this path for a broad universe). Quarters already present are re-parsed only if
    their ZIP is missing from the cache, so re-runs are cheap.
    """
    cikmap = news._ticker_cik_map()
    cik_to_symbol: dict[str, str] = {}
    for sym, cik in cikmap.items():
        if symbols is not None and sym not in set(symbols):
            continue
        cik_to_symbol.setdefault(str(cik).zfill(10), sym)
    keep_ciks = set(cik_to_symbol) if symbols is not None else None

    quarters = quarters_between(start_quarter, end_quarter)
    frames, got, missing = [], [], []
    for i, q in enumerate(quarters, 1):
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
            frames.append(df)
        got.append(q)
        log(f"  {q}: {len(df):7d} rows  ({i}/{len(quarters)})")

    if not frames:
        return {"rows": 0, "quarters": got, "missing": missing}
    allf = pd.concat(frames, ignore_index=True)
    # Same (symbol, tag, period, filing) key the companyfacts path produces.
    allf = allf.drop_duplicates(["symbol", "tag", "end", "start", "filed"], keep="last")
    n = store.upsert_facts(allf)
    return {"rows": int(len(allf)), "stored": int(n), "quarters": got,
            "missing": missing, "symbols": int(allf["symbol"].nunique())}
