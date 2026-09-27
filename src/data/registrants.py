"""The US registrant table: who is listed, where, and in what industry.

WHY. `universe.build_universe()` used to flatten a hand-written list of 105 tickers
picked in June 2026 - a universe chosen with hindsight, which is why
`docs/IMPROVEMENT_PLAN.md` records every backtest return as upward-biased. This module
replaces the seed with two free SEC files:

    company_tickers_exchange.json   ticker -> CIK, name, exchange (~10,400 rows)
    the FSDS quarterly `sub.txt`    CIK -> SIC, form types, first/last filing date

The second file is the load-bearing one, and not for the SIC alone: `sub.txt` lists
every filer that filed in that quarter, INCLUDING the ones that were later acquired or
went bust. Membership reconstructed from it is survivorship-honest, which the ticker
file (a snapshot of who is listed TODAY) is not. Both are already on disk - 46 cached
quarterly ZIPs - so building this table costs one HTTP request and ~30 seconds of
zipfile reads, and is O(1) in universe size.

The filtering ladder is deliberately auditable: every step logs its count, because
"10,400 in, 250 out" is otherwise an unfalsifiable claim. The steps drop, in order:
non-US-exchange listings (OTC and blanks), non-common-stock ticker shapes (warrants,
units, rights), registrants with no XBRL financial statements at all, funds/CEFs/BDCs
(SIC 6726) and blank-check shells (6770) - but NOT REITs (6798), which are operating
companies and belong in the universe - and finally anything whose only filings are
fund reports.

What this table does NOT do is apply the liquidity floor. That is
`universe.investable(as_of)`, because the floor is point-in-time and this table is not:
a registrant's exchange and SIC are close enough to constant that recomputing them
monthly is honest, while its price and share count are not.
"""
from __future__ import annotations

import json
import re
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

from .. import config, news
from . import sec_fsds, sic_sectors

_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"

# Exchanges that mean "listed US common stock". Anything else - OTC, blank, a foreign
# venue - is not tradeable through the broker this system advises on.
DEFAULT_EXCHANGES = ("Nasdaq", "NYSE", "NYSE American", "Cboe")

# Common stock only, INCLUDING share classes. A bare `^[A-Z]{1,5}$` looks tidy and
# silently deletes Berkshire, Brown-Forman, Moog, Lennar and every other dual-class
# issuer from a "broad US universe" - so a one-letter class suffix is allowed.
#
# The rule is read off the file rather than guessed. Of the 543 non-plain tickers on
# the four exchanges, the suffixes are: -P/-PA..-PZ (379, preferred series), -WT/-RW/
# -WTA (66, warrants), -UN (50, units), -RI (19, rights) and -A/-B/-C/-V (24, share
# classes). Every non-class suffix except a bare `-P` is two or more letters, so the
# regex handles them and only `-P` needs naming. Multi-class survivors collapse to one
# row per CIK later, in `universe.investable`.
_TICKER_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")
# Single-letter suffixes that are NOT a share class: preferred, warrant, unit, right.
_NON_CLASS_SUFFIXES = ("-P", "-W", "-U", "-R")
# Dotted forms some vendors use for the same instruments; absent from the SEC file
# today, cheap to keep out if it ever changes shape.
_DERIVATIVE_SUFFIXES = _NON_CLASS_SUFFIXES + (".WS", ".U", ".RT")

# Filers whose ONLY forms are these are investment companies, whatever their SIC says.
_FUND_ONLY_FORMS = {"N-CSR", "N-CSRS", "N-Q", "24F-2NT", "N-30D", "NT-NCSR"}

_REGISTRANTS = "registrants.parquet"
_SUB_INDEX = "sub_index.parquet"
_SUB_INDEX_META = "sub_index.json"
_SYMBOL_SICS = "symbol_sics.parquet"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


def _log_noop(msg: str) -> None:
    pass


# ------------------------------------------------------------------ exchange file
def load_exchange(force: bool = False) -> pd.DataFrame:
    """ticker/CIK/name/exchange for every SEC-registered listed company.

    NOTE the shape: unlike `company_tickers.json` (a dict of row objects, which
    `news._ticker_cik_map` reads), this file is columnar -
    {"fields": [...], "data": [[...], ...]} - so it cannot be parsed by the same code.
    Cached one file per calendar day, like every other SEC fetch here.
    """
    cf = Path(config.cache_dir()) / f"sec_exchange_{date.today().isoformat()}.json"
    payload: Any = None
    if cf.exists() and not force:
        try:
            payload = json.loads(cf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = None
    if payload is None:
        resp = requests.get(_EXCHANGE_URL, headers=news._sec_headers(), timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        try:
            cf.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass
    df = pd.DataFrame(payload.get("data", []), columns=payload.get("fields", []))
    if df.empty:
        return pd.DataFrame(columns=["cik", "name", "ticker", "exchange"])
    df["cik"] = df["cik"].astype("int64").astype(str).str.zfill(10)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["exchange"] = df["exchange"].fillna("").astype(str).str.strip()
    df["name"] = df["name"].fillna("").astype(str)
    return df[["cik", "name", "ticker", "exchange"]]


# --------------------------------------------------------------------- sub index
def _sub_index_paths(store) -> tuple[Path, Path]:
    d = store.root / "universe"
    return d / _SUB_INDEX, d / _SUB_INDEX_META


def build_sub_index(store, force: bool = False,
                    log: Callable[[str], None] = _log_noop) -> pd.DataFrame:
    """Per-CIK SIC, filing lifetime and form set, from every cached FSDS `sub.txt`.

    `sub.txt` is a few thousand rows per quarter and reads in milliseconds - the 600 MB
    file in those ZIPs is `num.txt`, which this never opens. Cached to parquet and
    rebuilt only when the set of cached quarters changes, so the monthly registrant
    refresh does not re-read 46 archives.
    """
    idx_path, meta_path = _sub_index_paths(store)
    zips = sorted(sec_fsds.cache_dir().glob("*.zip"))
    quarters = [p.stem for p in zips]
    if idx_path.exists() and not force:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
        if meta.get("quarters") == quarters:
            return pd.read_parquet(idx_path)

    frames = []
    for p in zips:
        try:
            with zipfile.ZipFile(p) as z:
                sub = sec_fsds._submissions(z)
        except (zipfile.BadZipFile, KeyError, ValueError) as exc:
            log(f"  sub index: {p.name} unreadable ({exc}) - skipped")
            continue
        frames.append(sub[["cik", "name", "sic", "form", "filed"]])
    if not frames:
        return pd.DataFrame(columns=["cik", "sec_name", "sic", "forms",
                                     "first_filed", "last_filed"])
    all_sub = pd.concat(frames, ignore_index=True)
    log(f"  sub index: {len(all_sub)} submissions over {len(zips)} quarters")

    g = all_sub.sort_values("filed", kind="stable").groupby("cik", sort=True)
    out = pd.DataFrame({
        "sec_name": g["name"].last(),
        # The SIC on the LATEST filing: a reclassification should move the company,
        # and the old code is not more true for being older.
        "sic": g["sic"].last(),
        "forms": g["form"].apply(lambda s: "|".join(sorted(set(s.dropna())))),
        "first_filed": g["filed"].min(),
        "last_filed": g["filed"].max(),
    }).reset_index()
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(idx_path, index=False)
    try:
        meta_path.write_text(json.dumps({"quarters": quarters}), encoding="utf-8")
    except OSError:
        pass
    return out


# ----------------------------------------------------------------------- ladder
def _broad(key: str, default: Any) -> Any:
    return (config.get("universe.broad", {}) or {}).get(key, default)


def build_candidates(store, force_exchange: bool = False,
                     log: Callable[[str], None] = _log_noop) -> tuple[pd.DataFrame, dict]:
    """Run the filtering ladder. Returns (candidates, funnel counts per step)."""
    exchanges = set(_broad("exchanges", list(DEFAULT_EXCHANGES)))
    exclude_sic = {int(s) for s in (_broad("exclude_sic", [6726, 6770]) or [])}

    df = load_exchange(force=force_exchange)
    funnel: dict[str, int] = {"sec_listed": len(df)}
    log(f"  ladder: {len(df)} listed SEC registrants")

    df = df[df["exchange"].isin(exchanges)]
    funnel["on_exchange"] = len(df)
    log(f"  ladder: {len(df)} on {sorted(exchanges)}")

    shape_ok = (df["ticker"].str.match(_TICKER_RE)
                & ~df["ticker"].str.endswith(_DERIVATIVE_SUFFIXES))
    df = df[shape_ok]
    funnel["common_stock_ticker"] = len(df)
    log(f"  ladder: {len(df)} common-stock ticker shapes "
        f"(preferreds/warrants/units/rights out, share classes kept)")

    sub = build_sub_index(store, log=log)
    df = df.merge(sub, on="cik", how="left")
    df = df[df["last_filed"].notna()]
    funnel["files_financials"] = len(df)
    log(f"  ladder: {len(df)} with XBRL financial statements on file")

    sic_num = pd.to_numeric(df["sic"], errors="coerce")
    df = df[~sic_num.isin(exclude_sic)]
    funnel["operating_company"] = len(df)
    log(f"  ladder: {len(df)} after dropping SIC {sorted(exclude_sic)} "
        f"(funds/CEFs/BDCs, blank checks; REITs 6798 kept)")

    forms = df["forms"].fillna("").str.split("|")
    fund_only = forms.map(lambda fs: bool(fs) and set(f for f in fs if f) <= _FUND_ONLY_FORMS
                          and any(f for f in fs))
    df = df[~fund_only]
    funnel["not_fund_reporter"] = len(df)
    log(f"  ladder: {len(df)} candidates (fund-only filers out)")

    cls = sic_sectors.classify_frame(df["sic"], symbols=df["ticker"])
    out = pd.DataFrame({
        "symbol": df["ticker"].values,
        "cik": df["cik"].values,
        "name": df["name"].where(df["name"].astype(bool), df["sec_name"]).values,
        "exchange": df["exchange"].values,
        "sic": df["sic"].values,
        "sector": cls["sector"].values,
        "industry_group": cls["industry_group"].values,
        "first_filed": pd.to_datetime(df["first_filed"]).values,
        "last_filed": pd.to_datetime(df["last_filed"]).values,
    })
    return out.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True), funnel


# ------------------------------------------------------------------- the table
def registrants_path(store) -> Path:
    return store.root / "universe" / _REGISTRANTS


def price_coverage(store, symbols: list[str] | None = None) -> pd.DataFrame:
    """first/last price date per symbol. Two projected columns off the price store -
    no pivot, no close values."""
    px = store.prices(symbols, columns=["close"])
    if px.empty:
        return pd.DataFrame(columns=["first_price_date", "last_price_date"])
    g = px.groupby("symbol")["date"]
    return pd.DataFrame({"first_price_date": g.min(), "last_price_date": g.max()})


def update_price_coverage(store) -> int:
    """Re-stamp first/last price date on the existing table after a price download.

    Those two columns are the cheap answer to "how much history does this candidate
    have?", and they are wrong the moment a download adds bars — so the download
    refreshes them rather than leaving a table that quietly disagrees with the store."""
    path = registrants_path(store)
    if not path.exists():
        return 0
    df = pd.read_parquet(path).drop(columns=["first_price_date", "last_price_date"],
                                    errors="ignore")
    cov = price_coverage(store, df["symbol"].tolist())
    df = df.join(cov, on="symbol") if not cov.empty else df.assign(
        first_price_date=pd.NaT, last_price_date=pd.NaT)
    df.to_parquet(path, index=False)
    return int(df["last_price_date"].notna().sum())


def refresh_registrants(store, force_exchange: bool = False,
                        log: Callable[[str], None] = _log_noop) -> dict[str, Any]:
    """Rebuild `universe/registrants.parquet`. Monthly cadence is plenty - a company's
    exchange and SIC move on a quarterly clock at best."""
    df, funnel = build_candidates(store, force_exchange=force_exchange, log=log)
    cov = price_coverage(store, df["symbol"].tolist())
    df = df.join(cov, on="symbol") if not cov.empty else df.assign(
        first_price_date=pd.NaT, last_price_date=pd.NaT)
    path = registrants_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    store._touch_manifest("registrants", len(df), {"funnel": funnel})
    return {"rows": len(df), "funnel": funnel, "path": str(path)}


def symbol_sics_path(store) -> Path:
    return store.root / "universe" / _SYMBOL_SICS


def load_symbol_sics(store) -> pd.DataFrame:
    """symbol -> SIC, sector, industry group for names looked up one by one."""
    path = symbol_sics_path(store)
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "cik", "sic", "sector", "industry_group"])
    return pd.read_parquet(path)


def fetch_sic(cik: str, timeout: float = 15) -> str | None:
    """A filer's SIC code from its SEC submissions JSON (None when unavailable)."""
    try:
        resp = requests.get(_SUBMISSIONS_URL.format(cik=str(cik).zfill(10)),
                            headers=news._sec_headers(), timeout=timeout)
        resp.raise_for_status()
        sic = resp.json().get("sic")
    except Exception:  # noqa: BLE001 - one unresolved name is not a failed refresh
        return None
    return (str(sic).strip() or None) if sic else None


def refresh_symbol_sics(store, symbols: list[str], pause: float = 0.12,
                        fetch: Callable[[str], str | None] | None = None,
                        cikmap: dict[str, str] | None = None,
                        log: Callable[[str], None] = _log_noop) -> dict[str, int]:
    """SIC -> sector for a HAND-PICKED list, one submissions request per new name.

    The registrant table is built from the bulk FSDS archives, which a curated setup
    never downloads, so every curated name used to land in sector "Unknown" -- and
    since valuation is ranked inside a sector, a whole board ranked its multiples as
    one undifferentiated peer group. SIC is close to constant, so a name is fetched
    once and cached; later refreshes only look up names that are new."""
    import time
    have = load_symbol_sics(store)
    known = set(have["symbol"]) if not have.empty else set()
    todo = [s.upper() for s in dict.fromkeys(symbols) if s.upper() not in known]
    if not todo:
        return {"fetched": 0, "cached": len(known), "unresolved": 0}
    cikmap = cikmap if cikmap is not None else news._ticker_cik_map()
    fetch = fetch or fetch_sic
    rows, unresolved = [], []
    for sym in todo:
        cik = cikmap.get(sym)
        sic = fetch(cik) if cik else None
        if fetch is fetch_sic and cik:
            time.sleep(pause)
        if not sic:
            unresolved.append(sym)
            continue
        rows.append({"symbol": sym, "cik": cik, "sic": sic})
    if rows:
        new = pd.DataFrame(rows)
        cls = sic_sectors.classify_frame(new["sic"], symbols=new["symbol"])
        new = pd.concat([new, cls], axis=1)
        out = pd.concat([have, new], ignore_index=True) if not have.empty else new
        path = symbol_sics_path(store)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.drop_duplicates("symbol", keep="last").to_parquet(path, index=False)
    if unresolved:
        log(f"  no SIC for: {unresolved}")
    return {"fetched": len(rows), "cached": len(known), "unresolved": len(unresolved)}


def load_registrants(store) -> pd.DataFrame:
    path = registrants_path(store)
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "cik", "name", "exchange", "sic",
                                     "sector", "industry_group"])
    return pd.read_parquet(path)


if __name__ == "__main__":
    from .history_store import HistoryStore
    res = refresh_registrants(HistoryStore(), log=print)
    print(json.dumps(res["funnel"], indent=2))
    reg = load_registrants(HistoryStore())
    print(reg["sector"].value_counts().to_string())
