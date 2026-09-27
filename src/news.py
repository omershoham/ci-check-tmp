"""Free news/catalyst gathering: Google News RSS + SEC EDGAR 8-K filings.

Used to feed the local-LLM signal stage (Stage 2). All network calls are
best-effort and degrade to an empty list on failure.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any
from urllib.parse import quote_plus

import feedparser
import requests

from . import config

_RECENT_DAYS = 30          # SEC 8-K lookback window
_NEWS_DAYS = 14            # headlines lookback (recent enough to reflect current sentiment)


def _sec_headers() -> dict[str, str]:
    """SEC fair-access policy requires a descriptive UA carrying a contact address.

    That address is personal, so it is NOT hardcoded here: set SEC_CONTACT_EMAIL in
    .env (git-ignored, preferred) or news.sec_contact in config.yaml. Without one the
    UA is still descriptive, but SEC throttles/blocks anonymous callers more readily,
    so set your own before relying on 8-K data.
    """
    contact = (os.environ.get("SEC_CONTACT_EMAIL")
               or config.get("news.sec_contact") or "").strip()
    return {"User-Agent": "Claude-Trader research tool "
                          f"({contact or 'contact unset - see .env.example'})"}


def google_news(query: str, limit: int = 8) -> list[dict[str, Any]]:
    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(f"{query} when:{_NEWS_DAYS}d")
        + "&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        # feedparser.parse(url) does its own urllib fetch with NO timeout — a single
        # stalled Google News response would wedge the whole scheduled run. Fetch with
        # an explicit timeout and hand the bytes to feedparser.
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for e in feed.entries[:limit]:
        out.append({
            "title": getattr(e, "title", ""),
            "link": getattr(e, "link", ""),
            "published": getattr(e, "published", ""),
            "source": "google_news",
        })
    return out


@lru_cache(maxsize=1)
def _company_tickers() -> dict[str, dict]:
    """SEC's company_tickers.json ({"0": {"cik_str", "ticker", "title"}, ...}), fetched
    at most once per process; {} on failure. The two maps below each keep their own
    daily cache file, so on a warm day this is never called at all."""
    try:
        resp = requests.get("https://www.sec.gov/files/company_tickers.json",
                            headers=_sec_headers(), timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:  # noqa: BLE001
        return {}


@lru_cache(maxsize=1)
def _ticker_cik_map() -> dict[str, str]:
    """Ticker -> zero-padded CIK, cached for the day (and memoised per process).

    `recent_8k` calls this once per symbol; without the memo a shortlist re-parsed a
    230 KB JSON 35 times, and a broad-universe candidate map would do it thousands."""
    cf = config.cache_dir() / f"sec_tickers_{date.today().isoformat()}.json"
    if cf.exists():
        try:
            return json.loads(cf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    data = _company_tickers()
    if not data:
        return {}
    mapping = {row["ticker"].upper(): f"{int(row['cik_str']):010d}"
               for row in data.values()}
    try:
        cf.write_text(json.dumps(mapping), encoding="utf-8")
    except OSError:
        pass
    return mapping


# Corporate suffixes and short joining words: capitalised/lower-cased rather than kept
# as acronyms by `display_name`'s "three letters or fewer is an acronym" rule.
_NAME_WORDS = {"INC": "Inc", "CO": "Co", "LTD": "Ltd", "PLC": "plc", "LLC": "LLC",
               "LP": "LP", "NV": "NV", "SA": "SA", "AG": "AG", "SE": "SE",
               "OF": "of", "AND": "and", "THE": "the", "FOR": "for", "ON": "on"}


def display_name(title: str) -> str:
    """An SEC registrant title made presentable: "COSTCO WHOLESALE CORP /NEW" ->
    "Costco Wholesale Corp". Mixed-case words ("Apple", "AbbVie") are the filer's own
    spelling and kept; all-caps words are capitalised, except the ones that are almost
    always acronyms: two letters or none of them vowels (GE, CVS), or carrying digits
    or symbols (3M, AT&T). Imperfect by design - Yahoo's name wins where it exists."""
    import re
    t = re.sub(r"\s+/[A-Z]{2,4}/?\s*$", "", (title or "").strip())  # "/DE/", "/NEW"
    words = []
    for i, w in enumerate(t.split()):
        core = w.strip(".,")
        if core != core.upper():
            words.append(w)
        elif core in _NAME_WORDS:
            new = _NAME_WORDS[core]
            new = new[0].upper() + new[1:] if i == 0 else new
            words.append(w.replace(core, new))
        elif (len(core) <= 2 or not set(core) & set("AEIOU")
              or not core.replace("-", "").replace(".", "").isalpha()):
            words.append(w)
        else:
            words.append("-".join(p.capitalize() for p in w.split("-")))
    return " ".join(words)


@lru_cache(maxsize=1)
def _ticker_name_map() -> dict[str, str]:
    """Ticker -> presentable company name from SEC's ticker file, cached for the day.

    The board's fallback display name for every name Yahoo's `.info` never reaches
    (all but the shortlist). Names change rarely, so offline the newest cached copy of
    ANY date is used before giving up; {} only when there has never been one."""
    cache = config.cache_dir()
    cf = cache / f"sec_ticker_names_{date.today().isoformat()}.json"
    if cf.exists():
        try:
            return json.loads(cf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    data = _company_tickers()
    if not data:
        for old in sorted(cache.glob("sec_ticker_names_*.json"), reverse=True):
            try:
                return json.loads(old.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
        return {}
    mapping: dict[str, str] = {}
    for row in data.values():
        tk = str(row.get("ticker") or "").upper()
        # The file lists a company's primary ticker first; keep that row's title.
        if tk and tk not in mapping and row.get("title"):
            mapping[tk] = display_name(str(row["title"]))
    try:
        cf.write_text(json.dumps(mapping), encoding="utf-8")
    except OSError:
        pass
    return mapping


# 8-K item codes -> what they actually mean. A bare "8-K filed" carries zero
# information; the item code is the difference between routine paperwork and a
# CFO resignation.
_8K_ITEMS = {
    "1.01": "material agreement entered", "1.02": "material agreement TERMINATED",
    "1.03": "bankruptcy/receivership", "2.01": "acquisition/disposition completed",
    "2.02": "results of operations (earnings)", "2.04": "debt acceleration triggered",
    "2.05": "exit/restructuring costs", "2.06": "material impairment",
    "3.01": "delisting/listing-standard notice", "4.01": "auditor change",
    "4.02": "financials can no longer be relied on", "5.02": "officer/director departure or appointment",
    "5.03": "charter/bylaws change", "7.01": "Reg FD disclosure",
    "8.01": "other material event", "9.01": "exhibits",
}


def _describe_items(items_str: str) -> str:
    codes = [c.strip() for c in (items_str or "").split(",") if c.strip()]
    named = [f"{c} {_8K_ITEMS[c]}" for c in codes if c in _8K_ITEMS and c != "9.01"]
    return "; ".join(named[:3])


def recent_8k(symbol: str, limit: int = 5) -> list[dict[str, Any]]:
    cik = _ticker_cik_map().get(symbol.upper())
    if not cik:
        return []
    try:
        resp = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                            headers=_sec_headers(), timeout=15)
        resp.raise_for_status()
        recent = resp.json().get("filings", {}).get("recent", {})
    except Exception:  # noqa: BLE001
        return []
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    docs = recent.get("primaryDocument", [])
    accs = recent.get("accessionNumber", [])
    items = recent.get("items", [])
    cutoff = (datetime.now(timezone.utc) - timedelta(days=_RECENT_DAYS)).date()
    out = []
    for i, form in enumerate(forms):
        if not form.startswith("8-K"):
            continue
        try:
            fdate = datetime.strptime(dates[i], "%Y-%m-%d").date()
        except (ValueError, IndexError):
            continue
        if fdate < cutoff:
            continue
        acc = accs[i].replace("-", "") if i < len(accs) else ""
        doc = docs[i] if i < len(docs) else ""
        link = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
                if acc and doc else "")
        desc = _describe_items(items[i] if i < len(items) else "")
        title = (f"SEC 8-K: {desc} (filed {dates[i]})" if desc
                 else f"SEC 8-K filed {dates[i]}")
        out.append({
            "title": title,
            "link": link,
            "published": dates[i],
            "source": "sec_8k",
        })
        if len(out) >= limit:
            break
    return out


def _similar(a: str, b: str, threshold: float = 0.6) -> bool:
    """Token-set Jaccard similarity — collapses syndicated near-duplicate headlines
    so one wire story can't fill most of the slots the LLM sees."""
    ta = {w for w in a.lower().split() if len(w) > 2}
    tb = {w for w in b.lower().split() if len(w) > 2}
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def _published_dt(it: dict) -> datetime:
    """Best-effort parse of an item's published date for sorting (fallback: epoch)."""
    s = it.get("published", "")
    try:
        from email.utils import parsedate_to_datetime
        d = parsedate_to_datetime(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        pass
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def get_news(symbol: str, name: str | None = None) -> list[dict[str, Any]]:
    """Combined, deduped recent items for one company, newest first.

    Two Google News queries: the plain company query (relevance-ranked, tends to surface
    price-target/listicle coverage) PLUS an adverse-news query. The second exists because
    the tape-moving story is often negative (lawsuit, probe, downgrade) and gets buried
    below promotional coverage in the plain query — e.g. the July 2026 DRAM antitrust
    class action never appeared in MU's top-8 plain results while the stock fell 20%.
    """
    query = f'{symbol} {name}' if name else symbol
    items = google_news(query)
    items += google_news(f'{query} (lawsuit OR investigation OR downgrade OR antitrust '
                         f'OR SEC probe OR guidance cut)', limit=6)
    items += recent_8k(symbol)
    seen, out = set(), []
    for it in items:
        key = it["title"].strip().lower()
        if not key or key in seen:
            continue
        # Fuzzy dedupe vs already-kept titles (syndicated stories reworded slightly).
        if any(_similar(key, prev["title"]) for prev in out):
            continue
        seen.add(key)
        out.append(it)
    out.sort(key=_published_dt, reverse=True)  # newest first so truncation drops old news
    return out[:14]


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "VRT"
    name = sys.argv[2] if len(sys.argv) > 2 else "Vertiv"
    print(f"News for {sym} ({name}):")
    for it in get_news(sym, name):
        print(f"  [{it['source']}] {it['title']}")
