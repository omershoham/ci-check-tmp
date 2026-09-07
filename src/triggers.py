"""Invalidation-trigger monitoring: turn deep-research 'invalidation' prose into
machine-checkable triggers, and have the daily runs evaluate them.

Why: every deep dive writes "what would break this thesis" — and nothing ever checked
it. A thesis can break on Tuesday and the owner finds out at the next deep dive. Now:
  1. After a portfolio deep-dive, ONE batched Claude call (no web) distills each
     holding's invalidation text + levels into structured triggers:
       price_below / price_above <threshold>, news_keyword <keyword>.
  2. Every pipeline run calls check_all() with fresh prices and headlines; a tripped
     trigger is marked 'triggered' and surfaces as a dashboard alert the same day.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

from . import deep_research, storage

_EXTRACT_TMPL = """Today is {today}. For each holding below, convert its thesis-invalidation
description into MACHINE-CHECKABLE triggers. Use only these kinds:
- "price_below": fires when the price drops below <threshold>
- "price_above": fires when the price rises above <threshold>  (rare — e.g. blow-off risk)
- "news_keyword": fires when a headline about the company contains <keyword>
  (choose 1-3 word, high-precision keywords: e.g. "contract cancelled", "guidance cut",
  "delisting", "dilution", "SEC investigation" — avoid generic words like "falls" or "stock")

Holdings:
{block}

Rules: 2-4 triggers per symbol; price triggers should be meaningful thesis-break levels
(use the invalidation text and the given support/add levels, NOT tiny moves); keywords must
be specific enough that a match almost certainly matters.

Output ONLY one JSON object:
{{"SYM": [{{"kind": "price_below", "threshold": <number>, "description": "<why this matters>"}},
          {{"kind": "news_keyword", "keyword": "<phrase>", "description": "..."}}], ...}}"""


def extract_and_store(run_id: int, positions: list[dict[str, Any]],
                      source: str = "portfolio") -> int:
    """One batched Claude call for all holdings; store the distilled triggers."""
    items = []
    for p in positions:
        inv = (p.get("invalidation") or "").strip()
        if not inv or inv.startswith("("):
            continue
        items.append(
            f"- {p['symbol']}: price {p.get('last')}; add_level {p.get('add_level')}; "
            f"trim_level {p.get('trim_level')}; invalidation: {inv[:400]}")
    if not items:
        return 0
    prompt = _EXTRACT_TMPL.format(today=date.today().isoformat(), block="\n".join(items))
    parsed, _ = deep_research._run_json(prompt)
    if not parsed:
        return 0
    n = 0
    # Every symbol that was dived gets its old same-source triggers superseded, even
    # if the new extraction produced nothing for it — otherwise stale price
    # thresholds from a previous thesis stay armed forever.
    dived_syms = {p["symbol"].upper() for p in positions}
    parsed_syms = {str(s).upper() for s in parsed}
    for sym in dived_syms - parsed_syms:
        storage.replace_triggers(sym, source, run_id, [])
    for sym, rows in parsed.items():
        if not isinstance(rows, list):
            continue
        clean = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            kind = str(r.get("kind", "")).strip()
            if kind in ("price_below", "price_above"):
                try:
                    thr = float(r.get("threshold"))
                except (TypeError, ValueError):
                    continue
                clean.append({"kind": kind, "threshold": thr,
                              "description": str(r.get("description", ""))[:200]})
            elif kind == "news_keyword":
                kw = str(r.get("keyword", "")).strip()
                if 2 <= len(kw) <= 60:
                    clean.append({"kind": kind, "keyword": kw,
                                  "description": str(r.get("description", ""))[:200]})
        # Empty `clean` still supersedes: the old thesis's triggers are obsolete.
        storage.replace_triggers(sym, source, run_id, clean)
        n += len(clean)
    return n


def check_all(prices: dict[str, float] | None = None,
              news_map: dict[str, list[dict]] | None = None) -> list[dict[str, Any]]:
    """Evaluate all active triggers; mark tripped ones and return them."""
    prices = prices or {}
    news_map = news_map or {}
    fired: list[dict[str, Any]] = []
    for t in storage.active_triggers():
        sym = t["symbol"]
        detail = None
        if t["kind"] in ("price_below", "price_above"):
            px = prices.get(sym)
            thr = t.get("threshold")
            if px is None or thr is None:
                continue
            if t["kind"] == "price_below" and px < thr:
                detail = f"price {px:.2f} < {thr:g}"
            elif t["kind"] == "price_above" and px > thr:
                detail = f"price {px:.2f} > {thr:g}"
        elif t["kind"] == "news_keyword":
            kw = (t.get("keyword") or "").lower()
            if not kw:
                continue
            for it in news_map.get(sym, []):
                title = (it.get("title") or "")
                if kw in title.lower():
                    detail = f"headline matched '{t['keyword']}': {title[:150]}"
                    break
        if detail:
            storage.fire_trigger(t["id"], detail)
            t["trigger_detail"] = detail
            fired.append(t)
    return fired


if __name__ == "__main__":
    print("active triggers:")
    for t in storage.active_triggers():
        print(f"  {t['symbol']:6} {t['kind']:12} thr={t.get('threshold')} "
              f"kw={t.get('keyword')} — {t.get('description','')[:60]}")
    print("recently fired:")
    for t in storage.recent_fired_triggers():
        print(f"  {t['symbol']:6} {t['triggered_at']} {t['trigger_detail']}")
