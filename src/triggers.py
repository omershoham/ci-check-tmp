"""Invalidation-trigger monitoring: turn deep-research 'invalidation' prose into
machine-checkable triggers, and have the daily runs evaluate them.

Why: every deep dive writes "what would break this thesis" — and nothing ever checked
it. A thesis can break on Tuesday and the owner finds out at the next deep dive. Now:
  1. After a portfolio deep-dive, ONE batched Claude call (no web) distills each
     holding's invalidation text + levels into structured triggers:
       price_below / price_above <threshold>, news_keyword <keyword>.
  2. Every pipeline run calls check_all() with fresh prices and headlines; a tripped
     trigger is marked 'triggered' and surfaces as a dashboard alert the same day.
  3. `extract_candidates` does the same for names you researched and did NOT buy
     (source='candidate'), so "good company, wrong price" is supervised instead of
     forgotten. Sources are independent: a candidate trigger is never superseded by a
     portfolio dive, or the other way round.
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


_CANDIDATE_TMPL = """Today is {today}. These are stocks I researched but do NOT own. For each,
give me MACHINE-CHECKABLE NEWS triggers — the headlines that would change my mind about
BUYING it. Use only:
- "news_keyword": fires when a headline about the company contains <keyword>
  (1-3 word, high-precision phrases: "guidance cut", "FDA approval", "contract award",
  "SEC investigation", "acquisition of" — never generic words like "falls" or "stock")

Candidates:
{block}

Rules: 1-3 keywords per symbol; each must be specific enough that a match almost
certainly matters to the thesis below it. Price levels are already handled — do NOT
return price triggers.

Output ONLY one JSON object:
{{"SYM": [{{"kind": "news_keyword", "keyword": "<phrase>", "description": "<why it matters>"}}], ...}}"""


def _price_triggers_from_report(rep: dict[str, Any]) -> list[dict[str, Any]]:
    """The price triggers a research report states OUTRIGHT, no LLM call needed.

    Phase 6 made `invalidation_level` a NUMBER, so the level that kills the thesis is
    already machine-checkable and paying for an extraction call to re-read it would be
    asking Claude to transcribe Claude. `entry_level` is accepted the same way
    `deal_likelihood` is — the prompt does not require it (PROMPT_VERSION stays 6), but
    reports that volunteer one get a real "buy if it pulls back to X" trigger out of it.
    """
    rows: list[dict[str, Any]] = []
    sym = str(rep.get("symbol") or "").upper()
    inv = _num(rep.get("invalidation_level"))
    if inv is not None:
        rows.append({
            "kind": "price_below", "threshold": inv,
            "description": f"{sym}: below {inv:g} the research thesis is invalidated — "
                           f"stop watching, do not buy the dip"})
    entry = _num(rep.get("entry_level"))
    last = _num(rep.get("latest_price_seen"))
    if entry is not None and entry != inv:
        # Which side to arm is decided by where the stated entry sits relative to the
        # price the analyst saw, not by the rating: an entry ABOVE the last price is a
        # breakout confirmation, one BELOW it is a pullback.
        kind = "price_above" if (last is not None and entry > last) else "price_below"
        rows.append({
            "kind": kind, "threshold": entry,
            "description": f"{sym}: stated entry level {entry:g} "
                           f"({'breakout' if kind == 'price_above' else 'pullback'})"})
    return rows


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def extract_candidates(run_id: int, reports: list[dict[str, Any]],
                       source: str = "candidate") -> int:
    """Arm triggers for researched names you do NOT own — "not yet, but buy if...".

    Why: `extract_and_store` is called only from the portfolio deep-dive, so only names
    already in the book were ever supervised. A great company researched at a bad price
    was rated `hold` and then forgotten until it happened to re-rank onto the board.

    Price levels come from the report's own numbers (free, exact); ONE batched Claude
    call adds news keywords for the whole cohort. The two sets are installed together in
    a single `replace_triggers` per symbol, so a failed or malformed LLM call cannot
    wipe the price triggers that were already derivable — the ordering bug that a
    "deterministic pass then LLM pass" implementation would have.

    Retention matches the portfolio path: re-researching a symbol supersedes ITS OWN
    source's previous set, empty extraction included, so a stale level never stays armed.
    """
    if not reports:
        return 0
    by_sym: dict[str, list[dict[str, Any]]] = {}
    block = []
    for rep in reports:
        sym = str(rep.get("symbol") or "").upper()
        if not sym:
            continue
        by_sym[sym] = _price_triggers_from_report(rep)
        thesis = str(rep.get("thesis") or "").strip()
        if thesis:
            block.append(f"- {sym}: rating {rep.get('rating')}; thesis: {thesis[:300]}")
    if not by_sym:
        return 0
    parsed = None
    if block:
        prompt = _CANDIDATE_TMPL.format(today=date.today().isoformat(),
                                        block="\n".join(block))
        parsed, _ = deep_research._run_json(prompt)
    for sym, rows in (parsed or {}).items():
        key = str(sym).upper()
        if key not in by_sym or not isinstance(rows, list):
            continue
        for r in rows:
            if not isinstance(r, dict) or str(r.get("kind", "")).strip() != "news_keyword":
                continue
            kw = str(r.get("keyword", "")).strip()
            if 2 <= len(kw) <= 60:
                by_sym[key].append({"kind": "news_keyword", "keyword": kw,
                                    "description": str(r.get("description", ""))[:200]})
    n = 0
    for sym, rows in by_sym.items():
        storage.replace_triggers(sym, source, run_id, rows)
        n += len(rows)
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
