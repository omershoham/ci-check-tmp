"""Map a scored/researched board row to a single display recommendation.

Six labels, plus `watch`: a researched, neutral-rated name you do NOT own that has an
armed candidate trigger or a stated entry level. It exists because `hold` means "you
own it, keep it", and reusing that word for "good company, wrong price" made the two
indistinguishable on the board — which is exactly the state where the picker is
supposed to be telling you to wait for a level.

Priority of evidence:
  1. Claude's explicit `rating` from deep research (most trustworthy).
  2. Else, derive from Claude `conviction` (0-10) if the name was researched.
  3. Else, a provisional rating from the cross-sectional quant score, marked as a
     screen-only guess (never says "sell" without research).

The screen-only rule cuts on `blended_pct` — the row's PERCENTILE inside its own
run — not on the absolute `blended` value, and the failure that prevents is
arithmetic, not cosmetic. `blended` is a weighted mean of per-pillar percentile
ranks, i.e. of ~uniforms, so its standard deviation is 0.289 * sqrt(sum w^2) / sum w:
0.167 with three balanced pillars, 0.145 with four, 0.130 with five. Against a fixed
0.66 cut that means P(buy) falls from ~19% to ~11% purely from switching a dark
pillar on, with ZERO change in the screen's skill — and it moves again with universe
size. A quantile cut is invariant to both: "the top 10% of today's board" means the
same thing at 105 names and at 1,500, with three pillars or with five.

Changing this threshold invalidates NO calibration statistic. `performance._bucket`
buckets on `research_reports.rating` (a screen-only pick is `screen_only` whatever
label it wore), and `claude_vs_screen`'s control cohort is defined by the pipeline's
finalist cut, not by this function. Only the dashboard's labels and sort move.
"""
from __future__ import annotations

from typing import Any

from . import config

# Canonical labels, badges, and a sort weight (higher = more bullish).
BADGE = {
    "strong_buy":  "🟩 Strong Buy",
    "buy":         "🟢 Buy",
    "hold":        "🟡 Hold",
    "watch":       "👀 Watch",
    "sell":        "🟠 Sell",
    "strong_sell": "🔴 Strong Sell",
    "avoid":       "⬜ Avoid",
    "unrated":     "❔ Unrated",
}
# `watch` sits between hold and avoid on purpose. It is NOT a weaker hold: `hold` means
# "you own this, keep it", so applying it to a name you do not own reads as an
# instruction to do nothing about a name you have no position in. `watch` means "good
# company, wrong price — there is an armed level that would change the answer", which
# is a strictly more actionable state than `avoid` and strictly less than owning it.
WEIGHT = {"strong_buy": 5, "buy": 4, "hold": 3, "watch": 2.5, "avoid": 2, "sell": 1.5,
          "strong_sell": 1, "unrated": 0, None: 0}

# LEGACY absolute cuts on `blended`. Used ONLY for rows scored before `blended_pct`
# existed (historical screen_scores rows, where the column is NULL). Never for new
# scoring — see the module docstring for why a fixed cut on `blended` is a bug.
LEGACY_BUY_SCORE = 0.66
LEGACY_HOLD_SCORE = 0.50


def blended_pct(row: dict[str, Any]) -> float | None:
    """The row's percentile inside its own run, wherever it happens to be carried.

    Fresh scoring rows put it top-level; rows rehydrated from SQLite carry it in the
    `blended_pct` column, and a row read straight out of `metrics_json` has it there.
    A None means "this row predates the field", which is what selects the legacy rule.
    """
    for v in (row.get("blended_pct"), (row.get("metrics") or {}).get("blended_pct")):
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def is_watch(row: dict[str, Any]) -> bool:
    """True when a neutral rating on a name you do NOT own has an armed level.

    `candidate_triggers` is the count of active source='candidate' triggers for the
    symbol (storage.get_board computes it); `entry_level` is a level the report stated
    outright. Either one means there is a price at which the answer changes, which is
    the whole difference between "watch" and "avoid".
    """
    if row.get("held_position"):
        return False
    return bool(row.get("candidate_triggers") or row.get("entry_level"))


def recommendation(row: dict[str, Any]) -> tuple[str, str]:
    """Return (label_key, basis) where basis is 'claude' or 'screen'."""
    rating = (row.get("rating") or "").strip().lower()
    if rating in BADGE:
        # A researched `hold` on a name you do not own is not "keep holding it" — it is
        # "not at this price". Say so, and only when there is an armed level that would
        # change the answer, so `watch` stays a claim about something checkable.
        if rating == "hold" and is_watch(row):
            return "watch", "claude"
        return rating, "claude"

    conv = row.get("conviction")
    if conv is not None:
        if conv >= 8:
            return "strong_buy", "claude"
        if conv >= 6.5:
            return "buy", "claude"
        if conv >= 5:
            return "watch" if is_watch(row) else "hold", "claude"
        return "sell", "claude"

    # Research was ATTEMPTED but produced no rating/conviction (error or parse fail):
    # mark unrated rather than letting it fall through to a misleading "avoid".
    raw = row.get("raw")
    if raw and (str(raw).startswith("error") or "could not parse" in str(row.get("thesis") or "")):
        return "unrated", "failed"

    # Provisional: quant screen only.
    pct = blended_pct(row)
    if pct is not None:
        if pct >= float(config.get("recommend.screen_buy_quantile", 0.90)):
            return "buy", "screen"
        if pct >= float(config.get("recommend.screen_hold_quantile", 0.60)):
            return "hold", "screen"
        return "avoid", "screen"
    b = row.get("blended") or 0.0
    if b >= LEGACY_BUY_SCORE:
        return "buy", "screen"
    if b >= LEGACY_HOLD_SCORE:
        return "hold", "screen"
    return "avoid", "screen"


def badge(row: dict[str, Any]) -> str:
    label, basis = recommendation(row)
    txt = BADGE.get(label, label)
    return txt + (" *" if basis == "screen" else "")


def sort_key(row: dict[str, Any]) -> float:
    """Label strength first, quant percentile as the tie-break.

    Without the tie-break every screen-only `avoid` sorts equal, so a 1,500-name board
    would come back in whatever order the caller happened to hold it — the label alone
    stopped being a total order the moment the universe got big.
    """
    label, _ = recommendation(row)
    w = WEIGHT.get(label, 0)
    tie = blended_pct(row)
    if tie is None:
        tie = float(row.get("blended") or 0.0)
    # Scaled to 0..0.4 because the smallest gap in WEIGHT is 0.5 (sell 1.5 -> avoid 2).
    # A wider tie-break would let a `sell` outrank an `avoid`, which is the opposite of
    # what this function is for.
    return float(w) + min(max(float(tie), 0.0), 1.0) * 0.4
