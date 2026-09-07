"""Map a scored/researched board row to a single display recommendation.

Priority of evidence:
  1. Claude's explicit `rating` from deep research (most trustworthy).
  2. Else, derive from Claude `conviction` (0-10) if the name was researched.
  3. Else, a provisional rating from the cross-sectional quant `blended` score,
     marked as a screen-only guess (never says "sell" without research).
"""
from __future__ import annotations

from typing import Any

# Canonical labels, badges, and a sort weight (higher = more bullish).
BADGE = {
    "strong_buy":  "🟩 Strong Buy",
    "buy":         "🟢 Buy",
    "hold":        "🟡 Hold",
    "sell":        "🟠 Sell",
    "strong_sell": "🔴 Strong Sell",
    "avoid":       "⬜ Avoid",
    "unrated":     "❔ Unrated",
}
WEIGHT = {"strong_buy": 5, "buy": 4, "hold": 3, "avoid": 2, "sell": 1.5,
          "strong_sell": 1, "unrated": 0, None: 0}


def recommendation(row: dict[str, Any]) -> tuple[str, str]:
    """Return (label_key, basis) where basis is 'claude' or 'screen'."""
    rating = (row.get("rating") or "").strip().lower()
    if rating in BADGE:
        return rating, "claude"

    conv = row.get("conviction")
    if conv is not None:
        if conv >= 8:
            return "strong_buy", "claude"
        if conv >= 6.5:
            return "buy", "claude"
        if conv >= 5:
            return "hold", "claude"
        return "sell", "claude"

    # Research was ATTEMPTED but produced no rating/conviction (error or parse fail):
    # mark unrated rather than letting it fall through to a misleading "avoid".
    raw = row.get("raw")
    if raw and (str(raw).startswith("error") or "could not parse" in str(row.get("thesis") or "")):
        return "unrated", "failed"

    # Provisional: quant screen only (blended is a 0..1 cross-sectional score).
    b = row.get("blended") or 0.0
    if b >= 0.66:
        return "buy", "screen"
    if b >= 0.50:
        return "hold", "screen"
    return "avoid", "screen"


def badge(row: dict[str, Any]) -> str:
    label, basis = recommendation(row)
    txt = BADGE.get(label, label)
    return txt + (" *" if basis == "screen" else "")


def sort_key(row: dict[str, Any]) -> float:
    label, _ = recommendation(row)
    return WEIGHT.get(label, 0)
