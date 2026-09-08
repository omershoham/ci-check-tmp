"""Cross-sectional pillar scoring -> blended conviction + tags.

Scores are cross-sectional percentile ranks within the day's universe (0..1).
Growth, quality and cash flow rank across the universe blended with the name's own
industry group. VALUATION does not: it ranks INSIDE the name's sector only, because a
multiple is only meaningful against the sector that shares its economics. A bank's P/E
of 11 is not "cheap" next to software at 60, and blending half a cross-sector rank
into the score imports a sector BET into what is supposed to be a stock-selection
signal.

The failure this prevents, concretely: EV/EBITDA does not exist for banks (interest is
the revenue), so the old universe-wide `_pct(...).fillna(0.5)` handed the entire
Financials sector a permanent neutral valuation — the screen could never surface a
cheap bank, silently, forever. Per-sector component lists (`SECTOR_METRICS`) replace
the metrics a sector genuinely lacks with the ones it is actually priced on: P/B for
banks, P/FFO for REITs.

Missing metrics degrade to a neutral 0.5 only while enough of the family is present;
below `scoring.min_fundamental_components` the subscore is NaN and the name's
fundamental pillar drops out of its composite entirely, rather than being fabricated.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import config, local_llm

# Bump whenever a pillar's DEFINITION changes (components, ranking method, signs).
# pillar_ic filters to the current version so pre-change scores can't contaminate
# the information-coefficient statistics.
SCORE_VERSION = 5  # v5: fundamentals served from the SEC store (no universe-wide
                   #     `.info`, so no forwardPE), valuation ranked sector-NEUTRAL,
                   #     n-dependent group shrinkage, per-sector component sets,
                   #     coverage floor (thin-data -> pillar dropped, not neutralised)
# v3 was: 12-1 momentum, sector-relative ranks, deduped fundamentals, sentiment-only
# sentiment pillar. v4 is RESERVED: it means "v3 + the learned (ML) pillar carrying
# non-zero weight", and rows were persisted under it. v5 + learned is therefore 6 —
# the version is derived from config so turning the pillar on is what bumps it, and
# rows scored with and without the pillar never mix inside pillar_ic.
_LEARNED_VERSION = {3: 4, 5: 6}


def learned_weight() -> float:
    try:
        return float((config.get("scoring.pillars", {}) or {}).get("learned", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def effective_score_version() -> int:
    if learned_weight() > 0:
        return _LEARNED_VERSION.get(SCORE_VERSION, SCORE_VERSION + 1)
    return SCORE_VERSION


# Which pillars a row was actually scored with. Persisted to screen_scores.pillar_set
# so pillar_ic can group on it: adding a pillar changes the composite's meaning even
# when no existing pillar's definition moved, and pooling the two is a silent error.
_PILLAR_ORDER = ("technical", "fundamental", "sentiment", "learned", "catalyst")


_DEFAULT_PILLARS = {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.35}


def pillar_set() -> str:
    """The WEIGHTED pillars, comma-joined. A pillar shipped dark is computed and
    persisted but contributes nothing to `blended`, so listing it would split the IC
    study's partition key for no change in the composite."""
    pillars = config.get("scoring.pillars", _DEFAULT_PILLARS) or _DEFAULT_PILLARS
    on = [p for p in _PILLAR_ORDER if float(pillars.get(p, 0) or 0) > 0]
    return ",".join(on)

# Component metrics per subscore. (column, higher_is_better)
# Deduped per CR#2: p_fcf removed (reciprocal of fcf_yield — one number was ~20% of
# the pillar); pegRatio removed (mixes valuation with growth, which has its own
# subscore); operatingMargins moved from CASHFLOW (it isn't a cash-flow metric) to
# QUALITY; returnOnAssets added as the profitability proxy (GP/A in spirit);
# fcf_margin gives cashflow a genuine second leg.
_VALUATION = [("trailingPE", False), ("forwardPE", False),
              ("enterpriseToEbitda", False), ("priceToSalesTrailing12Months", False)]
_CASHFLOW = [("fcf_yield", True), ("fcf_margin", True)]
_GROWTH = [("revenueGrowth", True), ("earningsGrowth", True)]
_QUALITY = [("returnOnEquity", True), ("returnOnAssets", True),
            ("grossMargins", True), ("operatingMargins", True), ("debtToEquity", False)]

# "Lower is better" VALUATION multiples where a non-positive value means the company
# is loss-making, not "cheap". These must be masked to NaN (→ neutral) before ranking,
# otherwise a negative P/E ranks as the cheapest name in the universe. (debtToEquity is
# deliberately NOT here: zero debt is legitimately best, so it must keep ranking.)
_MASK_NONPOS = {"trailingPE", "forwardPE", "pegRatio", "enterpriseToEbitda",
                "priceToSalesTrailing12Months", "p_fcf", "ev_sales",
                "priceToBook", "p_ffo"}

UNKNOWN_SECTOR = "Unknown"
FAMILIES = ("valuation", "cashflow", "growth", "quality")
_DEFAULT_METRICS: dict[str, list[tuple[str, bool]]] = {
    "valuation": _VALUATION, "cashflow": _CASHFLOW,
    "growth": _GROWTH, "quality": _QUALITY,
}

# Per-sector component sets. Only the families that DIFFER are listed; anything absent
# falls back to the default list above, including for an unmapped/"Unknown" sector.
# Each entry replaces a metric the sector structurally lacks (so ranking it produces a
# universal 0.5) with the one the sector is actually priced on.
SECTOR_METRICS: dict[str, dict[str, list[tuple[str, bool]]]] = {
    # Banks and insurers: EV is undefined (deposits and float are not "net debt"),
    # EBITDA is undefined (interest IS the revenue) and revenue multiples are noise.
    # Leverage is the business model, not a quality defect, so debtToEquity goes; and
    # "free cash flow" for a lender is an accounting artefact, so the family empties
    # out and its weight is redistributed over the families that do apply.
    "Financials": {
        "valuation": [("priceToBook", False), ("trailingPE", False), ("forwardPE", False)],
        "cashflow": [],
        "quality": [("returnOnEquity", True), ("returnOnAssets", True),
                    ("grossMargins", True), ("operatingMargins", True)],
    },
    # REITs: depreciation on buildings is the largest charge in the P&L and is largely
    # fictional, so P/E mis-ranks the whole sector — P/FFO is the sector's own metric.
    # Debt is the funding model of a REIT, so debtToEquity goes. Cash flow moves to FFO
    # for the same reason: development capex makes fcf_yield negative for a HEALTHY
    # REIT by construction, which is what pushed EQIX down 32 ranks on the first pass.
    "Real Estate": {
        "valuation": [("p_ffo", False), ("enterpriseToEbitda", False),
                      ("priceToSalesTrailing12Months", False)],
        "cashflow": [("ffo_yield", True), ("ffo_margin", True)],
        "quality": [("returnOnEquity", True), ("returnOnAssets", True),
                    ("grossMargins", True), ("operatingMargins", True)],
    },
    # Regulated utilities are permanently FCF-negative by design (rate base grows on
    # borrowed money), so fcf_yield ranks the whole sector as capital-destroying.
    "Utilities": {
        "valuation": [("trailingPE", False), ("enterpriseToEbitda", False)],
        "cashflow": [("fcf_margin", True)],
    },
    # Commodity earnings swing with the cycle, so a trailing P/E is highest at the
    # bottom and lowest at the top — exactly backwards. Not dropped, but down-weighted
    # from 2-of-4 (PE + fwd PE) to 1-of-3 of the valuation family.
    "Energy": {"valuation": [("enterpriseToEbitda", False),
                             ("priceToSalesTrailing12Months", False),
                             ("trailingPE", False)]},
    "Materials": {"valuation": [("enterpriseToEbitda", False),
                                ("priceToSalesTrailing12Months", False),
                                ("trailingPE", False)]},
}

# A name needs this many families with real data before the fundamental pillar means
# anything; below it the pillar is NaN and drops out of that name's composite.
_MIN_FAMILIES = 2


def sector_metrics(sector: str, family: str) -> list[tuple[str, bool]]:
    """The (column, higher_is_better) list one sector uses for one metric family."""
    return SECTOR_METRICS.get(sector, {}).get(family, _DEFAULT_METRICS[family])


def _shrinkage_k() -> float:
    try:
        return float(config.get("scoring.group_shrinkage_k", 10) or 10)
    except (TypeError, ValueError):
        return 10.0


def _min_components() -> int:
    try:
        return int(config.get("scoring.min_fundamental_components", 2) or 0)
    except (TypeError, ValueError):
        return 2


def _pct(series: pd.Series, higher_is_better: bool, mask_nonpositive: bool = False,
         groups: pd.Series | None = None, k: float | None = None,
         shrink_to: str = "universe") -> pd.Series:
    """Percentile rank, blended with the within-group rank when `groups` is given.

    The blend weight is n-DEPENDENT: λ(n) = n/(n+k), n = rankable values in the row's
    own group. A 3-name group's percentile can only take {0.33, 0.67, 1.0} yet used to
    carry a flat 50% of the score; it now carries 23%. k = `scoring.group_shrinkage_k`
    (default 10) reproduces the old flat 0.5 exactly at n=10, so this is a strict
    generalization of the previous rule, not a new one.

    `shrink_to` decides what a thin group is pulled TOWARDS:
      "universe" — the universe-wide rank. Right for growth/quality/cash flow: a 60%
                   gross margin is a high margin in any industry.
      "neutral"  — 0.5, i.e. "no opinion". Right for VALUATION, and the whole point of
                   sector-neutrality: a thin sector must not have its multiples ranked
                   against sectors that do not share its economics.
    """
    s = series.where(series > 0) if mask_nonpositive else series
    r = s.rank(pct=True)
    if groups is not None:
        rg = s.groupby(groups).rank(pct=True)
        n = s.notna().groupby(groups).transform("sum").astype(float)
        kk = _shrinkage_k() if k is None else float(k)
        # k=0 means "no shrinkage": the pure within-group percentile. The risk tags use
        # it, because a THRESHOLD on the shrunk score is not expressible — a group of
        # 12 can never score below 0.5*k/(n+k)=0.23, so "the most expensive 15% of its
        # sector" would silently never fire for any sector smaller than ~28 names.
        lam = (n > 0).astype(float) if kk <= 0 else n / (n + kk)
        base = 0.5 if shrink_to == "neutral" else r
        r = (1.0 - lam) * base + lam * rg
    if not higher_is_better:
        r = 1.0 - r
    return r.fillna(0.5)


def _component_percentiles(df: pd.DataFrame, components: list[tuple[str, bool]],
                           groups: pd.Series | None, k: float | None,
                           shrink_to: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(percentile frame, presence frame) over the components that carry any data.

    A column that is present but ALL-NaN is dropped, not ranked to a universal 0.5.
    Serving the pillar from the SEC store makes this load-bearing: `forwardPE` exists
    as a column on every snapshot and is empty on all of them, and ranking it would
    compress the valuation family's dispersion by 25% against the other families
    without changing a single rank inside it.
    """
    ranks: dict[str, pd.Series] = {}
    present: dict[str, pd.Series] = {}
    for col, hib in components:
        if col not in df.columns or col in ranks:
            continue
        raw = pd.to_numeric(df[col], errors="coerce")
        if col in _MASK_NONPOS:
            raw = raw.where(raw > 0)
        if not raw.notna().any():
            continue
        present[col] = raw.notna()
        ranks[col] = _pct(raw, hib, groups=groups, k=k, shrink_to=shrink_to)
    return (pd.DataFrame(ranks, index=df.index),
            pd.DataFrame(present, index=df.index))


def _subscore(df: pd.DataFrame, components: list[tuple[str, bool]],
              groups: pd.Series | None = None, min_components: int = 0,
              k: float | None = None, shrink_to: str = "universe") -> pd.Series:
    """One family's 0..1 subscore, or NaN when the name has fewer than
    `min_components` of the family's inputs. NaN rather than 0.5 is the point: a
    fabricated neutral is indistinguishable from a measured average, and it kept
    whole sectors sitting at the median of a screen that could never rank them."""
    return _family_score(df, components, groups, k, shrink_to, min_components)[0]


def _family_score(df, components, groups, k, shrink_to, min_components):
    """-> (subscore, components present per row, components configured per row)."""
    R, P = _component_percentiles(df, components, groups, k, shrink_to)
    zeros = pd.Series(0.0, index=df.index)
    if R.empty:
        empty = pd.Series(np.nan if min_components else 0.5, index=df.index)
        return empty, zeros, zeros
    n_present = P.sum(axis=1).astype(float)
    n_cols = pd.Series(float(len(R.columns)), index=df.index)
    sub = R.mean(axis=1)
    floor = min(int(min_components), len(R.columns))
    if floor:
        sub = sub.where(n_present >= floor)
    return sub, n_present, n_cols


def _sector_neutral_family(df: pd.DataFrame, family: str, sectors: pd.Series,
                           k: float | None, min_components: int):
    """Rank one family INSIDE each sector, each sector using its own component list.

    Every sector's rows come back as percentiles of their OWN peer group, so the
    outputs share a 0..1 scale without any cross-sector comparison ever taking place.
    That is what lets a cheap bank score 0.9 on valuation in a universe where the
    cheapest software name trades at 8x the bank's multiple."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    n_present = pd.Series(0.0, index=df.index)
    n_cols = pd.Series(0.0, index=df.index)
    for sec, idx in df.groupby(sectors, sort=False).groups.items():
        sub_df = df.loc[idx]
        grp = pd.Series(str(sec), index=sub_df.index)
        s, npres, ncol = _family_score(sub_df, sector_metrics(str(sec), family),
                                       grp, k, "neutral", min_components)
        out.loc[idx] = s
        n_present.loc[idx] = npres
        n_cols.loc[idx] = ncol
    return out, n_present, n_cols


def _blended_family(df: pd.DataFrame, family: str, sectors: pd.Series,
                    groups: pd.Series, k: float | None, min_components: int):
    """Growth / quality / cash flow: the universe rank blended with the within-group
    rank (as before), but each row averages only the components ITS sector uses."""
    cols: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for sec in dict.fromkeys(sectors.tolist()):
        for c, hib in sector_metrics(str(sec), family):
            if c not in seen:
                seen.add(c)
                cols.append((c, hib))
    R, P = _component_percentiles(df, cols, groups, k, "universe")
    zeros = pd.Series(0.0, index=df.index)
    if R.empty:
        return pd.Series(np.nan if min_components else 0.5, index=df.index), zeros, zeros
    use = pd.DataFrame(False, index=df.index, columns=R.columns)
    for sec, idx in df.groupby(sectors, sort=False).groups.items():
        want = [c for c, _ in sector_metrics(str(sec), family) if c in R.columns]
        if want:
            use.loc[idx, want] = True
    sub = R.where(use).mean(axis=1)
    n_present = (P & use).sum(axis=1).astype(float)
    n_cols = use.sum(axis=1).astype(float)
    if min_components:
        sub = sub.where(n_present >= np.minimum(min_components, n_cols))
    return sub, n_present, n_cols


def fundamental_detail(df: pd.DataFrame, subsectors: dict[str, str],
                       sectors: dict[str, str] | None = None,
                       tag_inputs: bool = True) -> dict[str, pd.Series]:
    """The fundamental pillar and its parts for a metrics frame indexed by symbol.

    ONE implementation, shared by the live screen (`score_universe`) and the backtest
    harness (`backtest.signals.fundamental_pillar`), so a replay can never drift from
    what the screen actually computed.

    `pillar` is NaN for a name backed by fewer than `_MIN_FAMILIES` families with real
    data; its composite is then re-normalized over the pillars that DO have data
    instead of the name being scored on an invented 0.5."""
    qw = config.get("scoring.quality.subweights", {})
    k = _shrinkage_k()
    mc = _min_components()
    grp = pd.Series({s: subsectors.get(s, "") for s in df.index})
    sec = pd.Series({s: ((sectors or {}).get(s) or UNKNOWN_SECTOR) for s in df.index})

    val, vn, vd = _sector_neutral_family(df, "valuation", sec, k, mc)
    cash, cn, cd = _blended_family(df, "cashflow", sec, grp, k, mc)
    grow, gn, gd = _blended_family(df, "growth", sec, grp, k, mc)
    qual, qn, qd = _blended_family(df, "quality", sec, grp, k, mc)

    parts = {"valuation": (val, qw.get("valuation", 0.3)),
             "cashflow": (cash, qw.get("cashflow", 0.3)),
             "growth": (grow, qw.get("growth", 0.2)),
             "quality": (qual, qw.get("quality", 0.2))}
    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    n_ok = pd.Series(0, index=df.index)
    for s, w in parts.values():
        ok = s.notna()
        num = num + s.fillna(0.0) * (float(w) * ok)
        den = den + float(w) * ok
        n_ok = n_ok + ok.astype(int)
    pillar = (num / den.replace(0.0, np.nan))
    if mc:
        pillar = pillar.where(n_ok >= _MIN_FAMILIES)
    denom = (vd + cd + gd + qd).replace(0.0, np.nan)
    coverage = ((vn + cn + gn + qn) / denom).fillna(0.0)
    # The UNSHRUNK within-sector valuation percentile. Scoring uses the shrunk `val`;
    # the risk tags need a plain percentile, because a fixed cut ("the most expensive
    # 15% of its sector") cannot be applied to a score that a thin sector compresses.
    val_pure = (_sector_neutral_family(df, "valuation", sec, 0.0, mc)[0] if tag_inputs
                else None)
    return {"pillar": pillar, "valuation": val, "cashflow": cash, "growth": grow,
            "quality": qual, "coverage": coverage, "families": n_ok,
            "valuation_in_sector": val_pure}


def fundamental_pillar(df: pd.DataFrame, subsectors: dict[str, str],
                       sectors: dict[str, str] | None = None) -> pd.Series:
    """The fundamental pillar (0..1, NaN where the data is too thin) — the exact
    computation `score_universe` uses, exposed so the backtest harness can replay the
    live pillar on point-in-time snapshots with the same code path."""
    return fundamental_detail(df, subsectors, sectors, tag_inputs=False)["pillar"]


def _r3(v) -> float | None:
    """Round for JSON, keeping "we could not measure this" as null, not 0.0."""
    return None if v is None or v != v else round(float(v), 3)


def _sentiment_pillar(sig: dict | None) -> float:
    """0..1 news-sentiment pillar: TONE ONLY. The catalyst vector measures EVENTS and
    has its own pillar; folding it in here would put the same evidence into the
    composite twice and make either pillar's information coefficient uninterpretable
    (which signal would be measured?)."""
    if not sig:
        return 0.5
    ss = sig.get("sentiment_score")
    return 0.5 if ss is None else float(ss) / 10.0


def catalyst_subweights() -> dict[str, float]:
    """Per-key weights for the catalyst pillar. EQUAL by default, deliberately.

    The deleted deal score put 35% on `subsector_relevance`, a hand-set prior that
    could never be wrong and never be measured. Hand-tuning these nine would repeat
    exactly that, so they stay equal until pillar_ic has the forward data to earn a
    change."""
    cfg = config.get("scoring.catalyst.subweights", {}) or {}
    return {k: float(cfg.get(k, 1.0) or 0.0) for k in local_llm.CATALYST_KEYS}


def _catalyst_pillar(sig: dict | None,
                     subweights: dict[str, float] | None = None) -> float | None:
    """Weighted mean of the eight positive catalyst keys minus `adverse_event`, each
    on 0..1, over the keys that were actually MEASURED.

    Returns None when no key was measured — a failed call, or a name never scanned.
    That is the genuine-0-vs-unknown discipline the old `_hs` carried: a measured 0
    means "the model read the headlines and found no such event" and must keep
    ranking below a measured 8, while an unmeasured key must not push a name down
    for evidence nobody ever gathered."""
    cat = (sig or {}).get("catalyst") or {}
    w = subweights if subweights is not None else catalyst_subweights()
    num = den = 0.0
    for k in local_llm.CATALYST_KEYS:
        v = cat.get(k)
        if v is None:
            continue
        wk = float(w.get(k, 1.0) or 0.0)
        if wk <= 0:
            continue
        den += wk
        num += wk * (float(v) / 10.0) * (-1.0 if k in local_llm.NEGATIVE_KEYS else 1.0)
    return None if den <= 0 else num / den


def score_universe(metrics: dict[str, dict],
                   subsectors: dict[str, str],
                   signals: dict[str, dict] | None = None,
                   technicals: dict[str, float] | None = None,
                   ml_scores: dict[str, float] | None = None,
                   sectors: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Three-pillar scoring (+ an optional fourth, learned pillar).

    metrics: symbol -> fundamentals dict.
    signals: symbol -> local_llm dict (catalyst vector, sentiment_score, ...).
    technicals: symbol -> technical_score (0..1).
    ml_scores: symbol -> raw score from the learned ranker (src/ml). Weighted only by
        config scoring.pillars.learned (default 0 = shipped dark); still persisted so the
        dashboard can show it. Missing/None -> neutral 0.5 for every name.
    sectors: symbol -> one of the 11 GICS-like sectors (data/sic_sectors). The
        VALUATION peer set. Unmapped names fall into "Unknown", which is treated as a
        sector of its own rather than being spread across the real ones.
    """
    if not metrics:
        return []
    signals = signals or {}
    technicals = technicals or {}
    ml_scores = ml_scores or {}
    sectors = sectors or {}
    df = pd.DataFrame(list(metrics.values()))
    df = df.set_index("symbol", drop=False)

    pillars = config.get("scoring.pillars", _DEFAULT_PILLARS)
    tagcfg = config.get("scoring.tags", {})

    # --- Fundamental pillar. ONE implementation, shared with the backtest replay.
    fd = fundamental_detail(df, subsectors, sectors)
    fundamental = fd["pillar"]
    val, cash, grow, qual = fd["valuation"], fd["cashflow"], fd["growth"], fd["quality"]
    fund_cov = fd["coverage"]
    val_in_sector = fd["valuation_in_sector"]

    # --- Catalyst pillar (SHIPPED DARK at scoring.pillars.catalyst = 0). Computed and
    # persisted from day one so the forward history accumulates while the pillar is
    # un-backtestable: no news archive exists, so it can only ever be earned forward.
    cw = catalyst_subweights()
    catalyst = pd.Series({s: _catalyst_pillar(signals.get(s), cw) for s in df.index},
                         dtype=float)

    # --- Technical & Sentiment pillars
    # technicals: symbol -> dict from technical.compute_many (has technical_score, state, rsi).
    technical = pd.Series({s: float((technicals.get(s) or {}).get("technical_score", 0.5))
                           for s in df.index})
    sentiment = pd.Series({s: _sentiment_pillar(signals.get(s)) for s in df.index})

    # --- Composite (the ranking score).
    # Each pillar is percentile-ranked across the universe BEFORE weighting. Without this
    # the three pillars sit on different scales (fundamental ~uniform, technical an absolute
    # rule score, sentiment a compressed LLM rescale), so the widest-dispersion pillar would
    # dominate regardless of the configured weights. Ranking makes the weights mean what they say.
    wT = pillars.get("technical", 0.35)
    wF = pillars.get("fundamental", 0.30)
    wS = pillars.get("sentiment", 0.35)
    wL = learned_weight()
    wC = float(pillars.get("catalyst", 0.0) or 0.0)
    technical_r = _pct(technical, higher_is_better=True)
    # NOT _pct: its fillna(0.5) is exactly what the coverage floor exists to prevent.
    # A thin-data name keeps a NaN rank here and the weight loop below drops the
    # pillar from ITS composite only.
    fundamental_r = fundamental.rank(pct=True)
    sentiment_r = _pct(sentiment, higher_is_better=True)
    learned = pd.Series({s: ml_scores.get(s) for s in df.index}, dtype=float)
    learned_r = _pct(learned, higher_is_better=True) if learned.notna().any() \
        else pd.Series(0.5, index=df.index)
    # _pct's fillna(0.5) is the right rule here: a name with NO measured catalyst key
    # sits in the middle of the cross-section, not at the bottom of it.
    catalyst_r = _pct(catalyst, higher_is_better=True) if catalyst.notna().any() \
        else pd.Series(0.5, index=df.index)
    # Normalize by the weight sum so the composite stays on the 0..1 scale when a user
    # turns the learned pillar on without re-balancing the other three (identical to
    # the old formula whenever the weights already sum to 1). The sum is PER ROW: a
    # name whose fundamental pillar is NaN is ranked on its other pillars re-scaled to
    # 1, not penalised for data we never had.
    wF_row = pd.Series(float(wF), index=df.index).where(fundamental.notna(), 0.0)
    wsum = (wT + wS + wL + wC) + wF_row
    wsum = wsum.where(wsum > 0, 1.0)
    composite = (technical_r * wT + fundamental_r.fillna(0.0) * wF_row
                 + sentiment_r * wS + learned_r * wL + catalyst_r * wC) / wsum

    # Sector-relative valuation: median PE / fwd PE / EV-EBITDA per sub-sector.
    df["_sub"] = [subsectors.get(s, "") for s in df.index]
    df["_sector"] = [sectors.get(s) or UNKNOWN_SECTOR for s in df.index]

    def _sec_med(col: str, by: str = "_sub") -> dict:
        if col not in df.columns:
            return {}
        v = pd.to_numeric(df[col], errors="coerce")
        v = v.where(v > 0)  # ignore negative/zero (loss-making) valuations
        return v.groupby(df[by]).median().to_dict()

    pe_med = _sec_med("trailingPE")
    fpe_med = _sec_med("forwardPE")      # {} once the pillar is SEC-served: fine
    ev_med = _sec_med("enterpriseToEbitda")
    pe_med_sector = _sec_med("trailingPE", by="_sector")

    rows: list[dict[str, Any]] = []
    q_thresh = tagcfg.get("quality_threshold", 0.6)
    require_fcf = config.get("scoring.quality.require_positive_fcf_for_tag", True)
    hr = tagcfg.get("higher_risk", {})
    # DEPRECATED (one release): a flat PE ceiling is a sector bet dressed as a risk
    # rule — 60x is ordinary for software and absurd for a utility. Superseded by
    # valuation_pct_floor / sector_pe_multiple below; remove after the next release.
    pe_ceiling = hr.get("pe_risk_ceiling", 60)
    val_floor = float(hr.get("valuation_pct_floor", 0.15) or 0)
    pe_mult = float(hr.get("sector_pe_multiple", 2.5) or 0)
    # A tail cut needs a tail. With 2 names in a sector, "the most expensive 15%" flags
    # one of the two every time, which is a coin toss, not a risk signal. Below
    # 1/val_floor members the percentile rule stands down and only the sector-median
    # multiple applies.
    min_sector_n = int(np.ceil(1.0 / val_floor)) if val_floor else 0
    sector_n = df["_sector"].value_counts().to_dict()
    earnings_soon_days = int(config.get("screen.earnings_soon_days", 7))
    pset = pillar_set()

    for s in df.index:
        m = metrics[s]
        fcf = m.get("freeCashflow")
        positive_fcf = bool(fcf and fcf > 0)
        pe = m.get("trailingPE")
        tech_info = technicals.get(s) or {}
        f_val = float(fundamental[s]) if fundamental[s] == fundamental[s] else None
        sec_name = sectors.get(s) or UNKNOWN_SECTOR

        tags: list[str] = []
        is_quality = (f_val is not None and f_val >= q_thresh
                      and (positive_fcf or not require_fcf))
        if is_quality:
            tags.append("quality")
        # "higher-risk" means unprofitable, or EXPENSIVE RELATIVE TO ITS OWN SECTOR.
        # The old rule also fired on `pe is None`, which flagged every pre-revenue
        # biotech and every loss-making growth name as risky for the same reason — a
        # missing number. That is a data fact, not a risk judgement: it now gets its
        # own neutral `no-earnings` tag and nothing else.
        higher_risk = False
        if hr.get("flag_if_unprofitable", True) and not m.get("profitable", False):
            higher_risk = True
        vs = val_in_sector[s]
        if (val_floor and vs == vs and vs <= val_floor
                and sector_n.get(sec_name, 0) >= min_sector_n):
            higher_risk = True
        smed = pe_med_sector.get(sec_name)
        if (pe_mult and isinstance(pe, (int, float)) and pe > 0
                and smed and smed == smed and pe > pe_mult * smed):
            higher_risk = True
        if pe_ceiling and isinstance(pe, (int, float)) and pe > pe_ceiling:
            higher_risk = True
        if higher_risk:
            tags.append("higher-risk")
        if pe is None:
            tags.append("no-earnings")
        if f_val is None:
            tags.append("thin-data")
        dte = m.get("days_to_earnings")
        if isinstance(dte, (int, float)) and 0 <= dte <= earnings_soon_days:
            tags.append("earnings-soon")

        sig = signals.get(s) or {}
        rows.append({
            "symbol": s,
            "name": m.get("name", s),
            "subsector": subsectors.get(s, ""),
            "sector": sec_name,
            "pillar_set": pset,
            # pillar scores (0..1)
            "technical": round(float(technical[s]), 4),
            "fundamental": round(f_val, 4) if f_val is not None else None,
            "sentiment": round(float(sentiment[s]), 4),
            "blended": round(float(composite[s]), 4),   # composite = ranking score
            # kept for back-compat / tags
            "quality": round(f_val, 4) if f_val is not None else None,
            "tags": tags,
            "metrics": {
                "price": m.get("price"),
                "marketCap": m.get("marketCap"),
                "trailingPE": pe,
                "forwardPE": m.get("forwardPE"),
                "p_fcf": m.get("p_fcf"),
                "fcf_yield": m.get("fcf_yield"),
                "revenueGrowth": m.get("revenueGrowth"),
                "grossMargins": m.get("grossMargins"),
                "returnOnEquity": m.get("returnOnEquity"),
                # pillar scores (persisted here so the dashboard can show them)
                "pillar_technical": round(float(technical[s]), 4),
                "pillar_fundamental": round(f_val, 4) if f_val is not None else None,
                "pillar_sentiment": round(float(sentiment[s]), 4),
                # learned pillar (dark until scoring.pillars.learned > 0)
                "pillar_learned": (round(float(learned[s]), 4)
                                   if s in learned.index and learned[s] == learned[s] else None),
                "pillar_learned_rank": round(float(learned_r[s]), 4) if learned.notna().any() else None,
                # catalyst pillar (dark until scoring.pillars.catalyst > 0). Persisted
                # with its RAW nine keys, so a later redefinition can be recomputed
                # over the accumulated history instead of only from the day it lands.
                "pillar_catalyst": _r3(catalyst[s]),
                "pillar_catalyst_rank": (round(float(catalyst_r[s]), 4)
                                         if catalyst.notna().any() else None),
                "catalyst": (signals.get(s) or {}).get("catalyst"),
                "signal_version": (signals.get(s) or {}).get("signal_version"),
                "tech_state": tech_info.get("state"),
                "rsi": tech_info.get("rsi"),
                "macd_hist": tech_info.get("macd_hist"),
                "pos_52w": tech_info.get("pos_52w"),
                "atr_pct": tech_info.get("atr_pct"),
                # trend setup + relative strength
                "setup": tech_info.get("setup"),
                "setup_label": tech_info.get("setup_label"),
                "signals": tech_info.get("signals"),
                "rs_3m": tech_info.get("rs_3m"),
                "rs_6m": tech_info.get("rs_6m"),
                "benchmark": tech_info.get("benchmark"),
                # modern valuation metrics
                "ev_ebitda": m.get("enterpriseToEbitda"),
                "ev_sales": m.get("ev_sales"),
                "ps": m.get("priceToSalesTrailing12Months"),
                "peg": m.get("pegRatio"),
                "rule_of_40": m.get("rule_of_40"),
                "net_cash": m.get("net_cash"),
                "earnings_date": m.get("earnings_date"),
                "days_to_earnings": m.get("days_to_earnings"),
                "priceToBook": m.get("priceToBook"),
                "p_ffo": m.get("p_ffo"),
                "cash_runway_quarters": m.get("cash_runway_quarters"),
                # sector-relative valuation (sub-sector medians)
                "sector_pe": pe_med.get(subsectors.get(s, "")),
                "sector_fwdpe": fpe_med.get(subsectors.get(s, "")),
                "sector_evebitda": ev_med.get(subsectors.get(s, "")),
                # GICS-like sector: the VALUATION peer set, and what the higher-risk
                # tag compares this name's P/E against.
                "sector": sec_name,
                "sector_pe_gics": _r3(pe_med_sector.get(sec_name)),
                # share of this name's own sector metric set that had real data
                "fund_coverage": round(float(fund_cov[s]), 3),
                "sentiment_label": sig.get("sentiment"),
                "sub_valuation": _r3(val[s]),
                "sub_cashflow": _r3(cash[s]),
                "sub_growth": _r3(grow[s]),
                "sub_quality": _r3(qual[s]),
            },
        })

    rows.sort(key=lambda r: r["blended"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


if __name__ == "__main__":
    from . import universe, fundamentals, technical
    syms = universe.symbols()[:12]
    print(f"Fetching {len(syms)} names for a scoring demo...")
    metr = fundamentals.fetch_many(syms)
    sub = universe.subsector_map()
    metr = {s: m for s, m in metr.items() if fundamentals.passes_floor(m)}
    techs = technical.compute_many(list(metr.keys()))
    print(f"{len(metr)} passed the liquidity floor")
    scored = score_universe(metr, sub, technicals=techs)
    print(f"\n{'rank':<4} {'sym':<6} {'comp':<6} {'tech':<6} {'fund':<6} {'sent':<6} tags")
    for r in scored[:15]:
        print(f"{r['rank']:<4} {r['symbol']:<6} {r['blended']:<6} "
              f"{r['technical']:<6} {r['fundamental']:<6} {r['sentiment']:<6} {','.join(r['tags'])}")
