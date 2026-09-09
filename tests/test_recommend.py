import numpy as np
import pandas as pd

from src import recommend


def test_explicit_rating_wins():
    assert recommend.recommendation({"rating": "Strong_Buy", "conviction": 1}) == ("strong_buy", "claude")


def test_conviction_thresholds():
    assert recommend.recommendation({"conviction": 8})[0] == "strong_buy"
    assert recommend.recommendation({"conviction": 6.5})[0] == "buy"
    assert recommend.recommendation({"conviction": 5})[0] == "hold"
    assert recommend.recommendation({"conviction": 4.9})[0] == "sell"


def test_failed_research_is_unrated_not_avoid():
    row = {"raw": "error: claude timed out", "blended": 0.2}
    assert recommend.recommendation(row) == ("unrated", "failed")
    row = {"thesis": "(could not parse JSON)", "raw": "{}", "blended": 0.9}
    assert recommend.recommendation(row) == ("unrated", "failed")


def test_screen_only_cuts_on_the_percentile():
    assert recommend.recommendation({"blended_pct": 0.95}) == ("buy", "screen")
    assert recommend.recommendation({"blended_pct": 0.90}) == ("buy", "screen")
    assert recommend.recommendation({"blended_pct": 0.75}) == ("hold", "screen")
    assert recommend.recommendation({"blended_pct": 0.59}) == ("avoid", "screen")
    # The percentile wins even when the absolute score would say otherwise: a 0.9
    # blended in a board where everything scores 0.9 is not a buy.
    assert recommend.recommendation({"blended": 0.9, "blended_pct": 0.2})[0] == "avoid"
    # A row rehydrated from metrics_json carries it there instead.
    assert recommend.recommendation({"metrics": {"blended_pct": 0.99}})[0] == "buy"
    assert recommend.badge({"blended_pct": 0.95}).endswith(" *")
    assert not recommend.badge({"rating": "buy"}).endswith("*")


def test_legacy_rows_without_a_percentile_keep_the_old_absolute_rule():
    """Rows scored before Phase 7 have blended_pct NULL. Re-labelling them with a
    quantile we cannot compute (their cross-section is gone) would rewrite history."""
    assert recommend.recommendation({"blended": 0.7}) == ("buy", "screen")
    assert recommend.recommendation({"blended": 0.55}) == ("hold", "screen")
    assert recommend.recommendation({"blended": 0.1}) == ("avoid", "screen")


def test_sort_key_orders_bullish_first():
    rows = [{"rating": "sell"}, {"rating": "strong_buy"}, {"blended": 0.55}]
    ordered = sorted(rows, key=recommend.sort_key, reverse=True)
    assert ordered[0]["rating"] == "strong_buy"
    assert ordered[-1]["rating"] == "sell"


def _synthetic_run(weights, n=2000, seed=7):
    """One run's worth of composite scores, built exactly the way score_universe does:
    each pillar is a percentile rank across the universe (a uniform), and `blended` is
    their weighted mean, normalized by the weight sum."""
    rng = np.random.default_rng(seed)
    ranks = {k: pd.Series(rng.permutation(n) / (n - 1.0)) for k in weights}
    wsum = sum(weights.values())
    blended = sum(ranks[k] * w for k, w in weights.items()) / wsum
    pct = blended.rank(pct=True)
    return [{"blended": round(float(b), 4), "blended_pct": round(float(p), 4)}
            for b, p in zip(blended, pct)]


# The two live pillar sets: today's three, and the five that exist the moment the
# catalyst and learned pillars are given weight (the plan's balanced case).
THREE = {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.35}
FIVE = {"technical": 0.20, "fundamental": 0.20, "sentiment": 0.20,
        "catalyst": 0.20, "learned": 0.20}


def test_quantile_rule_is_invariant_to_pillar_count_and_the_old_rule_was_not():
    """THE defect this threshold change exists to fix, demonstrated.

    `blended` is a weighted mean of ~uniforms, so its standard deviation is
    0.289 * sqrt(sum w^2) / sum w: ~0.167 with three balanced pillars, ~0.130 with
    five. Against a FIXED 0.66 cut that alone drops the buy count by a third or more —
    the screen appears to get pickier the day a dark pillar is switched on, with zero
    change in its skill. The quantile cut is invariant by construction.
    """
    three, five = _synthetic_run(THREE), _synthetic_run(FIVE)

    def n_buy(rows):
        return sum(1 for r in rows if recommend.recommendation(r)[0] == "buy")

    def n_buy_legacy(rows):
        return sum(1 for r in rows
                   if (r["blended"] or 0) >= recommend.LEGACY_BUY_SCORE)

    # Quantile rule: identical, and it is the configured top decile.
    assert n_buy(three) == n_buy(five)
    assert abs(n_buy(three) / len(three) - 0.10) < 0.01

    # Old absolute rule: ~40% fewer buys purely from the extra pillars.
    assert n_buy_legacy(five) < 0.7 * n_buy_legacy(three)


def test_quantile_rule_is_invariant_to_universe_size():
    """105 curated names and 1,500 broad ones must label the same SHARE of the board."""
    small, big = _synthetic_run(THREE, n=105, seed=3), _synthetic_run(THREE, n=1500, seed=3)

    def share(rows, label):
        return sum(1 for r in rows if recommend.recommendation(r)[0] == label) / len(rows)

    for label in ("buy", "hold", "avoid"):
        assert abs(share(small, label) - share(big, label)) < 0.03


def test_sort_key_breaks_label_ties_by_percentile_without_crossing_labels():
    rows = [{"rating": "sell"}, {"rating": "strong_buy"},
            {"blended_pct": 0.65}, {"blended_pct": 0.95}]
    ordered = sorted(rows, key=recommend.sort_key, reverse=True)
    assert ordered[0]["rating"] == "strong_buy"
    assert ordered[-1]["rating"] == "sell"
    # The smallest gap in WEIGHT is sell 1.5 -> avoid 2.0, so the tie-break must never
    # let the best possible `sell` cross the worst possible `avoid`.
    best_sell = recommend.sort_key({"rating": "sell", "blended_pct": 1.0})
    worst_avoid = recommend.sort_key({"blended_pct": 0.0})
    assert best_sell < worst_avoid
    # Within one label it does order by percentile.
    assert (recommend.sort_key({"blended_pct": 0.99})
            > recommend.sort_key({"blended_pct": 0.91}))


def test_watch_is_a_researched_hold_you_do_not_own_with_an_armed_level():
    """`hold` means "you own it, keep it". Reusing it for "good company, wrong price"
    made the two indistinguishable exactly where the picker should say "wait for X"."""
    assert recommend.recommendation({"rating": "hold", "candidate_triggers": 2}) == \
        ("watch", "claude")
    assert recommend.recommendation({"rating": "hold", "entry_level": 88.0})[0] == "watch"
    # You own it -> it stays a hold, whatever is armed.
    assert recommend.recommendation(
        {"rating": "hold", "candidate_triggers": 2, "held_position": 10})[0] == "hold"
    # Nothing armed -> no claim about a level, so no watch.
    assert recommend.recommendation({"rating": "hold"})[0] == "hold"
    # Conviction-derived neutral ratings go the same way.
    assert recommend.recommendation({"conviction": 5, "candidate_triggers": 1})[0] == "watch"
    # A screen-only row is never a watch: nothing researched it, nothing armed a level.
    assert recommend.recommendation({"blended_pct": 0.75})[0] == "hold"


def test_watch_sorts_between_hold_and_avoid():
    assert (recommend.WEIGHT["avoid"] < recommend.WEIGHT["watch"]
            < recommend.WEIGHT["hold"])
    # The tie-break must still not let a `watch` cross a `hold` or an `avoid` a `watch`.
    best_avoid = recommend.sort_key({"blended_pct": 0.599})   # just under the hold cut
    worst_watch = recommend.sort_key({"rating": "hold", "candidate_triggers": 1,
                                      "blended_pct": 0.0})
    assert best_avoid < worst_watch
    assert recommend.BADGE["watch"].endswith("Watch")
