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


def test_screen_only_fallback_and_badge_marker():
    assert recommend.recommendation({"blended": 0.7}) == ("buy", "screen")
    assert recommend.recommendation({"blended": 0.55}) == ("hold", "screen")
    assert recommend.recommendation({"blended": 0.1}) == ("avoid", "screen")
    assert recommend.badge({"blended": 0.7}).endswith(" *")
    assert not recommend.badge({"rating": "buy"}).endswith("*")


def test_sort_key_orders_bullish_first():
    rows = [{"rating": "sell"}, {"rating": "strong_buy"}, {"blended": 0.55}]
    ordered = sorted(rows, key=recommend.sort_key, reverse=True)
    assert ordered[0]["rating"] == "strong_buy"
    assert ordered[-1]["rating"] == "sell"
