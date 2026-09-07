from src import scoring


def _metrics():
    base = {"marketCap": 5e10, "price": 100.0, "avg_dollar_volume": 1e8, "freeCashflow": 1e9,
            "profitable": True, "trailingPE": 30, "forwardPE": 25, "enterpriseToEbitda": 20,
            "priceToSalesTrailing12Months": 8, "fcf_yield": 0.03, "fcf_margin": 0.2,
            "revenueGrowth": 0.2, "earningsGrowth": 0.2, "returnOnEquity": 0.3,
            "returnOnAssets": 0.1, "grossMargins": 0.6, "operatingMargins": 0.3,
            "debtToEquity": 20, "mom_50": 1.05, "mom_200": 1.1}
    out = {}
    for i, sym in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]):
        m = dict(base, symbol=sym, name=sym)
        m["trailingPE"] = 10 + 10 * i          # AAA cheapest ... FFF richest
        m["revenueGrowth"] = 0.05 * (i + 1)    # FFF fastest
        out[sym] = m
    out["FFF"]["trailingPE"] = -15             # loss-maker: negative PE must NOT rank cheapest
    out["FFF"]["profitable"] = False
    out["FFF"]["freeCashflow"] = -1e8
    out["EEE"]["days_to_earnings"] = 3
    return out


def _subs():
    return {"AAA": "semis_equipment", "BBB": "semis_equipment", "CCC": "cloud_ai_software",
            "DDD": "cloud_ai_software", "EEE": "datacenter_power", "FFF": "datacenter_power"}


def test_ranks_are_dense_and_scores_bounded():
    rows = scoring.score_universe(_metrics(), _subs())
    assert [r["rank"] for r in rows] == list(range(1, 7))
    for r in rows:
        for k in ("technical", "fundamental", "sentiment", "blended", "deal"):
            assert 0.0 <= r[k] <= 1.0
    assert rows == sorted(rows, key=lambda r: -r["blended"])


def test_tags_unprofitable_and_earnings_soon():
    by = {r["symbol"]: r for r in scoring.score_universe(_metrics(), _subs())}
    assert "higher-risk" in by["FFF"]["tags"]
    assert "quality" not in by["FFF"]["tags"]           # negative FCF blocks the quality tag
    assert "earnings-soon" in by["EEE"]["tags"]
    assert "earnings-soon" not in by["AAA"]["tags"]


def test_negative_pe_does_not_score_as_cheapest():
    by = {r["symbol"]: r for r in scoring.score_universe(_metrics(), _subs())}
    # Masked to neutral: FFF's valuation subscore must not beat the genuinely cheap AAA.
    assert by["FFF"]["metrics"]["sub_valuation"] <= by["AAA"]["metrics"]["sub_valuation"]


def test_sentiment_pillar_uses_tone_only():
    sig = {"AAA": {"sentiment_score": 9, "hyperscaler_signal": 0},
           "BBB": {"sentiment_score": 1, "hyperscaler_signal": 10}}
    by = {r["symbol"]: r for r in scoring.score_universe(_metrics(), _subs(), signals=sig)}
    assert by["AAA"]["sentiment"] > by["BBB"]["sentiment"]
    assert by["BBB"]["deal"] > by["AAA"]["deal"]          # relevance lives in the deal score
    assert by["CCC"]["sentiment"] == 0.5                  # unscanned -> neutral


def test_empty_universe():
    assert scoring.score_universe({}, {}) == []
