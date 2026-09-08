import pytest

from src import local_llm, scoring


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
        for k in ("technical", "fundamental", "sentiment", "blended"):
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


def _vec(**kw):
    """A full 9-key catalyst vector: unnamed keys are a MEASURED 0, not unknown."""
    return {k: kw.get(k, 0) for k in local_llm.CATALYST_KEYS}


def test_sentiment_pillar_uses_tone_only():
    sig = {"AAA": {"sentiment_score": 9, "catalyst": _vec()},
           "BBB": {"sentiment_score": 1, "catalyst": _vec(customer_deal=10)}}
    by = {r["symbol"]: r for r in scoring.score_universe(_metrics(), _subs(), signals=sig)}
    assert by["AAA"]["sentiment"] > by["BBB"]["sentiment"]
    # Events live in the catalyst pillar, never in the sentiment pillar.
    assert by["BBB"]["metrics"]["pillar_catalyst"] > by["AAA"]["metrics"]["pillar_catalyst"]
    assert by["CCC"]["sentiment"] == 0.5                  # unscanned -> neutral


# ------------------------------------------------------------- catalyst pillar
def test_catalyst_pillar_equal_weights_and_adverse_subtracts():
    w = scoring.catalyst_subweights()
    assert len(set(w.values())) == 1                      # equal by default, on purpose
    n = len(local_llm.CATALYST_KEYS)
    # One key at 10, the rest a measured 0 -> 1/n of the way up the scale.
    assert scoring._catalyst_pillar({"catalyst": _vec(customer_deal=10)}) == \
        pytest.approx(1.0 / n)
    # The negative axis SUBTRACTS: the same magnitude on adverse_event mirrors it.
    assert scoring._catalyst_pillar({"catalyst": _vec(adverse_event=10)}) == \
        pytest.approx(-1.0 / n)
    # ...and it can cancel a positive of the same size.
    assert scoring._catalyst_pillar(
        {"catalyst": _vec(customer_deal=8, adverse_event=8)}) == pytest.approx(0.0)


def test_catalyst_pillar_genuine_zero_is_not_unknown():
    """A measured all-zero vector is a real bottom-of-scale reading; an unmeasured one
    is None. Collapsing the two would score a name we never scanned as if we had."""
    assert scoring._catalyst_pillar({"catalyst": _vec()}) == 0.0
    assert scoring._catalyst_pillar({"catalyst": {k: None for k in local_llm.CATALYST_KEYS}}) is None
    assert scoring._catalyst_pillar({}) is None
    assert scoring._catalyst_pillar(None) is None
    # A partially-measured vector averages over the keys that WERE measured.
    assert scoring._catalyst_pillar({"catalyst": {"customer_deal": 10}}) == pytest.approx(1.0)


def test_catalyst_pillar_is_persisted_but_dark():
    sig = {"AAA": {"sentiment_score": 5, "catalyst": _vec(customer_deal=9),
                   "signal_version": 2}}
    rows = scoring.score_universe(_metrics(), _subs(), signals=sig)
    by = {r["symbol"]: r for r in rows}
    m = by["AAA"]["metrics"]
    assert m["pillar_catalyst"] is not None and m["signal_version"] == 2
    assert m["catalyst"]["customer_deal"] == 9            # raw keys persisted too
    assert by["BBB"]["metrics"]["pillar_catalyst"] is None   # never scanned
    # Weight 0 -> the composite must be exactly what it is without any signal at all.
    plain = {r["symbol"]: r["blended"] for r in scoring.score_universe(_metrics(), _subs())}
    assert all(r["blended"] == plain[r["symbol"]] for r in rows if r["symbol"] != "AAA")


def test_pillar_set_lists_weighted_pillars_only(monkeypatch):
    """A dark pillar must not enter pillar_set: it changes nothing in `blended`, and
    splitting the IC study's partition key for no change would strand the history."""
    monkeypatch.setattr(scoring.config, "get",
                        lambda k, d=None: ({"technical": 0.35, "fundamental": 0.30,
                                            "sentiment": 0.35, "catalyst": 0.0,
                                            "learned": 0.0}
                                           if k == "scoring.pillars" else d))
    assert scoring.pillar_set() == "technical,fundamental,sentiment"
    monkeypatch.setattr(scoring.config, "get",
                        lambda k, d=None: ({"technical": 0.35, "fundamental": 0.30,
                                            "sentiment": 0.25, "catalyst": 0.10}
                                           if k == "scoring.pillars" else d))
    assert scoring.pillar_set() == "technical,fundamental,sentiment,catalyst"


def test_deal_score_and_tags_are_gone():
    """The deal score was 35% static per-sector prior + a momentum term that
    double-counted the technical pillar. Nothing may resurrect it."""
    assert not hasattr(scoring, "SUBSECTOR_RELEVANCE")
    for r in scoring.score_universe(_metrics(), _subs()):
        assert "deal" not in r
        assert not ({"deal", "both"} & set(r["tags"]))


def test_empty_universe():
    assert scoring.score_universe({}, {}) == []
