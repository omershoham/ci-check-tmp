"""The catalyst vector's contract: the schema Ollama is constrained by, the dual
write that keeps the legacy column alive, and the theme pack that supplies the prose.

No network: the Ollama call is faked at the `requests.post` seam, so what is asserted
is our parsing and our schema, not the model's mood.
"""
from __future__ import annotations

import json

import pytest

from src import local_llm, theme


# ------------------------------------------------------------------- the schema
def test_all_nine_keys_are_required_by_the_schema():
    """Constrained decoding is the whole reason a 0 can be trusted: if a key were
    optional, an omitted key and a real 0 would be indistinguishable in the JSON."""
    cat = local_llm._CLASSIFY_SCHEMA["properties"]["catalyst"]
    assert tuple(cat["required"]) == local_llm.CATALYST_KEYS
    assert len(local_llm.CATALYST_KEYS) == 9
    for k in local_llm.CATALYST_KEYS:
        prop = cat["properties"][k]
        assert prop == {"type": "integer", "minimum": 0, "maximum": 10}
    assert "catalyst" in local_llm._CLASSIFY_SCHEMA["required"]
    assert local_llm.NEGATIVE_KEYS == {"adverse_event"}


def test_key_set_is_python_not_yaml():
    """Schema-key invariance: a theme varies prose only. If a pack could rename a key,
    two themes' signals could never be pooled and no IC could be computed on them."""
    for name in ("ai_infra", "general"):
        pack = theme.load(name)
        blob = json.dumps(pack).lower()
        for k in local_llm.CATALYST_KEYS:
            assert k not in blob, f"{name}.yaml must not define catalyst key {k}"


def test_classifier_is_theme_neutral(monkeypatch):
    """The classifier emits MEASURED FEATURES, so it must score the same evidence the
    same way under every pack — otherwise `customer_deal 8` means something different
    per theme and the two can never be pooled, which is the defect the single
    hyperscaler_signal had. Judgement (the pack's `catalyst_emphasis`) belongs in the
    Claude research prompt, where the output is an opinion, not a feature."""
    monkeypatch.setattr(theme, "configured_name", lambda: "ai_infra")
    sys_ai, p_ai = local_llm._system(), local_llm._prompt_tmpl()
    monkeypatch.setattr(theme, "configured_name", lambda: "general")
    sys_gen, p_gen = local_llm._system(), local_llm._prompt_tmpl()

    # The system prompt carries no theme at all.
    assert sys_ai == sys_gen
    for pack in ("ai_infra", "general"):
        assert theme.fragment("catalyst_emphasis", pack) not in sys_ai

    # The user prompt differs ONLY where the actor list is substituted: swapping one
    # pack's actors into the other's prompt reproduces it exactly.
    assert p_ai != p_gen
    assert p_ai.replace(theme.actors_text("ai_infra"),
                        theme.actors_text("general")) == p_gen


def test_regulatory_policy_is_favourable_only():
    """The key is directional. Without saying so the model scored an SEC probe as
    `regulatory_policy 5` — reading "regulatory" as a topic, not a direction — which
    double-counts an adverse event as a positive catalyst."""
    d = local_llm._KEY_DEFS["regulatory_policy"]
    assert "FAVOURABLE" in d and "adverse_event" in d
    assert "probes" in d and "score regulatory_policy 0" in d


def test_prompt_names_the_packs_actors(monkeypatch):
    monkeypatch.setattr(theme, "configured_name", lambda: "ai_infra")
    assert "NVIDIA" in local_llm._prompt_tmpl()
    monkeypatch.setattr(theme, "configured_name", lambda: "general")
    p = local_llm._prompt_tmpl()
    assert "NVIDIA" not in p and "government / defense primes" in p
    # Actors are RECOGNITION ("AWS is a large customer"), not weighting - so they are
    # the one thing a pack may put in a feature-producing prompt.
    # The key definitions themselves never move.
    for k in local_llm.CATALYST_KEYS:
        assert k in p


# ------------------------------------------------------------------- classify()
class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": json.dumps(self._p)}


def _fake_post(payload):
    def post(url, json=None, timeout=None):   # noqa: A002
        return _Resp(payload)
    return post


def _full(**kw):
    return {k: kw.get(k, 0) for k in local_llm.CATALYST_KEYS}


def test_classify_dual_writes_customer_deal_and_stamps_version(monkeypatch):
    monkeypatch.setattr(local_llm.requests, "post", _fake_post({
        "catalyst": _full(customer_deal=8, adverse_event=3),
        "sentiment": "bullish", "sentiment_score": 7,
        "catalysts": ["a deal"], "summary": "s"}))
    out = local_llm.classify("VRT", "Vertiv", ["[0d ago] headline"], check_available=False)
    assert out["catalyst"] == _full(customer_deal=8, adverse_event=3)
    assert out["hyperscaler_signal"] == out["catalyst"]["customer_deal"] == 8
    assert out["signal_version"] == local_llm.SIGNAL_VERSION == 2


def test_missing_key_is_unknown_not_zero(monkeypatch):
    """The schema promises every key, but the promise is the server's, not ours."""
    monkeypatch.setattr(local_llm.requests, "post", _fake_post({
        "catalyst": {"customer_deal": 5}, "sentiment": "neutral",
        "sentiment_score": 5, "catalysts": [], "summary": ""}))
    out = local_llm.classify("X", "X", ["h"], check_available=False)
    assert out["catalyst"]["customer_deal"] == 5
    assert out["catalyst"]["ma_event"] is None


def test_no_news_and_failure_paths_are_all_unknown(monkeypatch):
    no_news = local_llm.classify("X", "X", [], check_available=False)
    assert set(no_news["catalyst"].values()) == {None}
    assert no_news["sentiment_score"] == 5 and no_news["signal_version"] == 2

    def boom(*a, **k):
        raise local_llm.requests.RequestException("down")
    monkeypatch.setattr(local_llm.requests, "post", boom)
    err = local_llm.classify("X", "X", ["h"], check_available=False)
    assert set(err["catalyst"].values()) == {None}
    assert err["hyperscaler_signal"] is None and err["error"].startswith("error:")


# --------------------------------------------------------------------- storage
def test_signals_roundtrip_catalyst_json(storage, workspace):
    rid = storage.create_run("light")
    storage.upsert_tickers([{"symbol": "VRT", "name": "Vertiv", "subsector": "datacenter_power"}])
    storage.upsert_screen_scores(rid, [{"symbol": "VRT", "quality": 0.7, "blended": 0.7,
                                        "rank": 1, "tags": [], "metrics": {}}])
    vec = _full(customer_deal=8, adverse_event=2)
    storage.upsert_signals(rid, [{"symbol": "VRT", "catalyst": vec, "signal_version": 2,
                                  "hyperscaler_signal": 8, "catalysts": ["x"],
                                  "summary": "s"}])
    row = storage.get_board(rid)[0]
    assert row["catalyst_json"] == vec
    assert row["signal_version"] == 2
    assert row["hyperscaler_signal"] == row["catalyst_json"]["customer_deal"]
    # The deal column is frozen: the scorer no longer produces one.
    with storage.get_conn() as cn:
        assert cn.execute("SELECT deal FROM screen_scores WHERE run_id=?",
                          (rid,)).fetchone()["deal"] is None


# --------------------------------------------------- the replay must refuse to lie
def test_backtest_refuses_to_replay_a_weighted_unreplayable_pillar(monkeypatch):
    """At weight 0 the catalyst pillar is simply absent from the replayed composite
    and nothing is lost. At any weight > 0 the replay would keep running and quietly
    measure a DIFFERENT signal than the live screen - the Phase 4 parity bug in a
    form nobody would notice. It must raise instead."""
    from src.backtest import signals

    base = {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.35,
            "catalyst": 0.0, "learned": 0.0}
    assert signals.assert_replayable(base) == base          # dark: replay is honest

    for pillar in ("catalyst", "learned"):
        with pytest.raises(ValueError, match=f"scoring.pillars.{pillar}"):
            signals.assert_replayable(dict(base, **{pillar: 0.1}))

    # ...and it reads the live config when no weights are passed, so a weight set in
    # config.yaml cannot sneak past a caller that forgot to forward it.
    monkeypatch.setattr(signals.config, "get",
                        lambda k, d=None: (dict(base, catalyst=0.2)
                                           if k == "scoring.pillars" else d))
    with pytest.raises(ValueError, match="cannot replay the catalyst pillar"):
        signals.assert_replayable()


# ----------------------------------------------------------------- theme packs
def test_curated_universe_is_unchanged_by_the_move_to_a_pack():
    """The 105 curated tickers moved from config.yaml into themes/ai_infra.yaml. If a
    single name were dropped in the move, every backtest baseline would silently
    describe a different universe."""
    from src import universe
    syms = {s for lst in theme.curated_sectors("ai_infra").values() for s in lst}
    assert len(syms) == 105
    assert set(universe.symbols()) == syms
    for anchor in ("NVDA", "VRT", "ANET", "MSFT", "GOOG", "GOOGL", "ON"):
        assert anchor in syms
    assert set(theme.subsector_labels("ai_infra")) == set(theme.curated_sectors("ai_infra"))


def test_general_pack_is_theme_free():
    assert theme.curated_sectors("general") == {}
    assert "hyperscaler" not in theme.fragment("analyst_role", "general").lower()
    assert theme.board_name("general") and theme.actors("general")


def test_missing_pack_falls_back_instead_of_stopping_a_run():
    pack = theme.load("no_such_theme_pack")
    assert pack["name"] and pack["actors"] and pack["prompt_fragments"]["analyst_role"]


def test_theme_selection_follows_config(monkeypatch):
    cfg = {}
    monkeypatch.setattr(theme.config, "get", lambda k, d=None: cfg.get(k, d))
    cfg["universe.source"] = "curated"
    assert theme.configured_name() == "ai_infra"      # curated implies its theme
    cfg["universe.source"] = "broad"
    assert theme.configured_name() == "general"
    cfg["theme"] = "ai_infra"
    assert theme.configured_name() == "ai_infra"      # explicit always wins


def test_user_config_sectors_still_override_the_pack(monkeypatch):
    monkeypatch.setattr(theme.config, "get",
                        lambda k, d=None: ({"mine": ["AAA", "BBB"]}
                                           if k == "universe.sectors" else d))
    assert theme.curated_sectors() == {"mine": ["AAA", "BBB"]}
