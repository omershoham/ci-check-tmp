from src import triggers


def _install(storage, sym="NVDA", run_id=1):
    storage.replace_triggers(sym, "portfolio", run_id, [
        {"kind": "price_below", "threshold": 100.0, "description": "support broke"},
        {"kind": "price_above", "threshold": 200.0, "description": "blow-off"},
        {"kind": "news_keyword", "keyword": "guidance cut", "description": "thesis break"},
    ])


def test_price_below_fires_and_marks_triggered(storage):
    _install(storage)
    fired = triggers.check_all(prices={"NVDA": 95.0}, news_map={})
    assert [t["kind"] for t in fired] == ["price_below"]
    assert "95.00 < 100" in fired[0]["trigger_detail"]
    active = storage.active_triggers()
    assert all(t["kind"] != "price_below" for t in active)   # fired one is no longer active
    assert len(storage.recent_fired_triggers()) == 1


def test_news_keyword_is_case_insensitive_substring(storage):
    _install(storage)
    fired = triggers.check_all(prices={"NVDA": 150.0},
                               news_map={"NVDA": [{"title": "NVIDIA GUIDANCE CUT shocks"}]})
    assert [t["kind"] for t in fired] == ["news_keyword"]


def test_missing_price_does_not_fire(storage):
    _install(storage)
    assert triggers.check_all(prices={}, news_map={}) == []
    assert len(storage.active_triggers()) == 3


def test_replace_supersedes_same_source_only(storage):
    _install(storage)
    storage.replace_triggers("NVDA", "research", 2, [
        {"kind": "price_below", "threshold": 50.0, "description": "r"}])
    storage.replace_triggers("NVDA", "portfolio", 3, [])   # empty set still supersedes
    active = storage.active_triggers()
    assert len(active) == 1 and active[0]["source"] == "research"


# ---------------------------------------------------------------- candidate triggers
def test_candidate_price_triggers_come_from_the_report_not_an_llm_call(storage, monkeypatch):
    """Phase 6 made `invalidation_level` a NUMBER, so paying Claude to re-read it would
    be asking Claude to transcribe Claude. The LLM call only adds news keywords."""
    calls = []
    monkeypatch.setattr(triggers.deep_research, "_run_json",
                        lambda p: (calls.append(p) or ({}, "")))
    n = triggers.extract_candidates(1, [
        {"symbol": "NVDA", "rating": "hold", "thesis": "great, wrong price",
         "invalidation_level": 88.0, "entry_level": 95.0, "latest_price_seen": 120.0}])
    assert n == 2
    got = {(t["kind"], t["threshold"]) for t in storage.active_triggers()}
    assert got == {("price_below", 88.0), ("price_below", 95.0)}
    assert all(t["source"] == "candidate" for t in storage.active_triggers())
    assert len(calls) == 1                     # ONE batched call for the whole cohort


def test_a_stated_entry_above_the_last_price_arms_the_upside(storage, monkeypatch):
    monkeypatch.setattr(triggers.deep_research, "_run_json", lambda p: ({}, ""))
    triggers.extract_candidates(1, [
        {"symbol": "AMD", "rating": "hold", "thesis": "waiting on the breakout",
         "entry_level": 150.0, "latest_price_seen": 130.0}])
    assert [(t["kind"], t["threshold"]) for t in storage.active_triggers()] == \
        [("price_above", 150.0)]


def test_a_broken_llm_call_cannot_wipe_the_price_triggers(storage, monkeypatch):
    """The ordering bug a 'deterministic pass, then LLM pass' version would have: the
    LLM's own replace_triggers would supersede the levels installed a moment earlier."""
    monkeypatch.setattr(triggers.deep_research, "_run_json",
                        lambda p: (None, "error: claude timed out"))
    n = triggers.extract_candidates(1, [
        {"symbol": "NVDA", "rating": "hold", "thesis": "t", "invalidation_level": 88.0}])
    assert n == 1 and len(storage.active_triggers()) == 1


def test_news_keywords_are_merged_into_the_same_installed_set(storage, monkeypatch):
    monkeypatch.setattr(triggers.deep_research, "_run_json", lambda p: (
        {"NVDA": [{"kind": "news_keyword", "keyword": "guidance cut", "description": "d"},
                  {"kind": "price_below", "threshold": 5.0}],   # price rows are ignored
         "ZZZZ": [{"kind": "news_keyword", "keyword": "ignore me"}]}, ""))
    triggers.extract_candidates(1, [
        {"symbol": "NVDA", "rating": "hold", "thesis": "t", "invalidation_level": 88.0}])
    active = storage.active_triggers()
    assert {t["kind"] for t in active} == {"price_below", "news_keyword"}
    assert {t["symbol"] for t in active} == {"NVDA"}   # a symbol not researched is dropped


def test_re_researching_supersedes_only_the_candidate_source(storage, monkeypatch):
    monkeypatch.setattr(triggers.deep_research, "_run_json", lambda p: ({}, ""))
    _install(storage)                                  # 3 portfolio triggers on NVDA
    triggers.extract_candidates(1, [
        {"symbol": "NVDA", "rating": "hold", "thesis": "t", "invalidation_level": 88.0}])
    triggers.extract_candidates(2, [
        {"symbol": "NVDA", "rating": "hold", "thesis": "t", "invalidation_level": 70.0}])
    cand = [t for t in storage.active_triggers() if t["source"] == "candidate"]
    port = [t for t in storage.active_triggers() if t["source"] == "portfolio"]
    assert [t["threshold"] for t in cand] == [70.0]    # old level no longer armed
    assert len(port) == 3                              # the dive's triggers survive
