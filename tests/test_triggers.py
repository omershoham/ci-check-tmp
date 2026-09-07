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
