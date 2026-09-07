from src import health


def test_fresh_run_is_ok():
    hz = health.RunHealth()
    assert hz.overall == health.OK
    assert hz.problems() == []
    assert hz.summary() == "health ok"


def test_worsen_only_moves_down():
    hz = health.RunHealth()
    hz.degraded("broker", "offline")
    hz.ok("broker", "connected")          # cannot undo a degradation
    assert hz.stage("broker").status == health.DEGRADED
    assert hz.stage("broker").note == "offline"
    hz.failed("broker", "dead")
    hz.degraded("broker", "meh")           # cannot soften a failure
    assert hz.stage("broker").status == health.FAILED
    assert hz.stage("broker").note == "dead"


def test_overall_is_worst_stage_and_problems_sorted_worst_first():
    hz = health.RunHealth()
    hz.ok("fundamentals", "107/107")
    hz.degraded("sentiment", "ollama down")
    hz.failed("research", "no auth")
    assert hz.overall == health.FAILED
    assert [p[0] for p in hz.problems()] == ["research", "sentiment"]
    assert "research failed (no auth)" in hz.summary()


def test_json_round_trip_and_badge():
    hz = health.RunHealth()
    hz.degraded("broker", "IBKR offline")
    hz.ok("triggers", "3 checked")
    back = health.RunHealth.from_json(hz.to_json())
    assert back.overall == health.DEGRADED
    assert back.stage("broker").note == "IBKR offline"
    assert back.stage("triggers").status == health.OK
    assert health.RunHealth.from_json(None).overall == health.OK
    assert health.RunHealth.from_json("not json").overall == health.OK
    assert health.badge(None) == "⚪ unknown"
    assert health.badge("degraded").endswith("degraded")
