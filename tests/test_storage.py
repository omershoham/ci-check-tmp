from src import health


def test_init_db_is_idempotent(storage):
    storage.init_db()
    storage.init_db()
    with storage.get_conn() as cn:
        cols = {r["name"] for r in cn.execute("PRAGMA table_info(runs)")}
    assert {"health", "health_json", "progress", "heartbeat"} <= cols


def test_finish_run_persists_health(storage):
    rid = storage.create_run("light")
    hz = health.RunHealth()
    hz.degraded("broker", "IBKR offline")
    storage.finish_run(rid, "ok", "smoke", health=hz)
    row = storage.list_runs(1)[0]
    assert row["id"] == rid
    assert row["status"] == "ok"
    assert row["health"] == "degraded"
    back = health.RunHealth.from_json(row["health_json"])
    assert back.stage("broker").note == "IBKR offline"


def test_finish_run_without_health_leaves_null(storage):
    rid = storage.create_run("light")
    storage.finish_run(rid, "ok", "legacy caller")
    row = storage.list_runs(1)[0]
    assert row["health"] is None and row["health_json"] is None


def test_research_report_carries_prompt_provenance(storage):
    rid = storage.create_run("deep")
    storage.upsert_tickers([{"symbol": "NVDA", "name": "NVIDIA", "subsector": "semis_equipment"}])
    storage.upsert_screen_scores(rid, [{"symbol": "NVDA", "blended": 0.9, "rank": 1,
                                        "tags": [], "metrics": {}}])
    storage.upsert_research(rid, {"symbol": "NVDA", "rating": "buy", "conviction": 7,
                                  "prompt_version": 4, "model": "claude-test"})
    with storage.get_conn() as cn:
        r = cn.execute("SELECT prompt_version, model FROM research_reports "
                       "WHERE run_id=? AND symbol='NVDA'", (rid,)).fetchone()
    assert r["prompt_version"] == 4 and r["model"] == "claude-test"
    board = storage.get_board(rid)
    assert board[0]["rating"] == "buy"


def test_portfolio_run_health(storage):
    rid = storage.create_portfolio_run()
    hz = health.RunHealth()
    hz.failed("broker", "no holdings")
    storage.finish_portfolio_run(rid, "error", "none", health=hz)
    row = storage.latest_portfolio_run(status=None)
    assert row["health"] == "failed"
