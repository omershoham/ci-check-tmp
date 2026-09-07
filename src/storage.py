"""SQLite persistence: schema, connection, and upsert/query helpers.

One DB at paths.db. All writes go through small helpers so the pipeline and the
dashboard share a single source of truth.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable

from . import config


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Schema/migrations run once per process on first connection, so EVERY entry point
# (pipeline, dashboard, digest, learning CLI, ad-hoc scripts) sees the current
# columns. Before this, a reader that skipped init_db() could hit "no such column"
# right after an upgrade added one.
_schema_ready = False


def get_conn() -> sqlite3.Connection:
    global _schema_ready
    conn = sqlite3.connect(config.db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    # Wait up to 10s for a lock instead of raising 'database is locked' immediately when
    # the dashboard is reading while the pipeline writes.
    conn.execute("PRAGMA busy_timeout=10000;")
    if not _schema_ready:
        _schema_ready = True          # set first: init_db -> get_conn must not recurse
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    mode       TEXT NOT NULL,                 -- 'deep' | 'light'
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status     TEXT NOT NULL DEFAULT 'running', -- running|ok|error
    notes      TEXT
);

CREATE TABLE IF NOT EXISTS tickers (
    symbol    TEXT PRIMARY KEY,
    name      TEXT,
    subsector TEXT
);

CREATE TABLE IF NOT EXISTS screen_scores (
    run_id       INTEGER NOT NULL,
    symbol       TEXT NOT NULL,
    quality      REAL,
    deal         REAL,
    blended      REAL,
    rank         INTEGER,
    tags         TEXT,         -- JSON array of strings
    metrics_json TEXT,         -- JSON of raw fundamentals/subscores
    PRIMARY KEY (run_id, symbol),
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS signals (
    run_id            INTEGER NOT NULL,
    symbol            TEXT NOT NULL,
    hyperscaler_signal REAL,
    catalysts_json    TEXT,    -- JSON array
    summary           TEXT,
    PRIMARY KEY (run_id, symbol),
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS research_reports (
    run_id          INTEGER NOT NULL,
    symbol          TEXT NOT NULL,
    rating          TEXT,      -- strong_buy|buy|hold|sell|strong_sell
    conviction      REAL,
    deal_likelihood TEXT,
    thesis          TEXT,
    chart_pattern   TEXT,
    metrics_read    TEXT,
    data_as_of      TEXT,
    catalysts_json  TEXT,      -- JSON array
    risks_json      TEXT,      -- JSON array
    bull            TEXT,
    bear            TEXT,
    raw             TEXT,      -- raw Claude output (fallback if JSON parse fails)
    created_at      TEXT,
    PRIMARY KEY (run_id, symbol),
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS watchlist (
    symbol   TEXT PRIMARY KEY,
    status   TEXT NOT NULL,    -- 'pinned' | 'excluded'
    added_at TEXT
);

CREATE TABLE IF NOT EXISTS holdings (
    symbol       TEXT PRIMARY KEY,
    position     REAL,
    avg_cost     REAL,
    market_price REAL,
    updated_at   TEXT
);

CREATE TABLE IF NOT EXISTS quotes (
    symbol     TEXT PRIMARY KEY,
    last       REAL,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS picks (
    run_id           INTEGER NOT NULL,
    symbol           TEXT NOT NULL,
    surfaced_at      TEXT,
    price_at_surface REAL,
    PRIMARY KEY (run_id, symbol),
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

-- On-demand portfolio deep-dive (history retained for audit/change-tracking).
CREATE TABLE IF NOT EXISTS portfolio_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL DEFAULT 'running',  -- running|ok|error
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS portfolio_positions (
    run_id         INTEGER NOT NULL,
    symbol         TEXT NOT NULL,
    shares         REAL,
    avg_cost       REAL,
    last           REAL,
    pnl_pct        REAL,
    action         TEXT,   -- add|hold|trim|exit
    conviction     REAL,
    thesis_health  TEXT,
    valuation      TEXT,
    add_level      REAL,
    trim_level     REAL,
    invalidation   TEXT,
    risks_json     TEXT,
    catalysts_json TEXT,   -- JSON list of {date, event}
    data_as_of     TEXT,
    sources_json   TEXT,   -- JSON list of {title, date, url}
    validation_json TEXT,  -- {is_current, stale_items[], confidence, checked_as_of}
    raw            TEXT,
    PRIMARY KEY (run_id, symbol),
    FOREIGN KEY (run_id) REFERENCES portfolio_runs(id) ON DELETE CASCADE
);

-- Self-improvement loop: durable lessons distilled from graded past calls.
-- Injected into future research prompts so the analysis learns from outcomes.
CREATE TABLE IF NOT EXISTS lessons (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT,
    scope      TEXT,             -- 'global' | 'symbol' | 'pillar'
    symbol     TEXT,             -- for scope='symbol'
    lesson     TEXT NOT NULL,    -- the actionable takeaway
    evidence   TEXT,             -- which calls/outcomes support it
    status     TEXT NOT NULL DEFAULT 'active'  -- active | retired
);

-- Machine-checkable invalidation triggers distilled from deep-research theses.
-- The daily runs evaluate these so a broken thesis is flagged the day it breaks.
CREATE TABLE IF NOT EXISTS triggers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    source      TEXT,             -- 'portfolio' | 'research'
    run_id      INTEGER,          -- originating run (portfolio_runs.id or runs.id)
    kind        TEXT NOT NULL,    -- 'price_below' | 'price_above' | 'news_keyword'
    threshold   REAL,             -- for price triggers
    keyword     TEXT,             -- for news triggers (case-insensitive substring)
    description TEXT,             -- human-readable: what this trigger means
    status      TEXT NOT NULL DEFAULT 'active',  -- active | triggered | superseded
    created_at  TEXT,
    triggered_at TEXT,
    trigger_detail TEXT           -- what tripped it (price seen / headline matched)
);

CREATE TABLE IF NOT EXISTS portfolio_summary (
    run_id               INTEGER PRIMARY KEY,
    overall_risk         TEXT,
    concentration        TEXT,
    correlation          TEXT,
    diversification_gaps TEXT,
    weakest_link         TEXT,
    ranked_actions_json  TEXT,
    summary              TEXT,
    raw                  TEXT,
    FOREIGN KEY (run_id) REFERENCES portfolio_runs(id) ON DELETE CASCADE
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Lightweight additive migrations for DBs created before a column existed."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(research_reports)")}
    if "rating" not in cols:
        conn.execute("ALTER TABLE research_reports ADD COLUMN rating TEXT")
    if "chart_pattern" not in cols:
        conn.execute("ALTER TABLE research_reports ADD COLUMN chart_pattern TEXT")
    if "data_as_of" not in cols:
        conn.execute("ALTER TABLE research_reports ADD COLUMN data_as_of TEXT")
    if "metrics_read" not in cols:
        conn.execute("ALTER TABLE research_reports ADD COLUMN metrics_read TEXT")
    pcols = {r["name"] for r in conn.execute("PRAGMA table_info(portfolio_positions)")}
    if "weight_pct" not in pcols:
        conn.execute("ALTER TABLE portfolio_positions ADD COLUMN weight_pct REAL")
    if "target_weight_pct" not in pcols:
        conn.execute("ALTER TABLE portfolio_positions ADD COLUMN target_weight_pct REAL")
    if "action_note" not in pcols:
        conn.execute("ALTER TABLE portfolio_positions ADD COLUMN action_note TEXT")
    scols = {r["name"] for r in conn.execute("PRAGMA table_info(portfolio_summary)")}
    if "risk_json" not in scols:
        conn.execute("ALTER TABLE portfolio_summary ADD COLUMN risk_json TEXT")
    if "ballast_json" not in scols:
        conn.execute("ALTER TABLE portfolio_summary ADD COLUMN ballast_json TEXT")
    rcols = {r["name"] for r in conn.execute("PRAGMA table_info(research_reports)")}
    if "red_team_json" not in rcols:
        conn.execute("ALTER TABLE research_reports ADD COLUMN red_team_json TEXT")
    sscols = {r["name"] for r in conn.execute("PRAGMA table_info(screen_scores)")}
    if "score_version" not in sscols:
        conn.execute("ALTER TABLE screen_scores ADD COLUMN score_version INTEGER")
    lcols = {r["name"] for r in conn.execute("PRAGMA table_info(lessons)")}
    if lcols and "regime" not in lcols:
        conn.execute("ALTER TABLE lessons ADD COLUMN regime TEXT")
    for tbl in ("runs", "portfolio_runs"):
        cc = {r["name"] for r in conn.execute(f"PRAGMA table_info({tbl})")}
        if "progress" not in cc:
            conn.execute(f"ALTER TABLE {tbl} ADD COLUMN progress TEXT")
        if "heartbeat" not in cc:
            conn.execute(f"ALTER TABLE {tbl} ADD COLUMN heartbeat TEXT")
        # Per-stage data health (src/health.py): status says the process finished,
        # health says whether its inputs were whole. NULL on pre-feature runs.
        if "health" not in cc:
            conn.execute(f"ALTER TABLE {tbl} ADD COLUMN health TEXT")
        if "health_json" not in cc:
            conn.execute(f"ALTER TABLE {tbl} ADD COLUMN health_json TEXT")
    # Market regime at run time, so calibration can be cut by regime later (the
    # date-keyed regime cache is swept after 7 days and cannot be reconstructed).
    rcols = {r["name"] for r in conn.execute("PRAGMA table_info(runs)")}
    if "regime" not in rcols:
        conn.execute("ALTER TABLE runs ADD COLUMN regime TEXT")
    # Which prompt/model produced a report: calibration across prompt changes is
    # otherwise comparing different "analysts" as one (mirrors screen_scores.score_version).
    for tbl in ("research_reports", "portfolio_positions"):
        cc = {r["name"] for r in conn.execute(f"PRAGMA table_info({tbl})")}
        if "prompt_version" not in cc:
            conn.execute(f"ALTER TABLE {tbl} ADD COLUMN prompt_version INTEGER")
        if "model" not in cc:
            conn.execute(f"ALTER TABLE {tbl} ADD COLUMN model TEXT")


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


# ---------------------------------------------------------------- runs
def create_run(mode: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO runs(mode, started_at, status) VALUES (?,?,'running')",
            (mode, now_iso()),
        )
        return int(cur.lastrowid)


def set_progress(run_id: int, text: str, table: str = "runs") -> None:
    """Live progress heartbeat for a running pipeline / deep-dive. The dashboard
    shows it and flags a stall when the heartbeat goes quiet too long."""
    assert table in ("runs", "portfolio_runs")
    with get_conn() as conn:
        conn.execute(f"UPDATE {table} SET progress=?, heartbeat=? WHERE id=?",
                     (text[:200], now_iso(), run_id))


def finish_run(run_id: int, status: str = "ok", notes: str | None = None,
               health: Any | None = None, regime: str | None = None) -> None:
    """`health` is a src.health.RunHealth (or None for callers that don't track it);
    `regime` is the market regime label detected during the run (kept if None)."""
    h_overall = health.overall if health is not None else None
    h_json = health.to_json() if health is not None else None
    with get_conn() as conn:
        conn.execute(
            "UPDATE runs SET finished_at=?, status=?, notes=?, health=?, health_json=?, "
            "regime=COALESCE(?, regime) WHERE id=?",
            (now_iso(), status, notes, h_overall, h_json, regime, run_id),
        )


def mark_stale_runs(max_age_hours: float = 3.0) -> int:
    """Mark orphaned 'running' runs as errored — but ONLY those older than any
    plausible runtime. Without the age guard, the 14:00 scheduled pipeline would
    error-mark a manual portfolio deep-dive launched at 13:50 while it is still
    live (and the dashboard would then allow a second concurrent dive)."""
    cutoff = f"-{max_age_hours} hours"
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE runs SET status='error', finished_at=?, "
            "notes=COALESCE(notes,'')||' [auto: marked stale on next start]' "
            "WHERE status='running' AND datetime(started_at) < datetime('now', ?)",
            (now_iso(), cutoff))
        cur2 = conn.execute(
            "UPDATE portfolio_runs SET status='error', finished_at=?, "
            "notes=COALESCE(notes,'')||' [auto: marked stale]' "
            "WHERE status='running' AND datetime(started_at) < datetime('now', ?)",
            (now_iso(), cutoff))
        return int(cur.rowcount or 0) + int(cur2.rowcount or 0)


def score_history(symbol: str, limit: int = 12) -> list[tuple[int, float]]:
    """Composite score over recent completed runs, oldest→newest, for a sparkline."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT s.run_id, s.blended FROM screen_scores s
               JOIN runs r ON r.id = s.run_id
               WHERE s.symbol = ? AND r.status = 'ok' AND s.blended IS NOT NULL
               ORDER BY s.run_id DESC LIMIT ?""", (symbol.upper(), limit)).fetchall()
    return [(int(r["run_id"]), float(r["blended"])) for r in reversed(rows)]


def latest_run(mode: str | None = None) -> sqlite3.Row | None:
    q = "SELECT * FROM runs WHERE status='ok'"
    args: tuple = ()
    if mode:
        q += " AND mode=?"
        args = (mode,)
    q += " ORDER BY id DESC LIMIT 1"
    with get_conn() as conn:
        return conn.execute(q, args).fetchone()


def list_runs(limit: int = 30) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()


def runs_with_research() -> set[int]:
    """Run ids that contain at least one Claude deep-research report."""
    with get_conn() as conn:
        return {r["run_id"] for r in conn.execute(
            "SELECT DISTINCT run_id FROM research_reports")}


# ---------------------------------------------------------------- tickers
def upsert_tickers(rows: Iterable[dict[str, Any]]) -> None:
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO tickers(symbol,name,subsector) VALUES(:symbol,:name,:subsector)
               ON CONFLICT(symbol) DO UPDATE SET name=excluded.name, subsector=excluded.subsector""",
            list(rows),
        )


# ---------------------------------------------------------------- scores
def upsert_screen_scores(run_id: int, rows: Iterable[dict[str, Any]]) -> None:
    from .scoring import effective_score_version
    version = effective_score_version()
    payload = []
    for r in rows:
        payload.append({
            "run_id": run_id,
            "symbol": r["symbol"],
            "quality": r.get("quality"),
            "deal": r.get("deal"),
            "blended": r.get("blended"),
            "rank": r.get("rank"),
            "tags": json.dumps(r.get("tags", [])),
            "metrics_json": json.dumps(r.get("metrics", {})),
            "score_version": version,
        })
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO screen_scores(run_id,symbol,quality,deal,blended,rank,tags,metrics_json,score_version)
               VALUES(:run_id,:symbol,:quality,:deal,:blended,:rank,:tags,:metrics_json,:score_version)
               ON CONFLICT(run_id,symbol) DO UPDATE SET
                 quality=excluded.quality, deal=excluded.deal, blended=excluded.blended,
                 rank=excluded.rank, tags=excluded.tags, metrics_json=excluded.metrics_json,
                 score_version=excluded.score_version""",
            payload,
        )


def upsert_signals(run_id: int, rows: Iterable[dict[str, Any]]) -> None:
    payload = [{
        "run_id": run_id,
        "symbol": r["symbol"],
        "hyperscaler_signal": r.get("hyperscaler_signal"),
        "catalysts_json": json.dumps(r.get("catalysts", [])),
        "summary": r.get("summary"),
    } for r in rows]
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO signals(run_id,symbol,hyperscaler_signal,catalysts_json,summary)
               VALUES(:run_id,:symbol,:hyperscaler_signal,:catalysts_json,:summary)
               ON CONFLICT(run_id,symbol) DO UPDATE SET
                 hyperscaler_signal=excluded.hyperscaler_signal,
                 catalysts_json=excluded.catalysts_json, summary=excluded.summary""",
            payload,
        )


def upsert_research(run_id: int, r: dict[str, Any]) -> None:
    row = {
        "run_id": run_id,
        "symbol": r["symbol"],
        "rating": r.get("rating"),
        "conviction": r.get("conviction"),
        "deal_likelihood": r.get("deal_likelihood"),
        "thesis": r.get("thesis"),
        "chart_pattern": r.get("chart_pattern"),
        "metrics_read": r.get("metrics_read"),
        "data_as_of": r.get("data_as_of"),
        "catalysts_json": json.dumps(r.get("catalysts", [])),
        "risks_json": json.dumps(r.get("risks", [])),
        "red_team_json": json.dumps(r.get("red_team", {})),
        "bull": r.get("bull"),
        "bear": r.get("bear"),
        "raw": r.get("raw"),
        "prompt_version": r.get("prompt_version"),
        "model": r.get("model"),
        "created_at": now_iso(),
    }
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO research_reports
               (run_id,symbol,rating,conviction,deal_likelihood,thesis,chart_pattern,metrics_read,data_as_of,catalysts_json,risks_json,red_team_json,bull,bear,raw,prompt_version,model,created_at)
               VALUES(:run_id,:symbol,:rating,:conviction,:deal_likelihood,:thesis,:chart_pattern,:metrics_read,:data_as_of,:catalysts_json,:risks_json,:red_team_json,:bull,:bear,:raw,:prompt_version,:model,:created_at)
               ON CONFLICT(run_id,symbol) DO UPDATE SET
                 rating=excluded.rating, conviction=excluded.conviction,
                 deal_likelihood=excluded.deal_likelihood,
                 thesis=excluded.thesis, chart_pattern=excluded.chart_pattern,
                 metrics_read=excluded.metrics_read, data_as_of=excluded.data_as_of,
                 catalysts_json=excluded.catalysts_json,
                 risks_json=excluded.risks_json, red_team_json=excluded.red_team_json,
                 bull=excluded.bull, bear=excluded.bear,
                 raw=excluded.raw, prompt_version=excluded.prompt_version,
                 model=excluded.model, created_at=excluded.created_at""",
            row,
        )


# ---------------------------------------------------------------- board read
def get_board(run_id: int) -> list[dict[str, Any]]:
    """Joined view for the dashboard: scores + signals + research per symbol."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT s.symbol, t.name, t.subsector, s.quality, s.deal, s.blended,
                   s.rank, s.tags, s.metrics_json,
                   g.hyperscaler_signal, g.catalysts_json AS sig_catalysts, g.summary,
                   r.rating, r.conviction, r.deal_likelihood, r.thesis, r.chart_pattern,
                   r.metrics_read, r.data_as_of, r.created_at AS research_at,
                   r.catalysts_json AS res_catalysts, r.risks_json, r.red_team_json,
                   r.bull, r.bear, r.raw,
                   h.position AS held_position, h.avg_cost AS held_avg_cost,
                   q.last AS quote_last
            FROM screen_scores s
            LEFT JOIN tickers t ON t.symbol = s.symbol
            LEFT JOIN signals g ON g.run_id = s.run_id AND g.symbol = s.symbol
            LEFT JOIN research_reports r ON r.run_id = s.run_id AND r.symbol = s.symbol
            LEFT JOIN holdings h ON h.symbol = s.symbol
            LEFT JOIN quotes q ON q.symbol = s.symbol
            WHERE s.run_id = ?
            ORDER BY s.rank ASC
            """,
            (run_id,),
        ).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        for k in ("tags", "metrics_json", "sig_catalysts", "res_catalysts", "risks_json",
                  "red_team_json"):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except (json.JSONDecodeError, TypeError):
                    pass
        # Expose the parsed fundamentals/pillars under a stable 'metrics' key.
        d["metrics"] = d.pop("metrics_json", {}) or {}
        out.append(d)
    return out


# ---------------------------------------------------------------- watchlist / holdings / quotes
def set_watch(symbol: str, status: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO watchlist(symbol,status,added_at) VALUES(?,?,?)
               ON CONFLICT(symbol) DO UPDATE SET status=excluded.status, added_at=excluded.added_at""",
            (symbol.upper(), status, now_iso()),
        )


def get_watch() -> dict[str, str]:
    with get_conn() as conn:
        return {r["symbol"]: r["status"] for r in conn.execute("SELECT symbol,status FROM watchlist")}


def upsert_holdings(rows: Iterable[dict[str, Any]]) -> None:
    payload = [{
        "symbol": r["symbol"], "position": r.get("position"),
        "avg_cost": r.get("avg_cost"), "market_price": r.get("market_price"),
        "updated_at": now_iso(),
    } for r in rows]
    with get_conn() as conn:
        conn.execute("DELETE FROM holdings")
        # ON CONFLICT is defense-in-depth: get_positions() already aggregates per symbol,
        # but if a duplicate ever slips through this updates rather than crashing the run.
        conn.executemany(
            """INSERT INTO holdings(symbol,position,avg_cost,market_price,updated_at)
               VALUES(:symbol,:position,:avg_cost,:market_price,:updated_at)
               ON CONFLICT(symbol) DO UPDATE SET
                 position=excluded.position, avg_cost=excluded.avg_cost,
                 market_price=excluded.market_price, updated_at=excluded.updated_at""",
            payload,
        )


def upsert_quotes(rows: Iterable[dict[str, Any]]) -> None:
    payload = [{"symbol": r["symbol"], "last": r.get("last"), "updated_at": now_iso()} for r in rows]
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO quotes(symbol,last,updated_at) VALUES(:symbol,:last,:updated_at)
               ON CONFLICT(symbol) DO UPDATE SET last=excluded.last, updated_at=excluded.updated_at""",
            payload,
        )


def record_picks(run_id: int, rows: Iterable[dict[str, Any]]) -> None:
    payload = [{
        "run_id": run_id, "symbol": r["symbol"],
        "surfaced_at": now_iso(), "price_at_surface": r.get("price"),
    } for r in rows]
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO picks(run_id,symbol,surfaced_at,price_at_surface)
               VALUES(:run_id,:symbol,:surfaced_at,:price_at_surface)
               ON CONFLICT(run_id,symbol) DO NOTHING""",
            payload,
        )


# ---------------------------------------------------------------- lessons
def add_lessons(rows: Iterable[dict[str, Any]], regime: str | None = None) -> None:
    payload = [{"created_at": now_iso(), "scope": r.get("scope", "global"),
                "symbol": (r.get("symbol") or "").upper() or None,
                "lesson": r["lesson"], "evidence": r.get("evidence"),
                "regime": r.get("regime") or regime}
               for r in rows if r.get("lesson")]
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO lessons(created_at,scope,symbol,lesson,evidence,regime)
               VALUES(:created_at,:scope,:symbol,:lesson,:evidence,:regime)""", payload)


def active_lessons(scope: str | None = None, symbol: str | None = None,
                   limit: int = 8) -> list[dict[str, Any]]:
    q = "SELECT * FROM lessons WHERE status='active'"
    args: list = []
    if scope:
        q += " AND scope=?"; args.append(scope)
    if symbol:
        q += " AND symbol=?"; args.append(symbol.upper())
    q += " ORDER BY id DESC LIMIT ?"; args.append(limit)
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(q, args)]


def retire_old_lessons(keep_latest: int = 12, max_age_days: int = 56) -> None:
    """Keep the lesson set small and fresh — a lesson distilled from one selloff
    week must not steer prompts forever. Age-expiry means a truly durable lesson
    has to be re-confirmed by a later reflection to stay active."""
    with get_conn() as conn:
        conn.execute(
            """UPDATE lessons SET status='retired' WHERE scope='global' AND id NOT IN
               (SELECT id FROM lessons WHERE scope='global' AND status='active'
                ORDER BY id DESC LIMIT ?)""", (keep_latest,))
        conn.execute(
            "UPDATE lessons SET status='retired' WHERE status='active' AND "
            "date(created_at) < date('now', ?)", (f"-{int(max_age_days)} days",))


# ---------------------------------------------------------------- triggers
def replace_triggers(symbol: str, source: str, run_id: int,
                     rows: Iterable[dict[str, Any]]) -> None:
    """Supersede a symbol's previous active triggers FROM THE SAME SOURCE and install
    the new set (may be empty — a re-dived symbol whose new extraction produced no
    triggers must not keep stale price thresholds armed). Cross-source triggers
    survive: a research-installed trigger isn't wiped by a portfolio dive."""
    payload = [{
        "symbol": symbol.upper(), "source": source, "run_id": run_id,
        "kind": r.get("kind"), "threshold": r.get("threshold"),
        "keyword": r.get("keyword"), "description": r.get("description"),
        "created_at": now_iso(),
    } for r in rows if r.get("kind")]
    with get_conn() as conn:
        conn.execute("UPDATE triggers SET status='superseded' "
                     "WHERE symbol=? AND source=? AND status='active'",
                     (symbol.upper(), source))
        conn.executemany(
            """INSERT INTO triggers(symbol,source,run_id,kind,threshold,keyword,
                                    description,status,created_at)
               VALUES(:symbol,:source,:run_id,:kind,:threshold,:keyword,
                      :description,'active',:created_at)""",
            payload,
        )


def active_triggers() -> list[dict[str, Any]]:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM triggers WHERE status='active' ORDER BY symbol")]


def fire_trigger(trigger_id: int, detail: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE triggers SET status='triggered', triggered_at=?, "
                     "trigger_detail=? WHERE id=?", (now_iso(), detail[:300], trigger_id))


def recent_fired_triggers(days: int = 7) -> list[dict[str, Any]]:
    # Compare on date(): stored ISO has a 'T' separator that string-compares above
    # SQLite's space-separated datetime('now'), silently widening the window.
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM triggers WHERE status='triggered' AND date(triggered_at) >= "
            "date('now', ?) ORDER BY triggered_at DESC", (f"-{int(days)} days",))]


# ---------------------------------------------------------------- portfolio deep-dive
def create_portfolio_run() -> int:
    mark_stale_runs()  # sweep our own orphans too (killed dives otherwise stay 'running')
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO portfolio_runs(started_at, status) VALUES (?, 'running')", (now_iso(),))
        return int(cur.lastrowid)


def finish_portfolio_run(run_id: int, status: str = "ok", notes: str | None = None,
                         health: Any | None = None) -> None:
    h_overall = health.overall if health is not None else None
    h_json = health.to_json() if health is not None else None
    with get_conn() as conn:
        conn.execute("UPDATE portfolio_runs SET finished_at=?, status=?, notes=?, "
                     "health=?, health_json=? WHERE id=?",
                     (now_iso(), status, notes, h_overall, h_json, run_id))


def latest_portfolio_run(status: str | None = "ok") -> sqlite3.Row | None:
    q = "SELECT * FROM portfolio_runs"
    args: tuple = ()
    if status:
        q += " WHERE status=?"; args = (status,)
    q += " ORDER BY id DESC LIMIT 1"
    with get_conn() as conn:
        return conn.execute(q, args).fetchone()


def running_portfolio_run() -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM portfolio_runs WHERE status='running' ORDER BY id DESC LIMIT 1").fetchone()


def list_portfolio_runs(limit: int = 20) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM portfolio_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def upsert_portfolio_position(run_id: int, p: dict[str, Any]) -> None:
    row = {
        "run_id": run_id, "symbol": p["symbol"], "shares": p.get("shares"),
        "avg_cost": p.get("avg_cost"), "last": p.get("last"), "pnl_pct": p.get("pnl_pct"),
        "action": p.get("action"), "conviction": p.get("conviction"),
        "weight_pct": p.get("weight_pct"), "target_weight_pct": p.get("target_weight_pct"),
        "action_note": p.get("action_note"),
        "thesis_health": p.get("thesis_health"), "valuation": p.get("valuation"),
        "add_level": p.get("add_level"), "trim_level": p.get("trim_level"),
        "invalidation": p.get("invalidation"),
        "risks_json": json.dumps(p.get("risks", [])),
        "catalysts_json": json.dumps(p.get("catalysts", [])),
        "data_as_of": p.get("data_as_of"),
        "sources_json": json.dumps(p.get("sources", [])),
        "validation_json": json.dumps(p.get("validation", {})),
        "raw": p.get("raw"),
        "prompt_version": p.get("prompt_version"),
        "model": p.get("model"),
    }
    cols = ",".join(row.keys())
    ph = ",".join(f":{k}" for k in row)
    upd = ",".join(f"{k}=excluded.{k}" for k in row if k not in ("run_id", "symbol"))
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO portfolio_positions({cols}) VALUES({ph}) "
            f"ON CONFLICT(run_id,symbol) DO UPDATE SET {upd}", row)


def upsert_portfolio_summary(run_id: int, s: dict[str, Any]) -> None:
    row = {
        "run_id": run_id, "overall_risk": s.get("overall_risk"),
        "concentration": s.get("concentration"), "correlation": s.get("correlation"),
        "diversification_gaps": s.get("diversification_gaps"),
        "weakest_link": s.get("weakest_link"),
        "ranked_actions_json": json.dumps(s.get("ranked_actions", [])),
        "risk_json": json.dumps(s.get("risk_metrics", {})),
        "ballast_json": json.dumps(s.get("ballast_suggestions", [])),
        "summary": s.get("summary"), "raw": s.get("raw"),
    }
    cols = ",".join(row.keys())
    ph = ",".join(f":{k}" for k in row)
    upd = ",".join(f"{k}=excluded.{k}" for k in row if k != "run_id")
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO portfolio_summary({cols}) VALUES({ph}) "
            f"ON CONFLICT(run_id) DO UPDATE SET {upd}", row)


def get_portfolio(run_id: int) -> dict[str, Any]:
    with get_conn() as conn:
        positions = [dict(r) for r in conn.execute(
            "SELECT * FROM portfolio_positions WHERE run_id=? ORDER BY symbol", (run_id,)).fetchall()]
        summary = conn.execute("SELECT * FROM portfolio_summary WHERE run_id=?", (run_id,)).fetchone()
    for p in positions:
        for k in ("risks_json", "catalysts_json", "sources_json", "validation_json"):
            if p.get(k):
                try:
                    p[k] = json.loads(p[k])
                except (json.JSONDecodeError, TypeError):
                    pass
    summary = dict(summary) if summary else {}
    if summary.get("ranked_actions_json"):
        try:
            summary["ranked_actions_json"] = json.loads(summary["ranked_actions_json"])
        except (json.JSONDecodeError, TypeError):
            pass
    return {"positions": positions, "summary": summary}


if __name__ == "__main__":
    # Round-trip smoke test (M0 acceptance).
    init_db()
    rid = create_run("deep")
    upsert_tickers([{"symbol": "NVDA", "name": "NVIDIA", "subsector": "semis_equipment"}])
    upsert_screen_scores(rid, [{
        "symbol": "NVDA", "quality": 0.8, "deal": 0.9, "blended": 0.86, "rank": 1,
        "tags": ["deal", "quality", "both"], "metrics": {"pe": 45.0},
    }])
    upsert_signals(rid, [{"symbol": "NVDA", "hyperscaler_signal": 9.0,
                          "catalysts": ["new GPU deal"], "summary": "strong demand"}])
    upsert_research(rid, {"symbol": "NVDA", "conviction": 8.5, "deal_likelihood": "high",
                          "thesis": "AI backbone", "catalysts": ["earnings"], "risks": ["valuation"],
                          "bull": "dominant", "bear": "rich", "raw": "{}"})
    finish_run(rid, "ok", "smoke test")
    board = get_board(rid)
    assert board and board[0]["symbol"] == "NVDA", "round-trip failed"
    assert board[0]["tags"] == ["deal", "quality", "both"]
    assert board[0]["conviction"] == 8.5
    print("storage.py round-trip OK. run_id=", rid, "board rows=", len(board))
    print("Sample row:", {k: board[0][k] for k in ("symbol", "blended", "tags", "conviction")})
