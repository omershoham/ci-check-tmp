"""Dashboard chat: ask Claude questions grounded in EVERYTHING the system knows.

Runs on the same subscription path as deep research (`claude -p` via
deep_research._run_claude). Each question gets a compact, freshly-assembled snapshot
of the full system state — board, holdings & P&L, PM actions/targets, theses,
sell-signals/triggers, risk metrics, regime, calibration, lessons — so answers are
specific to THIS portfolio and THIS analysis, not generic market chat.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

from . import config, deep_research, learning, recommend, regime, storage

_SYSTEM_TMPL = """You are the analyst behind "Claude Trader", a personal AI-infra stock
research system. Today is {today}. The owner is asking you questions INSIDE the dashboard.
Below is the system's complete current state — its rankings, research, portfolio actions,
risk numbers and track record. Answer from THIS data first; be specific (cite tickers,
numbers, dates from the context). If something isn't in the context, say so rather than
inventing it. You may disagree with the system's own calls — say why. Keep answers tight:
short paragraphs or bullets, no preamble. Never present anything as financial advice —
this is research support for the owner's own decisions.

=== SYSTEM STATE ===
{context}
=== END SYSTEM STATE ===
{history}
OWNER'S QUESTION: {question}"""


def _fmt(v, nd=1):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "?"


def build_context(max_board: int = 15) -> str:
    """Compact snapshot of everything the dashboard knows (token-bounded)."""
    parts: list[str] = []

    # Regime
    r = regime.detect(use_cache=True)
    if r.get("regime"):
        parts.append(f"MARKET REGIME: {r['regime']} | SPY {r.get('bench_vs_200dma_pct')}% "
                     f"vs 200dma | breadth {r.get('breadth_pct', '?')}% | weak subsectors: "
                     f"{', '.join(r.get('weak_subsectors', []) or ['none'])}")

    # Latest board (top N + all held names)
    run = storage.latest_run()
    if run:
        board = storage.get_board(run["id"])
        parts.append(f"\nBOARD (run #{run['id']}, {str(run['finished_at'])[:16]}, "
                     f"top {max_board} of {len(board)} + held):")
        shown = set()
        for row in board:
            if len(shown) >= max_board and not row.get("held_position"):
                continue
            m = row.get("metrics") or {}
            lab, basis = recommend.recommendation(row)
            held = f" HELD {row['held_position']:g}sh" if row.get("held_position") else ""
            conv = f" conv={row['conviction']}" if row.get("conviction") is not None else ""
            parts.append(
                f"  #{row['rank']} {row['symbol']}{held}: {lab}({basis}){conv} "
                f"comp={_fmt(row.get('blended'), 2)} T/F/S="
                f"{_fmt(m.get('pillar_technical'), 2)}/{_fmt(m.get('pillar_fundamental'), 2)}"
                f"/{_fmt(m.get('pillar_sentiment'), 2)} PE={_fmt(m.get('trailingPE'))} "
                f"RSI={_fmt(m.get('rsi'), 0)} 12-1mom={_fmt(m.get('ret_252_21'), 0)}% "
                f"earn={m.get('earnings_date') or '?'}"
                + (f" | thesis: {str(row.get('thesis'))[:180]}" if row.get("thesis") else ""))
            shown.add(row["symbol"])

    # Portfolio deep-dive (actions, targets, theses, sell-signals)
    pf_run = storage.latest_portfolio_run("ok")
    if pf_run:
        pf = storage.get_portfolio(pf_run["id"])
        parts.append(f"\nPORTFOLIO DEEP-DIVE (#{pf_run['id']}, "
                     f"{str(pf_run['finished_at'])[:16]}):")
        for p in sorted(pf["positions"], key=lambda x: -(x.get("weight_pct") or 0)):
            parts.append(
                f"  {p['symbol']}: {p.get('action')} conv={p.get('conviction')} "
                f"weight {_fmt(p.get('weight_pct'))}%->target {_fmt(p.get('target_weight_pct'))}% "
                f"P&L {_fmt(p.get('pnl_pct'))}% add<={p.get('add_level')} "
                f"trim>={p.get('trim_level')}"
                + (f" | note: {str(p.get('action_note'))[:120]}" if p.get("action_note") else "")
                + (f" | health: {str(p.get('thesis_health'))[:120]}" if p.get("thesis_health") else "")
                + (f" | sell-if: {str(p.get('invalidation'))[:180]}" if p.get("invalidation") else ""))
        s = pf.get("summary") or {}
        if s.get("summary"):
            parts.append(f"  PORTFOLIO SUMMARY: {str(s['summary'])[:400]}")
        if s.get("weakest_link"):
            parts.append(f"  WEAKEST LINK: {str(s['weakest_link'])[:200]}")
        rj = s.get("risk_json")
        try:
            rm = json.loads(rj) if isinstance(rj, str) else (rj or {})
        except Exception:  # noqa: BLE001
            rm = {}
        if rm:
            parts.append(f"  RISK: eff_bets={rm.get('effective_bets')} "
                         f"(vol-aware {rm.get('vol_aware_bets')}) avg_corr="
                         f"{rm.get('avg_pairwise_corr')} beta_qqq={rm.get('beta_qqq')} "
                         f"SMH-20%->{rm.get('scenario_drawdown_pct')}% "
                         f"(stressed {rm.get('scenario_stressed_pct')}%, worst20d "
                         f"{rm.get('worst_20d_pct')}%)")
        bj = s.get("ballast_json")
        try:
            bl = json.loads(bj) if isinstance(bj, str) else (bj or [])
        except Exception:  # noqa: BLE001
            bl = []
        if bl:
            parts.append("  BALLAST SUGGESTIONS: " + "; ".join(
                f"{b.get('symbol')} {b.get('target_weight_pct')}%" for b in bl
                if isinstance(b, dict)))

    # Triggers
    fired = storage.recent_fired_triggers(days=7)
    if fired:
        parts.append("\nTRIGGERS FIRED (7d): " + "; ".join(
            f"{t['symbol']}: {t.get('trigger_detail')}" for t in fired[:6]))
    act = storage.active_triggers()
    if act:
        parts.append(f"ACTIVE TRIGGERS ({len(act)}): " + "; ".join(
            f"{t['symbol']} {t['kind']}"
            + (f"@{t['threshold']:g}" if t.get("threshold") else f"'{t.get('keyword')}'")
            for t in act[:15]))

    # Track record + lessons
    cal = learning.calibration_line()
    if cal:
        parts.append(f"\n{cal}")
    ba = learning.portfolio_batting_average()
    if ba:
        parts.append(ba)
    lessons = storage.active_lessons(scope="global", limit=6)
    if lessons:
        parts.append("ACTIVE LESSONS: " + " | ".join(ls["lesson"][:120] for ls in lessons))

    ctx = "\n".join(parts)
    return ctx[:14000]  # hard token bound


def ask(question: str, history: list[dict] | None = None,
        use_web: bool = False) -> str:
    """One chat turn. history = [{'role': 'user'|'assistant', 'content': str}, ...]."""
    hist_block = ""
    if history:
        lines = ["\nCONVERSATION SO FAR:"]
        for m in history[-6:]:  # last 3 exchanges
            who = "OWNER" if m.get("role") == "user" else "YOU"
            lines.append(f"{who}: {str(m.get('content'))[:800]}")
        hist_block = "\n".join(lines) + "\n"
    prompt = _SYSTEM_TMPL.format(today=date.today().isoformat(),
                                 context=build_context(),
                                 history=hist_block,
                                 question=question[:2000])
    tools = "WebSearch,WebFetch" if use_web else ""
    timeout = config.get("claude.chat_timeout_seconds", 240)
    try:
        text, _ = deep_research._run_claude(prompt, allowed_tools=tools, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return f"(chat failed: {exc})"
    return text.strip() or "(empty response)"


# ------------------------------------------------------------- investigate mode
_INVESTIGATE_SYSTEM = """You are the senior analyst behind "Claude Trader", working INSIDE
the project at {root}. Today is {today}. The owner is asking questions from the dashboard
and has given you FULL INVESTIGATIVE ACCESS: read any file, query the SQLite DB, run
Python, search the web. Investigate before you answer — don't guess from the snapshot
when you can check the source.

YOUR TOOLS AND THE TERRAIN:
- DB: data/trader.db (sqlite3). Tables: runs, screen_scores (pillars in metrics_json,
  score_version), research_reports (thesis/rating/conviction/red_team_json/raw),
  portfolio_runs, portfolio_positions (action/conviction/weight_pct/target_weight_pct/
  invalidation), portfolio_summary (risk_json/ballast_json), triggers, picks, lessons,
  holdings, quotes, watchlist. Query via: .venv/Scripts/python.exe -c "import sqlite3; ..."
- CODE: src/*.py — scoring.py (pillar math), technical.py (indicators), learning.py
  (grading/calibration/IC), performance.py (forward returns), risk.py, regime.py,
  triggers.py, deep_research.py (all research prompts), news.py, local_llm.py.
- Config/policy: config.yaml. Logs: logs/, data/*.log. Cache: data/cache/.
- Use WebSearch for anything about current market reality.

{corrections_block}

Answer style: tight, specific, cite what you checked (tables/files/queries). Not financial
advice. A quick state snapshot follows for orientation — but verify against the DB when
it matters.

=== SNAPSHOT ===
{context}
=== END SNAPSHOT ===

OWNER'S QUESTION: {question}"""

_CORRECTIONS_ALLOWED = """CORRECTIONS — you MAY fix stored analysis when the investigation genuinely warrants it:
- If you conclude a board rating/conviction (research_reports) or a portfolio action/
  conviction/target (portfolio_positions) is materially wrong, UPDATE the row via Python.
- EVERY correction MUST: (1) be announced explicitly in your answer ("CHANGED: MU buy->hold
  because ..."), (2) leave an audit trail — INSERT INTO lessons(created_at, scope, symbol,
  lesson, evidence, status) VALUES(datetime('now'), 'symbol', '<SYM>',
  '[chat correction <date>] <what changed and why>', '<evidence>', 'active').
- Never change: holdings, quotes, picks (ground truth), or any code/config file unless the
  owner explicitly asks in this conversation. Never place trades — the system is advisory.
- Be conservative: a correction needs evidence, not vibes. Disagreeing in prose is free;
  rewriting stored analysis is a real act."""

_CORRECTIONS_READONLY = """READ-ONLY SESSION: you have no shell and no write access. Investigate with Read/Grep/
Glob and the web, then ANSWER — if you conclude a stored rating or action is wrong, say so
and explain; the owner can enable corrections for a follow-up. Never place trades."""

# Default tool set is READ-ONLY. Raw web headlines flow into these prompts, so an agent
# with a shell is a prompt-injection surface; corrections (Bash + DB writes) are an
# explicit per-session opt-in from the dashboard.
_INVESTIGATE_TOOLS_READONLY = "Read,Grep,Glob,WebSearch,WebFetch"
_INVESTIGATE_TOOLS_WRITE = "Read,Grep,Glob,Bash,WebSearch,WebFetch"


def investigate(question: str, session_id: str | None = None,
                allow_corrections: bool = False) -> tuple[str, str | None]:
    """Agentic chat turn with read access to code+DB and web search. With
    `allow_corrections=True` the agent also gets Bash and audited correction rights.
    Returns (answer, session_id) — pass the session_id back on follow-ups so the agent
    keeps its investigation context (files read, queries run).
    """
    from . import llm as _llm
    if not getattr(_llm.get_provider(), "supports_sessions", False):
        return ("(Investigate mode needs the Claude Code CLI provider — llm.provider is "
                f"'{_llm.get_provider().name}', which has no file/shell tools. Use Quick mode.)",
                session_id)
    timeout = config.get("claude.chat_investigate_timeout_seconds", 900)
    tools = _INVESTIGATE_TOOLS_WRITE if allow_corrections else _INVESTIGATE_TOOLS_READONLY
    if session_id:
        # Resumed session already has the system contract + all prior investigation.
        prompt = question[:4000]
        extra = ["--resume", session_id]
    else:
        prompt = _INVESTIGATE_SYSTEM.format(
            root=str(config.ROOT), today=date.today().isoformat(),
            corrections_block=_CORRECTIONS_ALLOWED if allow_corrections else _CORRECTIONS_READONLY,
            context=build_context(), question=question[:4000])
        extra = []
    try:
        text, raw = deep_research._run_claude(prompt, allowed_tools=tools,
                                              timeout=timeout, extra_args=extra)
    except Exception as exc:  # noqa: BLE001
        return f"(investigation failed: {exc})", session_id
    sid = session_id
    try:
        sid = json.loads(raw).get("session_id") or session_id
    except (json.JSONDecodeError, AttributeError):
        pass
    return (text.strip() or "(empty response)"), sid


if __name__ == "__main__":
    print(build_context()[:3000])
    print("\n--- live ask ---")
    print(ask("Which holding is my biggest risk right now and why?"))
