# Code Review #3 — Structure & Scalability for a Self-Serve Platform

**Date:** 2026-09-07 · **Scope:** `src/` (20 modules, ~5,100 lines), `app/dashboard.py` (1,576), `scripts/`, config, packaging.
**Goal under review:** turn Claude Trader from a personal tool into a platform *other people* run with *their own* AI subscription, executing all analysis themselves.

This review is about architecture — how the code is shaped and whether that shape survives other people running it. Correctness bugs are listed only where they fell out of the structural reading (§6).

---

## 1. Verdict in one paragraph

The analytical core is in good shape: a clean four-stage funnel, one choke point for every Claude call, defensive degradation everywhere, config-driven tuning, versioned scoring, a genuinely thoughtful learning loop. What it is **not** yet is a product. Every assumption that makes it *yours* is baked in as a constant rather than a setting: one machine (Windows, `.venv\Scripts\python.exe`, `taskkill`), one user (one DB, one `holdings` table, one config file), one broker (IBKR via IBC), one LLM path (the `claude` CLI logged into *your* subscription), one GPU (Ollama with a specific model), and one investment thesis (AI-infra — the word "hyperscaler" is a database column). None of those is hard to fix individually; the work is that there are about twelve of them and they have to become explicit seams. The single most important structural gap for other people is **silent degradation**: a run reports `status=ok` while IBKR is down, Ollama is down, or Yahoo throttled you — the exact failure that hid your portfolio from the analysis for five weeks. A platform cannot ship that.

---

## 2. Decide the deployment model first

Everything below depends on which of these you mean by "platform":

| Model | What it means | Fit with "their own AI subscription" |
|---|---|---|
| **A. Self-hosted app** (recommended) | Each person installs it on their machine — `pipx install claude-trader`, `claude-trader doctor`, `claude-trader up`. Their `claude` login, their Ollama, their broker, their SQLite. | Natural. Same shape as today; N independent installs. |
| **B. Hosted multi-tenant SaaS** | You run a server; users sign in; jobs run per user. | Structurally a rewrite (user_id on every table, job queue, per-user secrets, no GPU per user). And the subscription-token model doesn't transfer: you can't legitimately run other people's Claude Code OAuth tokens on your server. They'd have to bring API keys, which changes the cost story the whole design was built around. |

**Recommendation: A.** It matches the stated goal, keeps costs on the user, and keeps you out of the business of holding people's broker credentials. Design so B isn't foreclosed (pass a `Workspace` object around instead of reading global module state), but don't build for it now.

Under model A, "scalability" has three meanings, in order of importance:
1. **Scales across machines** — installs cleanly on a stranger's Windows/macOS/Linux box with none of your paths. *(Today: no.)*
2. **Scales across interests** — someone can point it at biotech or energy instead of AI-infra. *(Today: no.)*
3. **Scales within a user** — a 500-name universe, a 30-position portfolio. *(Today: mostly yes, see §8.)*

---

## 3. What is already right — keep it

These are load-bearing for a platform; don't lose them while restructuring.

- **One LLM choke point.** Every Claude call goes through `deep_research._run_claude`. That makes the provider abstraction (§4, F1) a contained change instead of a hunt.
- **Degrade, don't die.** IBKR/Ollama/Yahoo down → neutral defaults and the run completes. Right instinct; needs to become *visible* degradation (F3).
- **Config-driven tuning.** Universe, weights, floors, policy, thresholds all live in `config.yaml`. The seam exists; it needs splitting into defaults vs. user (F2).
- **`SCORE_VERSION`** on `screen_scores` so a definition change doesn't contaminate the IC statistics. Extend the same discipline to prompts (F12).
- **Robust JSON salvage** (`_extract_json` scanning every brace; cheap reformat call instead of re-research). Expensive-call hygiene other tools get wrong.
- **Deadline-guarded downloads** (`_download_with_deadline`) after the Jul-13 hang. Yahoo will do this to every user.
- **WAL + `busy_timeout`** so the dashboard can read while the pipeline writes.
- **Progress heartbeat + stall detection** in `runs` / `portfolio_runs`.
- **Percentile-ranked pillars before weighting** so config weights mean what they say.
- **The learning loop's benchmark choice** (excess vs. equal-weight universe, not SPY) — the CR#2 fix that stopped measuring beta as skill.
- **`recommend.py`** as the single source of truth for label derivation, shared by dashboard, chat and digest.

---

## 4. Findings

Severity is *platform-readiness*, not "is it broken today". **Blocker** = other people cannot run it; **High** = they can, but it will bite them or you within weeks; **Medium** = debt that gets expensive as the codebase grows; **Low** = hygiene.

### F1 · Blocker — The AI provider is a subprocess to *your* CLI login
`deep_research._run_claude` shells out to `claude -p`, relies on `CLAUDE_CODE_OAUTH_TOKEN`, strips `ANTHROPIC_API_KEY`, kills via `taskkill /T` on `os.name == "nt"`, and works around the npm `.cmd` shim. `chat.py`, `triggers.py`, `learning.py`, `portfolio.py`, and two scripts all reach into that private function.

*Why it matters:* other users may have (a) Claude Code with a subscription, (b) an Anthropic API key, (c) neither, and want a different vendor. The subscription path also depends on the `claude` CLI being installed and interactively logged in — which is fine for a person, awkward for a product, and you should verify Anthropic's current terms on non-interactive use of a Claude Code subscription before advertising it as the supported path.

*Recommendation:* a `src/llm/` package:
```
llm/
  provider.py       # Protocol: complete(prompt, *, tools, timeout, json=False) -> Response(text, raw, session_id, model, usage)
  claude_cli.py     # today's code, made cross-platform (psutil / os.killpg)
  anthropic_api.py  # official SDK, BYO key; same Response shape
  ollama.py         # move local_llm's HTTP plumbing here
  json_utils.py     # _extract_json / _balanced_span / reformat
```
Callers get `llm.get_provider()` from config (`llm.provider: claude_cli | anthropic | ollama`). Web search becomes a capability flag the provider reports; if absent, the prompts' "use web search" lines must adapt (they're currently unconditional).

### F2 · Blocker — Config mixes shipped defaults with personal choices, at a fixed path
`config.yaml` holds both *app behavior* (pillar weights, floors, timeouts) and *your* choices (107-ticker universe, your position caps, your ballast list). `config.py` reads `ROOT / "config.yaml"` and `ROOT / ".env"`, `lru_cache`d, and every module calls `config.get(...)` on import-time module state. `paths.db` and `paths.cache` are project-relative.

*Recommendation:* `config.default.yaml` shipped in the package; user overrides in `$CLAUDE_TRADER_HOME/config.yaml` (default `~/.claude_trader/`), deep-merged; `.env` in the same home. Data, cache and logs under that home, never inside the package. Expose a `Settings`/`Workspace` object; keep `config.get` as a thin facade during the transition so 20 modules don't change at once. Validate on load (pydantic or a hand-rolled schema) so a typo in `scoring.pillars` fails loudly at startup instead of silently defaulting to 0.35.

### F3 · Blocker — "ok" means "the process didn't crash"
`pipeline.run` marks `status='ok'` if no exception escaped. Degradations are one log line each: `IBKR connected=False`, `local LLM available: False`, chunked-download fallback, `every local-LLM classification failed`. Runs #99–#113 were all `ok` and all portfolio-blind.

*Recommendation:* a per-stage health record persisted on the run — e.g. `health_json = {"fundamentals": "ok", "technicals": "degraded: 12 names skipped", "sentiment": "failed: ollama unreachable", "broker": "offline", "research": "ok 10/10"}` — with an overall `health` of `ok | degraded | failed` shown as a badge next to the run in the dashboard and in the launcher's freshness banner. `status` keeps meaning "process outcome"; `health` means "trust this output". This is the single highest-leverage change for other users, and it directly prevents a repeat of the five-week blind spot.

### F4 · Blocker — Not installable; Windows-shaped throughout
No `pyproject.toml`, no entry points, no lockfile; `requirements.txt` is `>=`-only (yfinance breaks on minor versions). `app/dashboard.py` does `sys.path.insert(0, ROOT)`; it and `scripts/` hardcode `.venv/Scripts/python.exe` in three places; `CREATE_NO_WINDOW`/`DETACHED_PROCESS`; launchers are PowerShell; `taskkill`.

*Recommendation:* package as `claude_trader/` with `[project.scripts]`: `claude-trader run --mode deep|light`, `claude-trader dashboard`, `claude-trader deepdive`, `claude-trader digest`, `claude-trader doctor`, `claude-trader up` (the launcher, in Python). Use `sys.executable` for child processes; `psutil` for tree-kill. Pin with a lockfile (`uv`/pip-tools). The PowerShell scripts remain as thin Windows conveniences that call the CLI.

### F5 · High — The thesis is hardcoded, not configured
"AI-infra" is structural, not thematic: `hyperscaler_signal` is a DB column, an Ollama JSON-schema field, and 45% of the deal score; `SUBSECTOR_RELEVANCE` and the four subsector keys live in `scoring.py`/`universe.py`; `_HYPERSCALERS` and the classifier system prompt name AWS/Azure/GCP; research prompts open with "US AI-infrastructure stocks"; `rs_benchmark: SMH` in config but `_THEME = "SMH"` hardcoded in `performance.py`; `deep_research` mentions VOO/QQQ as the ETF examples.

*Recommendation:* a **theme pack** — one YAML that the default install ships as `themes/ai_infra.yaml`:
```yaml
name: AI infrastructure
benchmark: SMH
subsectors: {semis_equipment: {label: ..., relevance: 0.95, tickers: [...]}, ...}
catalyst_signal:            # generalizes hyperscaler_signal
  name: hyperscaler_signal  # kept for DB back-compat
  question: "how likely is the company to benefit from large hyperscaler deals/capex"
  actors: [AWS, Azure, Google Cloud, Oracle, Meta, NVIDIA]
prompt_fragments: {analyst_role: "...", theme_context: "..."}
```
Scoring reads relevance from the pack; prompts are assembled from fragments; the classifier schema uses the pack's signal name. Rename the column to `theme_signal` behind a migration when convenient. Someone can then drop in `biotech.yaml` without touching Python.

### F6 · High — Two monolithic orchestrators, duplicated logic
`pipeline.run` is one 210-line function; stage boundaries exist only as `set_progress` strings. `portfolio.run` is the same shape at 190 lines. Consequences already visible: the finalist count is computed twice (`pipeline.py:136-137` and `201-202`); the red-team block is copy-pasted into `scripts/research_one.py:55-59`; the "held names not on the shortlist" logic appears in two forms; RSI/MACD math exists in both `technical.py` and `charts.py`.

*Recommendation:* stages as functions with typed I/O on a `RunContext` dataclass (`universe`, `metrics`, `techs`, `regime`, `board`, `signals`, `holdings`, `quotes`, `health`). `run()` becomes the sequencing. Wins: `research_one` becomes "run stage 3 for one symbol" instead of a re-implementation; stages are unit-testable with a fake provider; resume-from-stage becomes possible; the health record (F3) falls out naturally. Same treatment for `portfolio.run`.

### F7 · High — Background work coordinated by marker files and a `running` row
Dashboard buttons `Popen` detached processes; state is a `.marker` file in the cache dir (single research) or a `portfolio_runs.status='running'` row (deep-dive); staleness is inferred from age. `config.sweep_cache` only matches `*_*.json`, so a marker left by a hard-killed `research_one` is **never** swept — the button stays disabled until someone deletes the file by hand.

*Recommendation:* a `jobs` table — `id, kind, symbol, pid, started_at, heartbeat, status, error` — as the single mechanism for all background work (pipeline runs, deep-dives, single-name research, digest). Liveness = `pid` still alive *and* heartbeat fresh. The dashboard reads one table; the launcher can show "a deep run is in progress" the same way.

### F8 · High — No tests
Zero test files. The `__main__` blocks are manual smoke tests (and one of them shipped your portfolio — see CR of 2026-09-07). The pure modules — `scoring`, `technical._score_and_state`, `recommend`, `triggers.check_all`, `_extract_json`, `regime` classification, `apply_red_team`, `portfolio`'s policy backstop — are exactly the kind of code that breaks quietly when refactored.

*Recommendation:* `pytest` with (1) unit tests for the pure modules on fixed inputs, (2) a `FakeProvider` returning canned JSON so `pipeline.run` can execute end-to-end in seconds against a temp DB, (3) a storage round-trip test derived from the existing `__main__`. Not aiming for coverage; aiming for "the refactors in F1/F5/F6 can be done without fear."

### F9 · Medium — Storage: ad-hoc migrations, connection hygiene, SQL leaking into the UI
`_migrate` is a growing list of `if col not in cols: ALTER TABLE`, run on every `init_db()`, with no schema version. `with get_conn() as conn:` commits but does not close (sqlite3 semantics; CPython refcounting rescues it, PyPy wouldn't). The dashboard runs raw SQL in three places (`DELETE FROM watchlist`, `SELECT * FROM holdings`, `SELECT symbol,last FROM quotes`) bypassing `storage`; `portfolio.py` and `learning.py` do too.

*Recommendation:* `schema_version` table + numbered migration functions; `get_conn` as a real context manager (`contextlib.closing` or a `@contextmanager` that closes); every query behind a `storage` function. If model B ever happens, this is what makes Postgres a swap rather than a rewrite. Indices are fine at current volume (≈75k `screen_scores` rows/year).

### F10 · Medium — Dashboard is a 1,576-line script executed top to bottom
Module-level code, 23 `session_state` touches, five tabs inline, subprocess launches and cache-clearing inside render, two copies of the heartbeat/stall rendering. It works because one person maintains it.

*Recommendation:* `app/pages/` or per-tab modules with a `render_board(ctx)` signature; shared loaders in `app/data.py`; process launching via the `jobs` API (F7). Keep it Streamlit — it's the right tool for this — but stop it being one file.

### F11 · Medium — `print()` is the logging system
`_log()` prefixes and `flush=True`; no levels, no module loggers, no rotation; the launcher/Task Scheduler tee is what makes logs exist at all.

*Recommendation:* `logging` with per-module loggers, `RichHandler`/plain console + rotating file under the workspace home, level from config/CLI. The health record (F3) writes its summaries through the same channel.

### F12 · Medium — Prompts are unversioned and inline
Six large prompt templates live in `deep_research.py` (≈45% of the file), plus one each in `chat.py`, `learning.py`, `triggers.py`, `local_llm.py`. Research reports don't record which prompt or model produced them, so the calibration tab compares outputs of different "analysts" as if they were one.

*Recommendation:* `prompts/*.md` (or a `prompts.py`) with a `PROMPT_VERSION`; store `prompt_version` and `model` on `research_reports` and `portfolio_positions`, mirroring `SCORE_VERSION`. Cheap, and it makes the Accuracy tab honest across changes.

### F13 · Medium — Investigate mode hands an agent `Bash` with untrusted text in its prompt
`chat.investigate` runs `claude -p --allowedTools Read,Grep,Glob,Bash,WebSearch,WebFetch` with write rights to the DB; `build_context` and the research prompts embed **raw news headlines** fetched from the open web. For a single owner that's a deliberate trade-off. For a product it's a prompt-injection surface: a crafted headline about a held ticker reaches an agent that can run shell commands.

*Recommendation:* default Investigate to read-only tools; make `Bash`/DB-write an explicit per-session opt-in with a visible warning; consider a dedicated `claude-trader fix` path for corrections instead of free-form Bash. Document the trust model.

### F14 · Medium — Broker and market data are concrete, not interfaces
`ibkr.py` is already optional and defensive (good), but it is the *only* way holdings enter the system besides "IBKR was up last time". Many users won't have IBKR, and the IBC/Gateway install is the hardest part of your own setup. Market data is `yfinance` everywhere (`Ticker.info` per symbol, 107 sequential calls with a 0.4 s throttle).

*Recommendation:* `Broker` protocol with `IBKRBroker` and `ManualBroker` (CSV upload / editable table in the dashboard — the zero-dependency default). `MarketData` protocol wrapping yfinance so a paid/official source can be swapped in later without touching `fundamentals`/`technical`.

### F15 · Low — Small things
- `performance._MEMO` is a module dict keyed by date; in a long-lived Streamlit process it grows daily and never evicts.
- `_THEME = "SMH"` in `performance.py` ignores `technical.rs_benchmark` (folded into F5).
- `learning._REFLECT_TMPL` says "vs SPY" while grades are vs. universe — prompt/text drift.
- `market_calendar` comments assume one specific local timezone; harmless, but the schedule story should be "cron/Task Scheduler runs `claude-trader run`" with no timezone baked in.
- 40+ `except Exception` sites. Consistent with the degrade-don't-die design, acceptable *once F3 makes degradation visible*; without F3 they are where problems go to hide.

---

## 5. Target structure

```
claude_trader/
  __init__.py
  cli.py                 # run | dashboard | deepdive | digest | doctor | up
  settings.py            # Workspace: home, config (defaults ⊕ user), env, paths, theme
  health.py              # StageHealth, RunHealth (F3)
  jobs.py                # background job table + liveness (F7)
  llm/                   # provider protocol + claude_cli / anthropic_api / ollama (F1)
  prompts/               # versioned templates (F12)
  themes/ai_infra.yaml   # default theme pack (F5)
  data/                  # market data + broker protocols (F14): yfinance.py, ibkr.py, manual.py
  pipeline/              # stages/ + run.py (F6): screen.py, technicals.py, sentiment.py, research.py, overlay.py, triggers.py
  portfolio/             # deepdive stages
  analytics/             # scoring, technical, regime, risk, learning, performance, recommend (pure; testable)
  storage/               # schema.py, migrations/, repo functions (F9)
  dashboard/             # app.py + pages/ + data.py (F10)
tests/
pyproject.toml
```
Names are suggestions; the point is the seams: **settings, llm, theme, data, storage** become the five places a stranger's environment differs from yours.

---

## 6. Correctness items noticed along the way

Not the focus, but cheap to record:

1. **Stuck-disabled research button** — `research_inflight_*.marker` survives a hard kill and `sweep_cache` never removes it (`*_*.json` glob). Fix in F7 or at minimum date-stamp the marker so the sweep catches it.
2. **`with get_conn() as conn`** doesn't close connections — relies on refcounting. Wrap in `closing`.
3. **`performance._MEMO`** unbounded across days in the dashboard process.
4. **Finalist count** computed twice in `pipeline.run` with the same expression — harmless now, a drift bug waiting to happen.
5. **`_THEME` vs `technical.rs_benchmark`** — two sources of truth for the theme benchmark.
6. **`load_board.clear()` on every render tick** while a single-name research is in flight — correct, but it means auto-refresh re-queries the 5-way join every tick for the duration; fine at 100 rows, worth knowing.

---

## 7. Sequenced roadmap

Ordered so each step makes the next one safer. Effort is rough, for one person who knows the code.

**P0 — a stranger can install and trust it** (~2–3 weeks)
1. `pyproject.toml`, entry points, `sys.executable`, lockfile (F4) — 2 days
2. Workspace/settings split; `CLAUDE_TRADER_HOME`; validation (F2) — 3 days
3. Run health record + dashboard badge + `doctor` command (F3) — 3 days
4. LLM provider protocol; `claude_cli` + `anthropic_api`; Ollama behind the same protocol and *optional* (F1) — 4 days
5. pytest scaffold: pure-module tests + FakeProvider end-to-end run (F8) — 3 days

**P1 — it is a platform, not your tool** (~3–4 weeks)
6. Theme pack; de-hardcode AI-infra; `theme_signal` migration (F5) — 4 days
7. Stage-ify pipeline and portfolio; `RunContext`; dedupe `research_one` (F6) — 5 days
8. `jobs` table replacing markers/`running` rows (F7) — 2 days
9. Migrations with version table; connection hygiene; SQL out of the UI (F9) — 3 days
10. Dashboard split into pages; `logging` (F10, F11) — 4 days
11. `Broker`/`MarketData` protocols; manual holdings import (F14) — 3 days

**P2 — hardening** (~1 week)
12. Investigate-mode tool tightening + trust-model doc (F13) — 1 day
13. Prompt versioning on reports (F12) — 1 day
14. F15 items — 1 day

Do P0 before inviting anyone in. Steps 1–3 alone would already make *your own* installation safer.

---

## 8. Scalability envelope (within one user, model A)

Measured against today's code; where it breaks and what to change.

| Dimension | Today | Breaks at | Bottleneck | Fix |
|---|---|---|---|---|
| Universe size | 107 | ~400–500 | `fundamentals.fetch_many`: one `Ticker.info` HTTP call per name, sequential, 0.4 s throttle → ~10+ min at 500; Yahoo throttling risk rises | Batch endpoints / `MarketData` provider (F14); parallelism is limited by Yahoo, not CPU |
| Shortlist for sentiment | 35 + held | ~150 | Serial Ollama calls, GPU-bound, ~2–5 s each | Fine to raise; or cloud classifier via `llm` protocol |
| Claude research per run | 10 + held | cost/time, not code | ~1–3 min per name serial; budgeted by `deep_run_budget_seconds` | Already capped; parallelize only if provider allows and budget permits |
| Portfolio size | 9 | ~40 | `portfolio.run` serial: research + validation per holding ≈ 3–4 min each → 2+ hours | Batch validation; parallel per-holding research once F6 lands |
| DB size | 19 MB / 3 months | years | `research_reports.raw` full transcripts | Fine; archive `raw` after grading if it ever matters |
| Dashboard render | ~100 rows | ~1,000 rows | `get_board` 5-way join + Python JSON parse per row; Streamlit re-render | Paginate; server-side filter; fine for a long time |
| Cache dir | ~300 files/day, 7-day sweep | n/a | file-per-symbol-per-day JSON | Fine; move under workspace home |

Net: within-user scale is not the problem. Portability, configurability, and visible health are.

---

*Review method: full read of `src/*` (all 20 modules), `scripts/*`, config and packaging; structural read of `app/dashboard.py` (imports, loaders, chat dialog, launch paths, tab skeleton). No code was changed by this review.*
