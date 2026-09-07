# Claude Trader — System Improvement Plan (analysis quality first, platform-compatible)

## Context

Claude Trader is a working advisory pipeline (quant screen → local-LLM sentiment → Claude research → dashboard) that has run daily since 2026-06-17. CODE_REVIEW_3.md (2026-09-07) established the *structural* debt for turning it into a platform. This plan is about making the **analysis itself better and provably trustworthy**, informed by algorithmic-trading best practice, while keeping every change compatible with that platform roadmap so nothing is redone.

The trigger was the discovery that the system ran portfolio-blind for five weeks with every run reporting `ok`, plus the user's question of whether to add machine learning ("XGBoost on many features — learn patterns across stocks, then check them on each stock").

**Decisions confirmed with the user (2026-09-07):**
- Audience: **my system first, platform second** — changes must not conflict with CODE_REVIEW_3 §7.
- ML: a **pooled cross-sectional model** (learn patterns across the whole universe, score each stock) — *not* per-stock models. User named XGBoost; LightGBM is equivalent and has native ranking — allow either behind one interface.
- Data: **free sources (yfinance + IBKR)** for the platform. See "Data reality" below for what that can and cannot supply.
- **Advisory-only stays locked.** No execution. `ReadOnlyApi=yes` remains.

**What the DB holds today (measured):** 48 deep-run dates, 10,414 `screen_scores` rows, 506 rated `research_reports`, 493 `picks`, 7 portfolio dives / 59 position rows, 21 active lessons, 49 triggers (9 fired). That is ~2.5 months of the system's *own* labeled history — enough to start meta-labeling experiments, far too little to train a return model on. Price history from yfinance goes back 10+ years for every name, which is what the ranking model trains on.

---

## Data reality (answering "can we get it from IBKR and yfinance?")

| Need | yfinance | IBKR API | Verdict |
|---|---|---|---|
| Daily OHLCV, 10y+, current tickers | Yes (`yf.download`, batch) | Yes (`reqHistoricalData`, ~2,000 bars/request, 10 s pacing → slow for 100+ names) | **yfinance primary**, IBKR as cross-check |
| Historical (point-in-time) fundamentals | **No** — `.info` is current-only; any fundamentals feature built on it is look-ahead | **No** — needs a paid fundamentals subscription, still not point-in-time | **SEC Financial Statement Data Sets** (free, 2009→, quarterly zips, keyed by *filing date*) |
| Historical news/sentiment | No | No | Free tiers only (EODHD 20 calls/day, Finnhub). Treat sentiment as **live-only**; no backtest of the sentiment pillar |
| Delisted names (survivorship) | No | No | **Cannot be fixed for free.** Disclose: backtests run on the *current* curated universe and are upward-biased. Mitigation: freeze universe snapshots per date going forward; report the bias explicitly |

Sources: SEC DERA Financial Statement Data Sets (free, 2009–present, quarterly), TWS API historical-data limitations (2,000 bars/request, 10 s pacing), Retail Quant on point-in-time fundamentals, Norgate/Sharadar as the paid survivorship-free options if ever wanted.

---

## Best-practice gap map (what the field does vs. what this system does)

| Practice | State here | Track |
|---|---|---|
| Visible per-stage health; "ok" means trustworthy | `status=ok` on crash-free exit; degradation is a log line | **A** |
| Point-in-time data; no look-ahead | Live pipeline is fine (forward-looking). No historical store; fundamentals current-only | **C** |
| Walk-forward / purged validation for any learned or tuned signal | Pillar weights hand-set; `pillar_ic` is a statistical check but no backtest | **C, D** |
| Survivorship-aware universe | Curated list chosen June 2026 with hindsight | **C** (disclose, snapshot) |
| Transaction cost & turnover in evaluation | Prompts mention 0.3% round-trip; grading ignores costs | **B** |
| Deflated/probabilistic Sharpe, multiple-testing awareness | None | **C** |
| Meta-labeling: separate "direction" from "act/size" | Red-team is a prose version; no learned filter | **E** |
| Vol-targeting / fractional-Kelly caps / drawdown control | ATR-based risk hint, conviction caps, regime tightens 25% | **F** |
| LLM evaluation: control for memorization/beta | Live forward grading (good); no attribution of Claude's edge vs screen | **B** |
| Prompt/model versioning on every output | `SCORE_VERSION` only | **B** |
| Reproducibility: pinned deps, tests | None | **H** |

---

## Improvement tracks

### Track A — Trust the output (prerequisite for everything)
*From CODE_REVIEW_3 F3.* A run must say whether its data was whole.
- Add `health_json` + `health` (`ok|degraded|failed`) to `runs` and `portfolio_runs` via `storage._migrate`.
- `pipeline.run` records per-stage health: fundamentals (n fetched/n floor-passed, throttled?), technicals (n computed vs n neutral stubs), sentiment (Ollama up? error rate), broker (connected? n holdings), research (n/n, timeouts), triggers.
- Dashboard: health badge next to the run selector and in the header; the launcher's freshness banner reads it.
- `claude-trader doctor` (Python; the PowerShell launcher already probes ports) — checks venv, Ollama+model, `claude` CLI+auth, IBKR port, config validity, SEC contact set.
- Files: `src/storage.py`, `src/pipeline.py`, `src/portfolio.py`, `app/dashboard.py`, new `src/health.py`, new `scripts/doctor.py`.

### Track B — Honest measurement
- **Prompt + model versioning:** `prompt_version`, `model` columns on `research_reports` and `portfolio_positions`; `PROMPT_VERSION` constant next to the templates in `src/deep_research.py`. Accuracy tab groups by version.
- **Costs in grading:** `performance.pick_outcomes` subtracts a configurable round-trip cost (`evaluation.cost_bps`, default 30) for buy/strong_buy vs. the do-nothing alternative; `shadow_vs_actual` charges turnover on |target − actual| weight moved. Reuse the existing primitives as-is: `_fwd_return` (clamp-guarded, `base_override` for the pick-time price), `_price_history` (memoized per day), `_uni_mean_fwd`. Add eviction to `_MEMO` (keep only today's keys) while touching it.
- **Attribution of the Claude layer:** extend `performance.rating_calibration` with the screen-only cohort as the control (the July signal: screen-only +2.4% vs buys −0.5%) → a persistent "does Claude add value over the screen?" panel. Report decile spread and hit rate with n and a bootstrap CI (the existing 4w/1w windows).
- **Leakage note for LLM outputs:** document that Claude ratings are only valid *forward* (live) — never replay historical dates through Claude (knowledge-cutoff contamination). Guard: `deep_research.research_one` refuses a `today` override.
- Files: `src/performance.py`, `src/learning.py`, `src/storage.py`, `src/deep_research.py`, `app/dashboard.py` (Accuracy tab), `config.yaml` (`evaluation:` block).

### Track C — Point-in-time data layer + walk-forward backtest harness
*Design finalized from the independent design review — see "ML/backtest design" below.*

### Track D — Learned ranking pillar (pooled cross-sectional gradient boosting)
*See "ML/backtest design" below.* Plugs into `src/scoring.score_universe` as a 4th pillar (`scoring.pillars.learned`), `SCORE_VERSION` → 4, `pillar_ic` extended to it.

### Track E — Meta-labeling on the system's own history
*See "ML/backtest design" below.* Secondary classifier over (pillar scores, Claude rating/conviction, red-team haircut, regime) → P(call is right). Used as a **confidence badge and sizing multiplier**, never a veto.

### Track F — Risk & sizing (advisory)
- Portfolio **volatility target** (config `policy.vol_target_pct`, default 12%): `risk.portfolio_risk` already returns `ann_vol_pct` from the 90-day covariance — add `vol_scale = min(1, target/ann_vol)` to its output and one line to `risk_context_lines`; the PM-reconcile prompt then receives "current annualized vol X% vs target Y% → scale gross exposure by Z". While there, lift `_BENCH`/`_THEME` (`QQQ`/`SMH`, hardcoded in `risk.py:22-23`) into config next to `technical.rs_benchmark` — third copy of the theme benchmark (CR3 F5/F15).
- **Fractional-Kelly-style cap:** per-position cap = min(existing conviction cap, k · edge estimate) where the edge estimate comes from Track E's probability once it is significant; until then unchanged.
- **Drawdown rule:** if the shadow portfolio (Track B) is >X% below its 60-day high, caps tighten a further 25% (extends the regime mechanism in `deep_research.sizing_rules_text`).
- **Correlation-aware add sizing:** `risk.portfolio_risk` already yields effective bets and avg correlation; feed a "marginal correlation of this add" line into the deep-dive prompt.
- Files: `src/risk.py`, `src/deep_research.py` (`sizing_rules_text`), `src/portfolio.py`, `config.yaml`.

### Track G — LLM layer quality (platform-compatible)
- **Provider protocol** (CR3 F1): `src/llm/` with `claude_cli` (today's code) and `anthropic_api`; `local_llm` moves behind the same interface. Prerequisite for the `FakeProvider` used in tests and backtest replays.
- **Structured outputs:** keep JSON prompts; add per-template pydantic-style validation so a malformed field is rejected/reformatted instead of stored.
- **Red-team calibration monitor:** track survive-rate and haircut distribution per prompt version (the July overkill would have been visible in a week).
- **Injection surface:** Investigate mode defaults to read-only tools; Bash/DB-write is an explicit opt-in (CR3 F13).
- Files: new `src/llm/*`, `src/deep_research.py`, `src/chat.py`, `src/local_llm.py`, `app/dashboard.py`.

### Track H — Platform-compatible foundations (only what Tracks A–G need)
- `pyproject.toml` + entry points; `sys.executable` for child processes (CR3 F4).
- `Workspace`/settings with `CLAUDE_TRADER_HOME`; defaults ⊕ user config (CR3 F2) — the data store in Track C lives under it.
- `pytest` scaffold: pure-module tests (`scoring`, `technical._score_and_state`, `recommend`, `triggers.check_all`, `_extract_json`, regime classification, `portfolio` policy backstop) + `FakeProvider` end-to-end run (CR3 F8).
- Pin deps with a lockfile.

**Explicitly not doing:** per-stock predictors; order execution; hosted multi-tenant; fine-tuning an LLM; paid data (design the `MarketData` interface so Sharadar/Norgate can slot in later).

---

## ML/backtest design (Tracks C, D, E)

**Headline finding from the design review:** the system's *own* history is far too small for supervised ML — 295 gradeable picks over 53 symbols, ~10 independent months, one regime, and **zero sell/strong_sell labels** (304 hold / 202 buy / 3 strong_buy). Anything trained on system outputs is a calibration exercise. Anything trained on **prices + SEC point-in-time fundamentals** has 10 years × 100+ names and is real. So: the learned pillar trains on market data; meta-labeling ships as a *measurement* first and a model only after a sample gate.

### C1 · Point-in-time store
```
$CLAUDE_TRADER_HOME/data/history/
  prices/daily.parquet            # long: date, symbol, o/h/l/c, adj_close, volume, adj_factor
  fundamentals/facts.parquet      # long: cik, symbol, tag, ddate, qtrs, filed, value  (SEC FSDS num+sub)
  fundamentals/snapshots.parquet  # wide PIT: as_of_date, symbol, derived metrics (materialized monthly)
  universe/membership.parquet     # date, symbol, source (curated|sic), subsector
  backtests/<run_id>/, models/rank_v1/<fold_date>/, trials.jsonl, manifest.json
```
- Prices: yfinance 10y batch (reuse `technical._download_with_deadline`); store raw + adjusted (adjusted for returns, raw for historical `$5` floors).
- Fundamentals: **SEC Financial Statement Data Sets**, every fact keyed by **`filed`**, never `ddate`. Filter to universe CIKs on ingest (reuse `news._ticker_cik_map`). Derive market cap from `dei:EntityCommonStockSharesOutstanding` × adj-close.
- **One `derive(raw) -> metrics` function** (`src/analytics/metrics.py`) used by both the live `.info` path and the historical path, so the model sees identical definitions at train and inference.
- `MarketData` protocol (`src/data/market.py`): `prices()`, `fundamentals_asof(as_of)` (only `filed <= as_of`), `universe_asof(as_of, source)`, `latest_snapshot()` (live-only). `YFinanceLive` + `HistoryStore` implement it; `fundamentals.py`/`technical.py` keep their signatures.
- **Survivorship:** cannot be fixed for free. Guards: (1) headline metrics are *ranking* metrics vs the same-date universe (matches `performance.excess_uni_*`); (2) build a second, **PIT universe from SEC filers by SIC code** + size floor — includes names that later died; run every backtest on both and show them side by side; (3) mandatory disclosure line in reports and the Accuracy tab.

### C2 · Walk-forward harness (`src/backtest/`)
- **Hand-rolled pandas engine (~300 lines), not vectorbt.** Daily, long-only, rank-and-hold, monthly rebalance, ≤500 names = weights × returns matrix; vectorbt's order-level machinery is more than this needs and another pinned dependency. Keep vectorbt as an optional cross-check.
- Signals at close of t from data with `filed`/date ≤ t; fill at **t+1 open**; monthly rebalance (weekly for sensitivity). Costs 15 bps/side default (10 commission+spread, 5 slippage), report gross and net, turnover per rebalance.
- Benchmarks: equal-weight same-date universe, SMH, SPY — identical to `performance.py`. **Purge** training rows whose 21-day label overlaps the test window; **21-day embargo** after each fold; test cross-sections ≥21 trading days apart (extends `learning.pillar_ic`'s 30-calendar-day rule).
- Metrics per fold and pooled: rank IC + Fama-MacBeth t-stat, decile spread, top-decile-minus-universe, hit rate, **Deflated Sharpe** using the trial count from `trials.jsonl` (every configuration tried is appended — the anti-snooping guard), max drawdown, turnover, cost drag. `BacktestReport` dataclass → markdown + parquet; Accuracy tab reads it.
- **Baselines the harness must run first:** (a) equal weight, (b) 12-1 momentum alone, (c) **the current 3-pillar composite replayed historically** (technical + fundamental reconstructible; sentiment fixed at 0.5). (c) is the real prize: it answers *whether the existing screen has any IC at all* without waiting years of live runs. If it doesn't, re-weighting matters more than ML — decision point before M3.

### D · Learned ranking pillar
- **Pooled cross-sectional model**, rows = (date, symbol). Per-stock models rejected (~2,500 rows each, no cross-sectional information).
- **Features (~35, all PIT, rank-transformed per date):** from `technical._compute_one` — `rsi`, `macd_hist`, `atr_pct`, `pos_52w`, `vol_trend`, `ret_20/63/126`, `ret_252_21`, `mom_va`, price/SMA ratios, `rs_3m/6m`, days since cross; add 20/60d realized vol, 63d max drawdown, dollar-volume rank, sector-relative momentum (the `_pct(groups=)` idea). Fundamentals **only** from the SEC store: `fcf_yield`, `ev_sales`, gross margin, YoY revenue growth, accruals, net cash/mcap, days since last filing, expected-10-Q proximity. Regime features recomputed historically. **No news/LLM features** (no history).
- **Label:** 21-day forward return minus same-date universe mean, rank-transformed per date; 63-day as robustness check.
- **Model:** LightGBM `lambdarank` (group = date) primary; **XGBoost `reg:squarederror` on the ranked label as the allowed alternative** (user's preference; near-equivalent here). Shallow (`num_leaves` 15–31, `min_data_in_leaf` ≥200, `feature_fraction` 0.7, `lambda_l2` ≥1, ≤300 rounds, early-stop on the previous fold), monotone constraints where economics are unambiguous, SHAP per fold (sign flips across folds = snooping flag).
- **Schedule:** expanding window, first train ≥4 years, retrain monthly, purge+embargo as C2; each fold's model + features + metrics stored under `models/rank_v1/<fold_date>/`. Live inference = latest fold on today's features via the same `derive()`.
- **Integration:** `scoring.score_universe(..., ml_scores=None)`; fourth pillar `learned`, percentile-ranked like the others, weight `scoring.pillars.learned` **default 0.0 (shipped dark)**; `SCORE_VERSION → 4` only when weight > 0; persist `pillar_learned` + `model_version` in `metrics_json`; `deep_research.build_context` adds "ML ranking pillar: X (OOS IC 0.0Y)" so Claude knows how much to trust it; health record flags a missing/stale model as *degraded*.
- **Honest expectation:** rank IC ≈ 0.02–0.06, t-stat 2–4 over a decade, decile spread a few % annualized gross, with regime breaks (a published LightGBM ranker turned negative in the 2024 AI rotation). Net of costs at monthly turnover: thin positive or flat. Value is as a **weight adjuster and tie-breaker**, not an alpha engine.
- **Kill criteria (fixed now, not tuned later):** pooled IC t-stat < 2 on ≥8 non-overlapping test dates; <60% of folds with positive IC; Deflated Sharpe ≤ 0; >70% of spread from a single year; fails to beat baseline (c) by ≥0.01 IC; SHAP top-3 unstable across folds. Any one → pillar stays at weight 0 and the report says why.

### E · Meta-labeling (calibration first, model later)
- **Now:** extend `performance.rating_calibration` into a table by (rating × conviction bucket × regime) with **Wilson confidence intervals and n**; surface in the Accuracy tab and `learning.calibration_line`. Precision-with-CI is the honest version of "how much to trust this kind of call".
- **Gate for a model:** ≥1,000 gradeable picks across ≥3 regimes and ≥12 non-overlapping months (12–18 more months of daily runs — effective n today is ~50–80 after overlap). When met: **regularized logistic regression**, 5–7 features (`pillar_*`, `blended`, `rank`, `rating`, `conviction`, red-team `survives`/haircut, `regime`, `rsi`, `pos_52w`, `days_to_earnings`), purged CV by month. Not GBM.
- **Use:** confidence badge ("historical precision for this kind of call: 58%, n=41, CI 43–72%") and a sizing multiplier in [0.5, 1.0] on the conviction-weight cap; included in `learning_block` so Claude sees it. **Never a veto.**

### Five ways this goes wrong here — and the guard
1. **`.info` leaks into training.** `data/cache/fund_*.json` and `metrics_json` are current-state. *Guard:* the feature builder imports only `HistoryStore`; a test asserts `fundamentals_asof` never returns `filed > as_of`; `metrics_json` is allowed only for E (where it *is* PIT — written that day).
2. **Hindsight universe.** *Guard:* curated vs SIC-derived side by side; ranking metrics only as headline; disclosure line.
3. **Fake sample size** from overlapping labels and daily cross-sections (48 daily runs ≈ 2 independent monthly observations). *Guard:* purge + embargo; t-stats on non-overlapping dates; report effective n.
4. **Trial snooping.** *Guard:* `trials.jsonl` feeds Deflated Sharpe; kill criteria written before running.
5. **Grading-base mismatch** — picks graded from `price_at_surface` (≈ prior close), harness fills at t+1 open; 3,510 pre-v3 `screen_scores` rows would contaminate pillar features. *Guard:* one `label_return(px, t0, days, base)` shared by `performance.py` and the harness; filter `score_version >= 3` everywhere; state the fill convention in every report.

---

## Progress log

- **2026-09-07 · Phase 0 — done in this session:** `src/health.py` + per-stage health on `runs`/`portfolio_runs` (pipeline + portfolio orchestrators record it; dashboard badge + header warning; launcher "Last run" line; `scripts/doctor.py`); `PROMPT_VERSION`/`model` stamped on every report; `runs.regime` recorded; cost-aware grading (`evaluation.cost_bps`), `label_return` public primitive, `_MEMO` eviction; `claude_vs_screen` bootstrap attribution, `calibration_table` (Wilson CIs by rating×conviction×regime), `calibration_by_prompt_version`; shadow portfolio charges turnover; Accuracy tab panels; storage auto-migrates on first connection; `CLAUDE_TRADER_HOME` workspace (defaults ⊕ user overrides, replace-wholesale collections); `sys.executable` in dashboard launches; `tests/` (46 tests incl. offline FakeProvider deep run). Verified: real light run #117 → `health ok`, all 6 stages recorded; AppTest clean.
- **First honest numbers (live DB, 4w, net of 30 bp):** Claude buys −1.76% vs screen-only −1.12% → edge −0.64 pp, CI [−5.0, +3.8] (no evidence); 1w: −2.94 pp, CI [−5.56, −0.47] (significantly *negative*). Holds: 32% hit, −5.5% avg. Small control (n=22), one regime — treat as a flag for Phase 2's replay, not a verdict.
- **Deferred from Phase 0:** `pyproject.toml`/entry points — blocked on renaming the `src` package (CR3 P1), installing a package literally named `src` is not acceptable; lockfile pending the same.
- **2026-09-07 · Phases 1–4 + G, same session ("continue all phases"):**
  - **Phase 1 (PIT data):** `src/analytics/metrics.derive()` is now the single metric definition (live `fetch_one` parity-checked: 53/53 keys identical); `src/data/` — `MarketData` protocol, `HistoryStore` (parquet under `$CLAUDE_TRADER_HOME/data/history`), `yfinance_live` (10y raw+adjusted OHLCV), `sec_facts` (SEC **companyfacts** per CIK instead of 4 GB of FSDS zips; every fact keyed by `filed`; TTM chaining with derived Q4s carrying the 10-K filing date; year-earlier TTM on quarter boundaries; fast numpy path 0.7 s/symbol after profiling showed 55 s). Store holds 237k price rows (108 symbols, 2016-09→2026-09), 211k facts (105 CIKs; JNPR/CFLT have none — CFLT is delisted and should leave the universe). Pipeline records universe membership per run (going-forward survivorship record). `scripts/data_refresh.py`.
  - **Phase 2 (harness):** `src/backtest/` — pandas rank-and-hold engine (next-close fill, 15 bp/side, purge-ready), metrics (rank IC + Fama-MacBeth t, deciles, Deflated Sharpe with a `trials.jsonl` counter, drawdown, turnover), baseline signals (12-1 momentum, technical replay, fundamental replay on PIT snapshots, composite), `scripts/backtest.py`. **Results on the real store (2017-09→2026-07, 107 monthly dates):** momentum 12-1 IC 0.015 t 0.77 (portfolio +18%/yr vs EW but 80% of it 2023-25 — the hindsight-universe/AI-rally signature); **technical pillar IC 0.012 t 0.64, −3%/yr net vs EW, 141% turnover/rebalance → the pillar as built has no cross-sectional power at 21d and churns expensively.**
  - **Phase 3 (learned pillar, code complete, training pending snapshots):** `src/ml/features.py` (36 PIT features incl. SEC fundamentals + regime; `rank_transform`), `train.py` (LightGBM lambdarank / lgbm_reg / xgb_reg, expanding window, purge+embargo, time-ordered early stopping, gain-importance stability, **kill criteria fixed in code**), `infer.py` (`usable()` gate: verdict must pass unless `scoring.learned.force`), `scripts/train_ranker.py`; scoring gets `ml_scores` + `learned` pillar (weight 0, composite normalized by weight sum, `effective_score_version()` → 4 when weighted); pipeline computes/persists it with health notes; Accuracy tab verdict panel; doctor checks store + model.
  - **Phase 4 (sizing):** `risk.portfolio_risk` → `vol_scale` (policy.vol_target_pct 12), `drawdown_from_peak_pct`, per-name `marginal_corr`; benchmarks to config (`risk.benchmark`, `risk.theme_benchmark`); `sizing_rules_text(risk_ctx)` tightens caps 25% on drawdown (policy.drawdown_tighten_pct 10) and injects the vol-target scale; deep-dive prompts carry each holding's correlation to the rest of the book; risk computed BEFORE the dives.
  - **Track G (LLM layer):** `src/llm/` provider protocol — `claude_cli` (moved, psutil/killpg tree-kill) and `anthropic_api` (BYO key, web_search tool); `deep_research._run_claude` delegates; `_coerce_report` enum/bounds validation; Investigate mode **read-only by default** (Bash/DB-write is a dashboard opt-in); `performance.red_team_stats()` + dashboard red-team monitor.
  - Tests: 74 passing (`tests/`), incl. fast-vs-reference PIT parity, harness true-signal/noise controls, and a native-API walk-forward smoke (LightGBM's sklearn wrapper needed scikit-learn; switched to `lgb.train`).
  - **Phase 2 replay results (rest):** fundamental pillar IC 0.019 t 1.35 (+0.7%/yr vs EW, DSR 0.24); composite 3-pillar IC 0.021 t 1.36 over 2017→2026 (+3.8%/yr net vs EW, 92% turnover/month, best-year share 0.46) — but **IC 0.062, t 2.72 over the 2021-09→2026-07 window** used for the ML comparison. The screen has weak-but-real ranking power recently, none of it from the technical pillar.
  - **Phase 3 verdict (first training run, 59 folds 2021-09→2026-07): DO NOT SHIP.** OOS rank IC 0.0014, t 0.08, positive on 50% of dates, DSR 0.45 (6 trials), IC below the composite baseline by 0.06; early stopping at 3–8 rounds (the validation set finds nothing to learn). Portfolio metrics looked fine (+7%/yr vs EW, 62% hit) — exactly the "returns without ranking skill" pattern the ranking-first design exists to catch. Pillar stays at weight 0; verdict in `data/history/models/rank_v1/latest.json`, panel in the Accuracy tab.
  - Housekeeping: CFLT and JNPR removed from the universe (delisted — no prices, no SEC filings); `data/history/` gitignored; requirements + README updated.

## Sequencing (one developer; ~8–9 weeks total, phases 1–2 can overlap)

| Phase | Weeks | Delivers | Decision at the end |
|---|---|---|---|
| **0 · Trust & measure** — Tracks A, B, H(tests, settings) | 2 | Health record + badge, `doctor`, prompt/model versioning, cost-aware grading, Claude-vs-screen attribution panel, calibration table with CIs (E-now), `pytest` + `FakeProvider`, `Workspace`/`CLAUDE_TRADER_HOME` | Nothing learned ships before this exists |
| **1 · PIT data layer** — C1 | 2 | Price store, SEC FSDS ingest, shared `derive()`, `MarketData` protocol, `claude-trader data refresh`, PIT assertion test, SIC-derived universe | — |
| **2 · Harness + replay** — C2 | 1.5 | Engine, metrics, `trials.jsonl`, baselines (a)(b)(c) | **First honest report on the current 3-pillar screen's historical IC.** If ≈0, re-weighting > ML; revisit M3 scope |
| **3 · Learned pillar** — D | 2 | Features, folds, SHAP, kill-criteria evaluation, dark integration (weight 0), Accuracy-tab panel | Turn weight up only if all kill criteria pass |
| **4 · Sizing** — F | 1 | Vol target, drawdown rule, correlation-aware add line, benchmark constants to config | — |
| **Parallel · LLM layer** — G | ~1 | Provider protocol (any time after Phase 0), structured-output validation, red-team monitor, Investigate read-only default | — |
| **Later · Meta-label model** — E | 0.5 | Logistic model when the sample gate (≥1,000 picks, ≥3 regimes) is met — 12–18 months out | — |

Prerequisites from CODE_REVIEW_3 P0 that this plan absorbs: F2 (workspace), F3 (health), F8 (tests). Not prerequisites: F1 (provider — runs in parallel), F5 (theme pack — the SIC universe should read subsectors from it once it exists), F4 (packaging — parallel).

## Critical files

- **Modify:** `src/storage.py` (health cols, versions, `_MEMO`-style eviction), `src/pipeline.py` + `src/portfolio.py` (stage health), `src/performance.py` (costs, calibration CIs, shared `label_return`), `src/learning.py` (`calibration_line`, `learning_block`), `src/scoring.py` (`ml_scores`, 4th pillar, `SCORE_VERSION`), `src/technical.py` + `src/fundamentals.py` (call `MarketData`), `src/risk.py` (vol target, constants → config), `src/deep_research.py` (`PROMPT_VERSION`, `sizing_rules_text`, ML line in `build_context`), `src/chat.py` (read-only default), `app/dashboard.py` (health badge, Accuracy tab panels), `config.yaml` (`evaluation:`, `scoring.pillars.learned`, `policy.vol_target_pct`, benchmark keys).
- **New:** `src/health.py`, `src/settings.py`, `src/analytics/metrics.py` (`derive`), `src/data/market.py` + `history_store.py` + `yfinance_live.py` + `sec_fsds.py`, `src/backtest/` (engine, metrics, report), `src/ml/` (features, labels, train, infer), `src/llm/`, `scripts/doctor.py`, `scripts/data_refresh.py`, `tests/` (+ `FakeProvider`), `pyproject.toml`.
- **Reuse, don't rewrite:** `technical._compute_one/_rsi/_macd/_atr_pct/_detect_cross/_download_with_deadline`, `scoring._pct` (grouped percentile), `performance._fwd_return/_price_history/_uni_mean_fwd`, `learning.pillar_ic` (non-overlap rule + Fama-MacBeth), `regime.detect` classification logic, `news._ticker_cik_map`, `risk.portfolio_risk`, `deep_research._extract_json`.

## Explicitly not doing
Per-stock predictors; execution or paper orders (advisory stays locked; `shadow_vs_actual` is the paper-portfolio measure); hosted multi-tenant; LLM fine-tuning; paid data (interface allows Sharadar/Norgate later); news/sentiment backtesting (no free history); replaying historical dates through Claude (knowledge-cutoff contamination).

---

## Verification

- **Track A:** kill Ollama / stop the Gateway and run `--mode light`; the run must show `health=degraded` with the exact stage and reason in the dashboard badge; `doctor` must report the same.
- **Track B:** Accuracy tab shows per-prompt-version calibration; a synthetic run with `cost_bps=0` vs `30` changes buy-cohort excess by the expected amount.
- **Track C:** backtest of a *known* signal (12-1 momentum) over 2015→2026 reproduces the documented sign and magnitude; a deliberately leaked feature (e.g., current `.info` PE) is caught by the leakage test.
- **Track D:** out-of-sample rank IC and t-stat reported per walk-forward fold; pillar ships only if kill criteria pass; `SCORE_VERSION` bump verified by `pillar_ic` filtering.
- **Track E:** classifier evaluated with time-ordered split only; reported with n; badge appears in dashboard only above the sample gate.
- **Tests:** `pytest` green; `FakeProvider` end-to-end `pipeline.run("deep")` completes in <60 s on a temp DB.
- **No regression:** a real `light` run + dashboard AppTest after each phase (the project's existing verification pattern).
