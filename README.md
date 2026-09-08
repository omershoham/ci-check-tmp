# Claude Trader — Daily Stock Opportunity Helper

A locally-hosted assistant that surfaces US stock **opportunities** each day — quality at a
fair price, plus catalysts and events. **Advisory only**: it ranks ideas and writes research;
you trade manually in IBKR.

The universe is either a **curated list** or a **broad, survivorship-aware SEC-registrant
universe** (`universe.source` in `config.yaml`). A **theme pack** in `themes/` supplies the
board's name, the counterparties named in the prompts and, for the curated path, the ticker
lists; `themes/ai_infra.yaml` is the shipped AI-infrastructure theme and `themes/general.yaml`
is the theme-free default. What a pack never varies is the catalyst KEY SET — those nine keys
live in `src/local_llm.py`, so signals stay comparable across themes.

> ⚠️ **Not financial advice.** This tool is a personal research aid. It can be wrong.
> Do your own due diligence. Nothing here is a recommendation to buy or sell any security.

## How it works (4-stage funnel, cheap → expensive)

1. **Quant screen** (Python, free) — technical + sector-aware fundamental pillars → composite
   ranking score + tags.
2. **Local LLM** (Ollama, free) — news scan → a fixed **9-key catalyst vector** (customer deal,
   M&A, guidance, analyst action, product cycle, regulation, capital return/insider,
   management/ownership, adverse event) + news tone. The catalyst pillar ships **dark**
   (weight 0) and accumulates history until forward data earns it a weight.
3. **Claude deep research** (your subscription) — deep web research on ~10 finalists.
4. **Streamlit dashboard** — blended, tagged board + IBKR holdings/quotes overlay.

## Setup

```powershell
# 1. Create & activate a virtualenv (from the project root)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure subscription auth for headless Claude
claude setup-token          # paste the token into .env as CLAUDE_CODE_OAUTH_TOKEN
copy .env.example .env       # then edit .env
#   IMPORTANT: do NOT set ANTHROPIC_API_KEY anywhere (it forces per-token billing)

# 3. Install the local LLM (see scripts/install_ollama.md)
#    ollama pull qwen2.5:7b-instruct

# 4. (Optional) Start IB Gateway / TWS with the API enabled for holdings + live quotes
```

## Run — the button

Nothing runs on a schedule. Double-click **"Claude Trader"** on the Desktop when you want to
work; it starts Ollama, IB Gateway (via IBC) and the dashboard, opens the browser, and then
offers a menu:

```
  [1] Deep research run (Claude, ~15 min)
  [2] Light refresh (no Claude)
  [3] Weekly digest
  [4] Open dashboard in browser
  [5] Shut everything down (Gateway + dashboard)
  [6] Retry IB Gateway
  [Enter] Just leave it running
```

Each step is skipped if already up, so pressing it twice is harmless. Closing the window
leaves the Gateway and dashboard running — use `[5]` to stop them.

```powershell
.\scripts\launch.ps1                    # same thing from a shell
.\scripts\launch.ps1 -Run deep -NoMenu  # start the stack and go straight to a deep run
.\scripts\make_launcher_shortcut.ps1    # (re)create the Desktop button
```

Or drive the pipeline directly:

```powershell
python -m scripts.run_daily --mode deep    # full pipeline incl. Claude
python -m scripts.run_daily --mode light   # quick refresh
streamlit run app/dashboard.py             # open the dashboard
```

### Going back to always-on

The old unattended setup (scheduled Deep/Light/Watchdog/Digest tasks + logon shortcuts) is
disabled, not deleted: `scripts/restore_automation.ps1` brings it back,
`scripts/stop_automation.ps1` retires it again. `scripts/setup_windows_tasks.ps1` still
re-registers the tasks from scratch if they were ever removed.

## Tuning

All thresholds, weights, the universe, and floors live in `config.yaml` — no code changes needed.
Set `CLAUDE_TRADER_HOME` to keep your own `config.yaml` (overrides, deep-merged over the shipped
defaults), `.env` and `data/` outside the checkout.

## Trust, measurement, backtests, ML

```powershell
python -m scripts.doctor            # environment, auth, Ollama, IBKR, DB, history store, model
python -m pytest tests -q           # offline test suite (fake LLM/broker/data)
python -m scripts.data_refresh      # point-in-time store: 10y prices + SEC companyfacts + snapshots
python -m scripts.backtest --all    # walk-forward replays: momentum, technical, fundamental, composite
python -m scripts.train_ranker      # learned ranking pillar, walk-forward, with fixed kill criteria
```

- Every run records **per-stage health** (`runs.health`): `ok` means the inputs were whole; `degraded`
  names the stage that wasn't (Gateway offline, Ollama down, Yahoo throttled...). The dashboard, the
  launcher and `doctor` all show it.
- Every research report carries its **prompt version and model**; grading is **net of a round-trip
  cost** on buy calls; the Accuracy tab attributes the Claude layer's edge against a screen-only control
  with a bootstrap CI and shows precision-with-CI by call type.
- Backtests run on the point-in-time store only (SEC facts keyed by **filing date**, never `.info`),
  grade with the same forward-return convention as the live calibration, charge costs, and report
  rank IC / t-stat / Deflated Sharpe with the trial count. **The universe is hand-picked with
  hindsight — read the ranking metrics, not the returns.**
- The **learned pillar** (`scoring.pillars.learned`) ships at weight **0**. `train_ranker` prints a
  verdict against kill criteria fixed in `src/ml/train.py`; raise the weight only when every one
  passes. Any weight > 0 bumps `score_version` to 4 so old and new scores never mix in the statistics.

## Security — what stays on your machine

This repo is **code and configuration only**. Everything that identifies you or your
portfolio is git-ignored and never leaves the machine:

| Stays local | Why it must |
|---|---|
| `.env` | Claude subscription OAuth token, SEC contact address |
| `data/trader.db` (+ `-wal`/`-shm`) | positions, average costs, research reports, P&L history |
| `data/digests/` | names your actual holdings and target weights |
| `logs/`, `data/*.log`, `data/*.err` | run logs contain IBKR account numbers |
| `data/cache/` | scraped fundamentals, hundreds of files a day |
| `.claude/settings.local.json` | machine-local paths and tool allow-list |

**IBKR credentials are never in this repo.** The Gateway logs itself in through
[IBC](https://github.com/IbcAlpha/IBC), which keeps them in `C:\IBC\config.ini` outside the
project. Two settings there matter: `ReadOnlyApi=yes` (the API can read the account but
**cannot place trades** — this project is advisory only and never needs write access), and
`AcceptIncomingConnectionAction=accept`. The app talks to `127.0.0.1:4001` and connects
`readonly=True`.

**Claude auth** is the subscription token in `.env` only. Keep `ANTHROPIC_API_KEY` unset
everywhere or Claude Code silently switches to per-token API billing.

**The dashboard has no authentication** and binds to `127.0.0.1` on purpose. Don't bind it
to `0.0.0.0` or port-forward it — anyone who reaches it sees your positions.

### Before you commit

```powershell
.\scripts\preflight_public.ps1
```

Resolves the exact file set git would track and scans only those files for tokens, emails,
IBKR account numbers and absolute user paths; exits non-zero on any finding. Run it before
every push — a `git add -f`, a new log location, or a debug fixture pasted into a tracked
file is all it takes to leak something.

### If you forked this

Set `CLAUDE_CODE_OAUTH_TOKEN` and `SEC_CONTACT_EMAIL` in your own `.env` (copy
`.env.example`). The SEC contact is required by their fair-access policy — without it EDGAR
throttles you and the 8-K feed degrades quietly.
