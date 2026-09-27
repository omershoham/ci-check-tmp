# Setup and operations

## Requirements

- Python 3.11+
- One LLM for the research stage (`llm.provider` in `config.yaml`):
  - `claude_cli` (default) — the [Claude Code](https://claude.com/claude-code) CLI, logged
    in, on your subscription. Web search included; the only provider for the dashboard's
    Investigate mode.
  - `anthropic_api` — an Anthropic API key. Web search included.
  - `openai_compat` — any OpenAI-compatible endpoint. Pick a `preset` (`openai`,
    `openrouter`, `gemini`, `deepseek`, `ollama`, `lmstudio`), set `model`, and put the
    matching key in `.env` (local servers need none). Plain chat endpoints cannot browse:
    research then works from the screening context only, recency validation is skipped,
    and reports are labelled `[no-web]`. Set `web_search: true` only for a model that
    really searches (e.g. an OpenRouter `:online` model).

  ```yaml
  # fully local, no API key, no cost:
  llm:
    provider: openai_compat
    openai_compat: {preset: ollama, model: "qwen3.5:9b"}
  ```
- Optional: [Ollama](https://ollama.com) with the model named in `config.yaml`
  (`local_llm.model`, default `qwen3.5:9b`) for the news/catalyst stage. Without it the
  pipeline still runs; the sentiment pillar is neutral and the run's health says so.
- Optional: IB Gateway or TWS with the API enabled for live holdings and quotes. Without
  it, holdings come from the last stored snapshot and quotes from yfinance.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

copy .env.example .env                  # macOS/Linux: cp
# then add the credential for your llm.provider to .env, e.g.
claude setup-token                      # claude_cli: paste into .env as CLAUDE_CODE_OAUTH_TOKEN
```

Set `SEC_CONTACT_EMAIL` in `.env`: SEC EDGAR's fair-access policy wants a contact address
in every request's User-Agent, and throttles anonymous callers.

```powershell
python -m scripts.doctor                # checks everything below; fix what it flags
python -m pytest tests -q               # offline suite (fake LLM, broker and data)
```

## Build the point-in-time data store

The broad universe (the default, `universe.source: broad`) is built from SEC filings. Run
the steps in order; the first run downloads ~47 quarterly SEC archives and 10 years of
prices (about an hour, a few GB):

```powershell
python -m scripts.data_refresh --registrants        # SEC registrants -> ~5,800 candidates
python -m scripts.data_refresh --shares-bulk        # point-in-time share counts
python -m scripts.data_refresh --screen-prices      # trailing prices for the liquidity floor
python -m scripts.data_refresh --prices             # 10y daily bars
python -m scripts.data_refresh --facts-bulk         # financial statements (FSDS)
python -m scripts.data_refresh --snapshots          # monthly point-in-time fundamentals
python -m scripts.data_refresh --universe-history --start 2016-01-01
python -m scripts.data_quality                      # the data-quality gate
```

Prefer a hand-picked list? Set `universe.source: curated` and `theme: ai_infra` (or write
your own theme pack in `themes/`), then `python -m scripts.data_refresh` builds a small
store for just those names.

## Run

```powershell
python -m scripts.run_daily --mode light   # quant + news + local LLM, no Claude (~20 min at 1,500 names)
python -m scripts.run_daily --mode deep    # + Claude research on ~10 finalists
streamlit run app/dashboard.py             # http://localhost:8501
python -m scripts.backtest --all           # walk-forward replays of each signal
python -m scripts.train_ranker             # learned ranking pillar + its kill-criteria verdict
```

## Configuration

Everything tunable lives in `config.yaml`: universe, floors, pillar weights, sizing policy,
horizons, costs. Set `CLAUDE_TRADER_HOME` to keep your own `config.yaml` (deep-merged over
the shipped defaults), `.env` and `data/` outside the checkout. Collection-valued sections
(`universe.sectors`, `scoring.pillars`, the policy lists) are replaced wholesale by an
override, so "my list" means exactly your list.

## Windows convenience launcher

`scripts/launch.ps1` starts Ollama, IB Gateway (via IBC) and the dashboard — skipping
anything already up — opens the browser, and offers a menu (deep run, light run, digest,
shut down, retry Gateway). `scripts/make_launcher_shortcut.ps1` puts it on the Desktop.

Unattended operation is optional: `scripts/setup_windows_tasks.ps1` registers scheduled
deep/light runs, a Gateway watchdog and a weekly digest; `stop_automation.ps1` /
`restore_automation.ps1` disable and re-enable them. Set times in your local timezone
(deep ~07:00 ET pre-market, light ~16:15 ET after the close).

## IB Gateway via IBC (optional)

IBC can keep IB Gateway logged in unattended. Keep `ReadOnlyApi=yes` in IBC's `config.ini`
(the API can then read but never trade) and point `ibkr.port` in `config.yaml` at the
Gateway's API port (4001 live Gateway, 7496 live TWS). A full re-authentication after a
long outage triggers a two-factor prompt on your phone. See [SECURITY.md](SECURITY.md).
