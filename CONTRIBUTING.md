# Contributing

Thanks for looking. This project is small and opinionated: it exists to measure whether an
LLM research layer adds anything to a quant screen, so most of the rules below are about
not fooling ourselves. Read them before writing a signal.

Bug reports, data-quality reports and focused pull requests are all welcome. For anything
large (a new pillar, a new data source, a schema change), open an issue first so we can
agree on how it will be measured before you build it.

## Development setup

Python 3.11+.

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q            # fully offline: fake LLM, broker and data
python -m scripts.demo               # optional: a sandboxed board in ./demo, free data, no keys
```

The test suite never touches the network, your `.env` or your real data store:
`tests/conftest.py` redirects the database, cache and point-in-time store to pytest's temp
directory and fails any test that opens a store outside it. Tests that parse real cached SEC
archives are marked `slow` and skipped by default (`python -m pytest -m slow` to run them).

`python -m scripts.doctor` checks a real installation (config, LLM provider, Ollama, broker
port, data store, last run's health).

## Project layout

```
app/dashboard.py     Streamlit dashboard
src/                 the pipeline: universe, scoring, technical, news, local_llm,
                     deep_research, pipeline, performance, learning, health, ibkr, ...
  data/              point-in-time store, SEC FSDS / companyfacts loaders, registrants,
                     SIC sectors, the data-quality gate
  analytics/         derive(): the one metric definition shared by live screen and backtest
  backtest/          walk-forward rank engine, metrics (IC, Deflated Sharpe), baseline signals
  ml/                learned ranking pillar (features, walk-forward training, kill criteria)
  llm/               LLM provider seam (Claude CLI, Anthropic API, OpenAI-compatible)
scripts/             CLI entry points: python -m scripts.<name>
themes/              theme packs (board prose + optional curated universe)
tests/               offline test suite
docs/                setup, security, improvement plan, architecture review
```

## Research rules

These are the project's core, and PRs are reviewed against them.

1. **Point-in-time or nothing.** A feature may only use data that was knowable on the date
   it is scored. SEC fundamentals are keyed by the `filed` date, never the period end.
   Never use a data source that is not point-in-time (e.g. yfinance `.info`, current
   index membership, today's ticker list) in a feature that is backtested.
2. **Survivorship-aware universe.** Backtests run on the month-by-month universe rebuilt
   from SEC registrants, including companies that later delisted. A hand-picked list is
   allowed but must be disclosed as chosen with hindsight.
3. **A new signal comes with a walk-forward backtest and ships dark.** Add it as a baseline
   in `src/backtest/signals.py`, run it through `scripts/backtest.py`, and put the report
   numbers (rank IC and t-stat, top-decile vs equal-weight universe net of costs, turnover,
   Deflated Sharpe) in the PR. It enters `config.yaml` at weight `0.0` and is only
   switched on once it passes kill criteria written down *before* the results were seen
   (see `KILL_CRITERIA` in `src/ml/train.py` for the pattern).
4. **Every configuration counts.** Each backtest run is appended to `trials.jsonl`, and the
   number of distinct configurations deflates the Sharpe of whatever is kept. Don't delete
   or reset it to make a result look better, and report configurations you tried and
   dropped.
5. **Grade honestly.** Net of costs, against the equal-weight universe, on the horizon the
   idea claimed. Not against an index the theme happens to beat.
6. **Never grade an LLM on dates inside its training window.** LLM calls are graded
   forward only. Replaying past dates through a model that may have seen them is a memory
   test, not a backtest, and PRs that do it will be declined.
7. **Health is part of the output.** A stage that runs on partial inputs must mark the run
   `degraded` and say why (`src/health.py`), not fail silently or report `ok`.

## Adding a theme pack

A theme pack is a YAML file in `themes/`, selected by `theme:` in `config.yaml`. It owns
prose and, optionally, a curated universe:

```yaml
name: Example Board                 # shown in the dashboard
description: One sentence on what the theme covers.
actors:                             # counterparties whose deals matter; prompt prose only
  - large enterprise customers
prompt_fragments:
  analyst_role: >-
    an equity research analyst covering ...
  catalyst_emphasis: >-
    How to weigh catalysts in this theme.
subsectors:                         # {} for none; used when universe.source: curated
  some_group:
    label: Some Group
    symbols: [AAA, BBB]
```

A pack must never change the catalyst key set (`src/local_llm.CATALYST_KEYS`): the keys
are fixed so signals stay comparable across themes. Quote tickers YAML would misread
(`"ON"`). Start from `themes/general.yaml` or `themes/demo.yaml`.

## Adding an LLM provider

Providers live in `src/llm/`. Implement the `Provider` protocol from `src/llm/__init__.py`:
`name`, `supports_sessions`, `supports_web_search`, `model_label`, `complete(prompt, *,
tools, timeout, extra_args, model) -> Response` (raise `RuntimeError` on failure), and
`ready() -> (ok, detail)`, which must be offline and cheap. Then register it in
`get_provider()` and `PROVIDERS`, and add tests with a fake HTTP session (see
`tests/test_llm_providers.py`) so the suite stays offline.

Be honest about capabilities: if the endpoint cannot browse, `supports_web_search` must be
`False`, so research prompts are adapted and its reports are tracked separately as
`[no-web]`. If the endpoint already speaks OpenAI Chat Completions, add a preset to
`src/llm/openai_compat.py` instead of a new provider.

## Commits and pull requests

- Subjects are short and scoped: `area: summary`, e.g. `llm: OpenAI-compatible provider`,
  `dashboard: position-sizing guard used truthiness, NaN crashed the page`. The body says
  why, and what was measured if it changes a result.
- One logical change per PR. Include tests for behaviour changes.
- Before opening a PR, run the offline suite and the leak check:

  ```powershell
  python -m pytest tests -q
  .\scripts\preflight_public.ps1          # -NoHistory for a quick file-only scan
  ```

  The leak check scans for tokens, email addresses, broker account numbers, absolute user
  paths and commit identities. Commit with your GitHub noreply address. Never commit
  `.env`, anything under `data/` or `logs/`, or real holdings in examples.

## Out of scope

This project is **advisory only**. Broker access is read-only by design
(`ReadOnlyApi=yes`, `readonly=True`). Pull requests that add order placement, order
routing or any other trade execution will not be accepted, however well written.

By contributing you agree that your contributions are licensed under the [MIT License](LICENSE)
and that you will follow the [Code of Conduct](CODE_OF_CONDUCT.md).
