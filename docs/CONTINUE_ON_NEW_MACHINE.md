# Continuing on a new machine

What is in git: code, config defaults, docs, tests. What is **not** (by design — see
`.gitignore`): your Claude token, the database, the history store, logs, digests, and the
IBKR/IBC setup, which lives outside the project entirely.

## 1. Clone and install

```powershell
git clone <your-private-repo-url> Claude_Trader
cd Claude_Trader
python -m venv .venv
.\.venv\Scripts\Activate.ps1            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Secrets — `.env` (never committed)

```powershell
copy .env.example .env                  # macOS/Linux: cp
claude setup-token                      # paste the token into .env as CLAUDE_CODE_OAUTH_TOKEN
```
Also set `SEC_CONTACT_EMAIL` in `.env` (SEC EDGAR wants a contact address; the 8-K feed
degrades without it). Leave `ANTHROPIC_API_KEY` unset unless you switch `llm.provider` to
`anthropic_api` in `config.yaml`.

The Claude Code CLI must be installed and logged in on the new machine (`claude` on PATH).

## 3. Local LLM (optional but recommended)

```powershell
winget install Ollama.Ollama            # or https://ollama.com/download
ollama pull qwen3.5:9b                  # the model named in config.yaml local_llm.model
```
Without Ollama the pipeline still runs; the sentiment pillar is neutral and the run's
health shows `degraded: sentiment`.

## 4. IBKR (optional)

Holdings and quotes come from IB Gateway / TWS with the API enabled on `127.0.0.1:4001`
(`config.yaml` → `ibkr.port`; TWS uses 7496). On the original machine the Gateway is
auto-logged-in by IBC (`C:\IBC\config.ini`, `ReadOnlyApi=yes`) — that setup is machine-local
and must be redone by hand if you want it. Without a Gateway the pipeline runs with
`health degraded: broker` and the deep-dive uses the last stored holdings.

## 5. Verify, then build the history store

```powershell
python -m scripts.doctor                # every line should be OK or an expected WARN
python -m pytest tests -q               # offline suite, ~10 s
python -m scripts.data_refresh          # ~5 min: 10y prices + SEC companyfacts + snapshots
```

## 6. Run

```powershell
python -m scripts.run_daily --mode light          # quick refresh, no Claude
python -m scripts.run_daily --mode deep           # full research (uses your subscription)
streamlit run app/dashboard.py                    # http://localhost:8501
python -m scripts.backtest --all                  # walk-forward replays
python -m scripts.train_ranker                    # learned pillar + verdict
```

On Windows, `.\scripts\make_launcher_shortcut.ps1` recreates the Desktop button
(`scripts\launch.ps1` starts Ollama, the Gateway and the dashboard and shows a menu).

## 7. Where the thinking lives

- `docs/IMPROVEMENT_PLAN.md` — the approved plan, with a progress log and every
  backtest / training result so far
- `CODE_REVIEW_3.md` — the architecture review the plan came from
- `README.md` — commands and the trust/measurement model
