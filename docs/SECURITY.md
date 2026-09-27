# Security and privacy

This repository is **code and configuration only**. Everything that identifies a user or
their portfolio is git-ignored and never leaves the machine.

| Stays local | Why it must |
|---|---|
| `.env` | Claude subscription OAuth token, SEC contact address |
| `data/trader.db` (+ `-wal`/`-shm`) | positions, average costs, research reports, P&L history |
| `data/history/` | the point-in-time store, backtest reports, trained models (large, regenerable) |
| `data/digests/` | weekly digests name actual holdings and target weights |
| `logs/`, `data/*.log`, `data/*.err` | run logs can contain broker account numbers |
| `data/cache/` | scraped fundamentals, hundreds of files a day |
| `.claude/settings.local.json` | machine-local paths and tool allow-list |
| `.preflight_denylist*` | your private list of terms the leak check must never find |

## Broker access is read-only by design

Holdings and quotes come from Interactive Brokers through IB Gateway, optionally
auto-logged-in by [IBC](https://github.com/IbcAlpha/IBC). **Broker credentials are never in
this repository** — IBC keeps them in its own `config.ini`, outside the project. Two IBC
settings matter:

- `ReadOnlyApi=yes` — the API can read the account but **cannot place trades**. The project
  is advisory only and never needs write access.
- `AcceptIncomingConnectionAction=accept`

The app connects to `127.0.0.1:4001` with `readonly=True`, and every broker call runs under a
hard wall-clock deadline (`ibkr.total_timeout`) so a half-dead Gateway degrades the run
instead of hanging it.

## LLM credentials

With the default `llm.provider: claude_cli`, the only credential is the Claude Code
subscription token in `.env`. Keep `ANTHROPIC_API_KEY` unset everywhere: if it is set,
Claude Code silently switches to per-token API billing (the CLI provider strips it from the
subprocess environment as a second line of defence). With `llm.provider: anthropic_api` the
API key is the credential instead, also read from `.env`.

## The dashboard has no authentication

It binds to `127.0.0.1` on purpose. Don't bind it to `0.0.0.0` or port-forward it — anyone
who reaches it sees your positions.

The dashboard's **Investigate** chat mode runs an agent over the code and database. It is
**read-only by default**; shell access and database corrections are an explicit per-session
opt-in, because raw web headlines flow into its prompt (a prompt-injection surface).

## Before you push: the leak check

```powershell
.\scripts\preflight_public.ps1
```

It answers "if I push now, what leaves this machine?":

1. every tracked or trackable file is scanned for tokens, emails, broker account numbers
   and absolute user paths;
2. known-sensitive paths are confirmed ignored;
3. **history**: every commit author/committer must use a noreply address; commit messages
   are scanned for session URLs, emails, account numbers and tokens; and the content rules
   run over **every file version ever committed** — a secret deleted in a later commit is
   still public in the old one;
4. a **private denylist**: put your real name, personal emails, employer etc. one per line
   in `.preflight_denylist.txt` (git-ignored). The check searches files, history, messages
   and identities for them, so the public repo never contains the words it protects.

It exits non-zero on any finding. `-NoHistory` runs only the file checks (fast, for a
pre-commit hook).

For an independent second opinion, [gitleaks](https://github.com/gitleaks/gitleaks) over the
full history:

```
gitleaks git . --log-opts="--all" --redact
```

## Commit identity

Configure git to commit with your GitHub **noreply** address, and in GitHub → Settings →
Emails enable *Keep my email addresses private* and *Block command line pushes that expose
my email*. The second setting makes GitHub reject a push that would publish a personal
address, from any machine.

## Reporting a vulnerability

Please don't open a public issue. Use GitHub's private vulnerability reporting: the
repository's **Security** tab → **Report a vulnerability**. The report is visible only to
the maintainer until a fix is published.

Useful to include: what an attacker could reach (credentials, holdings, the dashboard, the
Investigate agent's tools), the steps or input that trigger it, and the affected commit.
Things in scope include anything that could leak `.env` or portfolio data, prompt injection
from news headlines that escapes the read-only defaults, and any path by which the broker
connection could place an order.
