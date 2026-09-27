## What and why

<!-- One logical change. Link the issue if there is one. -->

## How it was tested

<!-- Commands run, new tests added. For a result-changing PR: what was measured, before and after. -->

## Checklist

- [ ] `python -m pytest tests -q` passes, offline (no network, no real `.env` or data store)
- [ ] `scripts/preflight_public.ps1` run with no findings
- [ ] No secrets, account numbers, holdings, personal paths or email addresses in code, examples or commit messages
- [ ] Commit subjects follow `area: summary`
- [ ] Docs updated if behaviour or config changed

**If this adds or changes a signal, feature or data source:**

- [ ] Every input is point-in-time (knowable on the date it is scored)
- [ ] Walk-forward backtest via `scripts/backtest.py`; rank IC and t-stat, top decile vs equal-weight universe net of costs, turnover and Deflated Sharpe pasted below
- [ ] Ships dark (weight `0.0` in `config.yaml`) with kill criteria stated before the results
- [ ] No LLM is graded on dates inside its training window

<!-- Backtest evidence (or "n/a"): -->

This project is advisory only; PRs that add order placement or trade execution will not be accepted.
