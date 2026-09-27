# How this project compares

*As of 2026-09-27.* A side-by-side look at this project and four prominent open-source LLM
finance-agent projects, on the dimensions this project was built around, plus the ones where
the others are ahead.

## Ground rules

- Every cell about another project comes from that project's own repository (README,
  changelog, roadmap or source), linked to the exact commit that was read. Links go to a
  pinned commit so they keep pointing at what was read, even after the project changes.
- **"Not documented"** means the claim could not be found in what was read. It does not mean
  the feature is absent. Corrections are welcome: open an issue or PR with a link.
- **"N/A"** means the dimension does not fit the project's design (for example, a
  survivorship-aware universe for a tool that analyzes tickers you name).
- These projects move fast (TradingAgents shipped v0.4.0 and v0.5.0 within three weeks of
  each other). Treat this page as a dated snapshot.

## Projects and versions read

| Project | Stars (2026-09-27) | License | Commit read | What it is |
|---|---|---|---|---|
| **This project** | new | MIT | this repo | Advisory US-equity research: quant screen, then local-LLM news, then LLM research, with every call graded forward |
| [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) | 108,820 | Apache-2.0 | [`35543d0`](https://github.com/TauricResearch/TradingAgents/tree/35543d0248bf89fcb92b17a15858ad0c0e940687) (v0.5.x) | Multi-agent LLM trading framework (analysts, bull/bear researchers, trader, risk team, portfolio manager) on LangGraph |
| [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) | 63,767 | MIT | [`5d2c7ca`](https://github.com/virattt/ai-hedge-fund/tree/5d2c7ca2d02c6501692a58bb363dfab1916890ba) | LLM investor personas plus quant models, blended into a fund; a "v2" engine rebuild is in progress |
| [HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading) | 34,095 | MIT | [`0244ece`](https://github.com/HKUDS/Vibe-Trading/tree/0244eceaaf8d9e8ae79e2355836802acae3d1fc2) | Natural-language research workspace: strategy generation, many backtest engines, broker connectors |
| [AI4Finance-Foundation/FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) | 8,093 | Apache-2.0 | [`6d6ccd3`](https://github.com/AI4Finance-Foundation/FinRobot/tree/6d6ccd32c1b8b1904dc656cf06897438aba3daec) | Multi-agent equity-research report generation (DCF, comps, and similar), plus a macOS desktop app |

Star counts come from the GitHub REST API (`GET /repos/{owner}/{repo}`, field
`stargazers_count`), read on 2026-09-27.

Not included: [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) (21,297 stars), which
is a family of fine-tuned financial LLMs rather than a research or trading agent, and
[HKUDS/AI-Trader](https://github.com/HKUDS/AI-Trader) (22,577 stars), which is now a hosted
social trading platform for agents. Neither fits the comparison.

## Summary table

| Dimension | This project | TradingAgents | ai-hedge-fund | Vibe-Trading | FinRobot |
|---|---|---|---|---|---|
| Point-in-time fundamentals (keyed by filing date) | Yes, default | Yes, opt-in (v0.5.0) | Partial (in progress) | Yes (SEC fundamental factors) | Not documented |
| Survivorship-aware universe | Yes | N/A (you name tickers) | N/A (you name tickers) | Not documented | N/A (single-company reports) |
| Cost-net evaluation against a control | Yes | Benchmark alpha; costs not documented | Benchmark; no costs yet | Yes (costs + benchmark) | Not documented |
| Multiple-testing accounting | Yes (trial count, Deflated Sharpe) | Not documented | Planned (CPCV/PBO) | Library functions | Not documented |
| LLM calls graded forward only | Yes | Partial (live decision log; backtests replay past dates) | No (masked replay) | Different design | Not documented |
| Measured whether the LLM beats a non-LLM baseline | Yes (no evidence yet) | Not documented | Not documented | Not documented | Not documented |
| Local / offline models | Yes (Ollama, LM Studio) | Yes (Ollama, vLLM, LM Studio, llama.cpp) | Planned (Ollama) | Yes (Ollama) | Not documented |
| Bring your own provider | Yes (3 back ends, 6 presets) | Yes (15+) | Yes (7) | Yes (20+) | OpenAI documented |
| Broker integration | IBKR, read-only | None (simulated exchange) | Simulated only (IBKR/Alpaca planned) | 18 brokers | Not documented |
| Live order execution | No, by design | No | No (planned, opt-in) | Yes (opt-in, bounded, experimental) | Not documented |
| Free data only | Yes | Yes by default | No (Financial Datasets key) | Yes by default | No (FMP key) |
| Multi-agent debate | No (one researcher + red-team pass) | Yes | Yes (persona blend) | Yes | Yes |
| Asset classes / markets | US equities | Global equities + crypto | Stocks | Equities in 7+ markets, crypto, futures, forex, options | Equities |
| Interface | Streamlit dashboard, CLI | CLI | Terminal UI (web scaffold) | Web UI, CLI, REST, MCP, chat apps | Web app, macOS desktop app |

## Where the others are ahead

This project is narrow on purpose, and on several axes the others are well ahead:

- **Community and maturity.** TradingAgents and ai-hedge-fund have tens of thousands of
  users and many contributors. This project is new and has one maintainer.
- **Multi-agent reasoning.** TradingAgents' bull/bear debate and risk team, FinRobot's
  bull/bear/judge agents and ai-hedge-fund's investor personas are richer designs than this
  project's single researcher plus a devil's-advocate pass.
- **Breadth.** Vibe-Trading covers seven or more equity markets, crypto, futures, forex and
  options, with ten backtest engines, 18 broker connectors and a 462-alpha library.
  TradingAgents covers any market Yahoo Finance does. This project covers US equities only.
- **Valuation work.** FinRobot computes DCF, DDM, LBO, comps and WACC in deterministic code
  and writes analyst-style reports. This project does not build valuation models.
- **Execution.** Vibe-Trading can place live orders through a broker you authorize, bounded
  by limits you set. This project deliberately never trades.
- **Interfaces.** Vibe-Trading has a web UI, a REST API, an MCP server and chat-app
  channels. FinRobot ships a desktop app. ai-hedge-fund has a polished terminal UI with a
  live backtest board.

## Where this project is different

The measurement layer is what sets it apart:

1. **An explicit test of whether the LLM adds value.** LLM picks are compared with a
   screen-only control using a bootstrap confidence interval, and the answer is published
   even though it is currently unflattering (no evidence of an edge yet). None of the
   projects above documents a comparison like this.
2. **A broad universe with delisted names, rebuilt from SEC registrants month by month.** The
   other tools analyze tickers you choose, which is useful but cannot measure ranking skill
   across a cross-section.
3. **Trial counting.** Every distinct backtest configuration is logged, and the Deflated
   Sharpe ratio is computed against that count.
4. **LLM calls are never replayed over past dates.** They are graded only forward. This
   avoids the memorization problem that ai-hedge-fund's README describes, and that it
   mitigates by masking tickers and dates.

## Details and sources, per project

### TradingAgents

Base: [`TauricResearch/TradingAgents@35543d0`](https://github.com/TauricResearch/TradingAgents/tree/35543d0248bf89fcb92b17a15858ad0c0e940687)

| Dimension | Finding | Source |
|---|---|---|
| Point-in-time fundamentals | **Yes, opt-in, since v0.5.0 (2026-09-18).** SEC EDGAR fundamentals "as they stood on the run's date": unfiled periods are not served, and restated figures read as first reported. You enable it by adding `sec_edgar` to the vendor chain; the default fundamentals vendor is yfinance. v0.3.1 and v0.4.0 fixed look-ahead in other data paths. | [CHANGELOG 0.5.0](https://github.com/TauricResearch/TradingAgents/blob/35543d0248bf89fcb92b17a15858ad0c0e940687/CHANGELOG.md?plain=1#L54-L66), [README "Fundamentals as filed"](https://github.com/TauricResearch/TradingAgents/blob/35543d0248bf89fcb92b17a15858ad0c0e940687/README.md?plain=1#L264-L282), [default_config.py](https://github.com/TauricResearch/TradingAgents/blob/35543d0248bf89fcb92b17a15858ad0c0e940687/tradingagents/default_config.py#L138-L145) |
| Survivorship-aware universe | N/A. You pass tickers; there is no universe construction. | [README "Evaluating decisions over time"](https://github.com/TauricResearch/TradingAgents/blob/35543d0248bf89fcb92b17a15858ad0c0e940687/README.md?plain=1#L331-L349) |
| Cost-net evaluation vs a control | Backtest cells are "scored on realized alpha against the instrument's regional benchmark, grouped by rating". Transaction costs are not documented. | [README L349](https://github.com/TauricResearch/TradingAgents/blob/35543d0248bf89fcb92b17a15858ad0c0e940687/README.md?plain=1#L349) |
| Multiple-testing accounting | Not documented. | README, CHANGELOG |
| Forward-only grading of LLM calls | Partial. The always-on decision log scores each live decision's realized return on the next run for that ticker (forward). `run_backtest` also replays the pipeline over past dates, and the README does not discuss model training-cutoff contamination for those replays. | [README L309](https://github.com/TauricResearch/TradingAgents/blob/35543d0248bf89fcb92b17a15858ad0c0e940687/README.md?plain=1#L309), [README L331-L349](https://github.com/TauricResearch/TradingAgents/blob/35543d0248bf89fcb92b17a15858ad0c0e940687/README.md?plain=1#L331-L349) |
| LLM vs non-LLM baseline measured | Not documented. The README says backtest results "are not guaranteed to match any published figure". | [README L371](https://github.com/TauricResearch/TradingAgents/blob/35543d0248bf89fcb92b17a15858ad0c0e940687/README.md?plain=1#L371) |
| Local / offline models | Yes: Ollama, plus any OpenAI-compatible server (vLLM, LM Studio, llama.cpp). | [README L180-L184](https://github.com/TauricResearch/TradingAgents/blob/35543d0248bf89fcb92b17a15858ad0c0e940687/README.md?plain=1#L180-L184) |
| Bring your own provider | Yes: OpenAI, Google, Anthropic, xAI, DeepSeek, Qwen, GLM, MiniMax, OpenRouter, Mistral, Kimi, Groq, NVIDIA NIM, Azure OpenAI, Bedrock, Ollama. | [README L154-L175](https://github.com/TauricResearch/TradingAgents/blob/35543d0248bf89fcb92b17a15858ad0c0e940687/README.md?plain=1#L154-L175), [README L230](https://github.com/TauricResearch/TradingAgents/blob/35543d0248bf89fcb92b17a15858ad0c0e940687/README.md?plain=1#L230) |
| Broker integration | None. Approved orders go to "the simulated exchange". Current holdings can be passed in as input (`--portfolio`). | [README L104](https://github.com/TauricResearch/TradingAgents/blob/35543d0248bf89fcb92b17a15858ad0c0e940687/README.md?plain=1#L104), [CHANGELOG 0.5.0](https://github.com/TauricResearch/TradingAgents/blob/35543d0248bf89fcb92b17a15858ad0c0e940687/CHANGELOG.md?plain=1#L54-L66) |
| Live order execution | No. | [README L104](https://github.com/TauricResearch/TradingAgents/blob/35543d0248bf89fcb92b17a15858ad0c0e940687/README.md?plain=1#L104) |
| Free data only | Yes by default: yfinance for prices, fundamentals and news; FRED (free key) for macro; keyless Polymarket. Alpha Vantage is optional and needs a key. | [default_config.py L138-L145](https://github.com/TauricResearch/TradingAgents/blob/35543d0248bf89fcb92b17a15858ad0c0e940687/tradingagents/default_config.py#L138-L145) |
| Multi-agent debate | Yes: bullish and bearish researchers hold structured debates, and the number of rounds is configurable. | [README L89](https://github.com/TauricResearch/TradingAgents/blob/35543d0248bf89fcb92b17a15858ad0c0e940687/README.md?plain=1#L89) |
| Markets | Any Yahoo Finance market (US, HK, Tokyo, London, India, Canada, Australia, China A-shares) plus crypto. | [README L202-L212](https://github.com/TauricResearch/TradingAgents/blob/35543d0248bf89fcb92b17a15858ad0c0e940687/README.md?plain=1#L202-L212) |
| Interface | Interactive CLI, Python API, Docker. | [README L195-L200](https://github.com/TauricResearch/TradingAgents/blob/35543d0248bf89fcb92b17a15858ad0c0e940687/README.md?plain=1#L195-L200) |

### ai-hedge-fund

Base: [`virattt/ai-hedge-fund@5d2c7ca`](https://github.com/virattt/ai-hedge-fund/tree/5d2c7ca2d02c6501692a58bb363dfab1916890ba). The repository describes its engine
as a "v2" rebuild that is still in progress, and several items below are marked as planned in
its own roadmap.

| Dimension | Finding | Source |
|---|---|---|
| Point-in-time fundamentals | Partial. "Point-in-time by construction ... filters on filing date, not report period" is a stated principle. The ROADMAP marks "Point-in-time data correctness" as in progress, and the features module as partial. | [hedge_fund/README L77, L93-L95](https://github.com/virattt/ai-hedge-fund/blob/5d2c7ca2d02c6501692a58bb363dfab1916890ba/hedge_fund/README.md?plain=1#L93-L95), [ROADMAP L38](https://github.com/virattt/ai-hedge-fund/blob/5d2c7ca2d02c6501692a58bb363dfab1916890ba/ROADMAP.md?plain=1#L38) |
| Survivorship-aware universe | N/A. Tickers are supplied per run (`--tickers`). | [README](https://github.com/virattt/ai-hedge-fund/blob/5d2c7ca2d02c6501692a58bb363dfab1916890ba/README.md?plain=1#L60-L80) |
| Cost-net evaluation vs a control | Equity curve and excess return against the mandate's benchmark. Costs are not modeled yet: the simulated broker says "Slippage/costs are a declared future addition". | [backtesting/fund.py](https://github.com/virattt/ai-hedge-fund/blob/5d2c7ca2d02c6501692a58bb363dfab1916890ba/hedge_fund/backtesting/fund.py#L55-L58), [brokers/sim.py L5](https://github.com/virattt/ai-hedge-fund/blob/5d2c7ca2d02c6501692a58bb363dfab1916890ba/hedge_fund/brokers/sim.py#L5) |
| Multiple-testing accounting | Planned: "Validation gate: CPCV, probability of backtest overfitting (PBO)". | [ROADMAP L39](https://github.com/virattt/ai-hedge-fund/blob/5d2c7ca2d02c6501692a58bb363dfab1916890ba/ROADMAP.md?plain=1#L39) |
| Forward-only grading of LLM calls | No, but the risk is acknowledged and mitigated. Backtests replay past dates through the LLM with the ticker, industry and dates withheld. The README says this "reduces the recall without removing it" and that "a window after the model's training cutoff is the cleanest read". | [README L77](https://github.com/virattt/ai-hedge-fund/blob/5d2c7ca2d02c6501692a58bb363dfab1916890ba/README.md?plain=1#L77) |
| LLM vs non-LLM baseline measured | Not documented. The repo does include a non-LLM quant model (post-earnings drift) and an event-study module. | [hedge_fund/README](https://github.com/virattt/ai-hedge-fund/blob/5d2c7ca2d02c6501692a58bb363dfab1916890ba/hedge_fund/README.md?plain=1#L22-L28) |
| Local / offline models | Planned: "Ollama next: the free local path". | [ROADMAP L37](https://github.com/virattt/ai-hedge-fund/blob/5d2c7ca2d02c6501692a58bb363dfab1916890ba/ROADMAP.md?plain=1#L37) |
| Bring your own provider | Yes: Anthropic, OpenAI, DeepSeek, Google, xAI, Kimi, TypeSafe (Jev). | [README L43](https://github.com/virattt/ai-hedge-fund/blob/5d2c7ca2d02c6501692a58bb363dfab1916890ba/README.md?plain=1#L43) |
| Broker integration | Simulated broker only. Paper broker, and a live Interactive Brokers / Alpaca plugin ("opt-in, off by default"), are planned. | [ROADMAP L88-L91](https://github.com/virattt/ai-hedge-fund/blob/5d2c7ca2d02c6501692a58bb363dfab1916890ba/ROADMAP.md?plain=1#L88-L91) |
| Live order execution | No: "the system does not actually make any trades". | [README L8](https://github.com/virattt/ai-hedge-fund/blob/5d2c7ca2d02c6501692a58bb363dfab1916890ba/README.md?plain=1#L8) |
| Free data only | No. Requires a Financial Datasets API key for prices, fundamentals and earnings. | [README L42](https://github.com/virattt/ai-hedge-fund/blob/5d2c7ca2d02c6501692a58bb363dfab1916890ba/README.md?plain=1#L42) |
| Multi-agent | Yes: LLM investor personas (Buffett, Munger, Graham, Lynch, Druckenmiller) blended into strategies alongside quant models. | [hedge_fund/README L22-L28](https://github.com/virattt/ai-hedge-fund/blob/5d2c7ca2d02c6501692a58bb363dfab1916890ba/hedge_fund/README.md?plain=1#L22-L28) |
| Markets | Stocks. Other asset classes are not documented. | README |
| Interface | Textual terminal app with a fund builder and a live backtest board. A web dashboard is scaffolded. | [README](https://github.com/virattt/ai-hedge-fund/blob/5d2c7ca2d02c6501692a58bb363dfab1916890ba/README.md?plain=1#L48-L54), [ROADMAP L111](https://github.com/virattt/ai-hedge-fund/blob/5d2c7ca2d02c6501692a58bb363dfab1916890ba/ROADMAP.md?plain=1#L111) |

### Vibe-Trading

Base: [`HKUDS/Vibe-Trading@0244ece`](https://github.com/HKUDS/Vibe-Trading/tree/0244eceaaf8d9e8ae79e2355836802acae3d1fc2). Its README is over 2,000 lines long; the
findings below come from the sections cited.

| Dimension | Finding | Source |
|---|---|---|
| Point-in-time fundamentals | Yes for its fundamental factor family: "PIT-safe SEC company facts ... (filed-date anchored)". The data loaders are described as "point-in-time historical bars". | [README L687](https://github.com/HKUDS/Vibe-Trading/blob/0244eceaaf8d9e8ae79e2355836802acae3d1fc2/README.md?plain=1#L687), [README L312](https://github.com/HKUDS/Vibe-Trading/blob/0244eceaaf8d9e8ae79e2355836802acae3d1fc2/README.md?plain=1#L312) |
| Survivorship-aware universe | Not documented. Universes such as `csi300` and "S&P 500" appear in examples; whether historical constituents are used is not stated. | README |
| Cost-net evaluation vs a control | Yes: market-specific costs in the backtest engines, and benchmark comparison. | [README L699](https://github.com/HKUDS/Vibe-Trading/blob/0244eceaaf8d9e8ae79e2355836802acae3d1fc2/README.md?plain=1#L699), [README L464](https://github.com/HKUDS/Vibe-Trading/blob/0244eceaaf8d9e8ae79e2355836802acae3d1fc2/README.md?plain=1#L464) |
| Multiple-testing accounting | Library support: a `multipletesting` module ("Deflated significance") and purged cross-validation. Whether backtest trials are counted automatically is not documented. | [README L731](https://github.com/HKUDS/Vibe-Trading/blob/0244eceaaf8d9e8ae79e2355836802acae3d1fc2/README.md?plain=1#L731) |
| Forward-only grading of LLM calls | Different design. The LLM mostly writes strategy code, which is then backtested, so there is no per-date LLM call to grade. Grading of LLM judgment calls is not documented. | [README L445-L466](https://github.com/HKUDS/Vibe-Trading/blob/0244eceaaf8d9e8ae79e2355836802acae3d1fc2/README.md?plain=1#L445-L466) |
| LLM vs non-LLM baseline measured | Not documented. | README |
| Local / offline models | Yes: "run locally with Ollama (no key needed)". | [README L865](https://github.com/HKUDS/Vibe-Trading/blob/0244eceaaf8d9e8ae79e2355836802acae3d1fc2/README.md?plain=1#L865) |
| Bring your own provider | Yes, 20+ providers including OpenRouter (the shipped default), DeepSeek, Groq, Qwen, Kimi, GitHub Copilot and Ollama. | [README L871](https://github.com/HKUDS/Vibe-Trading/blob/0244eceaaf8d9e8ae79e2355836802acae3d1fc2/README.md?plain=1#L871) |
| Broker integration | Yes: 18 broker connectors (read and paper, with bounded live trading where supported). IBKR is read-only. There is also a read-only multi-broker portfolio page. | [README L622-L628](https://github.com/HKUDS/Vibe-Trading/blob/0244eceaaf8d9e8ae79e2355836802acae3d1fc2/README.md?plain=1#L622-L628) |
| Live order execution | Yes, opt-in. Live orders are "bounded by a user-defined mandate (symbol allowlist, order-size / exposure caps, daily trade cap, instant kill switch)". The disclaimer calls broker trading "experimental". | [README L332](https://github.com/HKUDS/Vibe-Trading/blob/0244eceaaf8d9e8ae79e2355836802acae3d1fc2/README.md?plain=1#L332), [README L624](https://github.com/HKUDS/Vibe-Trading/blob/0244eceaaf8d9e8ae79e2355836802acae3d1fc2/README.md?plain=1#L624), [README L2146](https://github.com/HKUDS/Vibe-Trading/blob/0244eceaaf8d9e8ae79e2355836802acae3d1fc2/README.md?plain=1#L2146) |
| Free data only | Yes by default: "23 built-in sources ... no key, no cost". Premium data is optional. | [README L543](https://github.com/HKUDS/Vibe-Trading/blob/0244eceaaf8d9e8ae79e2355836802acae3d1fc2/README.md?plain=1#L543) |
| Multi-agent | Yes: investment, quant, crypto and risk teams. | [README L297](https://github.com/HKUDS/Vibe-Trading/blob/0244eceaaf8d9e8ae79e2355836802acae3d1fc2/README.md?plain=1#L297) |
| Markets | A-share, HK, US, Canada, UK, India and Korea equities, plus crypto, futures, forex and options. | [README L310](https://github.com/HKUDS/Vibe-Trading/blob/0244eceaaf8d9e8ae79e2355836802acae3d1fc2/README.md?plain=1#L310) |
| Interface | Web UI, CLI, REST API, MCP server, and many chat-app channels. | README |

### FinRobot

Base: [`AI4Finance-Foundation/FinRobot@6d6ccd3`](https://github.com/AI4Finance-Foundation/FinRobot/tree/6d6ccd32c1b8b1904dc656cf06897438aba3daec). FinRobot focuses on equity-research
report generation, not on grading calls, so most measurement dimensions do not apply to it.
Its V2 is a hosted product whose source "has not yet been open-sourced".

| Dimension | Finding | Source |
|---|---|---|
| Point-in-time fundamentals | Not documented. SEC EDGAR is one of seven data providers. | [README L133](https://github.com/AI4Finance-Foundation/FinRobot/blob/6d6ccd32c1b8b1904dc656cf06897438aba3daec/README.md?plain=1#L133) |
| Survivorship-aware universe | N/A (single-company reports with peers). | [README L189-L210](https://github.com/AI4Finance-Foundation/FinRobot/blob/6d6ccd32c1b8b1904dc656cf06897438aba3daec/README.md?plain=1#L189-L210) |
| Cost-net evaluation / multiple testing / forward grading / LLM-vs-baseline | Not documented. | README |
| Local / offline models | Not documented. | README |
| Bring your own provider | An OpenAI key is documented for FinRobot Pro. The original (V0) framework uses an AutoGen `OAI_CONFIG_LIST`; other providers are not documented in the README. | [README L155-L165](https://github.com/AI4Finance-Foundation/FinRobot/blob/6d6ccd32c1b8b1904dc656cf06897438aba3daec/README.md?plain=1#L155-L165), [README L338-L349](https://github.com/AI4Finance-Foundation/FinRobot/blob/6d6ccd32c1b8b1904dc656cf06897438aba3daec/README.md?plain=1#L338-L349) |
| Broker integration / live execution | Not documented. The disclaimer says it should not be construed as "recommendations for live trading". | [README L416](https://github.com/AI4Finance-Foundation/FinRobot/blob/6d6ccd32c1b8b1904dc656cf06897438aba3daec/README.md?plain=1#L416) |
| Free data only | No: an FMP API key (and, for V0, Finnhub and sec-api keys). | [README L155-L165](https://github.com/AI4Finance-Foundation/FinRobot/blob/6d6ccd32c1b8b1904dc656cf06897438aba3daec/README.md?plain=1#L155-L165), [README L348-L349](https://github.com/AI4Finance-Foundation/FinRobot/blob/6d6ccd32c1b8b1904dc656cf06897438aba3daec/README.md?plain=1#L348-L349) |
| Multi-agent debate | Yes: a lead agent, five role agents, and bull, bear and judge debate agents (desktop app). | [README L88-L110](https://github.com/AI4Finance-Foundation/FinRobot/blob/6d6ccd32c1b8b1904dc656cf06897438aba3daec/README.md?plain=1#L88-L110) |
| Valuation | Deterministic DCF, DDM, LBO, comps, WACC and Monte Carlo; "Numbers are code-calculated. Narratives are LLM-assisted." | [README L112-L123](https://github.com/AI4Finance-Foundation/FinRobot/blob/6d6ccd32c1b8b1904dc656cf06897438aba3daec/README.md?plain=1#L112-L123) |
| Interface | Local web app, and a macOS (Apple Silicon) desktop app. | [README L49-L67](https://github.com/AI4Finance-Foundation/FinRobot/blob/6d6ccd32c1b8b1904dc656cf06897438aba3daec/README.md?plain=1#L49-L67), [README L167-L177](https://github.com/AI4Finance-Foundation/FinRobot/blob/6d6ccd32c1b8b1904dc656cf06897438aba3daec/README.md?plain=1#L167-L177) |

### This project

| Dimension | Finding | Where |
|---|---|---|
| Point-in-time fundamentals | Yes, the default. SEC companyfacts and Financial Statement Data Sets are keyed by the date each fact was filed; TTM values are chained and derived Q4s carry the 10-K filing date. | [`src/data/sec_facts.py`](../src/data/sec_facts.py), [`src/data/sec_fsds.py`](../src/data/sec_fsds.py) |
| Survivorship-aware universe | Yes. The universe is rebuilt monthly from SEC registrants with point-in-time floors; 1,464 of 2,964 ever-members exited. | [`src/data/registrants.py`](../src/data/registrants.py), [`src/universe.py`](../src/universe.py) |
| Cost-net evaluation vs a control | Yes. Backtests charge 15 bp per side; live grading charges a 30 bp round trip (`evaluation.cost_bps`). Both measure against the equal-weight universe, and the LLM's picks are also measured against a screen-only control. | [`src/backtest/engine.py`](../src/backtest/engine.py), [`src/performance.py`](../src/performance.py) |
| Multiple-testing accounting | Yes. Every distinct configuration is recorded in `trials.jsonl`, and the Deflated Sharpe ratio uses that count. | [`src/backtest/engine.py`](../src/backtest/engine.py), [`src/backtest/metrics.py`](../src/backtest/metrics.py) |
| Forward-only grading of LLM calls | Yes. Past dates are never replayed through an LLM. | [IMPROVEMENT_PLAN.md, "Explicitly not doing"](IMPROVEMENT_PLAN.md) |
| LLM vs non-LLM baseline measured | Yes: `claude_vs_screen`, a bootstrap CI. Current read: no evidence of an edge (4-week edge −0.6 pp, 95% CI −5.0 to +3.8, small control). | [`src/performance.py`](../src/performance.py) |
| Local / offline models | Yes. Ollama runs the news stage; research can run on Ollama or LM Studio through `openai_compat`. | [`src/local_llm.py`](../src/local_llm.py), [`src/llm/openai_compat.py`](../src/llm/openai_compat.py) |
| Bring your own provider | Yes: `claude_cli` (subscription), `anthropic_api`, and `openai_compat` with presets for OpenAI, OpenRouter, Gemini, DeepSeek, Ollama and LM Studio, or any `base_url`. | [`src/llm/`](../src/llm/) |
| Broker integration | IBKR, read-only (holdings and quotes); `ReadOnlyApi=yes` is recommended. | [`src/ibkr.py`](../src/ibkr.py), [SECURITY.md](SECURITY.md) |
| Live order execution | No, by design. | [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) |
| Free data only | Yes: SEC EDGAR, yfinance, and Google News RSS. (LLM usage is separate; a local model costs nothing.) | [`src/news.py`](../src/news.py), [`src/data/`](../src/data/) |
| Multi-agent debate | No. One researcher per finalist, plus a devil's-advocate pass on buys and a portfolio-manager pass on holdings. | [`src/deep_research.py`](../src/deep_research.py) |
| Markets | US equities only. | |
| Interface | Streamlit dashboard, plus CLI scripts. | [`app/dashboard.py`](../app/dashboard.py) |
