"""Stage 3: Claude deep research on finalists, via headless `claude -p`.

Runs on the Claude Code SUBSCRIPTION using CLAUDE_CODE_OAUTH_TOKEN. We strip
ANTHROPIC_API_KEY from the subprocess environment so billing never silently
falls back to per-token API usage. NOTE: we deliberately do NOT use `--bare`
(it disables OAuth/keychain auth and forces API-key billing).
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

from . import config

# Bump whenever any research/deep-dive template below changes in a way that could
# move ratings or convictions. Stored on every report so the Accuracy tab can group
# calibration by the "analyst" that actually produced it (mirrors scoring.SCORE_VERSION).
#   v4  2026-09-07  versioning introduced; templates as of the Jul-17 red-team
#                   recalibration ("stress test, not a veto"). Earlier reports: NULL.
PROMPT_VERSION = 4


def _stamp(report: dict) -> dict:
    """Attach the prompt/model provenance to a parsed report."""
    report["prompt_version"] = PROMPT_VERSION
    report["model"] = config.get("claude.model", "claude-opus-4-8")
    return report


_PROMPT_TMPL = """You are an equity research analyst focused on US AI-infrastructure stocks.
TODAY'S DATE IS {today}. Research {name} ({symbol}), sub-sector: {subsector}.

Use web search to find information that is CURRENT AS OF TODAY ({today}) — latest price, news
and filings from the last few days/weeks. Do NOT rely on memory; verify with current sources and
note the date of the data you use. Assess, for a medium-term (weeks-to-months) swing horizon:
1. Likelihood of the company landing or materially benefiting from large hyperscaler
   deals / capex (AWS, Azure, Google Cloud, Oracle/OCI, Meta, NVIDIA ecosystem).
2. Moat / vision / competitive position.
3. Concrete near-term catalysts and their rough timing.
4. Key risks (including valuation).
5. If the price is >20% below its recent high: diagnose WHY it fell (estimate cuts vs
   multiple compression vs dilution vs theme beta, with dates) and whether the cause
   impairs the thesis — "cheaper same story" and "story broke" demand opposite ratings.
{holding_note}
Context from local screening (may be partial):
{context}

Give an overall "rating" for the medium-term swing horizon:
- "strong_buy": high-conviction, attractive entry now
- "buy": positive, worth initiating/adding
- "hold": fairly valued / wait — neither add nor trim
- "sell": deteriorating thesis or overvalued — trim/exit
- "strong_sell": clear exit
Be willing to say hold/sell; do not default to buy. Weigh valuation against the catalysts.

Also read the CHART SETUP from recent price action (use web data): describe the current technical
pattern/structure (e.g. uptrend/breakout, base/consolidation, cup-and-handle, head-and-shoulders,
downtrend) and what it implies. Only name a classic pattern if it's clearly present, and note that
patterns are probabilistic, not certainties.

Also write a "metrics_read": a plain-language interpretation, SPECIFIC TO THIS STOCK, of how to read
its key numbers — valuation multiples (is its PE/EV-Sales high or low and WHY, vs its growth and its
sector), growth, margins/quality, and what its RSI / moving-average setup / relative strength say
right now. Explain what an investor should understand from these particular numbers for this name.

Output ONLY a single JSON object, no prose before or after, with exactly these keys:
{{
  "ticker": "{symbol}",
  "rating": "strong_buy" | "buy" | "hold" | "sell" | "strong_sell",
  "conviction": <number 0-10>,
  "deal_likelihood": "low" | "medium" | "high",
  "thesis": "<2-4 sentence summary>",
  "chart_pattern": "<current chart setup/pattern + what it implies + caveat>",
  "metrics_read": "<plain-language, stock-specific: how to read THIS name's valuation/growth/quality and technical numbers, and what they imply>",
  "catalysts": ["<short catalyst>", ...],
  "risks": ["<short risk>", ...],
  "bull": "<one sentence>",
  "bear": "<one sentence>",
  "data_as_of": "<most recent date of the data you actually used, YYYY-MM-DD>",
  "latest_price_seen": <the current price you found, number or null>
}}"""


def auth_status() -> dict[str, Any]:
    return {
        "claude_on_path": shutil.which("claude") is not None,
        "oauth_token_set": bool(config.oauth_token()),
        "api_key_set": config.api_key_is_set(),
    }


def _balanced_span(text: str, start: int) -> tuple[int, int] | None:
    """Return (start, end_exclusive) of the balanced {...} beginning at `start`."""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return start, i + 1
    return None


def _extract_json(text: str) -> dict | None:
    """Pull a JSON object out of a text blob.

    Scans EVERY '{' (not just the first), so a stray brace in prose before the JSON —
    e.g. "Based on {today}'s data..." — doesn't discard an otherwise-good, expensive
    Claude response. Prefers the last balanced object (the report is asked to come last).
    """
    candidates: list[dict] = []
    pos = text.find("{")
    while pos != -1:
        span = _balanced_span(text, pos)
        if span:
            try:
                obj = json.loads(text[span[0]:span[1]])
                if isinstance(obj, dict):
                    candidates.append(obj)
            except json.JSONDecodeError:
                pass
            pos = text.find("{", span[1])
        else:
            pos = text.find("{", pos + 1)
    if not candidates:
        return None
    # Prefer the richest object (most keys); on ties prefer the LAST — the report is
    # instructed to come last, and an earlier tie may be a prompt-echoed example.
    best = candidates[0]
    for c in candidates[1:]:
        if len(c) >= len(best):
            best = c
    return best


def _run_claude(prompt: str, allowed_tools: str | None = None,
                timeout: float | None = None,
                extra_args: list[str] | None = None) -> tuple[str, str]:
    """Return (result_text, raw_output). Raises on provider failure/timeout.

    The ONE call site for every LLM request in the system; it delegates to the
    provider selected by config `llm.provider` (src/llm). Kept as a module function so
    tests can monkeypatch it and so callers never import a provider directly.
    extra_args are CLI flags (e.g. --resume <session-id>); only the CLI provider
    supports them.
    """
    from . import llm as _llm
    resp = _llm.get_provider().complete(prompt, tools=allowed_tools, timeout=timeout,
                                        extra_args=extra_args)
    return resp.text, resp.raw


_RATINGS = ("strong_buy", "buy", "hold", "sell", "strong_sell")


def _coerce_report(parsed: dict) -> dict:
    """Structured-output validation for a research report: enum-check the rating,
    bound the conviction, list-ify list fields. A malformed field becomes None (which
    the dashboard renders as 'unrated') instead of poisoning stored analysis."""
    r = str(parsed.get("rating") or "").strip().lower().replace(" ", "_")
    parsed["rating"] = r if r in _RATINGS else None
    try:
        c = float(parsed.get("conviction"))
        parsed["conviction"] = max(0.0, min(10.0, c))
    except (TypeError, ValueError):
        parsed["conviction"] = None
    dl = str(parsed.get("deal_likelihood") or "").strip().lower()
    parsed["deal_likelihood"] = dl if dl in ("low", "medium", "high") else None
    for k in ("catalysts", "risks"):
        v = parsed.get(k)
        parsed[k] = [str(x) for x in v][:12] if isinstance(v, list) else ([str(v)] if v else [])
    return parsed


def research_one(symbol: str, name: str, subsector: str,
                 context: str = "", holding: dict | None = None) -> dict[str, Any]:
    """One live research call. `today` is ALWAYS the real date on purpose: Claude's
    ratings are only meaningful FORWARD. Replaying a past date through an LLM whose
    training data covers that period is not a backtest, it is a memory test — the
    walk-forward harness (plan Track C) must never call this with a historical date.
    """
    holding_note = ""
    if holding:
        holding_note = (
            f"\nNOTE: This is a CURRENT HOLDING — {holding.get('shares')} shares at "
            f"avg cost {holding.get('avg_cost')}. Rate it as keep/add/trim/exit "
            f"(hold/buy/strong_buy/sell/strong_sell) given today's price and valuation.\n")
    prompt = _PROMPT_TMPL.format(today=date.today().isoformat(), symbol=symbol, name=name,
                                 subsector=subsector or "n/a",
                                 context=context or "(none)",
                                 holding_note=holding_note)
    last_raw = ""
    result_text = ""
    for _ in (1, 2, 3):
        try:
            result_text, last_raw = _run_claude(prompt)
            break  # call succeeded; do NOT re-run the whole (expensive) research on parse issues
        except Exception as exc:  # noqa: BLE001
            last_raw = f"error: {exc}"
            result_text = ""
            continue  # retry ONLY on transient subprocess/launch/timeout failures
    parsed = _extract_json(result_text)
    if parsed is None and result_text:
        # The research call worked but the JSON didn't parse. Reformat the text we already
        # have with a cheap, no-web call instead of paying for the full research again.
        parsed = _reformat_json(result_text)
    if parsed:
        parsed["symbol"] = symbol
        parsed["raw"] = result_text or last_raw
        parsed.setdefault("catalysts", [])
        parsed.setdefault("risks", [])
        parsed.setdefault("rating", None)
        parsed.setdefault("chart_pattern", None)
        parsed.setdefault("metrics_read", None)
        parsed.setdefault("data_as_of", None)
        return _stamp(_coerce_report(parsed))
    return _stamp({"symbol": symbol, "raw": result_text or last_raw, "rating": None,
                   "conviction": None, "deal_likelihood": None,
                   "thesis": "(could not parse JSON)", "catalysts": [], "risks": []})


def _reformat_json(text: str) -> dict | None:
    """Cheap salvage: ask Claude (no web tools) to re-emit the JSON object from `text`."""
    prompt = ("Extract the single JSON object from the text below and output ONLY that "
              "JSON, no prose:\n\n" + text[:8000])
    try:
        result_text, _ = _run_claude(prompt, allowed_tools="")
    except Exception:  # noqa: BLE001
        return None
    return _extract_json(result_text)


def _run_json(prompt: str) -> tuple[dict | None, str]:
    """Run claude with retries on SUBPROCESS failures only; a successful call whose
    output fails to parse is salvaged with a cheap no-web reformat instead of
    re-running the full (expensive, possibly web-searching) prompt up to 3x."""
    last_raw = ""
    result_text = ""
    for _ in (1, 2, 3):
        try:
            result_text, last_raw = _run_claude(prompt)
            break
        except Exception as exc:  # noqa: BLE001
            last_raw = f"error: {exc}"
            result_text = ""
            continue
    if not result_text:
        return None, last_raw
    parsed = _extract_json(result_text)
    if parsed is None:
        parsed = _reformat_json(result_text)
    return parsed, result_text


# ---- Portfolio deep-dive (on-demand, deeper than the board research) ----
_DEEPDIVE_TMPL = """You are a senior PORTFOLIO MANAGER doing a deep dive on a CURRENT HOLDING.
Today's date is {today}. Use web search to gather the LATEST information (price, news, filings,
analyst views from the last few weeks). Cite the date of each key fact you rely on.

Holding: {name} ({symbol}), sub-sector {subsector}.
Position: {shares} shares at avg cost {avg_cost}; latest price {last}; unrealized P&L {pnl_pct}%.
PORTFOLIO CONTEXT — this position is {weight_pct}% of the total portfolio:
{portfolio_block}
{history_block}

Assess for a medium-term swing horizon, given the cost basis:
- Is the thesis intact, improving, or deteriorating?
- DRAWDOWN DIAGNOSIS (mandatory if the price is >20% below its recent high): do NOT just
  react to the lower price — decompose WHY it fell, with dates: how much is (a) estimate
  cuts / fundamental news, (b) multiple compression / sentiment, (c) dilution or supply,
  (d) sector/theme beta. Then judge: is the CAUSE thesis-relevant, or is this the same
  story at a cheaper price? "Fell a lot" is an observation, not a reason to sell; "fell
  because the thesis broke" is. Say explicitly which one this is.
- Valuation now (cheap/fair/rich) and why.
- Action is a PORTFOLIO decision, not a thesis grade. SIZING RULES (binding, from the
  owner's investment policy):
{sizing_rules}
  * "thesis intact but speculative and oversized" = TRIM, not hold.
- Action MUST also reflect TODAY'S price ({last}): use "add" only if you would buy MORE at the
  current price right now; if you'd only add on a pullback, use "hold" and put the dip level in
  add_level. Use "trim" only if you'd reduce at today's price; "exit" to sell out.
- Be genuinely willing to TRIM (e.g. a speculative position up a lot) or EXIT (broken thesis);
  do NOT default to hold/add.
- target_weight_pct: the portfolio weight you would move this position to NOW at today's price
  (0 = exit; can be above current weight if adding). Must be consistent with action and the
  sizing rules above.
- CONVICTION (0-10): calibrate and USE THE FULL RANGE — do NOT cluster at 6-7. Anchors:
  9-10 = exceptional/rare, 7-8 = strong & clear, 5-6 = average/balanced, 3-4 = weak/contested,
  0-2 = exit. Lower the conviction for speculative, highly volatile, or contested setups even if
  you'd hold — conviction reflects how clear-cut AND how risky THIS specific situation is, so a
  high-flyer up >80%, a beaten-down turnaround, a steady compounder, and a broad index fund
  should NOT all get the same score.
- If this is a broad index ETF (e.g. VOO, QQQ), treat it as a diversifier: action is usually
  hold, conviction reflects market-level attractiveness (typically modest), skip single-stock mechanics.
- Concrete add level and trim level (prices).
- What would INVALIDATE the thesis (specific triggers).
- Position-specific risks and dated upcoming catalysts.

Output ONLY one JSON object:
{{"symbol": "{symbol}", "action": "add"|"hold"|"trim"|"exit", "conviction": <0-10>,
  "target_weight_pct": <number>,
  "thesis_health": "<intact|improving|deteriorating - one sentence>",
  "valuation": "<cheap|fair|rich - one sentence>",
  "add_level": <number or null>, "trim_level": <number or null>,
  "invalidation": "<what breaks the thesis>", "risks": ["..."],
  "catalysts": [{{"date": "<YYYY-MM or date>", "event": "..."}}],
  "data_as_of": "<most recent date of data you found, YYYY-MM-DD>",
  "sources": [{{"title": "...", "date": "<YYYY-MM-DD>", "url": "..."}}]}}"""

_VALIDATE_TMPL = """Today's date is {today}. Below is a research report on {symbol} produced moments ago.
Use web search to VERIFY it is current as of today: check the latest price, any news in the last few
days, and that cited dates/facts are not stale or contradicted by newer information.

REPORT:
{report}

Output ONLY one JSON object:
{{"is_current": true|false, "stale_items": ["<each outdated or contradicted claim>"],
  "confidence": <0-10>, "checked_as_of": "{today}", "note": "<one sentence>"}}"""

_SYNTH_TMPL = """Today's date is {today}. You are a portfolio risk analyst. Holdings with their
per-position deep-dive summaries:

{holdings_block}

{risk_block}

{ballast_block}

{learning_block}

Analyze the PORTFOLIO as a whole (concentration, correlation/single-theme risk, gaps, weakest link).
Ground your correlation/concentration statements in the COMPUTED RISK METRICS above — cite the
numbers. If ballast candidates are listed, recommend 1-3 concrete diversifiers with target weights
(the owner's policy targets ~{min_ballast}% non-theme ballast).

Then RANK every holding best-to-worst on risk-adjusted attractiveness and assign each a
CONVICTION 0-10. You MUST use the full range and DIFFERENTIATE — they may not all be the same:
reserve 8-10 for your highest-conviction names and 2-4 for the weakest/most speculative
(e.g. a richly-valued high-flyer, a contested turnaround, or an index fund offering little edge).
If two names truly tie, break the tie. A flat set of 6-7s is not acceptable.

Output ONLY one JSON object:
{{"overall_risk": "<low|moderate|high - one sentence>",
  "concentration": "<position/sector concentration assessment>",
  "correlation": "<how correlated these names are; single-theme risk — cite the computed numbers>",
  "diversification_gaps": "<exposures that are missing>",
  "weakest_link": "<holding most at risk and why>",
  "ranked_actions": [{{"symbol": "...", "action": "add|hold|trim|exit", "conviction": <0-10>, "why": "..."}}],
  "ballast_suggestions": [{{"symbol": "...", "target_weight_pct": <number>, "why": "<one line>"}}],
  "summary": "<2-3 sentence portfolio takeaway>"}}"""


def sizing_rules_text(risk_ctx: dict | None = None) -> str:
    """Render the owner's policy (config: policy.*) as binding prompt rules.

    Mechanical (not prose-level) effects on the conviction-weight caps:
      * STRESS / RISK_OFF regime  -> caps x0.75
      * book drawdown from peak worse than policy.drawdown_tighten_pct -> caps x0.75 again
      * realized vol above policy.vol_target_pct -> gross exposure scaled by vol_scale
    `risk_ctx` is the dict from risk.portfolio_risk (drawdown, vol_scale) when available.
    """
    caps = config.get("policy.conviction_weight_caps", {3: 10, 5: 15}) or {}
    max_w = config.get("policy.max_position_weight", 20)
    min_conv = config.get("policy.high_weight_min_conviction", 7)
    from . import regime as _regime
    reg = (_regime.detect(use_cache=True) or {}).get("regime")
    tighten = reg in ("stress", "risk_off")
    factor = 0.75 if tighten else 1.0
    rc = risk_ctx or {}
    dd = rc.get("drawdown_from_peak_pct")
    dd_limit = float(config.get("policy.drawdown_tighten_pct", 10) or 0)
    dd_tighten = isinstance(dd, (int, float)) and dd_limit > 0 and dd <= -dd_limit
    if dd_tighten:
        factor *= 0.75
    lines = []
    prev = 0
    for conv_cap in sorted(caps):
        w = round(caps[conv_cap] * factor, 1)
        rng = f"conviction <={conv_cap}" if not prev else f"conviction {prev + 1}-{conv_cap}"
        lines.append(f"  * {rng} must not remain above ~{w:g}% of the portfolio "
                     f"(trim/exit regardless of thesis);")
        prev = conv_cap
    lines.append(f"  * any position >{round(max_w * factor):g}% of the portfolio requires "
                 f"conviction >={min_conv} to keep in full;")
    if tighten:
        lines.append(f"  * NOTE: caps are TIGHTENED 25% because the market regime is "
                     f"{reg.upper()} — risk comes off first, new adds need red-team-proof "
                     f"evidence;")
        lines.append("  * in this regime you MAY move weight to cash (targets need not "
                     "sum to current equity exposure).")
    if dd_tighten:
        lines.append(f"  * DRAWDOWN CONTROL: the book is {dd:g}% below its recent peak "
                     f"(limit {dd_limit:g}%) — caps tightened a further 25%; no new adds "
                     f"until the book stabilizes; weight MAY go to cash.")
    vs = rc.get("vol_scale")
    if isinstance(vs, (int, float)) and vs < 0.95:
        lines.append(f"  * VOLATILITY TARGET: realized vol {rc.get('ann_vol_pct')}% vs target "
                     f"{rc.get('vol_target_pct')}% — scale total equity exposure to ~{vs:.0%} "
                     f"of current (targets should sum to roughly {vs:.0%} of the current "
                     f"invested weight; the rest is cash).")
    risk_pct = config.get("policy.risk_per_position_pct", 1.0)
    lines.append(f"  * when sizing an ADD, risk no more than ~{risk_pct:g}% of portfolio "
                 f"value to a 2-ATR stop (wider-swinging names get smaller adds).")
    return "\n".join(lines)


def deep_dive_holding(symbol: str, name: str, subsector: str, holding: dict,
                      portfolio_block: str = "(not provided)",
                      history_block: str = "",
                      risk_ctx: dict | None = None) -> dict[str, Any]:
    prompt = _DEEPDIVE_TMPL.format(
        today=date.today().isoformat(), symbol=symbol, name=name, subsector=subsector or "n/a",
        shares=holding.get("shares"), avg_cost=holding.get("avg_cost"),
        last=holding.get("last"), pnl_pct=holding.get("pnl_pct"),
        weight_pct=holding.get("weight_pct", "?"),
        portfolio_block=portfolio_block,
        history_block=history_block,
        sizing_rules=sizing_rules_text(risk_ctx))
    parsed, raw = _run_json(prompt)
    if not parsed:
        return _stamp({"symbol": symbol, "action": None, "conviction": None,
                       "thesis_health": "(could not parse)", "risks": [], "catalysts": [],
                       "sources": [], "raw": raw})
    parsed["symbol"] = symbol
    parsed["raw"] = raw
    for k in ("risks", "catalysts", "sources"):
        parsed.setdefault(k, [])
    return _stamp(parsed)


def validate_recency(symbol: str, report: dict) -> dict[str, Any]:
    compact = {k: report.get(k) for k in
               ("action", "thesis_health", "valuation", "add_level", "trim_level",
                "catalysts", "data_as_of")}
    prompt = _VALIDATE_TMPL.format(today=date.today().isoformat(), symbol=symbol,
                                   report=json.dumps(compact, default=str))
    parsed, raw = _run_json(prompt)
    if not parsed:
        return {"is_current": None, "stale_items": [], "confidence": None,
                "checked_as_of": date.today().isoformat(), "note": "validation unavailable"}
    parsed.setdefault("stale_items", [])
    parsed["checked_as_of"] = parsed.get("checked_as_of") or date.today().isoformat()
    return parsed


_RANK_TMPL = """You are ranking a portfolio's holdings by RISK-ADJUSTED conviction. Holdings:
{block}

Assign each holding a conviction 0-10. HARD REQUIREMENTS:
- Use the FULL range and DIFFERENTIATE — do not give two holdings the same score unless truly identical.
- Reserve 8-10 for your highest-conviction names; 2-4 for the weakest / most speculative / most
  redundant (e.g. a richly-valued high-flyer up a lot, or an index fund that adds little edge).
- Every value MUST be an integer 0-10 — NEVER null.

Output ONLY a JSON object mapping each symbol to its integer, e.g. {{"AAA": 8, "BBB": 3, "CCC": 7}}
(AAA/BBB/CCC are placeholders — use the actual holding symbols from the list above)."""


def rank_convictions(summaries: list[dict]) -> dict[str, int]:
    """One focused call: return {symbol: conviction 0-10}, differentiated, no nulls."""
    block = "\n".join(
        f"- {h.get('symbol')}: action={h.get('action')}, {h.get('thesis_health','')}; "
        f"valuation {str(h.get('valuation',''))[:50]}; P&L {h.get('pnl_pct')}%; "
        f"weight {h.get('weight_pct', '?')}%"
        for h in summaries)
    parsed, _ = _run_json(_RANK_TMPL.format(block=block))
    out: dict[str, int] = {}
    for k, v in (parsed or {}).items():
        try:
            out[str(k).upper()] = max(0, min(10, int(round(float(v)))))
        except (TypeError, ValueError):
            continue
    return out


_REDTEAM_TMPL = """Today is {today}. You are a DEVIL'S ADVOCATE analyst. Another analyst just
rated {name} ({symbol}) "{rating}" (conviction {conviction}) with this thesis:

{thesis}

Their stated risks: {risks}

Your job: argue this is a SELL. Use web search to find the STRONGEST current bear
case — disconfirming evidence the buy thesis must survive: deteriorating data points,
competitive threats, valuation math, insider/analyst signals, adverse news the thesis
ignores. Do NOT restate the risks above — find what they MISSED.

Then judge honestly — you are a STRESS TEST, not a veto:
- "survives": false requires CONCRETE, DATED, NEW disconfirming evidence that the
  original analysis did not address AND that materially changes the risk/reward. A
  well-argued generic bear narrative, a valuation concern the thesis already weighs,
  or "the market is shaky" does NOT kill a rating. If your bear case is built from
  risks the analyst already listed, the rating SURVIVES.
- Calibrate the haircut, don't default: 0 = bear case was already priced into the
  analysis; 1 = a real but modest new concern; 2-3 = material NEW evidence (reserve
  for genuine finds). Most well-researched calls should survive with a 0-1 haircut —
  if you find yourself killing most ratings, you are grading your own eloquence, not
  the evidence.

Output ONLY one JSON object:
{{"bear_case": "<the strongest 2-4 sentence bear argument, with specifics>",
  "new_evidence": ["<disconfirming fact the original missed, with date>", ...],
  "survives": true|false,
  "suggested_rating": "strong_buy"|"buy"|"hold"|"sell"|"strong_sell",
  "confidence_haircut": <0-3, per the calibration above>,
  "note": "<one sentence: the single biggest vulnerability>"}}"""


def red_team(symbol: str, name: str, report: dict) -> dict[str, Any]:
    """Adversarial pass on a buy-rated report: find what the bull case missed."""
    prompt = _REDTEAM_TMPL.format(
        today=date.today().isoformat(), symbol=symbol, name=name,
        rating=report.get("rating"), conviction=report.get("conviction"),
        thesis=str(report.get("thesis", ""))[:800],
        risks="; ".join(str(r) for r in (report.get("risks") or [])[:6]))
    parsed, _ = _run_json(prompt)
    if not parsed:
        return {}
    parsed.setdefault("new_evidence", [])
    return parsed


def apply_red_team(report: dict, rt: dict) -> dict:
    """Fold the red-team verdict into the report: haircut conviction, downgrade if
    the rating didn't survive. Original values preserved in the red_team payload."""
    if not rt:
        return report
    report["red_team"] = rt
    try:
        haircut = max(0, min(3, int(rt.get("confidence_haircut") or 0)))
    except (TypeError, ValueError):
        haircut = 0
    conv = report.get("conviction")
    if isinstance(conv, (int, float)) and haircut:
        rt["original_conviction"] = conv
        report["conviction"] = max(0, round(conv - haircut, 1))
    if rt.get("survives") is False:
        sug = str(rt.get("suggested_rating", "")).lower()
        if sug in ("strong_buy", "buy", "hold", "sell", "strong_sell"):
            rt["original_rating"] = report.get("rating")
            report["rating"] = sug
    return report


_RECONCILE_TMPL = """Today is {today}. You are the PORTFOLIO MANAGER making the FINAL call after
reading each holding's single-name analysis. The analyst actions below were formed BEFORE the
convictions were cross-calibrated, so they can be inconsistent — your job is to reconcile them
into one coherent set of portfolio decisions at today's prices.

Holdings (analyst action, CALIBRATED conviction 0-10, current weight % of portfolio, P&L %):
{block}

YOUR PREVIOUS DECISIONS (from the last deep-dive — churn is expensive):
{prior_block}

{learning_block}

BINDING sizing rules (from the owner's investment policy):
{sizing_rules}
- redundant index exposure and the weakest names are the natural funding sources for rebalancing;
- respect the analyst's action when it is consistent with these rules; override it when not.
TURNOVER & TAX DISCIPLINE (binding):
- do NOT reverse a previous add->trim (or trim->add) within ~10 trading days unless an
  invalidation trigger fired or material new information arrived — name the reason if you do;
- ignore weight changes smaller than 2 percentage points (not worth the round trip);
- every round trip costs ~0.3% in spread/slippage, and gains on positions held under a year
  are taxed at the short-term rate — prefer trimming long-held lots and letting recent
  winners season when the decision is otherwise close.
- target_weight_pct = the weight to move to NOW (0 = exit). Targets should sum to roughly the
  current total equity exposure — this is a rebalance, not a liquidation — UNLESS the sizing
  rules above say the regime permits moving weight to cash.

EXECUTION TIMING (for every trim/exit — sizing says HOW MUCH, this says WHEN):
- "immediate": ONLY when the thesis broke, a sell-signal trigger fired, or new adverse
  information is still being priced in — get out before it gets worse.
- "staged": for SIZE-DISCIPLINE trims of an intact thesis (position merely too big).
  Never dump an oversold, heavily-shorted name into a capitulation low — that sells the
  bottom to the short-coverers. Stage it INTO STRENGTH: e.g. 1/3 now, 1/3 at a named
  bounce level, 1/3 after the next catalyst; or wait for stabilization (a close back
  above the 20-day average) before the first tranche. Give concrete levels.

Output ONLY one JSON object mapping every symbol:
{{"SYM": {{"action": "add"|"hold"|"trim"|"exit", "target_weight_pct": <number>,
  "why": "<one short sentence>",
  "execution": "immediate"|"staged",
  "execution_plan": "<for trims/exits: concrete tranches + levels; else empty>"}}, ...}}"""


def reconcile_actions(summaries: list[dict],
                      learning_block: str = "",
                      prior_block: str = "(no previous deep-dive)",
                      risk_ctx: dict | None = None) -> dict[str, dict]:
    """Final PM pass: reconcile per-name actions with calibrated convictions + weights.

    Exists because the per-position analyst finalizes its action BEFORE cross-holding
    conviction calibration, which is how 'hold, conviction 2' at 22% weight happens.
    Returns {symbol: {action, target_weight_pct, why}}.
    """
    block = "\n".join(
        f"- {h.get('symbol')}: analyst_action={h.get('action')}, "
        f"conviction={h.get('conviction')}, weight={h.get('weight_pct', '?')}%, "
        f"P&L {h.get('pnl_pct')}%; {str(h.get('thesis_health',''))[:80]}"
        for h in summaries)
    parsed, _ = _run_json(_RECONCILE_TMPL.format(today=date.today().isoformat(), block=block,
                                                 learning_block=learning_block,
                                                 prior_block=prior_block,
                                                 sizing_rules=sizing_rules_text(risk_ctx)))
    out: dict[str, dict] = {}
    for k, v in (parsed or {}).items():
        if not isinstance(v, dict):
            continue
        act = str(v.get("action", "")).strip().lower()
        if act not in ("add", "hold", "trim", "exit"):
            continue
        try:
            tw = float(v.get("target_weight_pct"))
        except (TypeError, ValueError):
            tw = None
        execu = str(v.get("execution", "")).strip().lower()
        out[str(k).upper()] = {"action": act, "target_weight_pct": tw,
                               "why": str(v.get("why", ""))[:200],
                               "execution": execu if execu in ("immediate", "staged") else None,
                               "execution_plan": str(v.get("execution_plan", ""))[:300]}
    return out


def portfolio_synthesis(summaries: list[dict], risk_block: str = "",
                        ballast_block: str = "",
                        learning_block: str = "") -> dict[str, Any]:
    block = "\n".join(
        f"- {h.get('symbol')}: action={h.get('action')}, conviction={h.get('conviction')}, "
        f"{h.get('thesis_health','')}; valuation {h.get('valuation','')}; "
        f"P&L {h.get('pnl_pct')}%, weight {h.get('weight_pct', '?')}% of portfolio"
        for h in summaries)
    prompt = _SYNTH_TMPL.format(
        today=date.today().isoformat(), holdings_block=block,
        risk_block=risk_block or "(computed risk metrics unavailable)",
        ballast_block=ballast_block or "(no ballast candidates provided)",
        learning_block=learning_block,
        min_ballast=config.get("policy.min_ballast_weight", 10))
    parsed, raw = _run_json(prompt)
    if not parsed:
        return {"summary": "(could not parse)", "ranked_actions": [], "raw": raw}
    parsed["raw"] = raw
    parsed.setdefault("ranked_actions", [])
    parsed.setdefault("ballast_suggestions", [])
    return parsed


def build_context(score_row: dict, signal_row: dict | None,
                  market_line: str = "") -> str:
    m = score_row.get("metrics", {}) or {}
    lines = []
    if market_line:
        lines.append(market_line)
    lines += [
        "Quant pillar scores (0-1, higher=better):",
        f"  - Technical: {score_row.get('technical')} (setup: {m.get('setup_label')}, "
        f"RSI {m.get('rsi')}, 3mo RS vs {m.get('benchmark')}: {m.get('rs_3m')}pp)",
        f"  - Fundamental: {score_row.get('fundamental')} "
        f"(PE={m.get('trailingPE')}, fwdPE={m.get('forwardPE')}, P/FCF={m.get('p_fcf')}, revGrowth={m.get('revenueGrowth')})",
        f"  - Sentiment: {score_row.get('sentiment')} (news tone: {m.get('sentiment_label')})",
        f"  - Composite: {score_row.get('blended')}; tags: {', '.join(score_row.get('tags', []))}",
        "Weigh all three pillars (technical setup, fundamentals/valuation, news sentiment) "
        "in your rating; flag when they disagree.",
    ]
    if signal_row:
        cats = signal_row.get("catalysts") or []
        lines.append(f"Recent catalysts (local news scan): {'; '.join(cats) or 'none'}")
        # Raw dated headlines — give Claude the actual news, not just a summary of it.
        # A tape-moving negative story (lawsuit, probe, downgrade) must reach the analyst
        # even if the local LLM's catalyst extraction glossed over it.
        heads = signal_row.get("headlines") or []
        if heads:
            lines.append("Recent headlines (dated, newest first) — investigate anything "
                         "material, especially adverse items:")
            lines.extend(f"  - {h}" for h in heads)
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    print("Auth status:", json.dumps(auth_status(), indent=2))
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        print("\nRunning a live research call for VRT (uses your subscription)...")
        out = research_one("VRT", "Vertiv", "datacenter_power",
                           "Quant blended 0.7; tags: deal, both; signal 8/10")
        print(json.dumps(out, indent=2)[:1500])
