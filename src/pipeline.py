"""Daily pipeline orchestration: chains the 4-stage funnel and persists a run.

Modes:
  deep  - full pipeline incl. Claude deep research on finalists (evening run).
  light - quant + local signals + IBKR refresh, NO Claude (pre-market run).
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

from . import (config, deep_research, fundamentals, health, ibkr, learning, local_llm,
               news, regime, scoring, storage, technical, triggers, universe)


def _log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def _gather_signals(shortlist: list[dict]) -> tuple[dict[str, dict], dict[str, list], bool]:
    """Run news + local-LLM classification for shortlist names.

    News I/O (RSS + SEC, network-bound) is fetched in parallel; the LLM classification
    is kept serial because it's GPU-bound (one model, one GPU).
    Returns (signals, news_map, llm_was_available) — the raw news items also feed
    trigger checking; the availability flag feeds the run's health record.
    """
    signals: dict[str, dict] = {}
    use_llm = local_llm.available()
    _log(f"local LLM available: {use_llm}")

    targets = [(r["symbol"], r.get("name", r["symbol"])) for r in shortlist]
    news_map: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(news.get_news, sym, name): sym for sym, name in targets}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                news_map[sym] = fut.result()
            except Exception:  # noqa: BLE001
                news_map[sym] = []

    n_err = 0
    for i, (sym, name) in enumerate(targets, 1):
        items = news_map.get(sym, [])
        sig = local_llm.classify_from_news(sym, name, items, check_available=use_llm)
        # Keep the raw dated headlines (in-run only; not persisted) so deep research can
        # hand Claude the actual news, not just the local LLM's summary of it.
        sig["headlines"] = [
            f"[{news._published_dt(it).strftime('%Y-%m-%d')}] {it.get('title','')}"
            for it in items[:10]]
        signals[sym] = sig
        if sig.get("error"):
            n_err += 1
        if i % 5 == 0 or i == len(targets):
            _log(f"  signals {i}/{len(targets)}")
    # A 100% error rate almost always means the model isn't pulled / Ollama is misconfigured;
    # surface it loudly instead of silently scoring every name with a neutral sentiment.
    if use_llm and targets and n_err == len(targets):
        _log("WARNING: every local-LLM classification failed — sentiment pillar is neutral "
             "for the whole board. Check that the model is pulled (`ollama list`).")
    return signals, news_map, use_llm


UNKNOWN_SECTOR = "Unknown"


def select_finalists(board: list[dict], n: int, per_sector_cap: int | None = None,
                     sectors: dict[str, str] | None = None) -> list[dict]:
    """Choose the `n` names Claude researches, at most `per_sector_cap` from any sector.

    `board` must already be ranked best-first (score_universe sorts it), so this is a
    single deterministic greedy pass: walk the ranking, take a name unless its sector
    is already full.

    The failure this prevents: on a 1,500-name universe the top 10 by composite land
    in whichever sector the composite currently favours — broad-250 was already 34%
    Information Technology — so the deep-research cohort becomes a sector bet, and
    `claude_vs_screen` degenerates into comparing one sector against the market rather
    than comparing an analyst against a screen. A cap makes the cohort a cross-section.

    The cap YIELDS rather than shrinking the cohort. If the capped pass cannot fill n
    (a single-sector universe — which is exactly what the curated ai_infra list is),
    the remaining slots go to the best names left over regardless of sector. A hard cap
    there would quietly cut the research budget from 10 names to 3, which is a
    different change from the one intended: with only one sector on the board there is
    no sector comparison left to degenerate.
    """
    if n <= 0 or not board:
        return []
    cap = int(per_sector_cap or 0)

    def _sec(row: dict) -> str:
        if sectors and row["symbol"] in sectors:
            return sectors[row["symbol"]] or UNKNOWN_SECTOR
        return row.get("sector") or UNKNOWN_SECTOR

    if cap <= 0:
        return board[:n]
    chosen: list[dict] = []
    taken: set[str] = set()
    counts: dict[str, int] = {}
    for r in board:
        if len(chosen) >= n:
            break
        s = _sec(r)
        if counts.get(s, 0) >= cap:
            continue
        chosen.append(r)
        taken.add(r["symbol"])
        counts[s] = counts.get(s, 0) + 1
    for r in board:                      # cap yields to the budget, see docstring
        if len(chosen) >= n:
            break
        if r["symbol"] not in taken:
            chosen.append(r)
            taken.add(r["symbol"])
    return chosen


def sector_mix(rows: list[dict]) -> str:
    """`Information Technology 3 · Health Care 2 · ...`, best-represented first.

    Goes in the run's health note because a finalist cohort that is 8/10 one sector is
    a fact about the RUN, not about any one name, and nothing else on the board says it.
    """
    counts: dict[str, int] = {}
    for r in rows:
        s = r.get("sector") or UNKNOWN_SECTOR
        counts[s] = counts.get(s, 0) + 1
    return " · ".join(f"{s} {c}" for s, c in
                      sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def run(mode: str = "deep") -> int:
    assert mode in ("deep", "light")
    storage.init_db()
    stale = storage.mark_stale_runs()  # clean up runs orphaned by a killed prior process
    if stale:
        _log(f"marked {stale} stale 'running' run(s) as error")
    swept = config.sweep_cache(max_age_days=7)
    if swept:
        _log(f"cache sweep: removed {swept} stale files")
    run_id = storage.create_run(mode)
    # Health is recorded per stage as we go; status=ok alone proved worthless
    # (5 weeks of 'ok' runs with no portfolio in them — see src/health.py).
    hz = health.RunHealth()
    _log(f"run #{run_id} mode={mode} starting")
    try:
        from .data.history_store import HistoryStore
        store = HistoryStore()
        # Stage 0: universe. In broad mode the SEC registrant table is thousands of
        # CANDIDATES and the point-in-time floor (price x SEC shares, dollar volume,
        # all from the local store) is what cuts it to the scorable set — BEFORE any
        # per-name network call. Doing it the other way round costs 5,000 sequential
        # `.info` requests to discard 95% of them.
        src = universe.configured_source()
        candidates = universe.build_universe()
        broad = src == "broad"
        storage.set_progress(run_id, f"stage 0/4: universe ({src})")
        uni = (universe.investable(date.today(), store=store, log=_log)
               if broad else candidates)
        by_symbol = {r["symbol"]: r for r in uni}
        subs = {r["symbol"]: r["subsector"] for r in uni}
        # The GICS-like sector is the VALUATION peer set (scoring ranks multiples
        # inside it, never across it). Curated names are SEC registrants too, so the
        # same table serves both sources; anything unmapped stays "Unknown", which
        # scoring treats as a sector of its own rather than spreading it around.
        secs = universe.sector_map(uni, store=store)
        syms = [r["symbol"] for r in uni]
        _log(f"universe ({src}): {len(syms)} tickers"
             + (f" of {len(candidates)} candidates" if broad else ""))
        if not syms:
            hz.failed("universe", f"0/{len(candidates)} candidates passed the "
                                  f"point-in-time floor")
        elif broad:
            hz.ok("universe", f"{len(syms)}/{len(candidates)} passed floor")
        else:
            hz.ok("universe", f"{len(syms)} curated tickers")
        # Record today's membership so future backtests know what was in the universe
        # on this date instead of reconstructing it with hindsight (survivorship guard).
        try:
            store.record_universe(date.today(), uni, source=src)
        except Exception as exc:  # noqa: BLE001
            _log(f"universe snapshot not recorded: {exc}")

        # Stage 1: fundamentals + floor, served from the point-in-time SEC store.
        # There is no per-name `.info` call here any more: that was 5,000 sequential
        # throttled requests to score a broad universe, and it made the live screen and
        # the backtest consume two different data sources. `enrich()` below adds the
        # two things SEC filings genuinely lack, for ~50 names instead of all of them.
        storage.set_progress(run_id, f"stage 1/4: fundamentals for {len(syms)} names")
        metrics_all = fundamentals.from_store(store, syms, date.today())
        metrics = {s: m for s, m in metrics_all.items() if fundamentals.passes_floor(m)}
        _log(f"passed liquidity floor: {len(metrics)}")
        n_snap = sum(1 for m in metrics_all.values() if not m.get("error"))
        cov = n_snap / len(syms) if syms else 0.0
        if not metrics:
            hz.failed("fundamentals", f"0/{len(syms)} names passed the floor "
                                      f"(SEC snapshot coverage {cov:.0%})")
        elif cov < 0.7:
            # Yahoo's mood is no longer the failure mode; a thin SEC store is. Coverage
            # of the floored universe is what says whether the pillar is real.
            hz.degraded("fundamentals", f"SEC snapshots cover {cov:.0%} of the "
                                        f"{len(syms)} names (<70%) — run "
                                        f"`scripts.data_refresh --facts-bulk --snapshots`")
        else:
            hz.ok("fundamentals", f"{len(metrics)}/{len(syms)} passed floor; "
                                  f"SEC snapshot coverage {cov:.0%}")
        storage.upsert_tickers([
            {"symbol": s, "name": m.get("name", s), "subsector": subs.get(s, ""),
             "sector": by_symbol.get(s, {}).get("sector") or secs.get(s),
             "industry_group": by_symbol.get(s, {}).get("industry_group")}
            for s, m in metrics.items()
        ])

        # Technical pillar for all passed-floor names (batch price-history download).
        # Light mode recomputes fresh (use_cache=False): its whole purpose is a price/
        # momentum refresh, and the deep run already populated today's technical cache,
        # so a cached light run would show identical prices/RSI to the morning.
        storage.set_progress(run_id, f"stage 1/4: technicals for {len(metrics)} names")
        techs = technical.compute_many(list(metrics.keys()), use_cache=(mode != "light"))
        _log(f"technicals computed: {len(techs)}")
        n_stub = sum(1 for t in techs.values() if t.get("insufficient_data"))
        if techs and n_stub == len(techs):
            hz.failed("technicals", f"all {n_stub} names fell back to neutral "
                                    f"(price download failed)")
        elif n_stub > 0.1 * max(len(techs), 1):
            hz.degraded("technicals", f"{n_stub}/{len(techs)} names neutral-stubbed "
                                      f"(download timeouts)")
        else:
            hz.ok("technicals", f"{len(techs) - n_stub}/{len(techs)} computed")

        # Market regime (uses the technicals just computed — no extra downloads for breadth)
        mkt = regime.detect(techs, subs)
        mkt_line = regime.context_line(mkt)
        if mkt:
            _log(f"regime: {mkt.get('regime')} | breadth {mkt.get('breadth_pct')}% "
                 f"| weak: {mkt.get('weak_subsectors') or 'none'}")
            hz.ok("regime", str(mkt.get("regime")))
        else:
            hz.degraded("regime", "benchmark download failed; Claude gets no regime context "
                                  "and sizing caps stay untightened")

        # Learned ranking pillar (src/ml): computed when a trained model exists, weighted
        # only when config says so AND the model passed its kill criteria.
        ml_scores: dict[str, float] = {}
        w_learned = scoring.learned_weight()
        if config.get("scoring.learned.enabled", True):
            try:
                from .ml import infer as ml_infer
                ok, why = ml_infer.usable()
                if ok or (w_learned == 0 and ml_infer.latest_meta()):
                    ml_scores, ml_info = ml_infer.score_today(list(metrics.keys()), subs)
                    _log(f"learned pillar: {len(ml_scores)} scores from model {ml_info.get('model_date')} "
                         f"(OOS IC {ml_info.get('oos_ic')}, verdict pass={ml_info.get('verdict_pass')}, "
                         f"weight {w_learned})")
                    hz.ok("learned", f"{len(ml_scores)} scored, weight {w_learned}")
                elif w_learned > 0:
                    hz.degraded("learned", f"weight {w_learned} configured but model unusable: {why} "
                                           f"— pillar neutral 0.5 for every name")
            except Exception as exc:  # noqa: BLE001
                _log(f"learned pillar unavailable: {exc}")
                if w_learned > 0:
                    hz.degraded("learned", f"inference failed ({str(exc)[:80]}) — pillar neutral")

        # Initial scoring (technical + fundamental; neutral sentiment) -> shortlist
        scored0 = scoring.score_universe(metrics, subs, technicals=techs,
                                         ml_scores=ml_scores, sectors=secs)
        scored0_by_sym = {r["symbol"]: r for r in scored0}
        shortlist_n = int(config.get("screen.shortlist_size", 35))
        shortlist = scored0[:shortlist_n]
        shortlist_syms = {r["symbol"] for r in shortlist}
        _log(f"shortlist: {len(shortlist)}")

        # Fetch holdings early so the news/sentiment scan also covers names you own.
        hold_early = ibkr.fetch_holdings()
        held_syms = [h["symbol"] for h in hold_early.get("holdings", [])]
        held_in_uni = [s for s in held_syms if s in metrics and s not in shortlist_syms]
        sentiment_targets = shortlist + [scored0_by_sym[s] for s in held_in_uni
                                         if s in scored0_by_sym]
        _log(f"IBKR holdings (early): connected={hold_early['connected']}, "
             f"held in-universe off-shortlist: {held_in_uni}")

        # The ONLY remaining `.info` calls: forward P/E, the analyst PEG, the exact
        # next-earnings timestamp and the display name — none of which exist in SEC
        # filings — for the shortlist plus what you hold. Bounded by shortlist_size,
        # so this stage costs the same whether the universe is 105 names or 1,500.
        enrich_syms = [r["symbol"] for r in sentiment_targets]
        storage.set_progress(run_id, f"stage 1/4: .info enrichment for {len(enrich_syms)} names")
        enriched = fundamentals.enrich(enrich_syms)
        n_enr_err = sum(1 for e in enriched.values() if e.get("error"))
        n_fpe = sum(1 for e in enriched.values() if e.get("forwardPE") is not None)
        _log(f"enriched {len(enriched) - n_enr_err}/{len(enriched)} names "
             f"({n_fpe} with a forward P/E)")
        for sym, extra in enriched.items():
            if sym in metrics:
                metrics[sym].update({k: v for k, v in extra.items() if k != "error"})
        if enrich_syms and n_enr_err > 0.2 * len(enrich_syms):
            hz.degraded("fundamentals", f"{n_enr_err}/{len(enrich_syms)} `.info` "
                                        f"enrichment calls failed (Yahoo throttling?) — "
                                        f"forward P/E missing for those names")

        # Stage 2: local-LLM news sentiment + signals for shortlist + held names
        storage.set_progress(run_id, f"stage 2/4: news + local-LLM signals for {len(sentiment_targets)} names")
        signals, news_map, llm_up = _gather_signals(sentiment_targets)
        n_sig_err = sum(1 for s in signals.values() if s.get("error"))
        if not llm_up:
            hz.degraded("sentiment", "Ollama unavailable — sentiment pillar neutral (0.5) "
                                     "for every name")
        elif signals and n_sig_err == len(signals):
            hz.degraded("sentiment", f"all {n_sig_err} local-LLM classifications failed "
                                     f"(model not pulled?) — pillar neutral")
        elif n_sig_err > 0.2 * max(len(signals), 1):
            hz.degraded("sentiment", f"{n_sig_err}/{len(signals)} classifications failed")
        else:
            hz.ok("sentiment", f"{len(signals) - n_sig_err}/{len(signals)} classified")

        # Re-score with all pillars -> final board
        board = scoring.score_universe(metrics, subs, signals=signals, technicals=techs,
                                       ml_scores=ml_scores, sectors=secs)

        # Close the funnel leak: a name that was NOT in the shortlist can still rank into
        # the finalist slots on technical+fundamental alone (it got a neutral sentiment).
        # Scan any such top names now and re-score once, so every finalist Claude sees was
        # actually news-scanned.
        if mode == "deep":
            finalist_n = min(int(config.get("screen.finalist_size", 10)),
                             int(config.get("claude.max_finalists", 10)))
            sector_cap = int(config.get("screen.max_finalists_per_sector", 0) or 0)
            # SAME selection as the final cut below. If this used the raw top-N while
            # the cut below stratified, the rescan would news-scan names that never
            # become finalists and miss the ones that do — reopening the exact funnel
            # leak it exists to close.
            unscanned = [r for r in select_finalists(board, finalist_n, sector_cap)
                         if r["symbol"] not in signals]
            if unscanned:
                _log(f"scanning {len(unscanned)} unscanned finalists: "
                     f"{[r['symbol'] for r in unscanned]}")
                extra_sigs, extra_news, _ = _gather_signals(unscanned)
                signals.update(extra_sigs)
                news_map.update(extra_news)
                board = scoring.score_universe(metrics, subs, signals=signals, technicals=techs,
                                               ml_scores=ml_scores, sectors=secs)

        storage.upsert_screen_scores(run_id, board)
        storage.upsert_signals(run_id, [
            {"symbol": s, **v} for s, v in signals.items()
        ])

        # The finalist cohort. Computed in BOTH modes even though only a deep run
        # researches it: the cut is what `claude_vs_screen` treats as the control
        # cohort, so its sector mix is a property of the run worth recording every
        # time, and a light run is the cheap way to see whether the cap is binding.
        finalist_n = min(int(config.get("screen.finalist_size", 10)),
                         int(config.get("claude.max_finalists", 10)))
        sector_cap = int(config.get("screen.max_finalists_per_sector", 0) or 0)
        top_finalists = select_finalists(board, finalist_n, sector_cap)
        fmix = sector_mix(top_finalists)
        _log(f"finalists ({len(top_finalists)}, cap {sector_cap or 'none'}/sector): "
             f"{[r['symbol'] for r in top_finalists]} — {fmix}")
        hz.ok("finalists", f"{len(top_finalists)} chosen, cap "
                           f"{sector_cap or 'none'}/sector — {fmix}")

        # Stage 4 (data overlay): IBKR holdings + quotes for the visible board
        board_syms = [r["symbol"] for r in board[:max(shortlist_n, 40)]]
        ib = ibkr.fetch_portfolio_and_quotes(board_syms)
        _log(f"IBKR connected={ib['connected']} ({ib['note']})")
        if ib["holdings"]:
            storage.upsert_holdings(ib["holdings"])
        if ib["quotes"]:
            storage.upsert_quotes(ib["quotes"])
        fresh: dict[str, float] = {}
        if not ib["connected"]:
            # Gateway down: refresh quotes via yfinance so the dashboard's prices/P&L
            # don't silently freeze at the last time IBKR was up (days, sometimes).
            fallback_syms = board_syms + [h["symbol"] for h in hold_early.get("holdings", [])]
            fresh = technical.yf_last_prices(list(dict.fromkeys(fallback_syms)))
            if fresh:
                storage.upsert_quotes([{"symbol": s, "last": p} for s, p in fresh.items()])
                _log(f"quotes refreshed via yfinance fallback for {len(fresh)} symbols")
        # The exact failure that hid the portfolio for 5 weeks: connected=False was a
        # single log line under a status=ok run. Now it is the run's health.
        if ib["connected"]:
            hz.ok("broker", f"IBKR connected; {len(ib['holdings'])} holdings, "
                            f"{len(ib['quotes'])} quotes")
        else:
            hz.degraded("broker", f"IBKR offline ({ib['note']}) — holdings from last "
                                  f"snapshot, {len(fresh)} quotes via yfinance fallback; "
                                  f"held names NOT researched")

        # Thesis supervision: evaluate active invalidation triggers against today's
        # prices + headlines. A broken thesis gets flagged the day it breaks.
        trig_syms = {t["symbol"] for t in storage.active_triggers()}
        price_map = {q["symbol"]: q["last"] for q in ib.get("quotes", []) if q.get("last")}
        price_map.update(fresh)  # gateway-down fallback quotes must count too
        for r in board:  # fill gaps with screen prices
            px = (r.get("metrics") or {}).get("price")
            if px and r["symbol"] not in price_map:
                price_map[r["symbol"]] = px
        # Off-universe/off-board names with armed triggers (e.g. held ballast ETFs)
        # would otherwise be invisible to both price AND news checks.
        missing_px = [s for s in trig_syms if s not in price_map]
        if missing_px:
            price_map.update(technical.yf_last_prices(missing_px))
        missing_news = [s for s in trig_syms if s not in news_map]
        if missing_news:
            for s in missing_news:
                try:
                    news_map[s] = news.get_news(s)
                except Exception:  # noqa: BLE001
                    news_map[s] = []
            _log(f"fetched news for {len(missing_news)} trigger-only symbols")
        fired = triggers.check_all(price_map, news_map)
        if fired:
            _log("!! TRIGGERS FIRED: " + "; ".join(
                f"{t['symbol']}: {t['trigger_detail']}" for t in fired))
        else:
            _log(f"triggers checked across {len(trig_syms)} symbols, none fired")
        unpriced = [s for s in trig_syms if s not in price_map]
        if unpriced:
            hz.degraded("triggers", f"no price for {len(unpriced)} triggered-symbol(s) "
                                    f"{unpriced[:5]} — price triggers unchecked")
        else:
            hz.ok("triggers", f"{len(trig_syms)} symbols checked, {len(fired)} fired")

        # Stage 3: Claude deep research on finalists (deep mode only)
        if mode == "deep":
            holdings_map = {h["symbol"]: h for h in ib.get("holdings", [])}
            finalists = list(top_finalists)
            fsyms = {r["symbol"] for r in finalists}
            # Always research held names that are in-universe, even if below the cutoff,
            # so every position you own gets a fresh Buy/Hold/Sell rating.
            held_extra = [r for r in board
                          if r["symbol"] in holdings_map and r["symbol"] not in fsyms]
            finalists += held_extra
            auth = deep_research.auth_status()
            _log(f"Claude auth: {auth}")
            _log(f"deep research on {len(finalists)} names "
                 f"({len(top_finalists)} top + {len(held_extra)} held); "
                 f"finalist sectors: {fmix}")
            if not auth.get("claude_on_path") or not auth.get("oauth_token_set"):
                hz.failed("research", "claude CLI missing or CLAUDE_CODE_OAUTH_TOKEN unset")
            # Overall wall-clock budget so a run of slow/timing-out finalists can't blow
            # past the Task Scheduler limit and leave the run stuck 'running'. 0 = no cap.
            budget = float(config.get("claude.deep_run_budget_seconds", 5400))
            t0 = time.monotonic()
            done = 0
            n_unrated = 0
            budget_hit = False
            per_call_timeout = float(config.get("claude.timeout_seconds", 600))
            claude_horizon: dict[str, str] = {}   # symbol -> the clock Claude named
            for i, row in enumerate(finalists, 1):
                remaining = budget - (time.monotonic() - t0) if budget else None
                # Stop while a full call (+ possible red team) still fits — otherwise
                # the Task Scheduler's hard 2h limit tree-kills us mid-research.
                if remaining is not None and remaining < per_call_timeout * 1.5:
                    _log(f"deep-research budget nearly exhausted ({remaining:.0f}s left); "
                         f"skipping {len(finalists) - i + 1} remaining names")
                    budget_hit = True
                    break
                sym = row["symbol"]
                storage.set_progress(run_id, f"stage 3/4: Claude research {i}/{len(finalists)}: {sym}")
                ctx = deep_research.build_context(row, signals.get(sym), market_line=mkt_line)
                # Outcome-feedback: the analyst sees its own track record on this name,
                # its global calibration, and lessons distilled from past misses.
                lb = learning.learning_block(sym)
                if lb:
                    ctx = lb + "\n\n" + ctx
                h = holdings_map.get(sym)
                holding = ({"shares": h.get("position"), "avg_cost": h.get("avg_cost")}
                           if h else None)
                report = deep_research.research_one(
                    sym, row.get("name", sym), row.get("subsector", ""), ctx, holding=holding)
                # The deterministic clock, recorded ALONGSIDE Claude's own choice (it is
                # never shown to Claude — see deep_research.research_one, so agreement
                # between the two stays a real measurement). It is also the fallback when
                # the report named no valid horizon, so every pick has a clock to be
                # graded on and horizon_source always says who chose it.
                report["quant_horizon"] = technical.default_horizon(
                    {**(row.get("metrics") or {}), "tags": row.get("tags")})
                report["horizon_source"] = "claude" if report.get("horizon") else "quant"
                if report.get("horizon"):
                    claude_horizon[sym] = report["horizon"]
                # Devil's-advocate pass on buy-rated names: the funnel pre-selects
                # momentum winners and the context primes Claude bullishly, so someone
                # must be paid to disagree. Downgrades/haircuts are applied + recorded.
                if (config.get("claude.red_team", True)
                        and str(report.get("rating", "")).lower() in ("buy", "strong_buy")):
                    rt = deep_research.red_team(sym, row.get("name", sym), report)
                    report = deep_research.apply_red_team(report, rt)
                    if rt:
                        _log(f"    red team: survives={rt.get('survives')} "
                             f"haircut={rt.get('confidence_haircut')} -> "
                             f"{report.get('rating')} conv={report.get('conviction')}")
                storage.upsert_research(run_id, report)
                done += 1
                if report.get("rating") is None:
                    n_unrated += 1
                _log(f"  research {i}/{len(finalists)}: {sym} "
                     f"rating={report.get('rating')} conviction={report.get('conviction')}")
                time.sleep(1)
            if finalists and done == 0:
                hz.failed("research", f"0/{len(finalists)} names researched")
            elif budget_hit:
                hz.degraded("research", f"wall-clock budget exhausted: {done}/{len(finalists)} "
                                        f"researched")
            elif n_unrated:
                hz.degraded("research", f"{n_unrated}/{done} research calls returned no "
                                        f"rating (timeouts/parse failures)")
            else:
                hz.ok("research", f"{done}/{len(finalists)} researched")
            # Snapshot ranked opportunity prices for forward tracking, PLUS the benchmark
            # price at the same instant so future excess-return vs SPY is computable.
            # The clock each pick will be graded on: Claude's when it named one, the
            # quant default otherwise (names the budget skipped, reports that returned no
            # valid horizon). Never invented for rows already in the table.
            picks = []
            for r in top_finalists:
                own = claude_horizon.get(r["symbol"])
                picks.append({
                    "symbol": r["symbol"],
                    "price": (r.get("metrics") or {}).get("price"),
                    "horizon": own or technical.default_horizon(
                        {**(r.get("metrics") or {}), "tags": r.get("tags")}),
                    "horizon_source": "claude" if own else "quant"})
            bench = config.get("technical.benchmark", "SPY")
            bench_price = technical.benchmark_last_price(bench)
            if bench_price is not None:
                picks.append({"symbol": bench, "price": bench_price})
            storage.record_picks(run_id, picks)
            _log(f"researched {done}/{len(finalists)} names; snapshotted {len(picks)} picks")

        storage.set_progress(run_id, "stage 4/4: finalizing")
        storage.finish_run(run_id, "ok",
                           f"{mode}: {len(board)} scored, signals={len(signals)}",
                           health=hz, regime=(mkt or {}).get("regime"))
        _log(f"run #{run_id} done ({len(board)} board rows) — {hz.summary()}")
        return run_id
    except Exception as exc:  # noqa: BLE001
        hz.failed("pipeline", str(exc)[:200])
        storage.finish_run(run_id, "error", str(exc)[:300], health=hz)
        _log(f"run #{run_id} FAILED: {exc}")
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description="Claude Trader daily pipeline")
    ap.add_argument("--mode", choices=["deep", "light"], default="deep")
    args = ap.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
