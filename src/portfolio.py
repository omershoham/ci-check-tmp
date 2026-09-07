"""On-demand portfolio deep-dive: richer per-holding research + recency
validation + a portfolio-level synthesis. Triggered from the dashboard button
(or `python -m scripts.portfolio_deepdive`); history is retained for auditing.
"""
from __future__ import annotations

from . import (config, deep_research, fundamentals, health, ibkr, learning, risk,
               storage, technical, triggers, universe)


def _log(msg: str) -> None:
    print(f"[portfolio] {msg}", flush=True)


def run() -> int:
    storage.init_db()
    rid = storage.create_portfolio_run()
    hz = health.RunHealth()
    _log(f"portfolio deep-dive #{rid} starting")
    try:
        ibd = ibkr.fetch_portfolio_and_quotes([])  # holdings + quotes for held names
        holdings = ibd.get("holdings", [])
        qmap = {q["symbol"]: q["last"] for q in ibd.get("quotes", [])}
        if holdings:
            hz.ok("broker", f"IBKR connected; {len(holdings)} holdings")
        else:
            # IBKR offline → fall back to the most recently fetched holdings/quotes in the DB
            # (refreshed by the daily board run), so the deep-dive still works when Gateway is down.
            with storage.get_conn() as cn:
                holdings = [{"symbol": r["symbol"], "position": r["position"], "avg_cost": r["avg_cost"]}
                            for r in cn.execute("SELECT symbol,position,avg_cost FROM holdings")]
                qmap = {r["symbol"]: r["last"] for r in cn.execute("SELECT symbol,last FROM quotes")}
            _log(f"IBKR offline ({ibd.get('note')}); using {len(holdings)} stored holdings")
            hz.degraded("broker", f"IBKR offline ({ibd.get('note')}); analyzed the "
                                  f"{len(holdings)} holdings from the last stored snapshot — "
                                  f"positions may have changed since")
        if not holdings:
            hz.failed("broker", "no holdings: IBKR offline and none stored")
            storage.finish_portfolio_run(rid, "error", "no holdings (IBKR offline and none stored)",
                                         health=hz)
            return rid
        # Stored quotes can be days old when the Gateway has been down — refresh via
        # yfinance so P&L context fed to Claude reflects today's prices, not stale ones.
        held_syms = [h["symbol"] for h in holdings]
        fresh = technical.yf_last_prices(held_syms)
        if fresh:
            qmap.update(fresh)
            storage.upsert_quotes([{"symbol": s, "last": p} for s, p in fresh.items()])
            _log(f"quotes refreshed via yfinance for {len(fresh)}/{len(held_syms)} holdings")
        subs = universe.subsector_map()

        # Portfolio weights FIRST — every downstream stage reasons about sizing.
        # (The old flow never computed weights, so a conviction-2 name at 22% of the
        # portfolio was analyzed exactly like an 8% one.)
        mv = {h["symbol"]: (qmap.get(h["symbol"]) or 0) * (h.get("position") or 0)
              for h in holdings}
        total_mv = sum(mv.values()) or 1.0
        weights = {s: round(v / total_mv * 100, 1) for s, v in mv.items()}
        _log(f"weights: {dict(sorted(weights.items(), key=lambda kv: -kv[1]))}")

        # Computed risk metrics FIRST (they only need weights): vol target, drawdown and
        # each name's marginal correlation feed every per-holding prompt and the sizing
        # rules, not just the end-of-run synthesis.
        _log("computing portfolio risk metrics")
        storage.set_progress(rid, "computing risk metrics", table="portfolio_runs")
        rm = risk.portfolio_risk(weights)
        risk_block = risk.risk_context_lines(rm)
        if rm:
            _log(f"risk: eff_bets={rm.get('effective_bets')} avg_corr={rm.get('avg_pairwise_corr')} "
                 f"vol={rm.get('ann_vol_pct')}% scale={rm.get('vol_scale')} "
                 f"dd={rm.get('drawdown_from_peak_pct')}%")
            hz.ok("risk", f"{rm.get('n_holdings')} names, {rm.get('effective_bets')} eff. bets, "
                          f"vol {rm.get('ann_vol_pct')}%")
        else:
            hz.degraded("risk", "price history unavailable — no computed risk metrics in the "
                                "prompts; sizing rules fall back to regime only")
        mcorr = (rm or {}).get("marginal_corr") or {}

        def _corr_note(sym: str) -> str:
            c = mcorr.get(sym)
            if c is None:
                return ""
            tag = "adds mostly the SAME risk" if c >= 0.7 else ("moderate overlap" if c >= 0.4 else "real diversifier")
            return f"; avg corr to rest of book {c:.2f} ({tag})"

        portfolio_block = "\n".join(
            f"  - {h['symbol']}: {weights.get(h['symbol'], 0)}% of portfolio "
            f"(P&L {((qmap.get(h['symbol'], 0) - h['avg_cost']) / h['avg_cost'] * 100) if (qmap.get(h['symbol']) and h.get('avg_cost')) else 0:+.0f}%"
            f"{_corr_note(h['symbol'])})"
            for h in sorted(holdings, key=lambda x: -weights.get(x["symbol"], 0)))

        summaries = []
        positions = []
        for i, h in enumerate(holdings, 1):
            sym = h["symbol"]
            last = qmap.get(sym)
            cost = h.get("avg_cost")
            pnl = ((last - cost) / cost * 100) if (last and cost) else None
            try:
                name = fundamentals.fetch_one(sym).get("name", sym)
            except Exception:  # noqa: BLE001
                name = sym
            hctx = {"shares": h.get("position"), "avg_cost": cost, "last": last,
                    "pnl_pct": round(pnl, 1) if pnl is not None else None,
                    "weight_pct": weights.get(sym)}

            _log(f"  deep dive {i}/{len(holdings)}: {sym} ({weights.get(sym)}% of pf)")
            storage.set_progress(rid, f"holding {i}/{len(holdings)}: {sym} — researching",
                                 table="portfolio_runs")
            report = deep_research.deep_dive_holding(
                sym, name, subs.get(sym, "Other"), hctx,
                portfolio_block=portfolio_block,
                history_block=learning.learning_block(sym),
                risk_ctx=rm)
            _log(f"    validating recency: {sym}")
            storage.set_progress(rid, f"holding {i}/{len(holdings)}: {sym} — validating recency",
                                 table="portfolio_runs")
            validation = deep_research.validate_recency(sym, report)

            pos = {**report, **hctx, "symbol": sym, "validation": validation}
            storage.upsert_portfolio_position(rid, pos)
            positions.append(pos)
            summaries.append({"symbol": sym, "action": report.get("action"),
                              "conviction": report.get("conviction"),
                              "thesis_health": report.get("thesis_health"),
                              "valuation": report.get("valuation"),
                              "pnl_pct": hctx["pnl_pct"],
                              "weight_pct": weights.get(sym)})

        n_noact = sum(1 for p in positions if p.get("action") is None)
        n_noval = sum(1 for p in positions if (p.get("validation") or {}).get("is_current") is None)
        if positions and n_noact == len(positions):
            hz.failed("research", f"0/{len(positions)} deep dives returned an action")
        elif n_noact:
            hz.degraded("research", f"{n_noact}/{len(positions)} deep dives returned no action "
                                    f"(timeouts/parse failures)")
        elif n_noval:
            hz.degraded("research", f"{n_noval}/{len(positions)} recency validations unavailable")
        else:
            hz.ok("research", f"{len(positions)} holdings dived + validated")

        _log(f"risk: eff_bets={rm.get('effective_bets')} avg_corr={rm.get('avg_pairwise_corr')} "
             f"beta_qqq={rm.get('beta_qqq')} smh_-20%={rm.get('scenario_drawdown_pct')}%")

        # Ballast candidates (non-AI diversifiers) with simple momentum context, so the
        # synthesis can recommend something concrete instead of "you need ballast".
        ballast_block = ""
        blist = config.get("policy.ballast_watchlist", []) or []
        if blist:
            bq = technical.yf_last_prices(list(blist))
            ballast_block = ("Ballast candidates available (non-AI diversifiers, latest price):\n"
                             + "\n".join(f"  - {s}: ${bq[s]:.2f}" for s in blist if s in bq))

        # Portfolio-level learning context (action batting average + calibration + lessons)
        pf_learn = learning.portfolio_learning_block()

        _log("portfolio-level synthesis")
        storage.set_progress(rid, "portfolio-level synthesis", table="portfolio_runs")
        synth = deep_research.portfolio_synthesis(summaries, risk_block=risk_block,
                                                  ballast_block=ballast_block,
                                                  learning_block=pf_learn)
        synth["risk_metrics"] = rm
        storage.upsert_portfolio_summary(rid, synth)
        if synth.get("summary") == "(could not parse)":
            hz.degraded("synthesis", "portfolio synthesis call failed to parse")

        # Cross-holding calibration: a dedicated ranking call assigns DIFFERENTIATED
        # convictions across all holdings (per-call scores cluster at 6-7). Write back.
        conv_map = deep_research.rank_convictions(summaries)
        _log(f"calibrated convictions: {conv_map}")
        if not conv_map:
            hz.degraded("synthesis", "conviction-ranking call failed — per-dive convictions "
                                     "used uncalibrated")
        for pos, summ in zip(positions, summaries):
            c = conv_map.get(pos["symbol"])
            if isinstance(c, (int, float)):
                pos["conviction"] = c
                summ["conviction"] = c

        # FINAL PM pass: actions were formed BEFORE calibration, so reconcile them with
        # the calibrated convictions + weights (this is what turns 'hold, conviction 2,
        # 22% weight' into the trim it should have been).
        # Previous dive's decisions -> turnover discipline (don't flip-flop daily).
        prev_run = storage.latest_portfolio_run("ok")
        prior_block = "(no previous deep-dive)"
        if prev_run and prev_run["id"] != rid:
            with storage.get_conn() as cn:
                prev_rows = cn.execute(
                    """SELECT symbol, action, target_weight_pct FROM portfolio_positions
                       WHERE run_id=? AND action IS NOT NULL""", (prev_run["id"],)).fetchall()
            if prev_rows:
                when = str(prev_run["finished_at"])[:10]
                prior_block = f"(dive of {when})\n" + "\n".join(
                    f"  - {x['symbol']}: {x['action']} -> target {x['target_weight_pct']}%"
                    for x in prev_rows)

        _log("reconciling actions with calibrated convictions + weights")
        storage.set_progress(rid, "PM reconcile: final actions + targets", table="portfolio_runs")
        final = deep_research.reconcile_actions(summaries, learning_block=pf_learn,
                                                prior_block=prior_block, risk_ctx=rm)
        if not final:
            hz.degraded("synthesis", "PM reconcile call failed — analyst actions kept, only "
                                     "the deterministic policy backstop applied")
        else:
            hz.ok("synthesis", f"reconciled {len(final)} actions")
        for pos in positions:
            sym = pos["symbol"]
            f = final.get(sym)
            if f:
                if f["action"] != (pos.get("action") or ""):
                    pos["action_note"] = (f"PM override: {f['why']} "
                                          f"(analyst said {pos.get('action')})")
                elif f.get("why"):
                    pos["action_note"] = f["why"]
                pos["action"] = f["action"]
                pos["target_weight_pct"] = f.get("target_weight_pct")
                if f.get("execution_plan"):
                    tag = "⏱ NOW" if f.get("execution") == "immediate" else "📆 STAGED"
                    pos["action_note"] = ((pos.get("action_note") or "") +
                                          f" | {tag}: {f['execution_plan']}").strip(" |")
            # Deterministic backstop — even if the reconcile call failed or missed a name,
            # a low-conviction oversized position must not surface as hold/add.
            # Caps come from config policy.conviction_weight_caps.
            w = pos.get("weight_pct") or 0
            conv = pos.get("conviction")
            caps = config.get("policy.conviction_weight_caps", {3: 10, 5: 15}) or {}
            cap = None
            if isinstance(conv, (int, float)):
                for conv_cap in sorted(caps):
                    if conv <= conv_cap:
                        cap = float(caps[conv_cap])
                        break
            if (cap is not None and w > cap * 1.5  # only intervene when clearly oversized
                    and pos.get("action") in ("hold", "add", None)):
                pos["action"] = "trim"
                pos["action_note"] = ((pos.get("action_note") or "") +
                                      f" [policy rule: conviction {conv} at {w}% weight "
                                      f"caps at {cap:g}%]").strip()
                # min(), not setdefault: a reconcile-provided target above the cap
                # must be lowered too, not just left in place.
                tw = pos.get("target_weight_pct")
                pos["target_weight_pct"] = min(tw, cap) if isinstance(tw, (int, float)) else cap
            storage.upsert_portfolio_position(rid, pos)
        _log("final actions: " + ", ".join(
            f"{p['symbol']}={p.get('action')}@{p.get('target_weight_pct')}%" for p in positions))

        # Distill each thesis's invalidation into machine-checkable triggers (one batched
        # Claude call) so the daily runs can flag a broken thesis the day it breaks.
        _log("extracting invalidation triggers")
        storage.set_progress(rid, "extracting sell-signal triggers", table="portfolio_runs")
        n_trig = triggers.extract_and_store(rid, positions, source="portfolio")
        _log(f"installed {n_trig} active triggers")
        if n_trig == 0 and any((p.get("invalidation") or "").strip() for p in positions):
            hz.degraded("triggers", "invalidation text present but 0 triggers extracted — "
                                    "thesis breaks will not be auto-detected")
        else:
            hz.ok("triggers", f"{n_trig} active triggers installed")

        storage.finish_portfolio_run(rid, "ok", f"{len(holdings)} holdings analyzed", health=hz)
        _log(f"portfolio deep-dive #{rid} done — {hz.summary()}")
        return rid
    except Exception as exc:  # noqa: BLE001
        hz.failed("pipeline", str(exc)[:200])
        storage.finish_portfolio_run(rid, "error", str(exc)[:300], health=hz)
        _log(f"FAILED: {exc}")
        raise


if __name__ == "__main__":
    run()
