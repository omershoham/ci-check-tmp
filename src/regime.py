"""Market regime detection: static pillar weights are wrong across regimes, and Claude
should know the tape it's rating into.

Signals (all free, from data already downloaded or one small batch):
  - benchmark (SPY) vs its 200-day MA  -> primary risk-on / risk-off switch
  - universe breadth: % of screened names above their 50-day MA -> stress detector
  - per-sub-sector breadth -> catches a targeted rout (e.g. memory names cracking
    while the index is flat, like the July 2026 DRAM-lawsuit selloff)
  - benchmark 20-day realized volatility

Output feeds: pipeline log, Claude research context (deep_research.build_context),
and the dashboard header chip. Cached per day next to the technical cache.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

import numpy as np
import yfinance as yf

from . import config


def _cache_file():
    return config.cache_dir() / f"regime_{date.today().isoformat()}.json"


def _state_file():
    # Rolling (not date-keyed) previous-regime state, for hysteresis across days.
    return config.cache_dir() / "regime_state.json"


def _prev_regime() -> str | None:
    try:
        return json.loads(_state_file().read_text(encoding="utf-8")).get("regime")
    except (OSError, json.JSONDecodeError):
        return None


def _save_state(regime_label: str) -> None:
    try:
        _state_file().write_text(json.dumps({"regime": regime_label,
                                             "as_of": date.today().isoformat()}),
                                 encoding="utf-8")
    except OSError:
        pass


def detect(technicals: dict[str, dict] | None = None,
           subsectors: dict[str, str] | None = None,
           use_cache: bool = True) -> dict[str, Any]:
    """Compute the market regime. `technicals` (from technical.compute_many) supplies
    breadth for free; without it, only the benchmark-level signals are computed."""
    if not config.get("regime.enabled", True):
        return {}
    cf = _cache_file()
    if use_cache and technicals is None and cf.exists():
        try:
            cached = json.loads(cf.read_text(encoding="utf-8"))
            # A breadth-less (partial) cache must not be served all day — it makes
            # the 'stress' classification unreachable until it's overwritten.
            if not cached.get("partial"):
                return cached
        except (json.JSONDecodeError, OSError):
            pass

    bench = config.get("regime.benchmark", "SPY")
    out: dict[str, Any] = {"benchmark": bench}
    try:
        df = yf.download(bench, period="1y", interval="1d",
                         auto_adjust=True, progress=False, threads=False)
        close = df["Close"]
        if hasattr(close, "columns"):  # MultiIndex frame
            close = close.iloc[:, 0]
        close = close.dropna()
        price = float(close.iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        rets = close.pct_change().dropna()
        vol20 = float(rets.tail(20).std() * np.sqrt(252) * 100)
        out.update({
            "bench_price": round(price, 2),
            "bench_vs_200dma_pct": round((price / sma200 - 1) * 100, 1),
            "bench_vs_50dma_pct": round((price / sma50 - 1) * 100, 1),
            "bench_vol20_ann_pct": round(vol20, 1),
        })
    except Exception:  # noqa: BLE001
        return {}

    # Breadth from the technicals the pipeline already computed (no extra downloads).
    if technicals:
        above50, total = 0, 0
        sub_above: dict[str, list[int]] = {}
        for sym, t in technicals.items():
            price, s50 = t.get("price"), t.get("sma50")
            if not (price and s50):
                continue
            total += 1
            up = 1 if price > s50 else 0
            above50 += up
            if subsectors:
                sub_above.setdefault(subsectors.get(sym, "other"), []).append(up)
        if total:
            out["breadth_pct"] = round(above50 / total * 100)
        if sub_above:
            out["subsector_breadth"] = {k: round(sum(v) / len(v) * 100)
                                        for k, v in sub_above.items() if v}

    # Classify — with HYSTERESIS: SPY oscillating ±0.5% around the 200dma must not
    # flip risk_on<->risk_off daily. Inside a ±1% band of the 200dma (and ±3pt band
    # of the breadth threshold) the previous day's regime is kept.
    stress_breadth = config.get("regime.breadth_threshold_pct", 40)
    band_200 = 1.0   # % around the 200dma
    band_breadth = 3  # pts around the breadth threshold
    vs200 = out.get("bench_vs_200dma_pct", 0)
    breadth = out.get("breadth_pct")
    prev = _prev_regime()

    if vs200 < -band_200:
        raw = "risk_off"
    elif vs200 <= band_200 and prev in ("risk_off", "stress", "neutral", "risk_on"):
        raw = prev  # inside the 200dma band: sticky
    elif breadth is not None and breadth < stress_breadth - band_breadth:
        raw = "stress"    # index fine, internals breaking (targeted rout)
    elif (breadth is not None and breadth < stress_breadth + band_breadth
          and prev == "stress"):
        raw = "stress"    # sticky exit from stress
    elif out.get("bench_vs_50dma_pct", 0) < 0 or (breadth is not None and breadth < 55):
        raw = "neutral"
    else:
        raw = "risk_on"
    out["regime"] = raw

    weak = [k for k, v in (out.get("subsector_breadth") or {}).items() if v < stress_breadth]
    if weak:
        out["weak_subsectors"] = weak
    if technicals is None:
        out["partial"] = True  # breadth-less computation; cache readers treat as a miss
    else:
        _save_state(raw)  # only full computations advance the hysteresis state

    try:
        cf.write_text(json.dumps(out), encoding="utf-8")
    except OSError:
        pass
    return out


def context_line(r: dict[str, Any]) -> str:
    """One prompt-ready line describing the regime for Claude's research context."""
    if not r or "regime" not in r:
        return ""
    parts = [f"MARKET REGIME: {r['regime'].upper()}",
             f"{r.get('benchmark','SPY')} {r.get('bench_vs_200dma_pct')}% vs 200dma"]
    if r.get("breadth_pct") is not None:
        parts.append(f"universe breadth {r['breadth_pct']}% above 50dma")
    if r.get("weak_subsectors"):
        parts.append("WEAK sub-sectors: " + ", ".join(r["weak_subsectors"]))
    line = " · ".join(parts)
    if r["regime"] in ("stress", "risk_off"):
        line += (". Be more conservative: require stronger evidence for buy/strong_buy, "
                 "prefer waiting for stabilization in the weak groups.")
    return line


if __name__ == "__main__":
    from . import technical, universe
    # `universe.symbols()` is the CANDIDATE list, which in broad mode is thousands of
    # names; breadth only ever meant "the universe the pipeline actually scores".
    uni = universe.current_universe()
    subs = {r["symbol"]: r["subsector"] for r in uni}
    techs = technical.compute_many([r["symbol"] for r in uni], use_cache=True)
    r = detect(techs, subs, use_cache=False)
    print(json.dumps(r, indent=2))
    print()
    print(context_line(r))
