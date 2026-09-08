"""Gate A: is the SEC-served fundamental pillar the same signal as the `.info`-served one?

    python -m scripts.compare_fundamental_source
    python -m scripts.compare_fundamental_source --top 20 --no-network

Phase 4 stopped the screen making one throttled yfinance `.info` request per name and
started serving the fundamental pillar from the local point-in-time SEC store. That is
a data-SOURCE swap on the input of a ranking signal, so it has to be measured, not
asserted: this script scores the same universe on the same day from both sources and
reports the Spearman rank correlation plus the names that moved.

Two numbers are printed, and they answer different questions:

  source-only   the CURRENT scoring definition applied to both metric dicts. Isolates
                the data swap.
  end-to-end    the OLD (v3) pillar on `.info` vs the NEW (v5) pillar on SEC facts —
                data swap AND definition change together. Expected to be lower; the
                definition change is the point of Phase 4, not a regression.

THE PASS CRITERION IS PER FAMILY, not on the whole pillar, because two of the four
families do not have a common definition across the two sources and so cannot agree:

  valuation, quality   same quantity both sides -> must agree, bar is 0.9. These are
                       the families that actually test the data swap.
  cash flow            `.info`'s `freeCashflow` is not TTM CFO minus TTM capex. For
                       many names it is a single QUARTER (MSFT 16.5B vs 67.0B TTM;
                       16.55 x 4 = 66.2), and SEC `cfo` matches `.info`'s own
                       `operatingCashflow` almost exactly (MSFT 1.000, AMZN 1.000).
                       The gap is Yahoo's capex leg, not the store.
  growth               `.info` reports most-recent-quarter year-over-year; the SEC
                       path computes TTM against the prior TTM. Different questions.

So a low whole-pillar Spearman here is a statement about the reference, and the two
families that CAN be compared are the gate.

The v3 pillar is reproduced inline below rather than imported, because it no longer
exists in `src/`. It is a frozen copy for measurement only — nothing scores with it.
"""
from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from src import fundamentals, scoring, universe
from src.data.history_store import HistoryStore

# --- frozen v3 fundamental pillar (pre-Phase-4), for the end-to-end comparison only.
_V3_VALUATION = [("trailingPE", False), ("forwardPE", False),
                 ("enterpriseToEbitda", False), ("priceToSalesTrailing12Months", False)]
_V3_CASHFLOW = [("fcf_yield", True), ("fcf_margin", True)]
_V3_GROWTH = [("revenueGrowth", True), ("earningsGrowth", True)]
_V3_QUALITY = [("returnOnEquity", True), ("returnOnAssets", True),
               ("grossMargins", True), ("operatingMargins", True), ("debtToEquity", False)]


def _v3_pct(s: pd.Series, hib: bool, mask: bool, groups: pd.Series) -> pd.Series:
    s = s.where(s > 0) if mask else s
    r = 0.5 * s.rank(pct=True) + 0.5 * s.groupby(groups).rank(pct=True)
    return (r if hib else 1.0 - r).fillna(0.5)


def _v3_sub(df: pd.DataFrame, comps, groups) -> pd.Series:
    cols = [_v3_pct(pd.to_numeric(df[c], errors="coerce"), hib,
                    c in scoring._MASK_NONPOS, groups)
            for c, hib in comps if c in df.columns]
    return pd.concat(cols, axis=1).mean(axis=1) if cols else pd.Series(0.5, index=df.index)


def v3_pillar(df: pd.DataFrame, subs: dict[str, str]) -> pd.Series:
    g = pd.Series({s: subs.get(s, "") for s in df.index})
    return (_v3_sub(df, _V3_VALUATION, g) * 0.30 + _v3_sub(df, _V3_CASHFLOW, g) * 0.30
            + _v3_sub(df, _V3_GROWTH, g) * 0.20 + _v3_sub(df, _V3_QUALITY, g) * 0.20)


_BAR = 0.9
_COMPARABLE = ("valuation", "quality")
_NOT_COMPARABLE = {
    "cashflow": "`.info` freeCashflow is often ONE QUARTER, not TTM CFO-capex "
                "(SEC cfo matches `.info` operatingCashflow ~1.00)",
    "growth": "`.info` is most-recent-quarter YoY; SEC is TTM vs prior TTM",
}

_WHY_COLS = ["forwardPE", "trailingPE", "enterpriseToEbitda",
             "priceToSalesTrailing12Months", "priceToBook", "p_ffo",
             "fcf_yield", "fcf_margin", "revenueGrowth", "earningsGrowth",
             "returnOnEquity", "returnOnAssets", "grossMargins", "operatingMargins",
             "debtToEquity"]


def _why(sym: str, info: dict, sec: dict, sector: str) -> str:
    """Name the inputs that differ, so a rank move can be attributed rather than guessed.

    `forwardPE` (always `.info`-only) and `p_ffo` (always SEC-only, and used only by
    Real Estate) are structural and reported once in the header, not per name."""
    bits = []
    used = {c for fam in scoring.FAMILIES
            for c, _ in scoring.sector_metrics(sector, fam)}
    for c in _WHY_COLS:
        if c in ("forwardPE", "p_ffo", "priceToBook") and c not in used:
            continue
        if c in ("p_ffo", "priceToBook") and sec.get(c) is not None and info.get(c) is None:
            continue
        a, b = info.get(c), sec.get(c)
        ha = a is not None and a == a
        hb = b is not None and b == b
        if ha and not hb:
            bits.append(f"{c}: .info only")
        elif hb and not ha:
            bits.append(f"{c}: SEC only")
        elif ha and hb and abs(float(a) - float(b)) > 0.25 * max(abs(float(a)), 1e-9):
            bits.append(f"{c} {float(a):.3g}->{float(b):.3g}")
    return f"[{sector}] " + ("; ".join(bits[:5]) or "inputs agree")


def main() -> None:
    ap = argparse.ArgumentParser(description="SEC-served vs .info-served fundamental pillar")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--no-network", action="store_true",
                    help="use only cached `.info` files (today's cache must exist)")
    args = ap.parse_args()

    store = HistoryStore()
    uni = universe.current_universe(store=store)
    syms = [r["symbol"] for r in uni]
    subs = {r["symbol"]: r["subsector"] for r in uni}
    secs = universe.sector_map(uni, store=store)
    today = date.today()

    sec_m = fundamentals.from_store(store, syms, today)
    info_m = (fundamentals.fetch_many(syms, use_cache=True) if not args.no_network
              else {s: fundamentals.fetch_one(s) for s in syms})
    # Compare only names BOTH sources can serve; anything else is a coverage question,
    # reported separately rather than silently changing the correlation.
    both = [s for s in syms
            if not sec_m[s].get("error") and not info_m.get(s, {}).get("error")
            and info_m.get(s, {}).get("price")]
    print(f"universe {len(syms)} | SEC-served {sum(1 for s in syms if not sec_m[s].get('error'))} "
          f"| .info-served {sum(1 for s in syms if not info_m.get(s, {}).get('error'))} "
          f"| compared {len(both)}")
    missing = [s for s in syms if s not in both]
    if missing:
        print(f"  not comparable: {missing}")

    sec_df = pd.DataFrame([sec_m[s] for s in both]).set_index("symbol", drop=False)
    inf_df = pd.DataFrame([info_m[s] for s in both]).set_index("symbol", drop=False)

    da = scoring.fundamental_detail(sec_df, subs, secs)
    db = scoring.fundamental_detail(inf_df, subs, secs)
    p_sec, p_inf = da["pillar"], db["pillar"]
    p_v3 = v3_pillar(inf_df, subs)

    rho_src = float(p_sec.rank().corr(p_inf.rank(), method="spearman"))
    rho_e2e = float(p_sec.rank().corr(p_v3.rank(), method="spearman"))
    print(f"\nSpearman(SEC-served, .info-served) — SAME definition = {rho_src:.4f} "
          f"(whole pillar; NOT the gate — see module docstring)")
    print(f"Spearman(SEC v5 pillar, .info v3 pillar) — end to end   = {rho_e2e:.4f}")

    # The gate. A low WHOLE-PILLAR correlation is only interpretable per family, and
    # only two families are measuring the same quantity on both sides.
    print("\nby family (Spearman):")
    fam_rho = {f: float(da[f].rank().corr(db[f].rank(), method="spearman"))
               for f in scoring.FAMILIES}
    passed = True
    for fam, rho in fam_rho.items():
        if fam in _COMPARABLE:
            verdict = "PASS" if rho > _BAR else "FAIL"
            passed &= rho > _BAR
            print(f"  {fam:<10} {rho:.4f}  {verdict} (bar {_BAR}) — same quantity both sides")
        else:
            print(f"  {fam:<10} {rho:.4f}  n/a  — {_NOT_COMPARABLE[fam]}")
    print(f"\nGATE ({' + '.join(_COMPARABLE)} > {_BAR}): "
          f"{'PASS' if passed else 'FAIL'}")
    print("\nper input — rho and median |gap| relative to `.info`:")
    for c in _WHY_COLS + ["freeCashflow", "totalRevenue", "marketCap"]:
        if c not in sec_df.columns or c not in inf_df.columns:
            continue
        a = pd.to_numeric(sec_df[c], errors="coerce")
        b = pd.to_numeric(inf_df[c], errors="coerce")
        m = a.notna() & b.notna()
        if int(m.sum()) < 10:
            continue
        rel = ((a[m] - b[m]).abs() / b[m].abs().clip(lower=1e-9)).median()
        print(f"  {c:<30} n={int(m.sum()):>3}  rho={a[m].corr(b[m], method='spearman'):.3f}"
              f"  med|rel|={float(rel):.3f}")

    cmp = pd.DataFrame({"sec": p_sec, "inf": p_inf, "v3": p_v3})
    cmp["d_rank"] = p_sec.rank(ascending=False) - p_inf.rank(ascending=False)

    def _line(sym, row, lead: str) -> str:
        d = row.d_rank
        dr = f"{int(d):+d}" if d == d else "n/a"
        return (f"{sym:<6} {lead} sec={row.sec:.4f} info={row.inf:.4f} rankΔ={dr}  "
                f"{_why(sym, info_m[sym], sec_m[sym], secs.get(sym, '?'))}")

    print(f"\n--- top {args.top} by SEC-served pillar ---")
    for i, (s, r) in enumerate(cmp.sort_values("sec", ascending=False).head(args.top).iterrows(), 1):
        print(f"{i:>3} " + _line(s, r, ""))
    print(f"\n--- top {args.top} by .info-served pillar ---")
    for i, (s, r) in enumerate(cmp.sort_values("inf", ascending=False).head(args.top).iterrows(), 1):
        print(f"{i:>3} " + _line(s, r, ""))
    print(f"\n--- {args.top} largest rank moves (SEC vs .info, same definition) ---")
    for s, r in cmp.reindex(cmp["d_rank"].abs().sort_values(ascending=False).index).head(args.top).iterrows():
        print(_line(s, r, ""))


if __name__ == "__main__":
    main()
