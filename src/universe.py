"""Build the AI-infra ticker universe from the curated config lists.

The curated per-sub-sector lists in config.yaml are the single source of truth — they
keep the universe transparent and stable. (ETF-holdings augmentation was considered but
dropped: issuer CSV scraping is too fragile to rely on.)
"""
from __future__ import annotations

from . import config

# Friendly sub-sector labels for display.
SUBSECTOR_LABELS = {
    "semis_equipment": "Semiconductors & Equipment",
    "datacenter_power": "Data Center & Power",
    "networking_optical": "Networking & Optical",
    "cloud_ai_software": "Cloud / AI Software",
}


def build_universe() -> list[dict[str, str]]:
    """Return [{symbol, subsector}] deduped, first sub-sector wins on overlap."""
    sectors = config.get("universe.sectors", {}) or {}
    seen: dict[str, str] = {}
    for subsector, symbols in sectors.items():
        for sym in symbols or []:
            sym = str(sym).strip().upper()
            if sym and sym not in seen:
                seen[sym] = subsector
    return [{"symbol": s, "subsector": ss} for s, ss in seen.items()]


def symbols() -> list[str]:
    return [r["symbol"] for r in build_universe()]


def subsector_map() -> dict[str, str]:
    return {r["symbol"]: r["subsector"] for r in build_universe()}


if __name__ == "__main__":
    u = build_universe()
    by_sub: dict[str, int] = {}
    for r in u:
        by_sub[r["subsector"]] = by_sub.get(r["subsector"], 0) + 1
    print(f"Universe: {len(u)} tickers")
    for ss, n in by_sub.items():
        print(f"  {SUBSECTOR_LABELS.get(ss, ss)}: {n}")
