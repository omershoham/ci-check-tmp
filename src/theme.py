"""Theme packs: the PROSE and (optionally) the curated universe of a screen.

A pack in `themes/*.yaml` owns what is genuinely theme-specific — the board's name,
the counterparties whose deals matter, the fragments injected into the local-LLM and
Claude prompts, and a curated ticker list. Selected by `theme:` in config.yaml.

WHAT A PACK MUST NEVER OWN: the catalyst key set. `local_llm.CATALYST_KEYS` is fixed
in Python precisely so that two themes' signals are the SAME measurement. A pack that
could rename `customer_deal` to `hyperscaler_signal` would make every cross-theme
pooling invalid and no information coefficient could ever be computed on the key —
which is exactly the failure the single `hyperscaler_signal` column produced.

The failure this module prevents concretely: a missing or malformed pack must not
stop a run. `load()` falls back to the built-in general pack and the screen keeps
scoring; only the prose degrades.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from . import config

THEMES_DIR = config.ROOT / "themes"
DEFAULT_THEME = "general"

# Used when the pack file is missing/unreadable, so prompts still have a role line.
_FALLBACK: dict[str, Any] = {
    "name": "Opportunity Board",
    "description": "US equities.",
    "actors": ["large customers", "hyperscalers", "government / defense primes",
               "major pharma partners", "telcos"],
    "prompt_fragments": {
        "analyst_role": "an equity research analyst covering US equities",
        "catalyst_emphasis": "Judge each catalyst on its own materiality to THIS "
                             "company's revenue or multiple.",
    },
    "subsectors": {},
}


def configured_name() -> str:
    """Which pack is active. Explicit `theme:` wins; otherwise the curated universe
    implies the theme it was hand-picked for, because a curated AI-infra list scored
    under the general pack would describe itself as having no theme."""
    name = config.get("theme")
    if name:
        return str(name).strip().lower()
    src = str(config.get("universe.source", "curated") or "curated").lower()
    return "ai_infra" if src == "curated" else DEFAULT_THEME


@lru_cache(maxsize=8)
def _load_file(name: str) -> dict[str, Any]:
    path = THEMES_DIR / f"{name}.yaml"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return dict(_FALLBACK)
    return data if isinstance(data, dict) else dict(_FALLBACK)


def load(name: str | None = None) -> dict[str, Any]:
    """The active pack, with every key the callers below expect present."""
    pack = dict(_FALLBACK)
    pack.update(_load_file(name or configured_name()))
    pack.setdefault("subsectors", {})
    frags = dict(_FALLBACK["prompt_fragments"])
    frags.update(pack.get("prompt_fragments") or {})
    pack["prompt_fragments"] = frags
    return pack


def board_name(name: str | None = None) -> str:
    return str(load(name).get("name") or _FALLBACK["name"])


def actors(name: str | None = None) -> list[str]:
    a = load(name).get("actors") or _FALLBACK["actors"]
    return [str(x) for x in a]


def actors_text(name: str | None = None) -> str:
    """The actor list as one prose clause for a prompt line."""
    return ", ".join(actors(name))


def fragment(key: str, name: str | None = None) -> str:
    """One prompt fragment, whitespace-collapsed so YAML folding can't inject
    newlines into the middle of a prompt sentence."""
    txt = str(load(name)["prompt_fragments"].get(key, "") or "")
    return " ".join(txt.split())


def subsectors(name: str | None = None) -> dict[str, dict[str, Any]]:
    subs = load(name).get("subsectors") or {}
    return subs if isinstance(subs, dict) else {}


def subsector_labels(name: str | None = None) -> dict[str, str]:
    return {k: str((v or {}).get("label") or k) for k, v in subsectors(name).items()}


def curated_sectors(name: str | None = None) -> dict[str, list[str]]:
    """subsector key -> ticker list, in pack order.

    A user `universe.sectors:` in config.yaml still wins (config._REPLACE_KEYS makes
    it a wholesale replacement), so an existing install that pinned its own universe
    keeps it; the pack is only the default.
    """
    override = config.get("universe.sectors") or {}
    if override:
        return {k: [str(s) for s in (v or [])] for k, v in override.items()}
    return {k: [str(s) for s in ((v or {}).get("symbols") or [])]
            for k, v in subsectors(name).items()}


if __name__ == "__main__":
    p = load()
    n = sum(len(v) for v in curated_sectors().values())
    print(f"theme: {configured_name()} — {p['name']}")
    print(f"actors: {actors_text()}")
    print(f"curated: {n} tickers across {len(curated_sectors())} sub-sectors")
