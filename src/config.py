"""Configuration loader: shipped defaults ⊕ per-user overrides, .env, and project paths.

Two locations matter:

  ROOT  - the code checkout. `config.yaml` here is the SHIPPED DEFAULTS file.
  HOME  - the user's workspace, `$CLAUDE_TRADER_HOME` (defaults to ROOT, so a
          single-directory install behaves exactly as before). Holds the user's
          `config.yaml` (overrides, deep-merged over the defaults), `.env`, and the
          `data/` tree (DB, cache, and later the point-in-time history store).

Splitting the two is what lets someone run this without editing files inside the
repo: they set CLAUDE_TRADER_HOME, drop a 10-line config.yaml with their universe
and policy, and never touch the shipped defaults.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root = parent of the src/ directory that contains this file.
ROOT = Path(__file__).resolve().parent.parent
HOME = Path(os.environ.get("CLAUDE_TRADER_HOME") or ROOT).expanduser().resolve()

DEFAULT_CONFIG_PATH = ROOT / "config.yaml"
USER_CONFIG_PATH = HOME / "config.yaml"     # same file as the defaults when HOME == ROOT
CONFIG_PATH = DEFAULT_CONFIG_PATH           # back-compat name

# .env: the workspace copy wins (dotenv never overrides variables already set, so
# loading HOME first gives it precedence), the checkout copy is the fallback.
load_dotenv(HOME / ".env")
if HOME != ROOT:
    load_dotenv(ROOT / ".env")


# Sections a user override REPLACES wholesale instead of merging into. These are
# collections (a universe, a watchlist, a cap ladder) where "my list" must mean
# exactly my list — a recursive merge would make it impossible to REMOVE a default
# sub-sector or ballast ETF from inside the user file.
_REPLACE_KEYS = frozenset({
    "universe.sectors",
    "policy.ballast_watchlist",
    "policy.conviction_weight_caps",
    "scoring.pillars",
})


def _deep_merge(base: Any, override: Any, path: str = "") -> Any:
    """Dicts merge recursively (scalars in the user file win, everything else keeps the
    default); lists and the _REPLACE_KEYS sections are replaced wholesale."""
    if path in _REPLACE_KEYS:
        return override
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            sub = f"{path}.{k}" if path else k
            out[k] = _deep_merge(base.get(k), v, sub) if k in base else v
        return out
    return override


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if USER_CONFIG_PATH != DEFAULT_CONFIG_PATH and USER_CONFIG_PATH.exists():
        with open(USER_CONFIG_PATH, "r", encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    return cfg


def get(path: str, default: Any = None) -> Any:
    """Dotted-path accessor, e.g. get('scoring.weights.deal')."""
    cur: Any = load_config()
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def abs_path(rel: str) -> Path:
    """Resolve a config-relative path (like paths.db) against the WORKSPACE home."""
    p = Path(rel)
    return p if p.is_absolute() else (HOME / p)


def db_path() -> Path:
    p = abs_path(get("paths.db", "data/trader.db"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = abs_path(get("paths.cache", "data/cache"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def sweep_cache(max_age_days: int = 7) -> int:
    """Delete date-stamped cache files older than max_age_days (the cache accrues
    ~300 files/day and nothing else ever prunes it). Returns files removed."""
    import re
    from datetime import date, timedelta
    cutoff = (date.today() - timedelta(days=max_age_days)).isoformat()
    n = 0
    for f in cache_dir().glob("*_*.json"):
        m = re.search(r"(\d{4}-\d{2}-\d{2})\.json$", f.name)
        if m and m.group(1) < cutoff:
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
    return n


def oauth_token() -> str | None:
    return os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")


def api_key_is_set() -> bool:
    """True if ANTHROPIC_API_KEY is present (would override subscription billing)."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


if __name__ == "__main__":
    cfg = load_config()
    print("Config loaded OK.")
    print("ROOT:", ROOT)
    print("HOME:", HOME, "(CLAUDE_TRADER_HOME)" if os.environ.get("CLAUDE_TRADER_HOME") else "(default = ROOT)")
    print("User overrides:", USER_CONFIG_PATH if USER_CONFIG_PATH.exists() and USER_CONFIG_PATH != DEFAULT_CONFIG_PATH else "(none)")
    print("DB path:", db_path())
    print("Cache dir:", cache_dir())
    from . import theme
    print("Theme:", theme.configured_name(), "-", theme.board_name())
    print("Sectors:", list(theme.curated_sectors().keys()))
    print("OAuth token set:", bool(oauth_token()))
    print("ANTHROPIC_API_KEY set (should be False):", api_key_is_set())
