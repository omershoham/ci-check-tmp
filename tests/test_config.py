import importlib

import yaml

from src import config


def test_default_home_is_root_and_behaviour_unchanged(monkeypatch):
    monkeypatch.delenv("CLAUDE_TRADER_HOME", raising=False)
    importlib.reload(config)
    try:
        assert config.HOME == config.ROOT
        assert config.USER_CONFIG_PATH == config.DEFAULT_CONFIG_PATH
        assert config.get("ibkr.port") == 4001
        assert config.abs_path("data/x.db") == config.ROOT / "data" / "x.db"
    finally:
        importlib.reload(config)


def test_workspace_home_overrides_and_relocates_data(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "ibkr": {"port": 4002},
        "universe": {"sectors": {"only": ["AAA", "BBB"]}},
        "paths": {"db": "data/mine.db"},
    }), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_TRADER_HOME", str(tmp_path))
    importlib.reload(config)
    try:
        assert config.HOME == tmp_path.resolve()
        assert config.get("ibkr.port") == 4002                    # overridden
        assert config.get("ibkr.client_id") == 17                 # default kept (deep merge)
        assert config.get("universe.sectors") == {"only": ["AAA", "BBB"]}  # list replaced, not appended
        assert config.get("scoring.pillars.technical") == 0.35     # untouched section intact
        assert config.db_path() == tmp_path.resolve() / "data" / "mine.db"
        assert config.cache_dir().is_relative_to(tmp_path.resolve())
    finally:
        monkeypatch.delenv("CLAUDE_TRADER_HOME", raising=False)
        importlib.reload(config)


def test_deep_merge_semantics():
    base = {"a": {"b": 1, "c": [1, 2]}, "d": 4}
    over = {"a": {"c": [9]}, "e": 5}
    out = config._deep_merge(base, over)
    assert out == {"a": {"b": 1, "c": [9]}, "d": 4, "e": 5}
    assert base == {"a": {"b": 1, "c": [1, 2]}, "d": 4}          # not mutated


def test_collection_sections_replace_not_merge():
    base = {"universe": {"sectors": {"semis": ["NVDA"], "power": ["VRT"]}, "other": 1},
            "policy": {"ballast_watchlist": ["TLT", "GLD"], "max_position_weight": 20}}
    over = {"universe": {"sectors": {"bio": ["AMGN"]}},
            "policy": {"ballast_watchlist": ["IEF"]}}
    out = config._deep_merge(base, over)
    assert out["universe"]["sectors"] == {"bio": ["AMGN"]}        # defaults removed, as intended
    assert out["universe"]["other"] == 1                          # sibling scalar kept
    assert out["policy"]["ballast_watchlist"] == ["IEF"]
    assert out["policy"]["max_position_weight"] == 20
