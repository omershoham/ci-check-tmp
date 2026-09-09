"""Shared fixtures: every test runs against a throwaway SQLite DB and cache dir.

`config.db_path` / `config.cache_dir` are the only two places the code learns where
data lives, so patching them isolates the whole storage layer per test.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Temp DB + cache dir; returns the tmp root. Schema is created."""
    db = tmp_path / "trader.db"
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(config, "db_path", lambda: db)
    monkeypatch.setattr(config, "cache_dir", lambda: cache)
    from src import storage
    # Each test gets a fresh DB file, so the once-per-process schema flag must reset.
    monkeypatch.setattr(storage, "_schema_ready", False)
    storage.init_db()
    return tmp_path


@pytest.fixture
def storage(workspace):
    from src import storage as _storage
    return _storage


@pytest.fixture(autouse=True)
def _restore_main_module():
    """Put `sys.modules['__main__']` back after every test.

    Streamlit's AppTest runs the app script by REPLACING the interpreter's `__main__`
    module with one whose `__file__` is app/dashboard.py, and never restores it. That
    leak is invisible until something spawns a process: `multiprocessing.spawn` hands
    each child `__main__.__file__` and the child re-executes it, so after a dashboard
    test every ProcessPoolExecutor worker tried to run the whole Streamlit dashboard.
    With CLAUDE_TRADER_HOME pointing at an empty data tree the app hits `st.stop()`,
    the worker dies during bootstrap, the pool breaks, and the pool's own shutdown
    deadlocks — which is why the suite hung at the serial-vs-parallel snapshot test
    (test_data_layer runs right after test_dashboard) and only under a HOME override.

    Autouse and unconditional: the next test to spawn a pool should not have to know
    which earlier test used Streamlit.
    """
    import sys
    main = sys.modules.get("__main__")
    yield
    if main is not None and sys.modules.get("__main__") is not main:
        sys.modules["__main__"] = main


@pytest.fixture
def curated(monkeypatch):
    """Pin the curated 105-name universe AND the pack that owns the list.

    The shipped default became `universe.source: broad` + `theme: general` at Phase 7,
    and the two are not independent: `source` picks the list-BUILDING path while the
    THEME owns the tickers, so pinning only the source yields an empty universe. Any
    test that is about the curated list has to say both, out loud, instead of
    inheriting whatever config.yaml currently ships.
    """
    from src import theme, universe
    monkeypatch.setattr(universe, "configured_source", lambda: "curated")
    monkeypatch.setattr(theme, "configured_name", lambda: "ai_infra")
    return "curated"
