"""Shared fixtures: every test runs against a throwaway SQLite DB, cache dir and
point-in-time store.

`config.db_path` / `config.cache_dir` / `config.history_root` are the only three places
the code learns where data lives, so patching them isolates the whole storage layer per
test. The history store was missing from that list until this change, and the offline
pipeline test called `record_universe(today, ..., source="curated")` straight into
`$CLAUDE_TRADER_HOME/data/history`: anyone who cloned this repo and ran `pytest`
silently mutated their own data. `_no_real_store` below makes that impossible to
reintroduce — it fails any test that resolves a store path outside pytest's tmp tree.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_store(request, tmp_path_factory, monkeypatch):
    """Fail loudly if a test resolves a history-store path outside pytest's tmp tree.

    The default redirect below is easy to defeat by accident — a test that constructs
    `HistoryStore()` before the fixture runs, or a helper that builds a path from
    `config.HOME` — and the resulting damage is invisible: parquet writes succeed, the
    user's real membership file grows a row, and the suite still passes. So the rule is
    enforced at the constructor rather than trusted at the seam.
    """
    from src.data.history_store import HistoryStore
    base = tmp_path_factory.getbasetemp().resolve()
    if request.node.get_closest_marker("real_store"):
        return
    root = tmp_path_factory.mktemp("history")
    monkeypatch.setattr(config, "history_root", lambda: root)
    orig = HistoryStore.__init__

    def guarded(self, root=None):
        orig(self, root)
        resolved = pathlib.Path(self.root).resolve()
        if base not in resolved.parents and resolved != base:
            raise AssertionError(
                f"test opened a history store OUTSIDE pytest's tmp tree: {resolved}. "
                f"Tests must never touch the user's real data — pass an explicit "
                f"`HistoryStore(tmp_path / ...)`, or mark the test `real_store`.")

    monkeypatch.setattr(HistoryStore, "__init__", guarded)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Temp DB + cache dir + history store; returns the tmp root. Schema is created."""
    db = tmp_path / "trader.db"
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(config, "db_path", lambda: db)
    monkeypatch.setattr(config, "cache_dir", lambda: cache)
    monkeypatch.setattr(config, "history_root", lambda: tmp_path / "history")
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
