"""The shared process pool must be faster, never different — and never a hang.

Phase 7 left the suite deadlocking under a CLAUDE_TRADER_HOME override. Cause: an
earlier Streamlit AppTest permanently replaces `sys.modules['__main__']` with the
dashboard script, `multiprocessing.spawn` re-executes `__main__.__file__` in every
child, and on an empty data tree that app calls `st.stop()` — so every worker died
during bootstrap, the pool broke, and `ProcessPoolExecutor.__exit__` deadlocked
joining a manager thread stuck closing the call queue. These tests pin both halves of
the fix.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import types

from src import parallel


def double(x):
    return x * 2


def die_in_worker(x):
    """Abrupt worker death — the shape that BREAKS a pool. A plain exception does not:
    it is captured and re-raised in the parent, which is the intended behaviour."""
    if mp.parent_process() is not None:
        os._exit(3)
    return x * 2


def boom(x):
    raise ValueError("worker said no")


def test_a_dying_worker_falls_back_to_serial_instead_of_hanging():
    logs: list[str] = []
    out = parallel.replay_symbols(die_in_worker, list(range(8)), workers=2,
                                  log=logs.append)
    assert out == [0, 2, 4, 6, 8, 10, 12, 14]
    assert any("process pool unavailable" in m for m in logs)


def test_a_raising_worker_propagates(monkeypatch):
    """The fallback must not swallow real errors: a bug in `fn` has to surface."""
    import pytest
    with pytest.raises(ValueError):
        parallel.replay_symbols(boom, [1, 2], workers=1)


def test_spawn_is_refused_when_main_has_been_hijacked(monkeypatch):
    fake = types.ModuleType("__main__")
    fake.__file__ = os.path.join(os.path.dirname(__file__), "..", "app", "dashboard.py")
    fake.__spec__ = None
    monkeypatch.setitem(sys.modules, "__main__", fake)
    why = parallel._spawn_is_safe()
    assert "dashboard.py" in why
    logs: list[str] = []
    # ...and the pool is never opened, so the results still come back, serially.
    assert parallel.replay_symbols(double, [1, 2, 3], workers=2, log=logs.append) == [2, 4, 6]
    assert any("__main__ has been replaced" in m for m in logs)


def test_spawn_is_allowed_for_a_normal_main():
    """A module imported by NAME (`python -m ...`, pytest) is passed by spec name, not
    by path, and is safe — the guard must not refuse every real run."""
    fake = types.ModuleType("__main__")
    fake.__file__ = "whatever.py"
    fake.__spec__ = types.SimpleNamespace(name="pytest.__main__")
    real = sys.modules["__main__"]
    sys.modules["__main__"] = fake
    try:
        assert parallel._spawn_is_safe() == ""
    finally:
        sys.modules["__main__"] = real
