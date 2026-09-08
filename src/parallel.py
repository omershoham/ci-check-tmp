"""One shared process pool for the per-symbol replays.

Three loops have the same shape — `data.sec_facts.build_snapshots`,
`ml.features.technical_replay` and `backtest.signals.technical_pillar` all walk symbols
independently, touch nothing shared, and were serial only because the store's I/O used
to dominate them. Once Phase 2 fixed the I/O they became the bottleneck, and they are
embarrassingly parallel: one symbol's replay cannot see another's.

Three rules this module exists to enforce, each recording a way to get it wrong:

  * `workers=1` runs the plain serial loop. That is what the tests exercise and what
    every "the refactor changed a number" investigation starts from; if the parallel
    path ever disagrees with it, the parallel path is wrong.
  * Results come back in INPUT order (`Executor.map`), never as they complete. The
    replays build frames whose column order follows insertion order, and the backtest
    engine breaks score ties with `sort_values(...).index[:n]` — so completion order
    would quietly change which names land in the top decile.
  * Windows starts workers with `spawn`: each one re-imports the module, so `fn` must
    be a module-level function and every payload must pickle. Never hand a worker the
    HistoryStore — pass the plain frames it needs, or every worker re-opens the store.

A broken pool (sandboxed CI, no fork/spawn permission) falls back to the serial loop
with a log line rather than taking the pipeline down.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Iterable, Sequence

# Leave two cores: one for the OS, one for whatever else the analyst is running. A
# fully saturated box makes the dashboard unusable during a refresh.
_RESERVED = 2
_ENV = "CLAUDE_TRADER_WORKERS"

# A spawned worker re-imports pandas and numpy — ~1.5s each on Windows, paid before it
# does anything. Below a few thousand symbol-dates that setup costs more than the work:
# the live screen replays 105 symbols on ONE date and measured 2.3s with a pool against
# 0.6s serial. Callers pass their own work estimate; the pool only opens above it.
_MIN_PARALLEL_WORK = 2_000


def default_workers() -> int:
    env = os.environ.get(_ENV)
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, (os.cpu_count() or 2) - _RESERVED)


def replay_symbols(fn: Callable[[Any], Any], payloads: Sequence[Any],
                   workers: int | None = None, log: Callable[[str], None] | None = None,
                   label: str = "replay", every: int = 20,
                   work: int | None = None) -> list[Any]:
    """Map `fn` over `payloads` and return the results in input order.

    `fn` must be importable by name from its own module (spawn re-imports it) and both
    its argument and its return value must pickle. `work` is the caller's estimate of
    total units (symbols x dates); below `_MIN_PARALLEL_WORK` the loop stays serial.
    An explicit `workers` overrides that judgement in both directions.
    """
    items = list(payloads)
    n = len(items)
    if n == 0:
        return []
    if workers is None and work is not None and work < _MIN_PARALLEL_WORK:
        return _serial(fn, items, log, label, every)
    w = default_workers() if workers is None else int(workers)
    w = max(1, min(w, n))
    if w == 1:
        return _serial(fn, items, log, label, every)
    try:
        out: list[Any] = []
        with ProcessPoolExecutor(max_workers=w) as ex:
            for i, r in enumerate(ex.map(fn, items), 1):
                out.append(r)
                _tick(log, label, i, n, every)
        return out
    except (BrokenProcessPool, OSError, PermissionError, ImportError) as exc:
        if log:
            log(f"  {label}: process pool unavailable ({exc}) — running serially")
        return _serial(fn, items, log, label, every)


def _serial(fn, items, log, label, every) -> list[Any]:
    out = []
    for i, p in enumerate(items, 1):
        out.append(fn(p))
        _tick(log, label, i, len(items), every)
    return out


def _tick(log, label: str, i: int, n: int, every: int) -> None:
    if log and (i % every == 0 or i == n):
        log(f"  {label} {i}/{n}")


def flatten(chunks: Iterable[Iterable[Any]]) -> list[Any]:
    """Concatenate per-symbol result lists in symbol order — the same list the serial
    loop would have appended to."""
    return [row for chunk in chunks for row in chunk]
