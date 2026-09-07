"""Environment + data health check. Run before a session, or when a run looks wrong:

    python -m scripts.doctor

Prints one line per check (OK / WARN / FAIL) and exits 1 if anything FAILed. The
launcher (scripts/launch.ps1) probes the same ports; this is the full version with
config, auth, model, DB and last-run health included.
"""
from __future__ import annotations

import importlib
import os
import shutil
import socket
import sys
from datetime import datetime, timezone

from src import config, health, storage

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"
_rows: list[tuple[str, str, str]] = []


def _rec(level: str, what: str, detail: str = "") -> None:
    _rows.append((level, what, detail))


def _port_open(host: str, port: int, timeout: float = 0.7) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_python() -> None:
    v = sys.version_info
    lvl = OK if (v.major, v.minor) >= (3, 11) else FAIL
    _rec(lvl, "python", f"{v.major}.{v.minor}.{v.micro} at {sys.executable}")


def check_packages() -> None:
    required = ["yfinance", "pandas", "numpy", "requests", "feedparser", "yaml", "dotenv",
                "plotly", "streamlit"]
    optional = ["ib_async", "pandas_market_calendars", "streamlit_autorefresh", "pytest"]
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    _rec(FAIL if missing else OK, "required packages",
         f"missing: {missing}" if missing else f"{len(required)} present")
    opt_missing = [m for m in optional if importlib.util.find_spec(m) is None]
    _rec(WARN if opt_missing else OK, "optional packages",
         f"missing: {opt_missing}" if opt_missing else "all present")


def check_config() -> None:
    try:
        cfg = config.load_config()
    except Exception as exc:  # noqa: BLE001
        _rec(FAIL, "config.yaml", f"does not load: {exc}")
        return
    pillars = (cfg.get("scoring") or {}).get("pillars") or {}
    tot = sum(float(v) for v in pillars.values()) if pillars else 0
    _rec(OK if abs(tot - 1.0) < 0.02 else WARN, "scoring.pillars",
         f"weights sum to {tot:.2f} ({', '.join(f'{k}={v}' for k, v in pillars.items())})")
    from src import universe
    n = len(universe.symbols())
    _rec(OK if n >= 20 else WARN, "universe", f"{n} tickers")
    _rec(OK if config.cache_dir().exists() and os.access(config.cache_dir(), os.W_OK) else FAIL,
         "cache dir", str(config.cache_dir()))


def check_auth() -> None:
    _rec(OK if config.oauth_token() else FAIL, "CLAUDE_CODE_OAUTH_TOKEN",
         "set in .env" if config.oauth_token() else "missing — run `claude setup-token`")
    _rec(FAIL if config.api_key_is_set() else OK, "ANTHROPIC_API_KEY",
         "SET — this forces per-token API billing; unset it" if config.api_key_is_set()
         else "unset (subscription path)")
    _rec(OK if shutil.which("claude") else FAIL, "claude CLI",
         shutil.which("claude") or "not on PATH")
    sec = os.environ.get("SEC_CONTACT_EMAIL") or config.get("news.sec_contact")
    _rec(OK if sec else WARN, "SEC contact",
         "set" if sec else "unset — EDGAR throttles anonymous callers (see .env.example)")


def check_ollama() -> None:
    from src import local_llm
    if not config.get("local_llm.enabled", True):
        _rec(WARN, "ollama", "disabled in config — sentiment pillar neutral")
        return
    if not local_llm.available():
        _rec(WARN, "ollama", f"unreachable at {config.get('local_llm.host')} — "
                             "sentiment pillar will be neutral")
        return
    model = config.get("local_llm.model", "")
    try:
        import requests
        tags = requests.get(f"{local_llm._host()}/api/tags", timeout=3).json()
        names = {m.get("name", "") for m in tags.get("models", [])}
        have = any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)
        _rec(OK if have else FAIL, "ollama model",
             f"{model} {'present' if have else 'NOT pulled — `ollama pull ' + model + '`'}")
    except Exception as exc:  # noqa: BLE001
        _rec(WARN, "ollama model", f"could not list models: {exc}")


def check_ibkr() -> None:
    if not config.get("ibkr.enabled", True):
        _rec(WARN, "ibkr", "disabled in config")
        return
    host, port = config.get("ibkr.host", "127.0.0.1"), int(config.get("ibkr.port", 4001))
    up = _port_open(host, port)
    _rec(OK if up else WARN, "ibkr gateway", f"{host}:{port} {'open' if up else 'closed — '
         'holdings/quotes will come from the last snapshot'}")


def check_db() -> None:
    try:
        storage.init_db()
        runs = storage.list_runs(1)
    except Exception as exc:  # noqa: BLE001
        _rec(FAIL, "database", f"{config.db_path()}: {exc}")
        return
    if not runs:
        _rec(WARN, "last run", "no runs yet")
        return
    r = runs[0]
    try:
        age_h = (datetime.now(timezone.utc)
                 - datetime.fromisoformat(r["started_at"])).total_seconds() / 3600
        age = f"{age_h:.0f}h ago"
    except (TypeError, ValueError):
        age = "?"
    h = r["health"] if "health" in r.keys() else None
    lvl = OK if (r["status"] == "ok" and h in (None, health.OK)) else \
        (FAIL if r["status"] == "error" or h == health.FAILED else WARN)
    detail = f"#{r['id']} {r['mode']} {age} status={r['status']} {health.badge(h)}"
    probs = health.RunHealth.from_json(r["health_json"] if "health_json" in r.keys()
                                       else None).problems()
    if probs:
        detail += " — " + "; ".join(f"{n}: {note}" for n, _, note in probs)
    _rec(lvl, "last run", detail)


def check_history_store() -> None:
    """Point-in-time store + learned model: present, fresh, verdict."""
    try:
        from src.data.history_store import HistoryStore
        man = HistoryStore().manifest()
    except Exception as exc:  # noqa: BLE001
        _rec(WARN, "history store", f"unreadable: {exc}")
        return
    if not man:
        _rec(WARN, "history store", "empty — run `python -m scripts.data_refresh` (backtests/ML need it)")
        return
    px = man.get("prices") or {}
    age = None
    if px.get("last"):
        try:
            age = (datetime.now(timezone.utc).date() - datetime.fromisoformat(px["last"]).date()).days
        except ValueError:
            pass
    _rec(OK if (age is not None and age <= 7) else WARN, "history prices",
         f"{px.get('symbols')} symbols to {px.get('last')}" + (f" ({age}d old)" if age is not None else ""))
    sn = man.get("snapshots") or {}
    _rec(OK if sn else WARN, "PIT snapshots",
         f"{sn.get('rows')} rows, {sn.get('dates')} month-ends to {sn.get('last')}" if sn else "none built")
    try:
        from src.ml import infer as ml_infer
        meta = ml_infer.latest_meta()
        w = config.get("scoring.pillars.learned", 0.0) or 0.0
        if not meta:
            _rec(OK, "learned pillar", "no model (pillar dark)")
        else:
            ok, why = ml_infer.usable()
            verdict = (meta.get("verdict") or {}).get("pass")
            if w > 0 and not ok:
                _rec(FAIL, "learned pillar", f"weight {w} but unusable: {why}")
            else:
                _rec(OK if (w == 0 or verdict) else WARN, "learned pillar",
                     f"model {meta.get('test_date')} · verdict {'PASS' if verdict else 'FAIL'} · weight {w}")
    except Exception as exc:  # noqa: BLE001
        _rec(WARN, "learned pillar", f"check failed: {exc}")


def main() -> int:
    for fn in (check_python, check_packages, check_config, check_auth, check_ollama,
               check_ibkr, check_db, check_history_store):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            _rec(FAIL, fn.__name__, f"check crashed: {exc}")
    width = max(len(w) for _, w, _ in _rows)
    print()
    for lvl, what, detail in _rows:
        print(f"  [{lvl}] {what.ljust(width)}  {detail}")
    fails = sum(1 for lvl, _, _ in _rows if lvl == FAIL)
    warns = sum(1 for lvl, _, _ in _rows if lvl == WARN)
    print()
    print(f"  {len(_rows)} checks: {fails} FAIL, {warns} WARN")
    print()
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
