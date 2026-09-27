"""One-command demo: a working board in a few minutes, on free data, in a sandbox.

    python -m scripts.demo                 # build, run a light (no-LLM-cost) pass, open the dashboard
    python -m scripts.demo --research      # also run LLM research on the top 3 (uses your llm.provider)
    python -m scripts.demo --reset         # wipe the demo workspace first
    python -m scripts.demo --no-dashboard  # build + run only
    python -m scripts.demo --host 0.0.0.0 --no-browser   # serve to other machines
                                           # (what the Docker image runs)

What it does:
  1. creates an isolated workspace (./demo by default, or --home) with its own
     config.yaml, database and data store — your real workspace is never touched;
  2. builds a small point-in-time store for ~35 US large caps (themes/demo.yaml):
     5 years of daily prices + SEC companyfacts + monthly fundamentals snapshots;
  3. runs the pipeline: light by default (quant screen + news, no LLM research, no
     cost), or deep with --research;
  4. opens the dashboard on port 8502 (8501 stays free for a real instance).

No broker is needed (IBKR is disabled in the demo). Ollama is used for the news stage
only if it is already running; otherwise that pillar is neutral and the run's health
says so.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import time
import webbrowser
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOME = ROOT / "demo"
PORT = 8502


def _say(msg: str) -> None:
    print(f"[demo] {msg}", flush=True)


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def demo_config(research: bool, use_ollama: bool) -> str:
    """The demo workspace's config.yaml — deep-merged OVER the shipped defaults."""
    start = date(date.today().year - 5, 1, 1).isoformat()
    return textwrap.dedent(f"""\
        # Written by scripts/demo.py — overrides deep-merged over the repo's config.yaml.
        theme: demo
        universe:
          source: curated
        screen:
          shortlist_size: 12          # names the news stage scans
          finalist_size: 3            # names that get LLM research (with --research)
          max_finalists_per_sector: 1
        claude:
          max_finalists: 3
          red_team: {"true" if research else "false"}
        local_llm:
          enabled: {"true" if use_ollama else "false"}
        ibkr:
          enabled: false              # no broker in the demo
        schedule:
          skip_when_market_closed: false
        demo:
          history_start: "{start}"
        """)


def run_step(title: str, args: list[str], env: dict[str, str]) -> float:
    _say(title)
    t0 = time.monotonic()
    proc = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=env)
    dt = time.monotonic() - t0
    if proc.returncode != 0:
        raise SystemExit(f"[demo] step failed ({' '.join(args)}), exit {proc.returncode}")
    _say(f"  done in {dt:.0f}s")
    return dt


def main() -> None:
    ap = argparse.ArgumentParser(description="One-command demo in an isolated workspace")
    ap.add_argument("--home", default=str(DEFAULT_HOME), help="demo workspace directory")
    ap.add_argument("--research", action="store_true",
                    help="also run LLM research on the top finalists (uses llm.provider)")
    ap.add_argument("--reset", action="store_true", help="empty the demo workspace first")
    ap.add_argument("--no-dashboard", action="store_true", help="don't launch the dashboard")
    ap.add_argument("--skip-data", action="store_true",
                    help="reuse the existing demo data store (re-run the pipeline only)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address the dashboard binds to (default 127.0.0.1, this machine "
                         "only; 0.0.0.0 serves every interface, e.g. inside a container)")
    ap.add_argument("--no-browser", action="store_true",
                    help="don't open a browser tab (headless machines, containers)")
    args = ap.parse_args()

    home = Path(args.home).expanduser().resolve()
    if home == ROOT.resolve():
        raise SystemExit("[demo] --home must not be the repo root (that is your real workspace)")
    if args.reset and home.exists():
        _say(f"resetting {home}")
        # Empty it rather than delete it: a container's /data is a mount point, which
        # cannot itself be removed.
        for child in home.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    home.mkdir(parents=True, exist_ok=True)

    # A refused local connection returns at once, so this is harmless where Ollama
    # cannot exist (a container): the news pillar is then neutral and health says so.
    use_ollama = _port_open("127.0.0.1", 11434)
    (home / "config.yaml").write_text(demo_config(args.research, use_ollama), encoding="utf-8")
    env = {**os.environ, "CLAUDE_TRADER_HOME": str(home), "PYTHONIOENCODING": "utf-8",
           "PYTHONUNBUFFERED": "1"}

    _say(f"workspace: {home}")
    _say(f"local LLM (Ollama) for the news stage: "
         f"{'found, will be used' if use_ollama else 'not running, news pillar will be neutral'}")

    if args.research:
        # Readiness check in a child so it sees the demo workspace's config.
        chk = subprocess.run(
            [sys.executable, "-c", "from src import llm; ok, d = llm.readiness(); "
             "print(('READY ' if ok else 'NOT READY ') + llm.provider_name() + ': ' + d)"],
            cwd=ROOT, env=env, capture_output=True, text=True)
        line = (chk.stdout or chk.stderr).strip().splitlines()[-1] if (chk.stdout or chk.stderr) else ""
        _say(f"LLM provider: {line}")
        if not line.startswith("READY"):
            raise SystemExit("[demo] --research needs a configured llm.provider — see docs/SETUP.md "
                             "(or run without --research for the free quant-only demo)")

    total = 0.0
    if not args.skip_data:
        start = date(date.today().year - 5, 1, 1).isoformat()
        total += run_step("building the data store (prices + SEC fundamentals, ~35 stocks)",
                          ["scripts.data_refresh", "--period", "5y", "--start", start], env)
    mode = "deep" if args.research else "light"
    total += run_step(f"running the pipeline ({mode})",
                      ["scripts.run_daily", "--mode", mode, "--force"], env)
    _say(f"demo ready in {total / 60:.1f} min")

    if args.no_dashboard:
        _say(f"launch later with:  CLAUDE_TRADER_HOME={home} streamlit run app/dashboard.py")
        return
    if _port_open("127.0.0.1", PORT):
        _say(f"port {PORT} is busy — is a demo dashboard already running? http://localhost:{PORT}")
        return
    _say(f"opening the dashboard at http://localhost:{PORT}  (Ctrl+C to stop)")
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "dashboard.py"),
           "--server.port", str(PORT), "--server.address", args.host,
           "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
    for _ in range(0 if args.no_browser else 60):
        if _port_open("127.0.0.1", PORT):
            webbrowser.open(f"http://localhost:{PORT}")
            break
        time.sleep(1)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
