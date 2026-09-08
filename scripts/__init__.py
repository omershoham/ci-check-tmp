"""CLI entry points (`python -m scripts.<name>`).

Importing this package is the one thing every script does first, so it is the single
place for console setup. Reports print em dashes, arrows and health-badge emoji; a
Windows console defaults to cp1252 and raises UnicodeEncodeError mid-report, which
hides the rest of the output and returns non-zero from a healthy run. Three scripts
hit this independently (doctor, backtest, ...) before it was fixed here once.
"""
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
