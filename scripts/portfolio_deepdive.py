"""Entry point for the on-demand portfolio deep-dive.

Usage:
    python -m scripts.portfolio_deepdive
"""
from __future__ import annotations

from src import portfolio


def main() -> None:
    portfolio.run()


if __name__ == "__main__":
    main()
