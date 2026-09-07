"""The `MarketData` seam.

Two implementations:
  * `data.history_store.HistoryStore` — point-in-time parquet store under the
    workspace (prices, SEC facts, monthly fundamentals snapshots, universe membership).
    Everything a backtest or a model may read comes from here and ONLY here.
  * `data.yfinance_live.YFinanceLive` — today's numbers for the live screen
    (`fundamentals.fetch_many`, batch price downloads). Current-state, never for training.

The protocol exists so a paid, survivorship-free source (Sharadar/Norgate) can be
dropped in later without touching `fundamentals.py`/`technical.py`/the harness.
"""
from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd


class MarketData(Protocol):
    def prices(self, symbols: list[str], start: date | None = None,
               end: date | None = None) -> pd.DataFrame:
        """Long frame: date, symbol, open, high, low, close, adj_close, volume."""

    def fundamentals_asof(self, symbols: list[str], as_of: date) -> pd.DataFrame:
        """Wide frame indexed by symbol with the derived metrics as they were KNOWABLE
        on `as_of` (every underlying fact filed on or before that date)."""

    def universe_asof(self, as_of: date, source: str = "curated") -> list[str]:
        """Symbols that were in the universe on `as_of` (curated snapshots or SIC-derived)."""


def history():
    """Default point-in-time store for the current workspace."""
    from .history_store import HistoryStore
    return HistoryStore()
