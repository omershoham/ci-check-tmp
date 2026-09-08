"""Point-in-time history store: parquet files under `$CLAUDE_TRADER_HOME/data/history`.

    prices/daily.parquet             date, symbol, open, high, low, close, adj_close, volume
    fundamentals/facts.parquet       symbol, cik, concept, tag, unit, start, end, val, filed, form, frame
    fundamentals/snapshots.parquet   as_of, symbol, <derived metrics>   (materialized monthly)
    universe/membership.parquet      as_of, symbol, subsector, source
    manifest.json                    row counts + last refresh per table

PIT rule enforced at read time: `fundamentals_asof(as_of)` serves the latest snapshot
whose `as_of` <= the requested date, and snapshots are built only from facts with
`filed <= as_of`. Nothing in here knows what `.info` says today.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .. import config

_PRICE_COLS = ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]


class HistoryStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else (config.HOME / "data" / "history")
        self.prices_path = self.root / "prices" / "daily.parquet"
        self.facts_path = self.root / "fundamentals" / "facts.parquet"
        self.snap_path = self.root / "fundamentals" / "snapshots.parquet"
        self.universe_path = self.root / "universe" / "membership.parquet"
        self.manifest_path = self.root / "manifest.json"

    # ------------------------------------------------------------------ io
    @staticmethod
    def _read(path: Path, columns: list[str] | None = None,
              filters: list | None = None) -> pd.DataFrame:
        """Read a parquet table, pushing column projection and row filters down to
        pyarrow so unread columns are never materialised."""
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path, columns=columns, filters=filters)

    @staticmethod
    def _write(path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def _touch_manifest(self, table: str, n_rows: int, extra: dict | None = None) -> None:
        man = {}
        if self.manifest_path.exists():
            try:
                man = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                man = {}
        man[table] = {"rows": int(n_rows),
                      "refreshed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      **(extra or {})}
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(man, indent=2), encoding="utf-8")

    def manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    # -------------------------------------------------------------- prices
    def upsert_prices(self, df: pd.DataFrame) -> int:
        """Merge a long price frame (same columns as _PRICE_COLS); newer rows win."""
        df = df[_PRICE_COLS].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        old = self._read(self.prices_path)
        if not old.empty:
            old["date"] = pd.to_datetime(old["date"])
            df = pd.concat([old, df], ignore_index=True)
        df = (df.sort_values(["symbol", "date"])
                .drop_duplicates(["symbol", "date"], keep="last")
                .reset_index(drop=True))
        self._write(self.prices_path, df)
        self._touch_manifest("prices", len(df),
                             {"symbols": int(df["symbol"].nunique()),
                              "first": str(df["date"].min().date()) if len(df) else None,
                              "last": str(df["date"].max().date()) if len(df) else None})
        return len(df)

    def prices(self, symbols: list[str] | None = None, start: date | None = None,
               end: date | None = None, columns: list[str] | None = None) -> pd.DataFrame:
        """Long price frame. `columns` projects (date/symbol are always included);
        symbol and date bounds are pushed into the parquet read rather than applied
        after materialising the whole table."""
        cols = None
        if columns:
            cols = list(dict.fromkeys(["date", "symbol", *columns]))
        filters: list = []
        if symbols:
            filters.append(("symbol", "in", set(symbols)))
        if start:
            filters.append(("date", ">=", pd.Timestamp(start)))
        if end:
            filters.append(("date", "<=", pd.Timestamp(end)))
        df = self._read(self.prices_path, columns=cols, filters=filters or None)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values(["symbol", "date"]).reset_index(drop=True)

    def last_date(self) -> pd.Timestamp | None:
        """Most recent price date in the store.

        Served from the manifest so callers that only need "how current is the store?"
        do not read (or pivot) the price table. Falls back to a read when the manifest
        is missing or stale-shaped.
        """
        last = (self.manifest().get("prices") or {}).get("last")
        if last:
            try:
                return pd.Timestamp(last)
            except ValueError:
                pass
        df = self._read(self.prices_path)
        if df.empty:
            return None
        return pd.Timestamp(pd.to_datetime(df["date"]).max())

    def close_matrix(self, symbols: list[str] | None = None, adjusted: bool = True,
                     start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """Wide date x symbol frame of (adjusted) closes — the harness's workhorse."""
        col = "adj_close" if adjusted else "close"
        df = self.prices(symbols, start, end, columns=[col])
        if df.empty:
            return df
        return df.pivot(index="date", columns="symbol", values=col).sort_index()

    # --------------------------------------------------------------- facts
    def upsert_facts(self, df: pd.DataFrame) -> int:
        old = self._read(self.facts_path)
        if not old.empty:
            # Replace each refreshed symbol wholesale (its full fact history came back).
            old = old[~old["symbol"].isin(set(df["symbol"]))]
            df = pd.concat([old, df], ignore_index=True)
        df = df.reset_index(drop=True)
        self._write(self.facts_path, df)
        self._touch_manifest("facts", len(df), {"symbols": int(df["symbol"].nunique())})
        return len(df)

    def facts(self, symbols: list[str] | None = None) -> pd.DataFrame:
        df = self._read(self.facts_path)
        if df.empty:
            return df
        if symbols:
            df = df[df["symbol"].isin(set(symbols))]
        for c in ("start", "end", "filed"):
            df[c] = pd.to_datetime(df[c])
        return df.reset_index(drop=True)

    # ----------------------------------------------------------- snapshots
    def write_snapshots(self, df: pd.DataFrame) -> int:
        df = df.copy()
        df["as_of"] = pd.to_datetime(df["as_of"]).dt.normalize()
        old = self._read(self.snap_path)
        if not old.empty:
            old["as_of"] = pd.to_datetime(old["as_of"])
            df = pd.concat([old, df], ignore_index=True)
        df = (df.sort_values(["symbol", "as_of"])
                .drop_duplicates(["symbol", "as_of"], keep="last").reset_index(drop=True))
        self._write(self.snap_path, df)
        self._touch_manifest("snapshots", len(df),
                             {"dates": int(df["as_of"].nunique()),
                              "first": str(df["as_of"].min().date()) if len(df) else None,
                              "last": str(df["as_of"].max().date()) if len(df) else None})
        return len(df)

    def snapshots(self) -> pd.DataFrame:
        df = self._read(self.snap_path)
        if not df.empty:
            df["as_of"] = pd.to_datetime(df["as_of"])
        return df

    def fundamentals_asof(self, symbols: list[str], as_of: date) -> pd.DataFrame:
        """Latest snapshot at or before `as_of` per symbol (PIT by construction: the
        snapshot for date D only used facts filed <= D). Indexed by symbol."""
        df = self.snapshots()
        if df.empty:
            return pd.DataFrame()
        df = df[(df["as_of"] <= pd.Timestamp(as_of)) & (df["symbol"].isin(set(symbols)))]
        if df.empty:
            return pd.DataFrame()
        # Serve only reasonably fresh snapshots: a symbol whose last snapshot is >120
        # days old (stopped filing / left the store) must not carry stale ratios forward.
        latest = df.sort_values("as_of").groupby("symbol").tail(1)
        fresh = latest[latest["as_of"] >= pd.Timestamp(as_of) - pd.Timedelta(days=120)]
        return fresh.set_index("symbol")

    # ------------------------------------------------------------ universe
    def record_universe(self, as_of: date, members: list[dict[str, str]],
                        source: str = "curated") -> int:
        """Append today's membership (dedup per date/source). Called by every pipeline
        run so, from now on, 'what was in the universe on date D' is a fact, not a
        reconstruction — the one survivorship fix that costs nothing."""
        rows = pd.DataFrame([{"as_of": pd.Timestamp(as_of).normalize(), "symbol": m["symbol"],
                              "subsector": m.get("subsector", ""), "source": source}
                             for m in members])
        old = self._read(self.universe_path)
        if not old.empty:
            old["as_of"] = pd.to_datetime(old["as_of"])
            old = old[~((old["as_of"] == pd.Timestamp(as_of).normalize()) & (old["source"] == source))]
            rows = pd.concat([old, rows], ignore_index=True)
        rows = rows.sort_values(["as_of", "source", "symbol"]).reset_index(drop=True)
        self._write(self.universe_path, rows)
        self._touch_manifest("universe", len(rows))
        return len(rows)

    def universe_asof(self, as_of: date, source: str = "curated") -> list[str]:
        """Membership on the latest recorded date <= as_of. Falls back to the current
        curated config when nothing was recorded yet (and says so via `universe_source`)."""
        df = self._read(self.universe_path)
        if not df.empty:
            df["as_of"] = pd.to_datetime(df["as_of"])
            df = df[(df["source"] == source) & (df["as_of"] <= pd.Timestamp(as_of))]
            if not df.empty:
                last = df["as_of"].max()
                return sorted(df[df["as_of"] == last]["symbol"].tolist())
        if source == "curated":
            from .. import universe
            return universe.symbols()
        return []

    def universe_source(self, as_of: date, source: str = "curated") -> str:
        df = self._read(self.universe_path)
        if df.empty:
            return "config (no recorded snapshots yet — hindsight universe)"
        df["as_of"] = pd.to_datetime(df["as_of"])
        df = df[(df["source"] == source) & (df["as_of"] <= pd.Timestamp(as_of))]
        return f"recorded snapshot {df['as_of'].max().date()}" if not df.empty else \
            "config (predates recorded snapshots — hindsight universe)"
