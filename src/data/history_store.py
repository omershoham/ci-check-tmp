"""Point-in-time history store: hive-partitioned parquet under `$CLAUDE_TRADER_HOME/data/history`.

    prices/daily/year=YYYY/part-NNNNN.parquet
        date, symbol, open, high, low, close, adj_close, volume
    fundamentals/facts/filed_quarter=YYYYqQ/part-NNNNN.parquet
        symbol, cik, concept, tag, unit, start, end, val, filed, form, frame, fy, fp
    fundamentals/snapshots/year=YYYY/part-NNNNN.parquet
        as_of, symbol, <derived metrics>            (materialized monthly)
    universe/membership.parquet
        as_of, symbol, subsector, source            (one row per name per run — stays one file)
    manifest.json
        row counts + last refresh per table

WHY PARTITIONED. The old layout kept one file per table and every write read it,
concatenated, sorted and rewrote it whole. At 105 names that is 240k price rows and
nobody notices. At the 1,500-2,500 names this store is being sized for it is 12-15M
rows, ~350MB of parquet and 2-3GB of peak RAM — paid to add the 2,500 rows of one
trading day. Partitioning turns the daily write into "append one file into one year
directory, then compact that directory", and turns every read into a directory prune
before a single byte of data is touched.

`facts` is keyed by `filed_quarter` (e.g. 2024q1) rather than by year because that is
the SEC FSDS ZIP's own key: one ZIP maps to exactly one directory, so re-ingesting a
quarter is idempotent by directory listing instead of by re-reading the table.

Sorting `(symbol, date)` inside each partition is not cosmetic — it is what makes the
per-row-group `symbol` statistics selective, so a one-symbol read skips most row
groups instead of decompressing the year.

The old single-file store is migrated in place on first access (`migrate_layout`); the
originals are kept as `*.parquet.migrated` because a bad migration must not cost a
ten-year re-download.

PIT rule enforced at read time: `fundamentals_asof(as_of)` serves the latest snapshot
whose `as_of` <= the requested date, and snapshots are built only from facts with
`filed <= as_of`. Nothing in here knows what `.info` says today.
"""
from __future__ import annotations

import json
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .. import config

_PRICE_COLS = ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]

# Explicit schemas for the two append-heavy tables. Without them a partition written
# from a frame whose `frame`/`fy` column happened to be all-None lands as parquet
# `null` and the next dataset read refuses to unify it with a `string` sibling — the
# failure mode that makes a partitioned store look corrupt weeks after the write.
_PRICE_SCHEMA = pa.schema([
    ("date", pa.timestamp("ns")), ("symbol", pa.string()),
    ("open", pa.float64()), ("high", pa.float64()), ("low", pa.float64()),
    ("close", pa.float64()), ("adj_close", pa.float64()), ("volume", pa.float64()),
])
_FACTS_COLS = ["symbol", "cik", "concept", "tag", "unit", "start", "end", "val",
               "filed", "form", "frame", "fy", "fp"]
_FACTS_STR = ("symbol", "cik", "concept", "tag", "unit", "form", "frame", "fp")
_FACTS_SCHEMA = pa.schema([
    ("symbol", pa.string()), ("cik", pa.string()), ("concept", pa.string()),
    ("tag", pa.string()), ("unit", pa.string()),
    ("start", pa.timestamp("ns")), ("end", pa.timestamp("ns")),
    ("val", pa.float64()), ("filed", pa.timestamp("ns")),
    ("form", pa.string()), ("frame", pa.string()),
    ("fy", pa.float64()), ("fp", pa.string()),
])

_YEAR_PART = ds.partitioning(pa.schema([("year", pa.int32())]), flavor="hive")
_QUARTER_PART = ds.partitioning(pa.schema([("filed_quarter", pa.string())]), flavor="hive")
_PARTITION_COLS = {"year", "filed_quarter"}

# Narrow rows (prices) tolerate big row groups; the wide tables are ~40 columns, where
# a 200k-row group is a 60MB decompression unit for a one-symbol read.
_RG_PRICES = 200_000
_RG_WIDE = 50_000
_PART_NAME = "part-{n:05d}.parquet"
_SOURCE_MARK = "_source_{source}.ok"


def _part_paths(directory: Path) -> list[Path]:
    """Part files of one partition, in write order (the ordinal is the tiebreaker
    `compact_prices` uses to decide which duplicate is 'later')."""
    return sorted(directory.glob("part-*.parquet")) if directory.exists() else []


def _quarter_label(ts: pd.Timestamp) -> str:
    return f"{ts.year}q{(ts.month - 1) // 3 + 1}"


def _filed_quarters(filed: pd.Series) -> pd.Series:
    """`filed` -> the FSDS quarter label that owns it. A row with no filing date is
    not knowable at any point in time; it is parked in `unknown` rather than guessed."""
    y = filed.dt.year.astype("Int64").astype("string")
    q = ((filed.dt.month - 1) // 3 + 1).astype("Int64").astype("string")
    return (y + "q" + q).where(filed.notna(), "unknown").astype(str)


class HistoryStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else (config.HOME / "data" / "history")
        self.prices_dir = self.root / "prices" / "daily"
        self.facts_dir = self.root / "fundamentals" / "facts"
        self.snap_dir = self.root / "fundamentals" / "snapshots"
        # Pre-partition single-file layout; read once by migrate_layout(), never after.
        self.prices_path = self.root / "prices" / "daily.parquet"
        self.facts_path = self.root / "fundamentals" / "facts.parquet"
        self.snap_path = self.root / "fundamentals" / "snapshots.parquet"
        self.universe_path = self.root / "universe" / "membership.parquet"
        self.manifest_path = self.root / "manifest.json"
        self._layout_checked = False

    # ------------------------------------------------------------------ io
    @staticmethod
    def _read(path: Path, columns: list[str] | None = None,
              filters: list | None = None) -> pd.DataFrame:
        """Read a single parquet file (universe membership, and the legacy tables
        during migration), pushing projection and row filters down to pyarrow."""
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path, columns=columns, filters=filters)

    @staticmethod
    def _write(path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    @staticmethod
    def _read_parts(directory: Path, partitioning, columns: list[str] | None = None,
                    filters: list | None = None) -> pd.DataFrame:
        """Read a partitioned directory (or one leaf of it, with `partitioning=None`).

        Column projection and the row filter go to the dataset scanner, so a filter on
        the partition key prunes whole directories and a filter on `symbol` skips row
        groups by their statistics. The partition columns themselves are never
        materialised — callers want the same frame the single file used to give them.

        Row order is fragment order (directories sorted, then part ordinal). That is
        load-bearing, not incidental: `_Tab.from_df` sorts facts by (end, filed) with a
        stable sort, so two facts filed on the same day are separated only by the order
        they come back in. `to_table` is ordered; `to_batches` under threads is not."""
        if not directory.exists() or not any(directory.rglob("part-*.parquet")):
            return pd.DataFrame()
        dataset = ds.dataset(directory, format="parquet", partitioning=partitioning)
        names = [n for n in dataset.schema.names if n not in _PARTITION_COLS]
        cols = [c for c in columns if c in names] if columns else names
        expr = pq.filters_to_expression(filters) if filters else None
        return dataset.to_table(columns=cols, filter=expr).to_pandas()

    @staticmethod
    def _write_part(directory: Path, table: pa.Table, replace: bool,
                    row_group_size: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        existing = _part_paths(directory)
        if replace:
            for p in existing:
                p.unlink()
            existing = []
        n = 0 if not existing else int(existing[-1].stem.split("-")[1]) + 1
        pq.write_table(table, directory / _PART_NAME.format(n=n), compression="zstd",
                       use_dictionary=True, row_group_size=row_group_size)

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

    # ----------------------------------------------------------- migration
    def _ensure_layout(self) -> None:
        """Migrate a pre-partition store the first time anything touches it, so an
        existing checkout keeps working without a manual step."""
        if self._layout_checked:
            return
        self._layout_checked = True
        if any(p.exists() for p in (self.prices_path, self.facts_path, self.snap_path)):
            self.migrate_layout()

    def migrate_layout(self, log=None) -> dict[str, int]:
        """Single-file store -> partitioned directories, in place and idempotent.

        The originals are renamed to `*.parquet.migrated` instead of deleted: a store
        is ten years of downloads and a migration bug must be recoverable locally."""
        self._layout_checked = True
        done: dict[str, int] = {}
        if self.prices_path.exists():
            df = self._read(self.prices_path)
            shutil.rmtree(self.prices_dir, ignore_errors=True)
            done["prices"] = self.append_prices(df) if not df.empty else 0
            self.prices_path.rename(self.prices_path.with_suffix(".parquet.migrated"))
        if self.facts_path.exists():
            df = self._read(self.facts_path)
            shutil.rmtree(self.facts_dir, ignore_errors=True)
            done["facts"] = self._write_facts_partitions(self._normalize_facts(df)) if not df.empty else 0
            self.refresh_facts_manifest()
            self.facts_path.rename(self.facts_path.with_suffix(".parquet.migrated"))
        if self.snap_path.exists():
            df = self._read(self.snap_path)
            shutil.rmtree(self.snap_dir, ignore_errors=True)
            done["snapshots"] = self.write_snapshots(df) if not df.empty else 0
            self.snap_path.rename(self.snap_path.with_suffix(".parquet.migrated"))
        if log and done:
            log(f"  migrated to partitioned layout: {done}")
        return done

    # -------------------------------------------------------------- prices
    @staticmethod
    def _normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
        df = df[_PRICE_COLS].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        return df

    def _price_years(self) -> list[int]:
        if not self.prices_dir.exists():
            return []
        return sorted(int(p.name.split("=", 1)[1]) for p in self.prices_dir.glob("year=*")
                      if _part_paths(p))

    def _price_stats(self) -> dict[str, Any]:
        """Row count and date range from parquet footers only — no data pages are read,
        so refreshing the manifest costs the same at 12M rows as at 240k."""
        rows, lo, hi = 0, None, None
        for p in sorted(self.prices_dir.rglob("part-*.parquet")) if self.prices_dir.exists() else []:
            md = pq.read_metadata(p)
            rows += md.num_rows
            try:
                i = md.schema.names.index("date")
            except ValueError:
                continue
            for rg in range(md.num_row_groups):
                st = md.row_group(rg).column(i).statistics
                if st is None or not st.has_min_max:
                    continue
                lo = st.min if lo is None else min(lo, st.min)
                hi = st.max if hi is None else max(hi, st.max)
        return {"rows": rows, "symbols": len(self.symbols_with_prices()),
                "first": str(pd.Timestamp(lo).date()) if lo is not None else None,
                "last": str(pd.Timestamp(hi).date()) if hi is not None else None}

    def _touch_prices_manifest(self) -> None:
        st = self._price_stats()
        self._touch_manifest("prices", st["rows"], {"symbols": st["symbols"],
                                                    "first": st["first"], "last": st["last"]})

    def append_prices(self, df: pd.DataFrame) -> int:
        """Append rows into the year partitions they touch. Reads no existing data.

        Duplicates are deliberately NOT resolved here — that is `compact_prices`'s job.
        The daily refresh is append-then-compact-the-touched-years, which is what
        removes the full-table rewrite from the hot path."""
        self._ensure_layout()
        df = self._normalize_prices(df)
        if df.empty:
            return 0
        n = 0
        for year, g in df.groupby(df["date"].dt.year, sort=True):
            g = g.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
            self._write_part(self.prices_dir / f"year={int(year)}",
                             pa.Table.from_pandas(g, schema=_PRICE_SCHEMA, preserve_index=False),
                             replace=False, row_group_size=_RG_PRICES)
            n += len(g)
        self._touch_prices_manifest()
        return n

    def compact_prices(self, years: list[int] | None = None) -> int:
        """Rewrite whole year partitions, keeping the LAST row per (symbol, date).

        "Last" means last written: part files carry an ordinal and are read back in
        that order, so a later append beats an earlier one — exactly what
        `upsert_prices` promised when it concatenated old-then-new and kept `last`.
        An `ingested_at` column would say the same thing at the price of a column in
        every one of 12M rows; the file ordinal is free."""
        self._ensure_layout()
        total = 0
        for y in sorted({int(x) for x in (years if years is not None else self._price_years())}):
            d = self.prices_dir / f"year={y}"
            parts = _part_paths(d)
            if not parts:
                continue
            frames = []
            for seq, p in enumerate(parts):
                f = pq.read_table(p).to_pandas()
                f["_seq"] = seq
                frames.append(f)
            g = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            before = len(g)
            g = (g.sort_values(["symbol", "date", "_seq"], kind="stable")
                   .drop_duplicates(["symbol", "date"], keep="last")
                   .drop(columns="_seq").reset_index(drop=True))
            total += len(g)
            if len(parts) == 1 and len(g) == before:
                continue                      # already one sorted, deduped file
            self._write_part(d, pa.Table.from_pandas(g, schema=_PRICE_SCHEMA, preserve_index=False),
                             replace=True, row_group_size=_RG_PRICES)
        self._touch_prices_manifest()
        return total

    def upsert_prices(self, df: pd.DataFrame) -> int:
        """Merge a long price frame (same columns as _PRICE_COLS); newer rows win.

        Semantics and return value (the table's total row count) are exactly what they
        were before partitioning; only the cost changed — append plus a compaction of
        the years the frame actually touched, never a rewrite of the whole store."""
        self._ensure_layout()
        d = self._normalize_prices(df)
        years = sorted({int(y) for y in d["date"].dt.year.unique()}) if not d.empty else []
        self.append_prices(d)
        self.compact_prices(years)
        return int(self._price_stats()["rows"])

    def prices(self, symbols: list[str] | None = None, start: date | None = None,
               end: date | None = None, columns: list[str] | None = None) -> pd.DataFrame:
        """Long price frame. `columns` projects (date/symbol are always included);
        symbol and date bounds are pushed into the scan, and the date bounds are
        restated as `year` bounds so whole partitions are skipped before any read."""
        self._ensure_layout()
        cols = None
        if columns:
            cols = list(dict.fromkeys(["date", "symbol", *columns]))
        filters: list = []
        if symbols:
            filters.append(("symbol", "in", sorted(set(symbols))))
        if start:
            filters += [("year", ">=", pd.Timestamp(start).year),
                        ("date", ">=", pd.Timestamp(start))]
        if end:
            filters += [("year", "<=", pd.Timestamp(end).year),
                        ("date", "<=", pd.Timestamp(end))]
        df = self._read_parts(self.prices_dir, _YEAR_PART, columns=cols, filters=filters or None)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)

    def symbols_with_prices(self) -> list[str]:
        """Distinct symbols in the price store.

        One dictionary-encoded column is read and nothing else. The callers that want
        this used to build the entire date x symbol pivot just to look at
        `closes.columns`, which at 2,500 names is a 6M-row pivot to answer a set
        membership question."""
        self._ensure_layout()
        if not self.prices_dir.exists() or not any(self.prices_dir.rglob("part-*.parquet")):
            return []
        dataset = ds.dataset(self.prices_dir, format="parquet", partitioning=_YEAR_PART)
        col = dataset.to_table(columns=["symbol"]).column("symbol")
        return sorted(str(v) for v in col.combine_chunks().unique().to_pylist() if v is not None)

    def last_date(self) -> pd.Timestamp | None:
        """Most recent price date in the store.

        Served from the manifest so callers that only need "how current is the store?"
        do not read (or pivot) the price table. Falls back to the parquet footers when
        the manifest is missing or stale-shaped.
        """
        last = (self.manifest().get("prices") or {}).get("last")
        if last:
            try:
                return pd.Timestamp(last)
            except ValueError:
                pass
        self._ensure_layout()
        hi = self._price_stats()["last"]
        return pd.Timestamp(hi) if hi else None

    def close_matrix(self, symbols: list[str] | None = None, adjusted: bool = True,
                     start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """Wide date x symbol frame of (adjusted) closes — the harness's workhorse.

        PASS `symbols`. `symbols=None` means "every symbol in the store", which makes
        the frame's DATE INDEX a function of unrelated downloads — and the backtest
        engine takes its rebalance calendar from that index. Analytic callers name what
        they score plus the benchmark columns they read; None is for exploration."""
        col = "adj_close" if adjusted else "close"
        df = self.prices(symbols, start, end, columns=[col])
        if df.empty:
            return df
        return df.pivot(index="date", columns="symbol", values=col).sort_index()

    # --------------------------------------------------------------- facts
    @staticmethod
    def _normalize_facts(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce an incoming fact frame onto the fixed schema. The two ingest paths
        disagree on types for free-text columns (companyfacts gives `fy` as a number,
        FSDS as a string), and a partitioned store cannot absorb that the way a single
        concat-and-rewrite file silently did."""
        out = pd.DataFrame({c: (df[c] if c in df.columns else None) for c in _FACTS_COLS},
                           index=df.index)
        for c in ("start", "end", "filed"):
            out[c] = pd.to_datetime(out[c], errors="coerce")
        for c in ("val", "fy"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        for c in _FACTS_STR:
            if out[c].dtype != object and str(out[c].dtype) not in ("str", "string"):
                out[c] = out[c].astype("string")
        return out.reset_index(drop=True)

    def _write_facts_partitions(self, d: pd.DataFrame) -> int:
        """Write already-normalised facts into their own `filed_quarter` directories,
        appending. Rows are routed by their OWN filed quarter, never by a label the
        caller passed: a row in the wrong directory is invisible to a pruned read."""
        n = 0
        for q, g in d.assign(_q=_filed_quarters(d["filed"])).groupby("_q", sort=True):
            g = g.drop(columns="_q")
            self._write_part(self.facts_dir / f"filed_quarter={q}",
                             pa.Table.from_pandas(g, schema=_FACTS_SCHEMA, preserve_index=False),
                             replace=False, row_group_size=_RG_WIDE)
            n += len(g)
        return n

    def facts_quarters(self, source: str | None = None) -> list[str]:
        """Quarter labels present in the store.

        `source="fsds"` narrows it to the quarters actually ingested from a DERA ZIP.
        The distinction matters: a `filed_quarter=` directory exists as soon as ANY
        path files a row into it, the per-CIK companyfacts path included, so bare
        directory presence would tell a bulk backfill it had already run when it had
        not — and the whole point of keying on the ZIP's own label is that re-running
        it is decided by a directory listing rather than by re-reading the table."""
        self._ensure_layout()
        if not self.facts_dir.exists():
            return []
        out = []
        for p in self.facts_dir.glob("filed_quarter=*"):
            if not _part_paths(p):
                continue
            if source and not (p / _SOURCE_MARK.format(source=source)).exists():
                continue
            out.append(p.name.split("=", 1)[1])
        return sorted(out)

    def _facts_stats(self) -> dict[str, Any]:
        rows = sum(pq.read_metadata(p).num_rows
                   for p in sorted(self.facts_dir.rglob("part-*.parquet"))) \
            if self.facts_dir.exists() else 0
        syms = 0
        if rows:
            dataset = ds.dataset(self.facts_dir, format="parquet", partitioning=_QUARTER_PART)
            syms = len(dataset.to_table(columns=["symbol"]).column("symbol")
                       .combine_chunks().unique())
        return {"rows": rows, "symbols": syms, "quarters": len(self.facts_quarters())}

    def refresh_facts_manifest(self) -> int:
        st = self._facts_stats()
        self._touch_manifest("facts", st["rows"],
                             {"symbols": st["symbols"], "quarters": st["quarters"]})
        return int(st["rows"])

    def append_facts_quarter(self, quarter: str, df: pd.DataFrame, replace: bool = False,
                             source: str | None = None, touch_manifest: bool = True) -> int:
        """Append one FSDS quarter's facts. Append-only by design: a published ZIP is
        immutable, so ingesting it is a directory write, not a table rewrite.

        `source` stamps a marker file so `facts_quarters(source=...)` can tell which
        quarters that path has covered. `replace=True` re-ingests a quarter whose PARSE
        changed (a tag added to FLOW_TAGS, say) — it EMPTIES the directory first, so
        anything else that had filed rows into that quarter goes with it and has to be
        re-run. `touch_manifest=False` lets a 40-quarter backfill pay the manifest's
        distinct-symbol scan once at the end instead of forty times."""
        self._ensure_layout()
        d = self._normalize_facts(df)
        if d.empty:
            return 0
        qdir = self.facts_dir / f"filed_quarter={quarter}"
        if replace:
            shutil.rmtree(qdir, ignore_errors=True)
        n = self._write_facts_partitions(d)
        if source and qdir.exists():
            # `_`-prefixed, so pyarrow's dataset discovery skips it (ignore_prefixes).
            (qdir / _SOURCE_MARK.format(source=source)).write_text(
                datetime.now(timezone.utc).isoformat(timespec="seconds"), encoding="utf-8")
        if touch_manifest:
            self.refresh_facts_manifest()
        return n

    def upsert_facts(self, df: pd.DataFrame) -> int:
        """Replace each refreshed symbol wholesale (its full fact history came back).

        Same semantics as before; the difference is that it now walks one
        `filed_quarter` directory at a time, so peak memory is one quarter rather than
        the whole fact table."""
        self._ensure_layout()
        new = self._normalize_facts(df)
        new["_q"] = _filed_quarters(new["filed"])
        syms = set(new["symbol"].dropna())
        for q in sorted(set(self.facts_quarters()) | set(new["_q"].unique())):
            d = self.facts_dir / f"filed_quarter={q}"
            old = self._read_parts(d, None)
            if not old.empty:
                old = old[~old["symbol"].isin(syms)]
            add = new[new["_q"] == q].drop(columns="_q")
            if old.empty and add.empty:
                shutil.rmtree(d, ignore_errors=True)     # every row in it was replaced
                continue
            # old-then-new, matching the concat order the single-file path used, so a
            # restatement filed the same day as its original still sorts after it.
            g = pd.concat([old, add], ignore_index=True) if not old.empty else add.reset_index(drop=True)
            self._write_part(d, pa.Table.from_pandas(g, schema=_FACTS_SCHEMA, preserve_index=False),
                             replace=True, row_group_size=_RG_WIDE)
        return self.refresh_facts_manifest()

    def facts(self, symbols: list[str] | None = None, start: date | None = None,
              end: date | None = None) -> pd.DataFrame:
        """Long fact frame. `start`/`end` bound the FILING date (the only date a PIT
        consumer may bound on) and are restated as `filed_quarter` bounds so whole
        quarters are pruned; the labels sort chronologically as strings."""
        self._ensure_layout()
        filters: list = []
        if symbols:
            filters.append(("symbol", "in", sorted(set(symbols))))
        if start:
            filters += [("filed_quarter", ">=", _quarter_label(pd.Timestamp(start))),
                        ("filed", ">=", pd.Timestamp(start))]
        if end:
            filters += [("filed_quarter", "<=", _quarter_label(pd.Timestamp(end))),
                        ("filed", "<=", pd.Timestamp(end))]
        df = self._read_parts(self.facts_dir, _QUARTER_PART, filters=filters or None)
        if df.empty:
            return df
        for c in ("start", "end", "filed"):
            df[c] = pd.to_datetime(df[c])
        return df.reset_index(drop=True)

    def shares_asof(self, symbols: list[str] | None, as_of: date) -> pd.Series:
        """Shares outstanding knowable on `as_of`, per symbol.

        The PIT rule in two clauses: `filed <= as_of` (it had to be public) and
        `end <= as_of` (the count had to have been measured). Among what survives,
        the latest filing wins. Vectorised over the whole cross-section on purpose —
        the broad-universe liquidity floor needs a market cap for thousands of names
        on every rebalance date, and a per-symbol loop there is minutes, not
        milliseconds."""
        self._ensure_layout()
        ts = pd.Timestamp(as_of)
        filters = [("concept", "==", "sharesOutstanding"),
                   ("filed_quarter", "<=", _quarter_label(ts)),
                   ("filed", "<=", ts), ("end", "<=", ts)]
        if symbols:
            filters.append(("symbol", "in", sorted(set(symbols))))
        df = self._read_parts(self.facts_dir, _QUARTER_PART,
                              columns=["symbol", "end", "val", "filed"], filters=filters)
        if df.empty:
            return pd.Series(dtype=float, name="sharesOutstanding")
        last = (df.sort_values(["symbol", "filed", "end"], kind="stable")
                  .groupby("symbol", sort=True).tail(1))
        return last.set_index("symbol")["val"].astype(float).rename("sharesOutstanding")

    # ----------------------------------------------------------- snapshots
    def _snapshot_years(self) -> list[int]:
        if not self.snap_dir.exists():
            return []
        return sorted(int(p.name.split("=", 1)[1]) for p in self.snap_dir.glob("year=*")
                      if _part_paths(p))

    def write_snapshots(self, df: pd.DataFrame) -> int:
        """Upsert monthly snapshots by year partition; newer rows per (symbol, as_of) win."""
        self._ensure_layout()
        d = df.copy()
        if d.empty:
            return int(self._touch_snapshots_manifest())
        d["as_of"] = pd.to_datetime(d["as_of"]).dt.normalize()
        touched = sorted({int(y) for y in d["as_of"].dt.year.unique()})
        existing = self._snapshot_years()
        # `derive()` gains metrics over time (Phase 4 adds three). A year written with
        # the new column set cannot sit next to a year without it, so a column change
        # rewrites every partition — cheap, because snapshots are monthly and small.
        old_cols = set(ds.dataset(self.snap_dir, format="parquet",
                                  partitioning=_YEAR_PART).schema.names) - _PARTITION_COLS \
            if existing else set(d.columns)
        years = sorted(set(touched) | set(existing)) if old_cols != set(d.columns) else touched
        frames: dict[int, pd.DataFrame] = {}
        for y in years:
            old = self._read_parts(self.snap_dir / f"year={y}", None)
            add = d[d["as_of"].dt.year == y]
            g = pd.concat([old, add], ignore_index=True) if not old.empty else add
            frames[y] = (g.sort_values(["symbol", "as_of"], kind="stable")
                          .drop_duplicates(["symbol", "as_of"], keep="last")
                          .reset_index(drop=True))
        if frames:
            # One arrow schema for every partition: a metric that is all-null in one
            # year would otherwise be parquet `null` there and `double` elsewhere, and
            # the dataset would refuse to unify them on the next read.
            schema = pa.Table.from_pandas(pd.concat(frames.values(), ignore_index=True),
                                          preserve_index=False).schema
            for y, g in frames.items():
                self._write_part(self.snap_dir / f"year={y}",
                                 pa.Table.from_pandas(g, schema=schema, preserve_index=False),
                                 replace=True, row_group_size=_RG_WIDE)
        return int(self._touch_snapshots_manifest())

    def _touch_snapshots_manifest(self) -> int:
        """One `as_of` column read serves all three manifest fields — the table is
        monthly, so it is thousands of rows, not millions."""
        col = self._read_parts(self.snap_dir, _YEAR_PART, columns=["as_of"])
        rows = len(col)
        extra: dict[str, Any] = {"dates": 0, "first": None, "last": None}
        if rows:
            a = pd.to_datetime(col["as_of"])
            extra = {"dates": int(a.nunique()), "first": str(a.min().date()),
                     "last": str(a.max().date())}
        self._touch_manifest("snapshots", rows, extra)
        return rows

    def snapshots(self, symbols: list[str] | None = None, start: date | None = None,
                  end: date | None = None) -> pd.DataFrame:
        self._ensure_layout()
        filters: list = []
        if symbols:
            filters.append(("symbol", "in", sorted(set(symbols))))
        if start:
            filters += [("year", ">=", pd.Timestamp(start).year),
                        ("as_of", ">=", pd.Timestamp(start))]
        if end:
            filters += [("year", "<=", pd.Timestamp(end).year),
                        ("as_of", "<=", pd.Timestamp(end))]
        df = self._read_parts(self.snap_dir, _YEAR_PART, filters=filters or None)
        if df.empty:
            return df
        df["as_of"] = pd.to_datetime(df["as_of"])
        # The same (symbol, as_of) order the single file was written in, so anything
        # downstream that happened to depend on row order sees what it always saw.
        return df.sort_values(["symbol", "as_of"], kind="stable").reset_index(drop=True)

    def fundamentals_asof(self, symbols: list[str], as_of: date) -> pd.DataFrame:
        """Latest snapshot at or before `as_of` per symbol (PIT by construction: the
        snapshot for date D only used facts filed <= D). Indexed by symbol."""
        ts = pd.Timestamp(as_of)
        # Only a snapshot inside the 120-day freshness window below can ever be served,
        # so read exactly that window — one year partition instead of the whole table.
        df = self.snapshots(symbols=symbols, start=(ts - pd.Timedelta(days=120)), end=ts)
        if df.empty:
            return pd.DataFrame()
        # Serve only reasonably fresh snapshots: a symbol whose last snapshot is >120
        # days old (stopped filing / left the store) must not carry stale ratios forward.
        latest = df.sort_values("as_of", kind="stable").groupby("symbol").tail(1)
        fresh = latest[latest["as_of"] >= ts - pd.Timedelta(days=120)]
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
