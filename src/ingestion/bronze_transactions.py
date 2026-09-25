"""Bronze ingestion for transactions: one landing file (one day) per run.

Reads landing/transactions/<date>.csv and writes
bronze/transactions/batch_date=<date>/part-0.parquet.

Design rules:
- Keep the data exactly as received: every column is stored as a string, and
  empty values stay empty strings (no NaN conversion). Typing happens in Silver.
- Add ingestion metadata columns, prefixed with "_" to separate them from
  business columns.
- Idempotent: re-running a day overwrites that day's partition, never appends.
- A missing landing file is an error, not an empty day.

Usage:
    python -m src.ingestion.bronze_transactions --date 2019-01-01
    python -m src.ingestion.bronze_transactions --start 2019-01-01 --end 2019-01-31
"""

import argparse
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa

from src.utils.paths import BRONZE_DIR, DATA_DIR, LANDING_DIR

log = logging.getLogger(__name__)

SOURCE_NAME = "kaggle_kartik2112"


def landing_path(batch_date: date, landing_dir: Path) -> Path:
    return landing_dir / f"{batch_date.isoformat()}.csv"


def partition_path(batch_date: date, bronze_dir: Path) -> Path:
    return bronze_dir / f"batch_date={batch_date.isoformat()}" / "part-0.parquet"


def ingest_day(
    batch_date: date,
    landing_dir: Path = LANDING_DIR / "transactions",
    bronze_dir: Path = BRONZE_DIR / "transactions",
) -> int:
    """Ingest one day's landing file into Bronze. Returns the number of rows written."""
    src = landing_path(batch_date, landing_dir)
    if not src.exists():
        raise FileNotFoundError(f"No landing file for {batch_date}: {src}")

    df = pd.read_csv(src, dtype=str, keep_default_na=False)

    # Explicit dtypes: on an empty day pandas cannot infer types from values and
    # would write "null" columns, giving that partition a different schema.
    def meta(value, dtype):
        return pd.Series(value, index=df.index, dtype=dtype)

    df["_ingested_at"] = meta(datetime.now(timezone.utc), "datetime64[us, UTC]")
    df["_source"] = meta(SOURCE_NAME, "str")
    df["_source_file"] = meta(_relative_to_data_dir(src), "str")
    df["_batch_date"] = meta(batch_date, pd.ArrowDtype(pa.date32()))

    # Write to a temp file, then rename over the target: a crash mid-write
    # never leaves a half-written partition behind.
    dest = partition_path(batch_date, bronze_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)

    log.info("%s: %d rows -> %s", batch_date, len(df), dest)
    return len(df)


def _relative_to_data_dir(path: Path) -> str:
    # Store a machine-independent path for lineage (not C:\Users\...).
    try:
        return path.resolve().relative_to(DATA_DIR.resolve()).as_posix()
    except ValueError:
        return path.name


def _date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", type=date.fromisoformat, help="single day to ingest")
    parser.add_argument("--start", type=date.fromisoformat, help="first day of a range (backfill)")
    parser.add_argument("--end", type=date.fromisoformat, help="last day of a range, inclusive")
    args = parser.parse_args()

    if args.date:
        days = [args.date]
    elif args.start and args.end:
        days = list(_date_range(args.start, args.end))
    else:
        parser.error("give either --date, or both --start and --end")

    total = sum(ingest_day(d) for d in days)
    log.info("ingested %d day(s), %d rows", len(days), total)


if __name__ == "__main__":
    main()
