"""Silver writer: read one Bronze partition, run its Silver transform, write Silver.

The transforms in src/silver are pure functions (no file I/O); this module is the
I/O around them. One call handles one partition:

    bronze/<source>/<partition>/part-0.parquet
        -> transform (src/silver)
        -> silver/<source>/<partition>/part-0.parquet

Design rules:
- Read the partition's file, not its folder: reading a partition folder can add
  the partition key (e.g. `vintage`) as an extra column, which the transforms'
  column contracts correctly reject.
- Transform first, write second: a SilverValidationError stops the run before
  anything is written, so a bad batch never overwrites a good Silver partition.
- Atomic, idempotent writes (write_parquet_atomic): a re-run overwrites its partition.
- The DataFrame returned by the transform is written unchanged, so its dtypes
  (including nullable Int64 and date32) come back identical when read.
- A missing Bronze partition is an error.
- The PII key is never printed or logged.

Usage:
    python -m src.pipeline.silver_jobs transactions --start 2019-01-01 --end 2020-12-31
    python -m src.pipeline.silver_jobs transactions --date 2019-01-01
    python -m src.pipeline.silver_jobs reference
"""

from __future__ import annotations

import argparse
import functools
import logging
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import TypeVar

import pandas as pd

from src.silver.census_acs import transform_census_acs
from src.silver.census_zcta import transform_census_zcta
from src.silver.holidays import transform_holidays
from src.silver.mcc import transform_mcc
from src.silver.transactions import SilverValidationError, transform_transactions
from src.utils.parquet import write_parquet_atomic
from src.utils.paths import BRONZE_DIR, SILVER_DIR

log = logging.getLogger(__name__)

PART_FILE = "part-0.parquet"

# Reference partitions built by the CLI (same years/vintage as the Bronze DAG).
HOLIDAY_COUNTRY = "US"
HOLIDAY_YEARS = [2019, 2020]
CENSUS_VINTAGE = 2019


# --- 1. Partition paths (relative to the Bronze / Silver root) ---------------

def transactions_partition(batch_date: date) -> str:
    return f"transactions/batch_date={batch_date.isoformat()}"


def census_acs_partition(vintage: int) -> str:
    return f"census_acs/vintage={vintage}"


def census_zcta_partition(vintage: int) -> str:
    return f"census_zcta/vintage={vintage}"


def holidays_partition(country: str, year: int) -> str:
    return f"holidays/country={country}/year={year}"


def mcc_partition(version: str) -> str:
    return f"mcc/version={version}"


def version_from_folder(folder_name: str) -> str:
    """'version=9675cfab77d6' -> '9675cfab77d6'."""
    key, sep, value = folder_name.partition("=")
    if key != "version" or not sep or not value:
        raise ValueError(f"not a version=<hash> folder name: {folder_name!r}")
    return value


def pinned_mcc_version() -> str:
    """Version of the pinned Bronze MCC partition, taken from its folder name.

    The folder comes from the Bronze ingestion code, so Silver always reads
    exactly the partition that Bronze wrote.
    """
    from src.ingestion.bronze_mcc import PINNED_COMMIT, partition_path

    return version_from_folder(partition_path(PINNED_COMMIT, Path(".")).parent.name)


# --- 2. Read / write -----------------------------------------------------------

def _read_bronze(partition: str, bronze_dir: Path) -> pd.DataFrame:
    path = bronze_dir / partition / PART_FILE
    if not path.exists():
        raise FileNotFoundError(f"No Bronze partition: {path}")
    return pd.read_parquet(path)


def _write_silver(silver: pd.DataFrame, partition: str, silver_dir: Path) -> int:
    dest = silver_dir / partition / PART_FILE
    write_parquet_atomic(silver, dest)
    log.info("%s: %d rows -> %s", partition, len(silver), dest)
    return len(silver)


# --- 3. One function per source -------------------------------------------------

def run_transactions_day(
    batch_date: date,
    key: bytes,
    bronze_dir: Path = BRONZE_DIR,
    silver_dir: Path = SILVER_DIR,
) -> int:
    """Build one day of Silver transactions. Returns the number of rows written.

    An empty Bronze day gives a 0-row file with the full schema
    (transform_transactions returns empty_silver_frame() for an empty batch).
    """
    partition = transactions_partition(batch_date)
    silver = transform_transactions(_read_bronze(partition, bronze_dir), key)
    return _write_silver(silver, partition, silver_dir)


def run_census_acs(
    vintage: int, bronze_dir: Path = BRONZE_DIR, silver_dir: Path = SILVER_DIR
) -> int:
    partition = census_acs_partition(vintage)
    silver = transform_census_acs(_read_bronze(partition, bronze_dir))
    return _write_silver(silver, partition, silver_dir)


def run_census_zcta(
    vintage: int, bronze_dir: Path = BRONZE_DIR, silver_dir: Path = SILVER_DIR
) -> int:
    partition = census_zcta_partition(vintage)
    silver = transform_census_zcta(_read_bronze(partition, bronze_dir), vintage)
    return _write_silver(silver, partition, silver_dir)


def run_holidays(
    country: str, year: int, bronze_dir: Path = BRONZE_DIR, silver_dir: Path = SILVER_DIR
) -> int:
    partition = holidays_partition(country, year)
    silver = transform_holidays(_read_bronze(partition, bronze_dir))
    return _write_silver(silver, partition, silver_dir)


def run_mcc(
    version: str, bronze_dir: Path = BRONZE_DIR, silver_dir: Path = SILVER_DIR
) -> int:
    partition = mcc_partition(version)
    silver = transform_mcc(_read_bronze(partition, bronze_dir), version)
    return _write_silver(silver, partition, silver_dir)


# --- 4. Fail fast on data errors (used by the Airflow tasks) --------------------

F = TypeVar("F", bound=Callable)


def fail_fast_on_validation_error(fail_with: type[Exception]) -> Callable[[F], F]:
    """Re-raise SilverValidationError as `fail_with`, keeping the message.

    In Airflow, `fail_with` is AirflowFailException: the task then fails at once
    instead of retrying, because retrying cannot fix bad data. Other errors
    (I/O, a container restart) are left alone, so the task's retries apply.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except SilverValidationError as error:
                raise fail_with(f"Silver validation failed: {error}") from error
        return wrapper  # type: ignore[return-value]
    return decorator


# --- 5. CLI ------------------------------------------------------------------------

def _date_range(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="source", required=True)

    tx = sub.add_parser("transactions", help="daily transactions partitions")
    tx.add_argument("--date", type=date.fromisoformat, help="single day")
    tx.add_argument("--start", type=date.fromisoformat, help="first day of a range")
    tx.add_argument("--end", type=date.fromisoformat, help="last day of a range, inclusive")

    sub.add_parser("reference", help="holidays, Census ACS, Census Gazetteer and MCC")
    args = parser.parse_args()

    if args.source == "transactions":
        if args.date:
            days = [args.date]
        elif args.start and args.end:
            days = list(_date_range(args.start, args.end))
        else:
            parser.error("give either --date, or both --start and --end")

        from dotenv import load_dotenv

        from src.silver.transactions import load_pii_key
        from src.utils.paths import PROJECT_ROOT

        load_dotenv(PROJECT_ROOT / ".env")  # does not override variables already set
        key = load_pii_key()  # once per invocation; never printed
        total = sum(run_transactions_day(day, key) for day in days)
        log.info("Silver transactions: %d day(s), %d rows", len(days), total)

    else:
        total = sum(run_holidays(HOLIDAY_COUNTRY, year) for year in HOLIDAY_YEARS)
        total += run_census_acs(CENSUS_VINTAGE)
        total += run_census_zcta(CENSUS_VINTAGE)
        total += run_mcc(pinned_mcc_version())
        log.info("Silver reference data: %d rows", total)


if __name__ == "__main__":
    main()
