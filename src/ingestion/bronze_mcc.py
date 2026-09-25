"""Bronze ingestion for merchant category codes (MCC).

Source: github.com/greggles/mcc-codes (Unlicense), file mcc_codes.csv, pinned to
a commit so every run downloads identical bytes. Written to
bronze/mcc/version=<short sha>/part-0.parquet.

Design rules:
- Keep the data exactly as received: every column is a string (MCCs have
  leading zeros, e.g. "0742"), empty values stay empty strings.
- _content_sha256 records a checksum of the downloaded bytes, so it is
  verifiable which file was loaded.
- Pinned version: upgrading the source is a deliberate one-line change; the old
  version stays in its own partition.
- Validate before writing, so a bad download never overwrites a good partition.
- Idempotent: re-running a version overwrites its partition.

Note: transactions carry a `category` (e.g. grocery_pos), not an MCC. The
category -> MCC mapping is a separate, hand-written seed, not part of this source.

Usage:
    python -m src.ingestion.bronze_mcc
"""

import argparse
import hashlib
import io
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.utils.http import TIMEOUT_SECONDS, make_session
from src.utils.parquet import write_parquet_atomic
from src.utils.paths import BRONZE_DIR

log = logging.getLogger(__name__)

SOURCE_NAME = "github_greggles_mcc_codes"
PINNED_COMMIT = "9675cfab77d6160397d2b84c0ebc934439841929"  # 2019-02-03, last change to mcc_codes.csv
URL_TEMPLATE = "https://raw.githubusercontent.com/greggles/mcc-codes/{commit}/mcc_codes.csv"


def partition_path(commit: str, bronze_dir: Path) -> Path:
    return bronze_dir / f"version={commit[:12]}" / "part-0.parquet"


def ingest(
    commit: str = PINNED_COMMIT,
    bronze_dir: Path = BRONZE_DIR / "mcc",
    session: requests.Session | None = None,
) -> int:
    """Download one version of the MCC file into Bronze. Returns the number of rows."""
    session = session or make_session()
    url = URL_TEMPLATE.format(commit=commit)

    response = session.get(url, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    content = response.content

    df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    if "mcc" not in df.columns or df.empty:
        raise ValueError(f"Unexpected MCC file from {url}: columns={list(df.columns)}, rows={len(df)}")

    def meta(value, dtype):
        return pd.Series(value, index=df.index, dtype=dtype)

    df["_ingested_at"] = meta(datetime.now(timezone.utc), "datetime64[us, UTC]")
    df["_source"] = meta(SOURCE_NAME, "str")
    df["_source_url"] = meta(url, "str")
    df["_content_sha256"] = meta(hashlib.sha256(content).hexdigest(), "str")

    dest = partition_path(commit, bronze_dir)
    write_parquet_atomic(df, dest)

    log.info("MCC %s: %d rows -> %s", commit[:12], len(df), dest)
    return len(df)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--commit", default=PINNED_COMMIT, help="source repo commit to load")
    args = parser.parse_args()
    ingest(args.commit)


if __name__ == "__main__":
    main()
