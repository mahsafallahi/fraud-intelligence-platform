"""Bronze ingestion for US Census ZIP geography (Gazetteer ZCTA file).

Source: Census Bureau Gazetteer files (public domain, no API key). One zip per
vintage, containing a tab-separated file with one row per ZIP Code Tabulation
Area (ZCTA): GEOID, land/water area and the internal point (lat/long).
Written to bronze/census_zcta/vintage=<YYYY>/part-0.parquet.

Design rules:
- Keep the data exactly as received: every column is a string (GEOIDs have
  leading zeros, e.g. "00601"), and the file's trailing whitespace is kept
  (including in the last column name). Trimming is Silver's job.
- _content_sha256 records a checksum of the downloaded zip.
- Validate before writing, so a bad download never overwrites a good partition.
- Idempotent: re-running a vintage overwrites its partition.

Note: ZCTAs approximate ZIP codes; transaction ZIPs must be zero-padded to five
digits in Silver before joining (58 of them lost their leading zero upstream).

Usage:
    python -m src.ingestion.bronze_census_zcta --vintage 2019
"""

import argparse
import hashlib
import io
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.utils.http import TIMEOUT_SECONDS, make_session
from src.utils.parquet import write_parquet_atomic
from src.utils.paths import BRONZE_DIR

log = logging.getLogger(__name__)

SOURCE_NAME = "census_gazetteer_zcta"
URL_TEMPLATE = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "{vintage}_Gazetteer/{vintage}_Gaz_zcta_national.zip"
)


def partition_path(vintage: int, bronze_dir: Path) -> Path:
    return bronze_dir / f"vintage={vintage}" / "part-0.parquet"


def ingest(
    vintage: int = 2019,
    bronze_dir: Path = BRONZE_DIR / "census_zcta",
    session: requests.Session | None = None,
) -> int:
    """Download one Gazetteer vintage into Bronze. Returns the number of rows."""
    session = session or make_session()
    url = URL_TEMPLATE.format(vintage=vintage)

    response = session.get(url, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    content = response.content

    member, text = _extract_single_txt(content, url)
    df = pd.read_csv(io.BytesIO(text), sep="\t", dtype=str, keep_default_na=False)
    if "GEOID" not in df.columns or df.empty:
        raise ValueError(f"Unexpected Gazetteer file from {url}: columns={list(df.columns)}, rows={len(df)}")

    def meta(value, dtype):
        return pd.Series(value, index=df.index, dtype=dtype)

    df["_ingested_at"] = meta(datetime.now(timezone.utc), "datetime64[us, UTC]")
    df["_source"] = meta(SOURCE_NAME, "str")
    df["_source_url"] = meta(url, "str")
    df["_source_file"] = meta(member, "str")
    df["_content_sha256"] = meta(hashlib.sha256(content).hexdigest(), "str")

    dest = partition_path(vintage, bronze_dir)
    write_parquet_atomic(df, dest)

    log.info("Gazetteer %d: %d ZCTAs -> %s", vintage, len(df), dest)
    return len(df)


def _extract_single_txt(content: bytes, url: str) -> tuple[str, bytes]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as e:
        raise ValueError(f"Response from {url} is not a zip file") from e
    txt_files = [n for n in archive.namelist() if n.endswith(".txt")]
    if len(txt_files) != 1:
        raise ValueError(f"Expected exactly one .txt file in {url}, found {archive.namelist()}")
    return txt_files[0], archive.read(txt_files[0])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vintage", type=int, default=2019)
    args = parser.parse_args()
    ingest(args.vintage)


if __name__ == "__main__":
    main()
