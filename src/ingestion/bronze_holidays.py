"""Bronze ingestion for public holidays from the Nager.Date API (free, no key).

One API call per (country, year) -> one row in
bronze/holidays/country=<CC>/year=<YYYY>/part-0.parquet.

Design rules:
- Keep the data exactly as received: the raw response body is stored as a
  single string column. Parsing the JSON into one row per holiday is Silver's job,
  so the Bronze schema never changes, whatever the API returns.
- Validate before writing (HTTP 200, a non-empty JSON list), so a bad response
  never overwrites a good partition.
- Retry transient failures (network errors, 429, 5xx); fail immediately on
  anything else (e.g. 404 for an unknown country).
- Idempotent: re-running a (country, year) overwrites its partition.

Usage:
    python -m src.ingestion.bronze_holidays --years 2019 2020
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.parquet import write_parquet_atomic
from src.utils.paths import BRONZE_DIR

log = logging.getLogger(__name__)

SOURCE_NAME = "nager_date_api"
BASE_URL = "https://date.nager.at/api/v3/PublicHolidays"
TIMEOUT_SECONDS = 30


def make_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=1,  # waits 1s, 2s, 4s between attempts
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def partition_path(country: str, year: int, bronze_dir: Path) -> Path:
    return bronze_dir / f"country={country}" / f"year={year}" / "part-0.parquet"


def ingest_year(
    year: int,
    country: str = "US",
    bronze_dir: Path = BRONZE_DIR / "holidays",
    session: requests.Session | None = None,
) -> int:
    """Fetch one year of holidays into Bronze. Returns the number of holidays received."""
    session = session or make_session()
    url = f"{BASE_URL}/{year}/{country}"

    response = session.get(url, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    body = response.text
    n_holidays = _validate(body, url)

    df = pd.DataFrame({
        "response_body": pd.Series([body], dtype="str"),
        "_ingested_at": pd.Series([datetime.now(timezone.utc)], dtype="datetime64[us, UTC]"),
        "_source": pd.Series([SOURCE_NAME], dtype="str"),
        "_request_url": pd.Series([url], dtype="str"),
        "_country": pd.Series([country], dtype="str"),
        "_year": pd.Series([year], dtype="int32"),
    })
    dest = partition_path(country, year, bronze_dir)
    write_parquet_atomic(df, dest)

    log.info("%s %d: %d holidays -> %s", country, year, n_holidays, dest)
    return n_holidays


def _validate(body: str, url: str) -> int:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Response from {url} is not valid JSON") from e
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty JSON list from {url}, got: {body[:200]}")
    return len(payload)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--country", default="US")
    args = parser.parse_args()

    session = make_session()
    total = sum(ingest_year(y, args.country, session=session) for y in args.years)
    log.info("ingested %d year(s), %d holidays", len(args.years), total)


if __name__ == "__main__":
    main()
