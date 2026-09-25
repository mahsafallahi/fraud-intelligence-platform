"""Bronze ingestion for US Census demographics per ZIP (ACS 5-year, Census Data API).

One API call per vintage returns every ZIP Code Tabulation Area (ZCTA) with the
variables below, as a JSON array of arrays (first row = header). Stored as one
row in bronze/census_acs/vintage=<YYYY>/part-0.parquet.

Needs an API key in the CENSUS_API_KEY environment variable (the CLI loads it
from .env; an already-set environment variable wins, e.g. in Docker).

Design rules:
- Keep the data exactly as received: the raw response body is stored as a
  single string column. Parsing and cleaning is Silver's job (note: the Census
  uses -666666666 for "not available"; Silver must turn it into null).
- The key is sent as a URL parameter, so it must never reach storage, logs or
  error messages: the stored URL has the key removed, errors are re-raised
  redacted, and urllib3 (which logs URLs on retries/debug) gets a redacting
  log filter.
- Validate before writing, so a bad response never overwrites a good partition.
- Idempotent: re-running a vintage overwrites its partition.

Usage:
    python -m src.ingestion.bronze_census_acs --vintage 2019
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.http import TIMEOUT_SECONDS, make_session
from src.utils.parquet import write_parquet_atomic
from src.utils.paths import BRONZE_DIR, PROJECT_ROOT
from src.utils.secrets import mask_secret_in_logs, redact

log = logging.getLogger(__name__)

SOURCE_NAME = "census_acs5_api"
KEY_ENV_VAR = "CENSUS_API_KEY"
URL_TEMPLATE = "https://api.census.gov/data/{vintage}/acs/acs5"
GEOGRAPHY = "zip code tabulation area"
VARIABLES = [
    "NAME",
    "B01003_001E",  # total population
    "B19013_001E",  # median household income (USD)
    "B01002_001E",  # median age
]
# Loggers that may print request URLs (and therefore the key).
URL_LOGGERS = ["urllib3.connectionpool", "urllib3.util.retry"]


class CensusAPIError(RuntimeError):
    """Request to the Census API failed (message is redacted)."""


def partition_path(vintage: int, bronze_dir: Path) -> Path:
    return bronze_dir / f"vintage={vintage}" / "part-0.parquet"


def ingest(
    vintage: int = 2019,
    bronze_dir: Path = BRONZE_DIR / "census_acs",
    session: requests.Session | None = None,
    api_key: str | None = None,
) -> int:
    """Fetch one ACS 5-year vintage into Bronze. Returns the number of ZCTAs received."""
    api_key = api_key or os.getenv(KEY_ENV_VAR)
    if not api_key:
        raise CensusAPIError(f"{KEY_ENV_VAR} is not set (add it to .env)")
    mask_secret_in_logs(api_key, URL_LOGGERS)

    session = session or make_session()
    url = URL_TEMPLATE.format(vintage=vintage)
    params = {"get": ",".join(VARIABLES), "for": f"{GEOGRAPHY}:*"}

    try:
        response = session.get(url, params={**params, "key": api_key}, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as e:
        # `from None` drops the original exception, whose message contains the key.
        raise CensusAPIError(redact(f"{type(e).__name__}: {e}", api_key)) from None

    body = response.text
    n_zctas = _validate(body, url)

    df = pd.DataFrame({
        "response_body": pd.Series([body], dtype="str"),
        "_ingested_at": pd.Series([datetime.now(timezone.utc)], dtype="datetime64[us, UTC]"),
        "_source": pd.Series([SOURCE_NAME], dtype="str"),
        "_request_url": pd.Series([_url_without_key(url, params)], dtype="str"),
        "_vintage": pd.Series([vintage], dtype="int32"),
    })
    dest = partition_path(vintage, bronze_dir)
    write_parquet_atomic(df, dest)

    log.info("ACS %d: %d ZCTAs -> %s", vintage, n_zctas, dest)
    return n_zctas


def _url_without_key(url: str, params: dict) -> str:
    return requests.Request("GET", url, params=params).prepare().url


def _validate(body: str, url: str) -> int:
    # Messages show only the start of the body; the key is never echoed by the
    # API, and the URL here has no key.
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise ValueError(f"Response from {url} is not valid JSON: {body[:100]!r}") from None
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[0], list):
        raise ValueError(f"Expected a header row plus data rows from {url}")
    missing = set(VARIABLES + [GEOGRAPHY]) - set(payload[0])
    if missing:
        raise ValueError(f"Response from {url} is missing columns: {sorted(missing)}")
    return len(payload) - 1


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vintage", type=int, default=2019)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")  # does not override variables already set
    ingest(args.vintage)


if __name__ == "__main__":
    main()
