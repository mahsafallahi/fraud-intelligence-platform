"""Silver transform for public holidays (Nager.Date API).

Bronze keeps one row per country and year: the raw JSON list returned by the API.
Silver flattens it to one row per holiday and state:
- a nationwide ("global") holiday gives one row with state_code = null;
- a state-level holiday gives one row per listed state ("US-CA" -> "CA").
The same holiday can appear more than once on a day in the source (for example a
nationwide Columbus Day plus a state-level Columbus Day list), so the key is
(holiday_date, name, state_code), not the date alone. Pure function, no file I/O.
"""

from __future__ import annotations

import datetime
import json
import re

import pandas as pd

from src.silver.transactions import SilverValidationError

# --- 1. Contract -------------------------------------------------------------

EXPECTED_BRONZE_COLUMNS = frozenset(
    ["response_body", "_ingested_at", "_source", "_request_url", "_country", "_year"]
)
EXPECTED_ITEM_KEYS = frozenset(
    ["date", "localName", "name", "countryCode", "fixed", "global",
     "counties", "launchYear", "types"]
)
KEY = ["holiday_date", "name", "state_code"]

SILVER_SCHEMA: dict[str, str] = {
    "holiday_date": "date32[pyarrow]",
    "name": "str",
    "country_code": "str",
    "state_code": "str",  # null for nationwide holidays
    "is_global": "bool",
    "types": "str",  # sorted and comma-separated, e.g. "Bank,Public"
    "_ingested_at": "datetime64[us, UTC]",
    "_source": "str",
    "_year": "int16",
}


# --- 2. Helpers --------------------------------------------------------------

def _fail(message: str) -> None:
    raise SilverValidationError(message)


def _parse_response(body: object, where: str) -> list:
    try:
        items = json.loads(body)
    except (TypeError, ValueError) as error:
        raise SilverValidationError(f"{where}: response_body is not valid JSON: {error}") from error
    if not isinstance(items, list) or not items:
        _fail(f"{where}: response_body must be a non-empty JSON list")
    return items


def _flatten_item(item: object, country: str, year: int) -> list[dict]:
    """One API holiday -> one Silver row (nationwide) or one row per state."""
    where = f"{country} {year}"
    if not isinstance(item, dict) or set(item) != EXPECTED_ITEM_KEYS:
        _fail(f"{where}: holiday fields changed: {item!r}")

    name = item["name"]
    if not isinstance(name, str) or not name.strip():
        _fail(f"{where}: holiday without a name: {item!r}")
    name = name.strip()

    raw_date = item["date"]
    if not isinstance(raw_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_date):
        _fail(f"{where}: {name}: date {raw_date!r} is not YYYY-MM-DD")
    try:
        day = datetime.date.fromisoformat(raw_date)
    except ValueError as error:
        raise SilverValidationError(f"{where}: {name}: date {raw_date!r} is not a real day") from error
    if day.year != year:
        _fail(f"{where}: {name}: date {day} is outside year {year}")

    if item["countryCode"] != country:
        _fail(f"{where}: {name}: countryCode {item['countryCode']!r} does not match {country}")

    is_global = item["global"]
    if not isinstance(is_global, bool):
        _fail(f"{where}: {name}: global must be true or false, got {is_global!r}")

    types = item["types"]
    if not isinstance(types, list) or not types or not all(isinstance(t, str) and t for t in types):
        _fail(f"{where}: {name}: types must be a non-empty list of names")

    base = {
        "holiday_date": day,
        "name": name,
        "country_code": country,
        "is_global": is_global,
        "types": ",".join(sorted(types)),
    }

    counties = item["counties"]
    if is_global:
        if counties is not None:
            _fail(f"{where}: {name}: a nationwide holiday must not list states")
        return [{**base, "state_code": None}]

    if not isinstance(counties, list) or not counties:
        _fail(f"{where}: {name}: a state-level holiday must list at least one state")
    rows = []
    for code in counties:
        match = re.fullmatch(rf"{country}-([A-Z]{{2}})", code) if isinstance(code, str) else None
        if match is None:
            _fail(f"{where}: {name}: state code {code!r} is not like {country}-XX")
        rows.append({**base, "state_code": match.group(1)})
    return rows


# --- 3. The transform --------------------------------------------------------

def transform_holidays(bronze: pd.DataFrame) -> pd.DataFrame:
    """Turn Bronze holiday rows (one API response per country and year) into Silver rows."""
    actual = set(bronze.columns)
    if actual != EXPECTED_BRONZE_COLUMNS:
        _fail(
            f"Bronze columns changed. Missing: {sorted(EXPECTED_BRONZE_COLUMNS - actual)}. "
            f"Unexpected: {sorted(actual - EXPECTED_BRONZE_COLUMNS)}."
        )
    if bronze.empty:
        _fail("the Bronze holidays partition has no rows")
    if bronze[["_country", "_year"]].duplicated().any():
        _fail("more than one Bronze row for the same country and year")

    rows = []
    for record in bronze.to_dict("records"):
        country = record["_country"]
        year = int(record["_year"])
        if not isinstance(country, str) or not re.fullmatch(r"[A-Z]{2}", country):
            _fail(f"_country {country!r} is not a 2-letter code")
        where = f"{country} {year}"
        for item in _parse_response(record["response_body"], where):
            for row in _flatten_item(item, country, year):
                row["_ingested_at"] = record["_ingested_at"]
                row["_source"] = record["_source"]
                row["_year"] = year
                rows.append(row)

    silver = pd.DataFrame(rows, columns=list(SILVER_SCHEMA))

    dupes = silver.duplicated(subset=KEY, keep=False)
    if dupes.any():
        examples = silver.loc[dupes, KEY].head(3).values.tolist()
        _fail(f"{int(dupes.sum())} row(s) share holiday_date, name and state_code, e.g. {examples}")

    silver["_ingested_at"] = pd.to_datetime(silver["_ingested_at"], utc=True)
    silver = silver.sort_values(KEY, na_position="first").reset_index(drop=True)
    return silver.astype(SILVER_SCHEMA)