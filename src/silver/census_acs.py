"""Silver transform for Census ACS 5-year data (one vintage per Bronze partition).

Bronze stores the raw API response: one row whose response_body is a JSON
array of arrays (first row = header). Silver turns it into one typed row per
ZIP code tabulation area (ZCTA). Pure function, no file I/O.
"""

from __future__ import annotations

import json

import pandas as pd

from src.silver.transactions import SilverValidationError

# --- 1. Contract -------------------------------------------------------------

EXPECTED_BRONZE_COLUMNS = frozenset(
    ["response_body", "_ingested_at", "_source", "_request_url", "_vintage"]
)

EXPECTED_HEADER = [
    "NAME",
    "B01003_001E",  # total population
    "B19013_001E",  # median household income (dollars)
    "B01002_001E",  # median age (years)
    "state",
    "zip code tabulation area",
]

# Census annotation codes: numbers that mean "no estimate", not real values.
CENSUS_SENTINELS = frozenset(
    {-999999999, -888888888, -666666666, -555555555, -333333333, -222222222}
)

SILVER_SCHEMA: dict[str, str] = {
    "zip": "str",
    "state_fips": "str",
    "population": "int64",
    "median_household_income": "Int64",  # nullable integer
    "median_age": "float64",  # NaN when there is no estimate
    "_ingested_at": "datetime64[us, UTC]",
    "_source": "str",
    "_vintage": "int16",
}


# --- 2. Validation helpers ---------------------------------------------------

def _fail(message: str) -> None:
    raise SilverValidationError(message)


def _check(bad: pd.Series, column: str, problem: str, values: pd.Series) -> None:
    count = int(bad.sum())
    if count:
        _fail(f"{column}: {count} row(s) {problem}, e.g. {values[bad].head(3).tolist()}")


def _parse_number(
    values: pd.Series, column: str, allow_sentinel: bool, whole: bool
) -> pd.Series:
    """Parse Census numbers; sentinel codes become missing only if allowed."""
    parsed = pd.to_numeric(values, errors="coerce")
    _check(parsed.isna(), column, "is not a number", values)

    is_sentinel = parsed.isin(CENSUS_SENTINELS)
    if not allow_sentinel:
        _check(is_sentinel, column, "has a Census 'no estimate' code", values)
    _check((parsed < 0) & ~is_sentinel, column, "is negative", values)
    if whole:
        _check((parsed % 1 != 0) & ~is_sentinel, column, "is not a whole number", values)

    return parsed.mask(is_sentinel)


# --- 3. The transform --------------------------------------------------------

def transform_census_acs(bronze: pd.DataFrame) -> pd.DataFrame:
    """Turn one Bronze ACS partition (one API response) into Silver rows."""
    actual = set(bronze.columns)
    if actual != EXPECTED_BRONZE_COLUMNS:
        _fail(
            f"Bronze columns changed. Missing: {sorted(EXPECTED_BRONZE_COLUMNS - actual)}. "
            f"Unexpected: {sorted(actual - EXPECTED_BRONZE_COLUMNS)}."
        )
    if len(bronze) != 1:
        _fail(f"expected exactly 1 Bronze row (one API response), got {len(bronze)}")

    meta = bronze.iloc[0]
    try:
        rows = json.loads(meta["response_body"])
    except (TypeError, ValueError) as error:
        raise SilverValidationError(f"response_body is not valid JSON: {error}") from error

    if not isinstance(rows, list) or len(rows) < 2:
        _fail("response_body must be a JSON list with a header and at least one data row")
    header, data = rows[0], rows[1:]
    if header != EXPECTED_HEADER:
        _fail(f"Census header changed: {header}")
    bad_rows = [
        i for i, row in enumerate(data, start=1)
        if not isinstance(row, list) or len(row) != len(header)
    ]
    if bad_rows:
        _fail(f"{len(bad_rows)} data row(s) do not have {len(header)} values, e.g. {bad_rows[:3]}")

    raw = pd.DataFrame(data, columns=header, dtype="str")

    zip_code = raw["zip code tabulation area"]
    _check(~zip_code.str.fullmatch(r"\d{5}"), "zip code tabulation area",
           "is not 5 digits", zip_code)
    _check(zip_code.duplicated(keep=False), "zip code tabulation area",
           "is duplicated", zip_code)

    state = raw["state"]
    _check(~state.str.fullmatch(r"\d{2}"), "state", "is not a 2-digit FIPS code", state)

    _check(raw["NAME"] != "ZCTA5 " + zip_code, "NAME",
           "does not match its ZIP (columns may be shifted)", raw["NAME"])

    silver = pd.DataFrame({
        "zip": zip_code,
        "state_fips": state,
        "population": _parse_number(
            raw["B01003_001E"], "B01003_001E", allow_sentinel=False, whole=True),
        "median_household_income": _parse_number(
            raw["B19013_001E"], "B19013_001E", allow_sentinel=True, whole=True),
        "median_age": _parse_number(
            raw["B01002_001E"], "B01002_001E", allow_sentinel=True, whole=False),
    })
    silver["_ingested_at"] = pd.to_datetime(meta["_ingested_at"], utc=True)
    silver["_source"] = meta["_source"]
    silver["_vintage"] = int(meta["_vintage"])

    silver = silver.sort_values("zip").reset_index(drop=True)
    return silver[list(SILVER_SCHEMA)].astype(SILVER_SCHEMA)
