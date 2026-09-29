"""Silver transform for the Census Gazetteer ZCTA file (one vintage per Bronze partition).

Bronze keeps the file as delivered: every value a string, and the last column
(INTPTLONG) padded with trailing spaces in its name and in all of its values.
Silver trims, types and validates it: one row per ZIP code tabulation area.
Pure function, no file I/O.
"""

from __future__ import annotations

import pandas as pd

from src.silver.transactions import SilverValidationError

# --- 1. Contract -------------------------------------------------------------

BUSINESS_COLUMNS = [
    "GEOID", "ALAND", "AWATER", "ALAND_SQMI", "AWATER_SQMI", "INTPTLAT", "INTPTLONG",
]
METADATA_COLUMNS = [
    "_ingested_at", "_source", "_source_url", "_source_file", "_content_sha256",
]
EXPECTED_BRONZE_COLUMNS = frozenset(BUSINESS_COLUMNS + METADATA_COLUMNS)

# ALAND_SQMI and AWATER_SQMI are rounded copies of ALAND and AWATER: dropped.
SILVER_SCHEMA: dict[str, str] = {
    "zip": "str",
    "land_area_m2": "int64",
    "water_area_m2": "int64",
    "centroid_lat": "float64",
    "centroid_long": "float64",
    "_ingested_at": "datetime64[us, UTC]",
    "_source": "str",
    "_content_sha256": "str",
    "_vintage": "int16",
}


# --- 2. Validation helpers ---------------------------------------------------

def _fail(message: str) -> None:
    raise SilverValidationError(message)


def _check(bad: pd.Series, column: str, problem: str, values: pd.Series) -> None:
    count = int(bad.sum())
    if count:
        _fail(f"{column}: {count} row(s) {problem}, e.g. {values[bad].head(3).tolist()}")


def _parse_float(values: pd.Series, column: str, low: float, high: float) -> pd.Series:
    parsed = pd.to_numeric(values, errors="coerce")
    _check(parsed.isna(), column, "is not a number", values)
    _check((parsed < low) | (parsed > high), column, f"is outside [{low}, {high}]", values)
    return parsed.astype("float64")


# --- 3. The transform --------------------------------------------------------

def transform_census_zcta(bronze: pd.DataFrame, vintage: int) -> pd.DataFrame:
    """Turn one Bronze Gazetteer partition into Silver rows.

    `vintage` comes from the partition folder name (vintage=2019), because the
    Gazetteer file itself has no year column.
    """
    if not isinstance(vintage, int) or not 2000 <= vintage <= 2100:
        _fail(f"vintage {vintage!r} is not a plausible year")

    df = bronze.rename(columns=lambda name: name.strip())
    if df.columns.duplicated().any():
        _fail("two Bronze columns have the same name after trimming spaces")
    actual = set(df.columns)
    if actual != EXPECTED_BRONZE_COLUMNS:
        _fail(
            f"Bronze columns changed. Missing: {sorted(EXPECTED_BRONZE_COLUMNS - actual)}. "
            f"Unexpected: {sorted(actual - EXPECTED_BRONZE_COLUMNS)}."
        )
    if df.empty:
        _fail("the Bronze Gazetteer partition has no rows")

    biz = {column: df[column].str.strip() for column in BUSINESS_COLUMNS}
    for column, values in biz.items():
        _check(values.isna() | values.eq(""), column, "is null or empty", values)

    zip_code = biz["GEOID"]
    _check(~zip_code.str.fullmatch(r"\d{5}"), "GEOID", "is not 5 digits", zip_code)
    _check(zip_code.duplicated(keep=False), "GEOID", "is duplicated", zip_code)
    for column in ["ALAND", "AWATER"]:
        _check(~biz[column].str.fullmatch(r"\d+"), column,
               "is not a whole non-negative number", biz[column])

    silver = pd.DataFrame({
        "zip": zip_code,
        "land_area_m2": biz["ALAND"].astype("int64"),
        "water_area_m2": biz["AWATER"].astype("int64"),
        "centroid_lat": _parse_float(biz["INTPTLAT"], "INTPTLAT", -90, 90),
        "centroid_long": _parse_float(biz["INTPTLONG"], "INTPTLONG", -180, 180),
        "_ingested_at": pd.to_datetime(df["_ingested_at"], utc=True),
        "_source": df["_source"],
        "_content_sha256": df["_content_sha256"],
    })
    silver["_vintage"] = vintage

    silver = silver.sort_values("zip").reset_index(drop=True)
    return silver[list(SILVER_SCHEMA)].astype(SILVER_SCHEMA)


