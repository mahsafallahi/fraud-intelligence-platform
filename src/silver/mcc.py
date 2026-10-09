"""Silver transform for the MCC reference list (one pinned version per Bronze partition).

Bronze keeps the file as delivered: every value a string, four description
columns from different sources (three of them with empty values) and an IRS
reporting column. Silver keeps one row per merchant category code with its
edited description, trimmed and validated.
Pure function, no file I/O.
"""

from __future__ import annotations

import re

import pandas as pd

from src.silver.transactions import SilverValidationError

# --- 1. Contract -------------------------------------------------------------

BUSINESS_COLUMNS = [
    "mcc", "edited_description", "combined_description",
    "usda_description", "irs_description", "irs_reportable",
]
METADATA_COLUMNS = ["_ingested_at", "_source", "_source_url", "_content_sha256"]
EXPECTED_BRONZE_COLUMNS = frozenset(BUSINESS_COLUMNS + METADATA_COLUMNS)

# Only edited_description is complete (no empty values); the other three
# descriptions and irs_reportable are not needed downstream: dropped.
SILVER_SCHEMA: dict[str, str] = {
    "mcc": "str",
    "description": "str",
    "_ingested_at": "datetime64[us, UTC]",
    "_source": "str",
    "_content_sha256": "str",
    "_version": "str",
}


# --- 2. Validation helpers ---------------------------------------------------

def _fail(message: str) -> None:
    raise SilverValidationError(message)


def _check(bad: pd.Series, column: str, problem: str, values: pd.Series) -> None:
    count = int(bad.sum())
    if count:
        _fail(f"{column}: {count} row(s) {problem}, e.g. {values[bad].head(3).tolist()}")


# --- 3. The transform --------------------------------------------------------

def transform_mcc(bronze: pd.DataFrame, version: str) -> pd.DataFrame:
    """Turn one Bronze MCC partition into Silver rows.

    `version` comes from the partition folder name (version=<commit hash>),
    because the file itself does not say which commit it was taken from.
    """
    if not isinstance(version, str) or not re.fullmatch(r"[0-9a-f]{7,40}", version):
        _fail(f"version {version!r} is not a commit hash")

    actual = set(bronze.columns)
    if actual != EXPECTED_BRONZE_COLUMNS:
        _fail(
            f"Bronze columns changed. Missing: {sorted(EXPECTED_BRONZE_COLUMNS - actual)}. "
            f"Unexpected: {sorted(actual - EXPECTED_BRONZE_COLUMNS)}."
        )
    if bronze.empty:
        _fail("the Bronze MCC partition has no rows")

    code = bronze["mcc"].str.strip()
    description = bronze["edited_description"].str.strip()
    _check(code.isna() | code.eq(""), "mcc", "is null or empty", code)
    _check(description.isna() | description.eq(""), "edited_description",
           "is null or empty", description)
    _check(~code.str.fullmatch(r"\d{4}"), "mcc", "is not 4 digits", code)
    _check(code.duplicated(keep=False), "mcc", "is duplicated", code)

    silver = pd.DataFrame({
        "mcc": code,
        "description": description,
        "_ingested_at": pd.to_datetime(bronze["_ingested_at"], utc=True),
        "_source": bronze["_source"],
        "_content_sha256": bronze["_content_sha256"],
    })
    silver["_version"] = version

    silver = silver.sort_values("mcc").reset_index(drop=True)
    return silver[list(SILVER_SCHEMA)].astype(SILVER_SCHEMA)