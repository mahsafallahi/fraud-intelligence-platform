"""Tests for the Silver Census ACS transform (src/silver/census_acs.py)."""

import json
import re

import pandas as pd
import pytest

from src.silver.census_acs import EXPECTED_HEADER, SILVER_SCHEMA, transform_census_acs
from src.silver.transactions import SilverValidationError

# --- 1. Fake inputs -----------------------------------------------------------

def acs_row(zip_code="00601", pop="17113", income="14361", age="41.9", state="72", name=None):
    """One data row exactly as the Census API returns it (all strings)."""
    return [name or f"ZCTA5 {zip_code}", pop, income, age, state, zip_code]


def make_bronze(data_rows=None, body=None):
    """One Bronze row: the raw API response plus metadata, typed like the real Parquet."""
    if body is None:
        rows = data_rows if data_rows is not None else [acs_row()]
        body = json.dumps([EXPECTED_HEADER] + rows)
    return pd.DataFrame({
        "response_body": pd.Series([body], dtype="str"),
        "_ingested_at": [pd.Timestamp("2026-09-25 21:30:58", tz="UTC")],
        "_source": pd.Series(["census_acs5_api"], dtype="str"),
        "_request_url": pd.Series(["https://api.census.gov/data/2019/acs/acs5?get=NAME"], dtype="str"),
        "_vintage": pd.Series([2019], dtype="int32"),
    })


# --- 2. Happy path ------------------------------------------------------------

def test_valid_row_gives_exact_schema_and_values():
    out = transform_census_acs(make_bronze())

    assert {c: str(t) for c, t in out.dtypes.items()} == SILVER_SCHEMA
    row = out.iloc[0]
    assert row["zip"] == "00601"
    assert row["state_fips"] == "72"
    assert row["population"] == 17113
    assert row["median_household_income"] == 14361
    assert row["median_age"] == 41.9
    assert row["_vintage"] == 2019


def test_rows_are_sorted_by_zip():
    out = transform_census_acs(make_bronze([acs_row("00603"), acs_row("00601")]))
    assert out["zip"].tolist() == ["00601", "00603"]


def test_capped_income_values_are_kept_unchanged():
    out = transform_census_acs(make_bronze([
        acs_row("00601", income="250001"),
        acs_row("00602", income="2499"),
    ]))
    assert out["median_household_income"].tolist() == [250001, 2499]


# --- 3. Census "no estimate" codes ------------------------------------------------

@pytest.mark.parametrize("code", ["-666666666", "-999999999", "-222222222"])
def test_income_no_estimate_code_becomes_missing(code):
    out = transform_census_acs(make_bronze([acs_row(income=code)]))
    assert pd.isna(out.loc[0, "median_household_income"])


def test_age_no_estimate_code_becomes_missing():
    out = transform_census_acs(make_bronze([acs_row(age="-666666666")]))
    assert pd.isna(out.loc[0, "median_age"])


# --- 4. Bad values fail loudly --------------------------------------------------

@pytest.mark.parametrize(
    ("field", "value", "column"),
    [
        ("pop", "-666666666", "B01003_001E"),
        ("income", "-5", "B19013_001E"),
        ("income", "abc", "B19013_001E"),
        ("pop", "3.5", "B01003_001E"),
        ("zip_code", "0601", "zip code tabulation area"),
        ("state", "7", "state"),
    ],
)
def test_bad_value_raises_clear_error(field, value, column):
    bronze = make_bronze([acs_row(**{field: value})])
    with pytest.raises(SilverValidationError, match=f"^{re.escape(column)}:"):
        transform_census_acs(bronze)


# --- 5. Structure problems fail loudly ------------------------------------------------

def test_duplicate_zip_raises():
    with pytest.raises(SilverValidationError, match="is duplicated"):
        transform_census_acs(make_bronze([acs_row(), acs_row()]))


def test_name_that_does_not_match_zip_raises():
    with pytest.raises(SilverValidationError, match="^NAME:"):
        transform_census_acs(make_bronze([acs_row(name="ZCTA5 99999")]))


def test_changed_header_raises():
    body = json.dumps([EXPECTED_HEADER[:-1] + ["zcta"], acs_row()])
    with pytest.raises(SilverValidationError, match="header changed"):
        transform_census_acs(make_bronze(body=body))


def test_row_with_missing_value_raises():
    with pytest.raises(SilverValidationError, match="do not have 6 values"):
        transform_census_acs(make_bronze([acs_row()[:5]]))


def test_invalid_json_raises():
    with pytest.raises(SilverValidationError, match="not valid JSON"):
        transform_census_acs(make_bronze(body="<html>Missing Key</html>"))


def test_more_than_one_bronze_row_raises():
    bronze = make_bronze()
    with pytest.raises(SilverValidationError, match="exactly 1 Bronze row"):
        transform_census_acs(pd.concat([bronze, bronze], ignore_index=True))


def test_unexpected_column_raises():
    bronze = make_bronze()
    bronze["extra"] = "x"
    with pytest.raises(SilverValidationError, match=re.escape("Unexpected: ['extra']")):
        transform_census_acs(bronze)