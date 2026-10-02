"""Tests for the Silver holidays transform (src/silver/holidays.py)."""

import json
import re

import pandas as pd
import pytest

from src.silver.holidays import SILVER_SCHEMA, transform_holidays
from src.silver.transactions import SilverValidationError

# --- 1. Fake inputs -----------------------------------------------------------

def holiday(date="2019-01-01", name="New Year's Day", is_global=True,
            counties=None, types=("Public", "Bank")):
    """One holiday exactly as the Nager.Date API returns it."""
    return {
        "date": date,
        "localName": name,
        "name": name,
        "countryCode": "US",
        "fixed": False,
        "global": is_global,
        "counties": counties,
        "launchYear": None,
        "types": list(types),
    }


def make_bronze(responses=None):
    """Bronze rows: one raw API response per year, typed like the real Parquet."""
    responses = responses if responses is not None else {2019: [holiday()]}
    years = list(responses)
    return pd.DataFrame({
        "response_body": pd.Series([json.dumps(responses[y]) for y in years], dtype="str"),
        "_ingested_at": [pd.Timestamp("2026-09-25 21:30:00", tz="UTC")] * len(years),
        "_source": pd.Series(["nager_date_api"] * len(years), dtype="str"),
        "_request_url": pd.Series(
            [f"https://date.nager.at/api/v3/PublicHolidays/{y}/US" for y in years], dtype="str"),
        "_country": pd.Series(["US"] * len(years), dtype="str"),
        "_year": pd.Series(years, dtype="int32"),
    })


# --- 2. Happy path ------------------------------------------------------------

def test_nationwide_holiday_gives_one_row_with_no_state():
    out = transform_holidays(make_bronze())

    assert list(out.columns) == list(SILVER_SCHEMA)
    assert out.astype(SILVER_SCHEMA).dtypes.equals(out.dtypes)
    assert len(out) == 1
    row = out.iloc[0]
    assert str(row["holiday_date"]) == "2019-01-01"
    assert pd.isna(row["state_code"])
    assert row["is_global"]
    assert row["types"] == "Bank,Public"
    assert row["_year"] == 2019


def test_state_level_holiday_gives_one_row_per_state_without_prefix():
    item = holiday("2019-05-08", "Truman Day", is_global=False, counties=["US-MO", "US-CA"])
    out = transform_holidays(make_bronze({2019: [item]}))
    assert out["state_code"].tolist() == ["CA", "MO"]
    assert not out["is_global"].any()


def test_nationwide_and_state_level_versions_on_same_day_are_both_kept():
    nationwide = holiday("2019-10-14", "Columbus Day", types=("Bank",))
    by_state = holiday("2019-10-14", "Columbus Day", is_global=False,
                       counties=["US-AL", "US-AZ"], types=("Public",))
    out = transform_holidays(make_bronze({2019: [nationwide, by_state]}))

    assert len(out) == 3
    assert out["state_code"].isna().tolist() == [True, False, False]
    assert out["types"].tolist() == ["Bank", "Public", "Public"]


def test_two_years_in_one_batch():
    out = transform_holidays(make_bronze({
        2019: [holiday()],
        2020: [holiday(date="2020-01-01")],
    }))
    assert out.groupby("_year").size().to_dict() == {2019: 1, 2020: 1}


def test_rows_are_sorted_by_date():
    out = transform_holidays(make_bronze({2019: [
        holiday("2019-12-25", "Christmas Day"),
        holiday("2019-01-01", "New Year's Day"),
    ]}))
    assert out["name"].tolist() == ["New Year's Day", "Christmas Day"]


# --- 3. Bad holidays fail loudly --------------------------------------------------

BAD_ITEMS = [
    ({**holiday(), "counties": ["US-CA"]}, "must not list states"),
    (holiday(is_global=False, counties=None), "at least one state"),
    (holiday(is_global=False, counties=[]), "at least one state"),
    (holiday(is_global=False, counties=["CA"]), "is not like US-XX"),
    (holiday(date="2019/01/01"), "is not YYYY-MM-DD"),
    (holiday(date="2019-02-30"), "is not a real day"),
    (holiday(date="2020-01-01"), "outside year 2019"),
    ({**holiday(), "countryCode": "CA"}, "does not match US"),
    ({**holiday(), "global": "yes"}, "must be true or false"),
    (holiday(types=()), "types must be"),
    ({**holiday(), "extra": 1}, "fields changed"),
    (holiday(name=" "), "without a name"),
]


@pytest.mark.parametrize(
    ("item", "message"),
    BAD_ITEMS,
    ids=["global-with-states", "no-states", "empty-states", "bad-state-code",
         "bad-date-format", "impossible-date", "wrong-year", "wrong-country",
         "global-not-bool", "no-types", "extra-field", "no-name"],
)
def test_bad_holiday_raises_clear_error(item, message):
    with pytest.raises(SilverValidationError, match=re.escape(message)):
        transform_holidays(make_bronze({2019: [item]}))


def test_same_holiday_and_state_twice_raises():
    with pytest.raises(SilverValidationError, match="share holiday_date, name and state_code"):
        transform_holidays(make_bronze({2019: [holiday(), holiday()]}))


# --- 4. Structure problems fail loudly ----------------------------------------------

def test_invalid_json_raises():
    bronze = make_bronze()
    bronze["response_body"] = pd.Series(["<html>error</html>"], dtype="str")
    with pytest.raises(SilverValidationError, match="not valid JSON"):
        transform_holidays(bronze)


def test_empty_holiday_list_raises():
    with pytest.raises(SilverValidationError, match="non-empty JSON list"):
        transform_holidays(make_bronze({2019: []}))


def test_two_bronze_rows_for_same_year_raise():
    bronze = make_bronze()
    with pytest.raises(SilverValidationError, match="same country and year"):
        transform_holidays(pd.concat([bronze, bronze], ignore_index=True))


def test_unexpected_column_raises():
    bronze = make_bronze()
    bronze["extra"] = "x"
    with pytest.raises(SilverValidationError, match=re.escape("Unexpected: ['extra']")):
        transform_holidays(bronze)


def test_empty_bronze_raises():
    with pytest.raises(SilverValidationError, match="has no rows"):
        transform_holidays(make_bronze({}))