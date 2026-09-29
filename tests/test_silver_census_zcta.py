"""Tests for the Silver Census Gazetteer ZCTA transform (src/silver/census_zcta.py)."""

import re

import pandas as pd
import pytest

from src.silver.census_zcta import SILVER_SCHEMA, transform_census_zcta
from src.silver.transactions import SilverValidationError

# --- 1. Fake inputs -----------------------------------------------------------

# The real file pads the last column name and all of its values with spaces.
PADDED_LONG = "INTPTLONG" + " " * 20


def zcta_row(geoid="00601", aland="166659747", awater="799292",
             lat="18.180555", lon="-66.749961"):
    """One Gazetteer row as Bronze stores it (all strings, last column padded)."""
    return {
        "GEOID": geoid,
        "ALAND": aland,
        "AWATER": awater,
        "ALAND_SQMI": "64.348",
        "AWATER_SQMI": "0.309",
        "INTPTLAT": lat,
        PADDED_LONG: lon + " " * 10,
    }


def make_bronze(rows=None):
    rows = rows if rows is not None else [zcta_row()]
    df = pd.DataFrame(rows, columns=list(zcta_row())).astype("str")
    df["_ingested_at"] = pd.Timestamp("2026-09-25 21:30:51", tz="UTC")
    df["_source"] = "census_gazetteer_zcta"
    df["_source_url"] = "https://www2.census.gov/geo/docs/2019_Gaz_zcta_national.zip"
    df["_source_file"] = "2019_Gaz_zcta_national.txt"
    df["_content_sha256"] = "a" * 64
    return df


# --- 2. Happy path ------------------------------------------------------------

def test_valid_row_gives_exact_schema_and_trimmed_values():
    out = transform_census_zcta(make_bronze(), vintage=2019)

    assert {c: str(t) for c, t in out.dtypes.items()} == SILVER_SCHEMA
    row = out.iloc[0]
    assert row["zip"] == "00601"
    assert row["land_area_m2"] == 166659747
    assert row["water_area_m2"] == 799292
    assert row["centroid_lat"] == 18.180555
    assert row["centroid_long"] == -66.749961
    assert row["_vintage"] == 2019


def test_rounded_square_mile_columns_are_dropped():
    out = transform_census_zcta(make_bronze(), vintage=2019)
    assert "ALAND_SQMI" not in out.columns
    assert "AWATER_SQMI" not in out.columns


@pytest.mark.parametrize(
    ("geoid", "lat", "lon"),
    [("96799", "-14.223174", "-169.517743"), ("96910", "13.452852", "144.747191")],
    ids=["american-samoa", "guam"],
)
def test_island_territories_are_valid_coordinates(geoid, lat, lon):
    out = transform_census_zcta(make_bronze([zcta_row(geoid=geoid, lat=lat, lon=lon)]), vintage=2019)
    assert out.loc[0, "centroid_lat"] == float(lat)
    assert out.loc[0, "centroid_long"] == float(lon)


def test_rows_are_sorted_by_zip():
    out = transform_census_zcta(make_bronze([zcta_row("00603"), zcta_row("00601")]), vintage=2019)
    assert out["zip"].tolist() == ["00601", "00603"]


# --- 3. Bad values fail loudly --------------------------------------------------

@pytest.mark.parametrize(
    ("field", "value", "column"),
    [
        ("geoid", "601", "GEOID"),
        ("aland", "-5", "ALAND"),
        ("aland", "", "ALAND"),
        ("awater", "1.5", "AWATER"),
        ("lat", "95", "INTPTLAT"),
        ("lon", "abc", "INTPTLONG"),
    ],
)
def test_bad_value_raises_clear_error(field, value, column):
    bronze = make_bronze([zcta_row(**{field: value})])
    with pytest.raises(SilverValidationError, match=f"^{column}:"):
        transform_census_zcta(bronze, vintage=2019)


def test_duplicate_zip_raises():
    with pytest.raises(SilverValidationError, match="is duplicated"):
        transform_census_zcta(make_bronze([zcta_row(), zcta_row()]), vintage=2019)


# --- 4. Structure problems fail loudly ----------------------------------------------

def test_missing_column_raises():
    bronze = make_bronze().drop(columns=["ALAND"])
    with pytest.raises(SilverValidationError, match=re.escape("Missing: ['ALAND']")):
        transform_census_zcta(bronze, vintage=2019)


def test_unexpected_column_raises():
    bronze = make_bronze()
    bronze["extra"] = "x"
    with pytest.raises(SilverValidationError, match=re.escape("Unexpected: ['extra']")):
        transform_census_zcta(bronze, vintage=2019)


def test_two_columns_with_same_name_after_trimming_raise():
    bronze = make_bronze()
    bronze["GEOID "] = "00601"
    with pytest.raises(SilverValidationError, match="same name after trimming"):
        transform_census_zcta(bronze, vintage=2019)


def test_empty_partition_raises():
    with pytest.raises(SilverValidationError, match="has no rows"):
        transform_census_zcta(make_bronze([]), vintage=2019)


@pytest.mark.parametrize("vintage", [1999, 2101, "2019"])
def test_implausible_vintage_raises(vintage):
    with pytest.raises(SilverValidationError, match="not a plausible year"):
        transform_census_zcta(make_bronze(), vintage=vintage)