"""Tests for the Silver MCC transform."""

from __future__ import annotations

import pandas as pd
import pytest

from src.silver.mcc import SILVER_SCHEMA, transform_mcc
from src.silver.transactions import SilverValidationError

VERSION = "9675cfab77d6"

BRONZE_COLUMNS = [
    "mcc", "edited_description", "combined_description",
    "usda_description", "irs_description", "irs_reportable",
    "_ingested_at", "_source", "_source_url", "_content_sha256",
]

DEFAULT_ROWS = [
    {"mcc": "5411", "edited_description": "Grocery Stores, Supermarkets"},
    {"mcc": "0742", "edited_description": "Veterinary Services "},
    {"mcc": "5999", "edited_description": " Miscellaneous and Specialty Retail Stores"},
]


def make_bronze(rows: list[dict] | None = None) -> pd.DataFrame:
    """Build a small Bronze MCC table; each row overrides the defaults."""
    records = []
    for row in DEFAULT_ROWS if rows is None else rows:
        record = {
            "mcc": "5411",
            "edited_description": "Some description",
            "combined_description": "",
            "usda_description": "",
            "irs_description": "",
            "irs_reportable": "Yes",
            "_ingested_at": pd.Timestamp("2026-09-25 21:30:50", tz="UTC"),
            "_source": "github_greggles_mcc_codes",
            "_source_url": "https://example.test/mcc_codes.csv",
            "_content_sha256": "a" * 64,
        }
        record.update(row)
        records.append(record)
    return pd.DataFrame(records, columns=BRONZE_COLUMNS)


# --- Happy path ---------------------------------------------------------------

def test_output_matches_schema():
    silver = transform_mcc(make_bronze(), VERSION)
    assert list(silver.columns) == list(SILVER_SCHEMA)
    assert {c: str(t) for c, t in silver.dtypes.items()} == SILVER_SCHEMA
    assert len(silver) == 3


def test_values_are_trimmed_and_sorted():
    silver = transform_mcc(make_bronze(), VERSION)
    assert silver["mcc"].tolist() == ["0742", "5411", "5999"]
    assert silver["description"].tolist() == [
        "Veterinary Services",
        "Grocery Stores, Supermarkets",
        "Miscellaneous and Specialty Retail Stores",
    ]


def test_leading_zero_is_kept():
    silver = transform_mcc(make_bronze([{"mcc": " 0742 "}]), VERSION)
    assert silver["mcc"].tolist() == ["0742"]


def test_version_and_metadata_are_carried():
    silver = transform_mcc(make_bronze(), VERSION)
    assert silver["_version"].eq(VERSION).all()
    assert silver["_source"].eq("github_greggles_mcc_codes").all()
    assert silver["_content_sha256"].eq("a" * 64).all()


def test_input_is_not_modified():
    bronze = make_bronze()
    before = bronze.copy()
    transform_mcc(bronze, VERSION)
    pd.testing.assert_frame_equal(bronze, before)


# --- Rejections ---------------------------------------------------------------

@pytest.mark.parametrize("version", [12345, "main", "9675CFAB77D6", "", None])
def test_bad_version_is_rejected(version):
    with pytest.raises(SilverValidationError, match="not a commit hash"):
        transform_mcc(make_bronze(), version)


def test_missing_column_is_rejected():
    bronze = make_bronze().drop(columns=["edited_description"])
    with pytest.raises(SilverValidationError, match="Bronze columns changed"):
        transform_mcc(bronze, VERSION)


def test_unexpected_column_is_rejected():
    bronze = make_bronze().assign(version=VERSION)
    with pytest.raises(SilverValidationError, match="Bronze columns changed"):
        transform_mcc(bronze, VERSION)


def test_empty_partition_is_rejected():
    with pytest.raises(SilverValidationError, match="has no rows"):
        transform_mcc(make_bronze([]), VERSION)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"mcc": "541"}, "is not 4 digits"),
        ({"mcc": "54A1"}, "is not 4 digits"),
        ({"mcc": None}, "is null or empty"),
        ({"edited_description": "   "}, "is null or empty"),
    ],
)
def test_bad_values_are_rejected(row, message):
    with pytest.raises(SilverValidationError, match=message):
        transform_mcc(make_bronze([row]), VERSION)


def test_duplicate_code_is_rejected():
    bronze = make_bronze([{"mcc": "5411"}, {"mcc": "5411 "}])
    with pytest.raises(SilverValidationError, match="is duplicated"):
        transform_mcc(bronze, VERSION)