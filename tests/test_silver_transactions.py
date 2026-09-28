"""Tests for the Silver transactions transform (src/silver/transactions.py)."""

import datetime
import hashlib
import hmac
import re

import pandas as pd
import pytest

from src.silver.transactions import (
    SILVER_SCHEMA,
    SilverValidationError,
    empty_silver_frame,
    load_pii_key,
    transform_transactions,
)

# --- 1. Fake inputs -----------------------------------------------------------

# Fake keys used only in tests. Never the real PII_HASH_KEY.
KEY = bytes(range(32))
OTHER_KEY = bytes(range(1, 33))

CARD = "4000123412341234"


def make_row(**overrides: str) -> dict[str, str]:
    """One valid Bronze row; every business value is a string."""
    row = {
        "Unnamed: 0": "0",
        "trans_date_trans_time": "2019-01-01 00:00:18",
        "cc_num": CARD,
        "merchant": "fraud_Test Shop",
        "category": "misc_net",
        "amt": "4.97",
        "first": "Test",
        "last": "Person",
        "gender": "F",
        "street": "1 Test Street",
        "city": "Testville",
        "state": "NC",
        "zip": "28654",
        "lat": "36.0788",
        "long": "-81.1781",
        "city_pop": "3495",
        "job": "Tester",
        "dob": "1988-03-09",
        "trans_num": "t0001",
        "unix_time": "1325376018",
        "merch_lat": "36.011293",
        "merch_long": "-82.048315",
        "is_fraud": "0",
    }
    row.update(overrides)
    return row


def make_bronze(rows: list[dict[str, str]]) -> pd.DataFrame:
    """A Bronze batch with the same column types as the real Parquet files."""
    df = pd.DataFrame(rows, columns=list(make_row())).astype("str")
    df["_ingested_at"] = pd.Timestamp("2026-09-26 10:00:00", tz="UTC")
    df["_source"] = "kaggle"
    df["_source_file"] = "test.csv"
    df["_batch_date"] = pd.Series(
        [datetime.date(2019, 1, 1)] * len(df), index=df.index, dtype="date32[pyarrow]"
    )
    return df


# --- 2. Happy path ------------------------------------------------------------

def test_valid_row_gives_exact_schema_and_values():
    out = transform_transactions(make_bronze([make_row()]), KEY)

    assert list(out.columns) == list(SILVER_SCHEMA)
    assert out.dtypes.equals(empty_silver_frame().dtypes)
    row = out.iloc[0]
    assert row["amt"] == 4.97
    assert row["birth_year"] == 1988
    assert row["merchant"] == "Test Shop"
    assert row["is_fraud"] == 0
    assert row["trans_ts"] == pd.Timestamp("2019-01-01 00:00:18")


def test_personal_and_useless_columns_are_gone():
    out = transform_transactions(make_bronze([make_row()]), KEY)
    for column in ["Unnamed: 0", "unix_time", "first", "last", "street", "cc_num", "dob"]:
        assert column not in out.columns


def test_card_hash_is_hmac_sha256_of_card_number():
    out = transform_transactions(make_bronze([make_row()]), KEY)
    expected = hmac.new(KEY, CARD.encode("utf-8"), hashlib.sha256).hexdigest()
    assert out.loc[0, "card_hash"] == expected


def test_different_key_gives_different_hash():
    bronze = make_bronze([make_row()])
    first = transform_transactions(bronze, KEY).loc[0, "card_hash"]
    second = transform_transactions(bronze, OTHER_KEY).loc[0, "card_hash"]
    assert first != second


def test_zip_that_lost_its_leading_zero_is_padded():
    out = transform_transactions(make_bronze([make_row(zip="1257")]), KEY)
    assert out.loc[0, "zip"] == "01257"


def test_fraud_label_one_becomes_integer_one():
    out = transform_transactions(make_bronze([make_row(is_fraud="1")]), KEY)
    assert out.loc[0, "is_fraud"] == 1


def test_rows_are_sorted_by_transaction_time():
    late = make_row(trans_num="t2", trans_date_trans_time="2019-01-01 23:00:00")
    early = make_row(trans_num="t1", trans_date_trans_time="2019-01-01 01:00:00")
    out = transform_transactions(make_bronze([late, early]), KEY)
    assert out["trans_num"].tolist() == ["t1", "t2"]


def test_empty_batch_keeps_the_full_schema():
    out = transform_transactions(make_bronze([]), KEY)
    assert len(out) == 0
    assert list(out.columns) == list(SILVER_SCHEMA)
    assert out.dtypes.equals(empty_silver_frame().dtypes)


# --- 3. Bad values fail loudly --------------------------------------------------

@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("is_fraud", "2"),
        ("amt", "abc"),
        ("amt", "-5"),
        ("zip", "123456"),
        ("gender", "X"),
        ("state", "nc"),
        ("lat", "95"),
        ("merch_long", "-200"),
        ("city_pop", "3.5"),
        ("trans_date_trans_time", "2019/01/01 00:00:18"),
        ("dob", "2030-01-01"),
        ("city", ""),
    ],
)
def test_bad_value_raises_clear_error(column, bad_value):
    bronze = make_bronze([make_row(**{column: bad_value})])
    with pytest.raises(SilverValidationError, match=f"^{column}:"):
        transform_transactions(bronze, KEY)


def test_merchant_that_is_only_the_prefix_raises():
    bronze = make_bronze([make_row(merchant="fraud_")])
    with pytest.raises(SilverValidationError, match="^merchant:"):
        transform_transactions(bronze, KEY)


def test_duplicate_trans_num_raises():
    bronze = make_bronze([make_row(), make_row()])
    with pytest.raises(SilverValidationError, match="^trans_num:"):
        transform_transactions(bronze, KEY)


def test_missing_column_raises():
    bronze = make_bronze([make_row()]).drop(columns=["amt"])
    with pytest.raises(SilverValidationError, match=re.escape("Missing: ['amt']")):
        transform_transactions(bronze, KEY)


def test_unexpected_column_raises():
    bronze = make_bronze([make_row()])
    bronze["batch_date"] = "2019-01-01"
    with pytest.raises(SilverValidationError, match=re.escape("Unexpected: ['batch_date']")):
        transform_transactions(bronze, KEY)


# --- 4. Privacy and the key ---------------------------------------------------------

@pytest.mark.parametrize(
    ("column", "secret_value"),
    [("cc_num", "4000ABCD12341234"), ("dob", "1988-13-45")],
)
def test_error_message_does_not_leak_personal_data(column, secret_value):
    bronze = make_bronze([make_row(**{column: secret_value})])
    with pytest.raises(SilverValidationError) as error:
        transform_transactions(bronze, KEY)
    assert secret_value not in str(error.value)


def test_hex_string_instead_of_bytes_is_rejected():
    with pytest.raises(TypeError):
        transform_transactions(make_bronze([make_row()]), KEY.hex())


def test_load_pii_key_returns_32_bytes():
    assert load_pii_key({"PII_HASH_KEY": "ab" * 32}) == bytes.fromhex("ab" * 32)


@pytest.mark.parametrize(
    "value",
    ["", "zz" * 32, "ab" * 16],
    ids=["missing", "not-hex", "too-short"],
)
def test_load_pii_key_fails_clearly(value):
    with pytest.raises(RuntimeError, match="PII_HASH_KEY") as error:
        load_pii_key({"PII_HASH_KEY": value})
    if value:
        assert value not in str(error.value)