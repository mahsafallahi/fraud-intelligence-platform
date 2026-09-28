"""Silver transform for credit-card transactions.

One Bronze batch in (every business column is a string), one typed and
validated Silver DataFrame out. No file I/O here: the caller (a notebook
or an Airflow task) reads Bronze and writes Silver.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Mapping

import numpy as np
import pandas as pd

# --- 1. Constants and schema -------------------------------------------------

PII_KEY_ENV_VAR = "PII_HASH_KEY"
PII_KEY_BYTES = 32  # 64 hex characters in .env

METADATA_COLUMNS = ["_ingested_at", "_source", "_source_file", "_batch_date"]

BRONZE_BUSINESS_COLUMNS = [
    "Unnamed: 0", "trans_date_trans_time", "cc_num", "merchant", "category",
    "amt", "first", "last", "gender", "street", "city", "state", "zip",
    "lat", "long", "city_pop", "job", "dob", "trans_num", "unix_time",
    "merch_lat", "merch_long", "is_fraud",
]
EXPECTED_BRONZE_COLUMNS = frozenset(BRONZE_BUSINESS_COLUMNS + METADATA_COLUMNS)

# Dropped on purpose (reasons in docs/PROJECT_STATUS.md).
DROPPED_COLUMNS = ["Unnamed: 0", "unix_time", "first", "last", "street"]

# Output columns, in this exact order, with these exact types.
SILVER_SCHEMA: dict[str, str] = {
    "trans_num": "str",
    "trans_ts": "datetime64[us]",
    "card_hash": "str",
    "merchant": "str",
    "category": "str",
    "amt": "float64",
    "gender": "str",
    "city": "str",
    "state": "str",
    "zip": "str",
    "lat": "float64",
    "long": "float64",
    "city_pop": "int64",
    "job": "str",
    "birth_year": "int16",
    "merch_lat": "float64",
    "merch_long": "float64",
    "is_fraud": "int8",
    "_ingested_at": "datetime64[us, UTC]",
    "_source": "str",
    "_source_file": "str",
        "_batch_date": "date32[pyarrow]",
}

TS_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"
MERCHANT_PREFIX = "fraud_"
VALID_GENDERS = ("F", "M")
VALID_LABELS = ("0", "1")
MIN_BIRTH_YEAR = 1900


# --- 2. Errors and the hash key ----------------------------------------------

class SilverValidationError(ValueError):
    """A Bronze batch contains values the Silver layer refuses to accept."""


def load_pii_key(env: Mapping[str, str] | None = None) -> bytes:
    """Read PII_HASH_KEY (64 hex characters) and return it as 32 bytes.

    The key value is never included in any message.
    """
    source = os.environ if env is None else env
    raw = (source.get(PII_KEY_ENV_VAR) or "").strip()
    if not raw:
        raise RuntimeError(
            f"{PII_KEY_ENV_VAR} is not set. Add it to .env as 64 hex characters."
        )
    try:
        key = bytes.fromhex(raw)
    except ValueError:
        raise RuntimeError(f"{PII_KEY_ENV_VAR} is not valid hex.") from None
    if len(key) != PII_KEY_BYTES:
        raise RuntimeError(
            f"{PII_KEY_ENV_VAR} must be {PII_KEY_BYTES * 2} hex characters, "
            f"got {len(raw)}."
        )
    return key


def _check_key(key: object) -> None:
    if not isinstance(key, bytes) or len(key) != PII_KEY_BYTES:
        raise TypeError(
            f"key must be {PII_KEY_BYTES} bytes; load it with load_pii_key()."
        )


# --- 3. Validation helpers ---------------------------------------------------

def _check(
    bad: pd.Series,
    column: str,
    problem: str,
    values: pd.Series,
    show_examples: bool = True,
) -> None:
    """Raise SilverValidationError if any row is flagged in `bad`."""
    count = int(bad.sum())
    if count == 0:
        return
    message = f"{column}: {count} row(s) {problem}"
    if show_examples:
        message += f", e.g. {values[bad].head(3).tolist()}"
    raise SilverValidationError(message)


def _check_columns(bronze: pd.DataFrame) -> None:
    actual = set(bronze.columns)
    missing = sorted(EXPECTED_BRONZE_COLUMNS - actual)
    unexpected = sorted(actual - EXPECTED_BRONZE_COLUMNS)
    if missing or unexpected:
        raise SilverValidationError(
            f"Bronze columns changed. Missing: {missing}. Unexpected: {unexpected}."
        )


def _check_no_missing(df: pd.DataFrame) -> None:
    for column in df.columns:
        values = df[column]
        missing = values.isna()
        if column not in METADATA_COLUMNS:
            missing = missing | values.str.strip().eq("")
        _check(missing, column, "is null or empty", values, show_examples=False)


def _parse_datetime(
    values: pd.Series, column: str, fmt: str, show_examples: bool = True
) -> pd.Series:
    parsed = pd.to_datetime(values, format=fmt, errors="coerce")
    _check(parsed.isna(), column, f"does not match {fmt}", values, show_examples)
    return parsed


def _parse_float(
    values: pd.Series,
    column: str,
    low: float | None = None,
    high: float | None = None,
) -> pd.Series:
    parsed = pd.to_numeric(values, errors="coerce").astype("float64")
    _check(~np.isfinite(parsed), column, "is not a finite number", values)
    if low is not None:
        _check(parsed < low, column, f"is below {low}", values)
    if high is not None:
        _check(parsed > high, column, f"is above {high}", values)
    return parsed


# --- 4. Hashing --------------------------------------------------------------

def _hash_cards(card_numbers: pd.Series, key: bytes) -> pd.Series:
    """HMAC-SHA256 of each card number; each distinct card is hashed once."""
    lookup = {
        number: hmac.new(key, number.encode("utf-8"), hashlib.sha256).hexdigest()
        for number in card_numbers.unique()
    }
    return card_numbers.map(lookup)


# --- 5. The transform --------------------------------------------------------

def transform_transactions(bronze: pd.DataFrame, key: bytes) -> pd.DataFrame:
    """Turn one Bronze transactions batch into a Silver DataFrame.

    Raises SilverValidationError on any value Silver will not accept.
    """
    _check_key(key)
    _check_columns(bronze)
    if bronze.empty:
        return empty_silver_frame()

    df = bronze.drop(columns=DROPPED_COLUMNS)
    _check_no_missing(df)

    trans_ts = _parse_datetime(
        df["trans_date_trans_time"], "trans_date_trans_time", TS_FORMAT
    )

    dob = _parse_datetime(df["dob"], "dob", DATE_FORMAT, show_examples=False)
    _check(dob.dt.year < MIN_BIRTH_YEAR, "dob", f"is before {MIN_BIRTH_YEAR}",
           df["dob"], show_examples=False)
    _check(dob > trans_ts, "dob", "is after the transaction time",
           df["dob"], show_examples=False)

    _check(~df["cc_num"].str.fullmatch(r"\d+"), "cc_num", "is not digits only",
           df["cc_num"], show_examples=False)

    _check(~df["zip"].str.fullmatch(r"\d{1,5}"), "zip", "is not 1-5 digits", df["zip"])
    zip_code = df["zip"].str.zfill(5)

    _check(~df["is_fraud"].isin(VALID_LABELS), "is_fraud", "is not '0' or '1'",
           df["is_fraud"])
    _check(~df["gender"].isin(VALID_GENDERS), "gender", "is not 'F' or 'M'",
           df["gender"])
    _check(~df["state"].str.fullmatch(r"[A-Z]{2}"), "state",
           "is not a 2-letter code", df["state"])
    _check(~df["city_pop"].str.fullmatch(r"\d+"), "city_pop",
           "is not a whole number", df["city_pop"])
    _check(df["trans_num"].duplicated(keep=False), "trans_num",
           "is duplicated in this batch", df["trans_num"])

    merchant = df["merchant"].str.removeprefix(MERCHANT_PREFIX)
    _check(merchant.str.strip().eq(""), "merchant",
           "is empty after removing the prefix", df["merchant"])

    amt = _parse_float(df["amt"], "amt")
    _check(amt <= 0, "amt", "is not positive", df["amt"])

    silver = pd.DataFrame({
        "trans_num": df["trans_num"],
        "trans_ts": trans_ts,
        "card_hash": _hash_cards(df["cc_num"], key),
        "merchant": merchant,
        "category": df["category"],
        "amt": amt,
        "gender": df["gender"],
        "city": df["city"],
        "state": df["state"],
        "zip": zip_code,
        "lat": _parse_float(df["lat"], "lat", -90, 90),
        "long": _parse_float(df["long"], "long", -180, 180),
        "city_pop": df["city_pop"].astype("int64"),
        "job": df["job"],
        "birth_year": dob.dt.year,
        "merch_lat": _parse_float(df["merch_lat"], "merch_lat", -90, 90),
        "merch_long": _parse_float(df["merch_long"], "merch_long", -180, 180),
        "is_fraud": df["is_fraud"].astype("int8"),
        "_ingested_at": pd.to_datetime(df["_ingested_at"], utc=True),
        "_source": df["_source"],
        "_source_file": df["_source_file"],
                "_batch_date": df["_batch_date"],
    })

    silver = silver.sort_values(["trans_ts", "trans_num"]).reset_index(drop=True)
    return _enforce_schema(silver)


# --- 6. Schema helpers -------------------------------------------------------

def empty_silver_frame() -> pd.DataFrame:
    """Zero rows, but the exact Silver columns and types."""
    return pd.DataFrame(
        {column: pd.Series(dtype=dtype) for column, dtype in SILVER_SCHEMA.items()}
    )


def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    return df[list(SILVER_SCHEMA)].astype(SILVER_SCHEMA)