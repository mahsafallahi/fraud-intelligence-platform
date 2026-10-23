"""Tests for src/models/split.py — boundary days are where off-by-one bugs live."""

from datetime import date

import pandas as pd
import pytest

from src.models.split import (
    GAP,
    GAP_DAYS,
    TEST,
    TEST_START,
    TRAIN,
    TRAIN_END,
    VALID_END,
    VALID_START,
    VALIDATION,
    assign_split,
)


@pytest.mark.parametrize(
    "ts, expected",
    [
        ("2019-01-01 00:00:00", TRAIN),
        ("2020-06-23 23:59:59", TRAIN),
        ("2020-06-24 00:00:00", GAP),
        ("2020-06-30 23:59:59", GAP),
        ("2020-07-01 00:00:00", VALIDATION),
        ("2020-09-23 23:59:59", VALIDATION),
        ("2020-09-24 00:00:00", GAP),
        ("2020-09-30 23:59:59", GAP),
        ("2020-10-01 00:00:00", TEST),
        ("2020-12-31 23:59:59", TEST),
    ],
)
def test_boundary_days(ts, expected):
    result = assign_split(pd.Series(pd.to_datetime([ts])))
    assert result.iloc[0] == expected


def test_gaps_are_exactly_seven_days():
    assert (VALID_START - TRAIN_END).days - 1 == GAP_DAYS
    assert (TEST_START - VALID_END).days - 1 == GAP_DAYS


def test_keeps_index_and_name():
    ts = pd.Series(pd.to_datetime(["2019-05-01", "2020-11-01"]), index=[10, 42])
    result = assign_split(ts)
    assert list(result.index) == [10, 42]
    assert result.name == "split"
    assert list(result) == [TRAIN, TEST]


def test_accepts_plain_dates():
    ts = pd.Series([date(2020, 6, 23), date(2020, 7, 1)])
    assert list(assign_split(ts)) == [TRAIN, VALIDATION]


def test_empty_input_gives_empty_output():
    result = assign_split(pd.Series([], dtype="datetime64[ns]"))
    assert len(result) == 0


def test_missing_timestamp_fails_loudly():
    ts = pd.Series(pd.to_datetime(["2019-05-01", None]))
    with pytest.raises(ValueError, match="missing"):
        assign_split(ts)


@pytest.mark.parametrize("ts", ["2018-12-31 23:59:59", "2021-01-01 00:00:00"])
def test_out_of_range_fails_loudly(ts):
    with pytest.raises(ValueError, match="outside"):
        assign_split(pd.Series(pd.to_datetime([ts])))