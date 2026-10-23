"""Tests for src/models/features.py, including the leakage test."""

import numpy as np
import pandas as pd
import pytest

from src.models.features import CATEGORIES, FEATURE_COLUMNS, build_features


def make_frame(rows, index=None):
    """Tiny fake model frame: each row needs card_key, trans_ts and amt."""
    base = {
        "category": "travel", "channel": "unknown", "birth_year": 1980,
        "city_pop": 1000, "median_household_income": 50000, "has_income": True,
        "distance_km": 10.0, "is_public_holiday_in_state": False,
        "is_nationwide_public_holiday": False, "is_bank_holiday": False,
    }
    frame = pd.DataFrame([{**base, **r} for r in rows], index=index)
    frame["trans_ts"] = pd.to_datetime(frame["trans_ts"])
    return frame


def test_output_columns_and_original_order():
    frame = make_frame(
        [
            {"card_key": "A", "trans_ts": "2020-01-01 12:00", "amt": 5.0},
            {"card_key": "A", "trans_ts": "2020-01-01 10:00", "amt": 6.0},
            {"card_key": "B", "trans_ts": "2020-01-01 11:00", "amt": 7.0},
        ],
        index=[7, 3, 5],
    )
    result = build_features(frame)
    assert list(result.index) == [7, 3, 5]
    assert list(result.columns) == FEATURE_COLUMNS


def test_secs_since_prev_stays_within_each_card():
    frame = make_frame([
        {"card_key": "A", "trans_ts": "2020-01-01 10:00", "amt": 1.0},
        {"card_key": "B", "trans_ts": "2020-01-01 10:10", "amt": 1.0},
        {"card_key": "A", "trans_ts": "2020-01-01 10:20", "amt": 1.0},
        {"card_key": "B", "trans_ts": "2020-01-01 11:00", "amt": 1.0},
    ])
    secs = build_features(frame)["secs_since_prev"]
    assert np.isnan(secs[0]) and np.isnan(secs[1])
    assert secs[2] == 1200 and secs[3] == 3000


def test_window_counts_exclude_current_and_boundary():
    frame = make_frame([
        {"card_key": "A", "trans_ts": "2020-01-01 00:00", "amt": 1.0},
        {"card_key": "A", "trans_ts": "2020-01-01 00:30", "amt": 1.0},
        {"card_key": "A", "trans_ts": "2020-01-01 00:59", "amt": 1.0},
        {"card_key": "A", "trans_ts": "2020-01-01 01:30", "amt": 1.0},
        {"card_key": "A", "trans_ts": "2020-01-02 00:59", "amt": 1.0},
    ])
    result = build_features(frame)
    assert list(result["card_txn_1h"]) == [0, 1, 2, 1, 0]
    assert list(result["card_txn_24h"]) == [0, 1, 2, 3, 1]


def test_amt_vs_card_median_needs_five_earlier_transactions():
    amounts = [10.0, 20.0, 30.0, 40.0, 50.0, 100.0]
    frame = make_frame([
        {"card_key": "A", "trans_ts": f"2020-01-01 0{i}:00", "amt": a}
        for i, a in enumerate(amounts)
    ])
    ratio = build_features(frame)["amt_vs_card_median"]
    assert ratio[:5].isna().all()
    assert ratio[5] == pytest.approx(100.0 / 30.0)


def test_card_prev_txn_count():
    frame = make_frame([
        {"card_key": "A", "trans_ts": "2020-01-01 10:00", "amt": 1.0},
        {"card_key": "A", "trans_ts": "2020-01-01 11:00", "amt": 1.0},
        {"card_key": "B", "trans_ts": "2020-01-01 12:00", "amt": 1.0},
        {"card_key": "A", "trans_ts": "2020-01-01 13:00", "amt": 1.0},
    ])
    assert list(build_features(frame)["card_prev_txn_count"]) == [0, 1, 0, 2]


def test_same_second_transactions_do_not_see_each_other():
    earlier = [
        {"card_key": "A", "trans_ts": f"2020-01-01 0{i}:00", "amt": 10.0 * (i + 1)}
        for i in range(6)  # 00:00 .. 05:00, amounts 10 .. 60
    ]
    frame = make_frame(earlier + [
        {"card_key": "A", "trans_ts": "2020-01-01 05:30", "amt": 100.0},
        {"card_key": "A", "trans_ts": "2020-01-01 05:30", "amt": 200.0},
        {"card_key": "A", "trans_ts": "2020-01-01 06:00", "amt": 1.0},
    ])
    result = build_features(frame)
    pair, after = result.iloc[6:8], result.iloc[8]

    # Both simultaneous rows see exactly the same, strictly earlier history.
    assert list(pair["secs_since_prev"]) == [1800, 1800]
    assert list(pair["card_prev_txn_count"]) == [6, 6]
    assert list(pair["card_txn_1h"]) == [1, 1]
    assert list(pair["card_txn_24h"]) == [6, 6]
    assert pair["amt_vs_card_median"].tolist() == pytest.approx([100 / 35, 200 / 35])

    # The next transaction sees both of them.
    assert after["secs_since_prev"] == 1800
    assert after["card_prev_txn_count"] == 8
    assert after["card_txn_1h"] == 2


def test_hour_night_and_day_of_week():
    frame = make_frame([
        {"card_key": "A", "trans_ts": "2020-06-01 23:15", "amt": 1.0},  # Monday
        {"card_key": "A", "trans_ts": "2020-06-02 03:59", "amt": 1.0},
        {"card_key": "A", "trans_ts": "2020-06-02 04:00", "amt": 1.0},
        {"card_key": "A", "trans_ts": "2020-06-06 12:00", "amt": 1.0},  # Saturday
    ])
    result = build_features(frame)
    assert list(result["hour"]) == [23, 3, 4, 12]
    assert list(result["is_night"]) == [1, 1, 0, 0]
    assert list(result["day_of_week"]) == [0, 1, 1, 5]


def test_age_uses_transaction_year():
    frame = make_frame([
        {"card_key": "A", "trans_ts": "2019-03-01 10:00", "amt": 1.0},
        {"card_key": "A", "trans_ts": "2020-03-01 10:00", "amt": 1.0},
    ])
    assert list(build_features(frame)["age"]) == [39, 40]


def test_flags_and_missing_income_stay_missing():
    frame = make_frame([
        {"card_key": "A", "trans_ts": "2020-01-01 10:00", "amt": 1.0,
         "median_household_income": None, "has_income": False,
         "is_bank_holiday": True},
    ])
    result = build_features(frame)
    assert np.isnan(result["median_household_income"].iloc[0])
    assert result["has_income"].iloc[0] == 0
    assert result["is_bank_holiday"].iloc[0] == 1


def test_category_codes_are_fixed():
    frame = make_frame([
        {"card_key": "A", "trans_ts": "2020-01-01 10:00", "amt": 1.0},
    ])
    category = build_features(frame)["category"]
    assert list(category.cat.categories) == CATEGORIES
    assert category.cat.codes.iloc[0] == CATEGORIES.index("travel")


def test_future_rows_do_not_change_past_features():
    """Leakage test: features for rows up to a cutoff must not depend on later rows."""
    rng = np.random.default_rng(42)
    rows = []
    for card in ["A", "B", "C"]:
        seconds = rng.choice(10 * 24 * 3600, size=60, replace=False)
        for s in seconds:
            rows.append({
                "card_key": card,
                "trans_ts": pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=int(s)),
                "amt": float(rng.uniform(1, 500)),
                "category": str(rng.choice(CATEGORIES)),
            })
    rows.append({**rows[0], "amt": 999.0})  # a same-second pair, as in the real data
    frame = make_frame(rows).sample(frac=1, random_state=0)  # unsorted input
    cutoff = pd.Timestamp("2020-01-05 12:00")
    past = frame[frame["trans_ts"] <= cutoff]

    full = build_features(frame).loc[past.index]
    truncated = build_features(past)
    pd.testing.assert_frame_equal(full, truncated)

    # Positive control: a card-wide mean (uses the future) DOES change.
    naive_full = frame.groupby("card_key")["amt"].transform("mean").loc[past.index]
    naive_past = past.groupby("card_key")["amt"].transform("mean")
    assert not np.allclose(naive_full, naive_past)


def test_missing_column_fails_loudly():
    frame = make_frame([
        {"card_key": "A", "trans_ts": "2020-01-01 10:00", "amt": 1.0},
    ]).drop(columns="distance_km")
    with pytest.raises(ValueError, match="missing columns"):
        build_features(frame)


def test_unknown_category_fails_loudly():
    frame = make_frame([
        {"card_key": "A", "trans_ts": "2020-01-01 10:00", "amt": 1.0,
         "category": "crypto"},
    ])
    with pytest.raises(ValueError, match="unknown or missing category"):
        build_features(frame)