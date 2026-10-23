"""Time-based train / validation / test split for the fraud model.

All split boundaries live here, in one place, so the notebooks, the
feature code, training, calibration and the API use exactly the same dates.

Design (decided 2026-10-23 from monthly volumes in notebook 04):
- train       2019-01-01 .. 2020-06-23
- gap         2020-06-24 .. 2020-06-30   (7 days)
- validation  2020-07-01 .. 2020-09-23
- gap         2020-09-24 .. 2020-09-30   (7 days)
- test        2020-10-01 .. 2020-12-31

The 7-day gaps are longer than twice the longest fraud episode of a card
(2.89 days), so no episode can sit on both sides of a boundary. Gap rows are
excluded from evaluation but are still used as card history for features.
"""

import numpy as np
import pandas as pd

TRAIN_START = pd.Timestamp("2019-01-01")
TRAIN_END = pd.Timestamp("2020-06-23")
VALID_START = pd.Timestamp("2020-07-01")
VALID_END = pd.Timestamp("2020-09-23")
TEST_START = pd.Timestamp("2020-10-01")
TEST_END = pd.Timestamp("2020-12-31")

GAP_DAYS = 7

TRAIN = "train"
GAP = "gap"
VALIDATION = "validation"
TEST = "test"


def assign_split(ts: pd.Series) -> pd.Series:
    """Return the split label (train / gap / validation / test) for each timestamp.

    Comparison is by calendar day, so 23:59 on the last train day is still train.
    Raises ValueError on missing values or dates outside 2019-01-01 .. 2020-12-31,
    so a bad row fails loudly instead of silently dropping out of evaluation.
    """
    day = pd.to_datetime(ts).dt.normalize()

    if day.isna().any():
        raise ValueError(f"{int(day.isna().sum())} missing timestamps")

    outside = (day < TRAIN_START) | (day > TEST_END)
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} timestamps outside "
            f"{TRAIN_START.date()} .. {TEST_END.date()}"
        )

    labels = np.select(
        [
            day <= TRAIN_END,
            (day >= VALID_START) & (day <= VALID_END),
            day >= TEST_START,
        ],
        [TRAIN, VALIDATION, TEST],
        default=GAP,
    )
    return pd.Series(labels, index=ts.index, name="split")