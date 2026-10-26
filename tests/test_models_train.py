"""Tests for src/models/train.py."""

import numpy as np
import pandas as pd
import pytest

from src.models.features import CARD_HISTORY_FEATURES, CATEGORIES, CHANNELS, FEATURE_COLUMNS
from src.models.split import GAP_DAYS
from src.models.train import (
    INNER_TRAIN_END,
    INNER_VALID_START,
    PARAM_GRID,
    fit_final_lgbm,
    inner_split,
    make_lgbm,
    make_logistic,
    model_columns,
    tune_lgbm,
)


def fake_training_data(n=3000, seed=0):
    """Random features over the training period, with fraud driven by amount and night."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.uniform(0, 10, n) for c in FEATURE_COLUMNS})
    X["amt"] = rng.uniform(1, 1000, n)
    X["is_night"] = rng.integers(0, 2, n)
    X["category"] = pd.Categorical(rng.choice(CATEGORIES, n), categories=CATEGORIES)
    X["channel"] = pd.Categorical(rng.choice(CHANNELS, n), categories=CHANNELS)
    for col in ["secs_since_prev", "amt_vs_card_median", "median_household_income"]:
        X.loc[rng.random(n) < 0.05, col] = np.nan
    ts = pd.Timestamp("2019-01-01") + pd.to_timedelta(
        rng.integers(0, 539 * 86400, n), unit="s"
    )
    logit = (X["amt"] - 600) / 100 + 2 * X["is_night"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y.to_numpy(), pd.Series(ts)


def test_model_columns_with_and_without_card_history():
    assert model_columns(True) == FEATURE_COLUMNS
    without = model_columns(False)
    assert len(without) == len(FEATURE_COLUMNS) - len(CARD_HISTORY_FEATURES)
    assert not set(CARD_HISTORY_FEATURES) & set(without)


def test_inner_split_boundaries():
    ts = pd.to_datetime([
        "2020-03-24 23:59:59", "2020-03-25 00:00:00", "2020-03-31 23:59:59",
        "2020-04-01 00:00:00", "2020-06-23 23:59:59",
    ])
    tr, va = inner_split(ts)
    assert list(tr) == [True, False, False, False, False]
    assert list(va) == [False, False, False, True, True]


def test_inner_gap_is_seven_days():
    assert (INNER_VALID_START - INNER_TRAIN_END).days - 1 == GAP_DAYS


def test_inner_split_rejects_rows_after_training():
    with pytest.raises(ValueError, match="after the training period"):
        inner_split(pd.to_datetime(["2020-06-24 00:00:00"]))


def test_make_lgbm_weighting_and_params():
    weighted = make_lgbm(True, num_leaves=15)
    plain = make_lgbm(False)
    assert weighted.get_params()["class_weight"] == "balanced"
    assert plain.get_params()["class_weight"] is None
    assert weighted.get_params()["num_leaves"] == 15
    assert weighted.get_params()["metric"] == "average_precision"


def test_logistic_handles_missing_values_and_gives_probabilities():
    X, y, _ = fake_training_data(n=500)
    cols = model_columns(False)
    model = make_logistic(cols).fit(X[cols], y)
    proba = model.predict_proba(X[cols])[:, 1]
    assert proba.shape == (500,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_tune_lgbm_tries_every_combination_best_first():
    X, y, ts = fake_training_data()
    table = tune_lgbm(X, y, ts, weighted=False)
    assert len(table) == np.prod([len(v) for v in PARAM_GRID.values()])
    assert table["inner_ap"].is_monotonic_decreasing
    assert (table["best_iteration"] >= 1).all()


def test_fit_final_uses_the_tuned_settings():
    X, y, _ = fake_training_data(n=500)
    best = {"num_leaves": 15, "min_child_samples": 50, "best_iteration": 37}
    model = fit_final_lgbm(X, y, weighted=True, best=best)
    params = model.get_params()
    assert params["n_estimators"] == 37
    assert params["num_leaves"] == 15
    assert params["class_weight"] == "balanced"
    assert model.booster_.num_trees() == 37