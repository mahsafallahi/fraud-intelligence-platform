"""Tests for src/models/xgb.py."""

import numpy as np
import pandas as pd

from src.models.features import CATEGORIES, CHANNELS
from src.models.train import FINAL_COLUMNS
from src.models.xgb import XGB_GRID, fit_final_xgb, make_xgb, tune_xgb


def fake_training_data(n=3000, seed=0):
    """Random F features over the training period, with fraud driven by amount and night."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.uniform(0, 10, n) for c in FINAL_COLUMNS})
    X["amt"] = rng.uniform(1, 1000, n)
    X["is_night"] = rng.integers(0, 2, n)
    X["category"] = pd.Categorical(rng.choice(CATEGORIES, n), categories=CATEGORIES)
    X["channel"] = pd.Categorical(rng.choice(CHANNELS, n), categories=CHANNELS)
    X.loc[rng.random(n) < 0.05, "secs_since_prev"] = np.nan
    ts = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 539 * 86400, n), unit="s")
    logit = (X["amt"] - 600) / 100 + 2 * X["is_night"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y.to_numpy(), pd.Series(ts)


def test_make_xgb_settings():
    params = make_xgb(max_depth=4).get_params()
    assert params["max_depth"] == 4
    assert params["enable_categorical"] is True and params["tree_method"] == "hist"
    assert params["eval_metric"] == "aucpr" and params["scale_pos_weight"] is None


def test_tune_xgb_tries_every_combination_best_first():
    X, y, ts = fake_training_data()
    table = tune_xgb(X, y, ts)
    assert len(table) == np.prod([len(v) for v in XGB_GRID.values()])
    assert table["inner_ap"].is_monotonic_decreasing
    assert (table["best_iteration"] >= 1).all()


def test_fit_final_uses_the_tuned_settings_and_categoricals():
    X, y, _ = fake_training_data(n=800)
    best = {"max_depth": 4, "min_child_weight": 0.25, "best_iteration": 23}
    model = fit_final_xgb(X, y, best)
    assert model.get_params()["n_estimators"] == 23
    assert model.get_booster().num_boosted_rounds() == 23
    proba = model.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_same_data_gives_identical_predictions():
    X, y, _ = fake_training_data(n=800)
    best = {"max_depth": 4, "min_child_weight": 0.25, "best_iteration": 15}
    a = fit_final_xgb(X, y, best).predict_proba(X)[:, 1]
    b = fit_final_xgb(X, y, best).predict_proba(X)[:, 1]
    assert np.array_equal(a, b)
