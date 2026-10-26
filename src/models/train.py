"""Model training for the fraud detector (Phase 4, steps 5-6).

Design (decided 2026-10-25):
- Main model: LightGBM (native categorical features and missing values, fast on
  1.3M rows). XGBoost is used later only as a robustness check.
- Four variants: class weighting {off, on} x card-history features {off, on}.
- Hyperparameters and the number of trees are chosen on an INNER time split
  inside the training period: inner train up to 2020-03-24, a 7-day gap, inner
  validation 2020-04-01 .. 2020-06-23. The final model is then refit on the
  whole training period with the chosen settings. The validation quarter stays
  clean for comparing variants, fitting the calibrator and choosing the
  threshold; the test quarter is used once, at the end.
- Small, deliberate grid: with about 100 independent fraud episodes per quarter,
  a large search would tune to noise.
- Early stopping uses average precision (PR-AUC), the same ruler as the
  evaluation; the default log-loss would change meaning under class weights.
- Logistic regression baseline: log of skewed features, median imputation with
  missing-value indicators, scaling and one-hot categories, all fitted on the
  training rows only.
"""

import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.models.features import CARD_HISTORY_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS
from src.models.split import GAP_DAYS, TRAIN_END

INNER_VALID_START = pd.Timestamp("2020-04-01")
INNER_TRAIN_END = INNER_VALID_START - pd.Timedelta(days=GAP_DAYS + 1)  # 2020-03-24

SEED = 42
LOG_FEATURES = ["amt", "secs_since_prev", "amt_vs_card_median", "city_pop"]
BASE_PARAMS = {
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "metric": "average_precision",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}
PARAM_GRID = {"num_leaves": [15, 63], "min_child_samples": [50, 500]}
EARLY_STOPPING_ROUNDS = 100


def model_columns(with_card_history: bool) -> list:
    """Feature columns for a variant: all 19, or 18 without card history."""
    if with_card_history:
        return list(FEATURE_COLUMNS)
    return [c for c in FEATURE_COLUMNS if c not in CARD_HISTORY_FEATURES]


def inner_split(ts) -> tuple:
    """Boolean masks (inner_train, inner_valid) for rows of the training period."""
    day = pd.to_datetime(pd.Series(ts)).dt.normalize()
    if (day > TRAIN_END).any():
        raise ValueError("rows after the training period")
    return (day <= INNER_TRAIN_END).to_numpy(), (day >= INNER_VALID_START).to_numpy()


def make_logistic(columns: list) -> Pipeline:
    """Logistic regression baseline with preprocessing fitted on training rows only."""
    log_cols = [c for c in LOG_FEATURES if c in columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in columns]
    plain = [c for c in columns if c not in log_cols + cat_cols]
    prep = ColumnTransformer([
        ("log", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                              FunctionTransformer(np.log1p), StandardScaler()), log_cols),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                              StandardScaler()), plain),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000))])


def make_lgbm(weighted: bool, **params) -> lgb.LGBMClassifier:
    """LightGBM classifier; weighted=True balances the classes in the loss."""
    return lgb.LGBMClassifier(
        **{**BASE_PARAMS, **params},
        class_weight="balanced" if weighted else None,
    )


def tune_lgbm(X: pd.DataFrame, y, ts, weighted: bool) -> pd.DataFrame:
    """Try every PARAM_GRID combination on the inner split, with early stopping.

    Returns one row per combination, best first: the parameters, the best number
    of trees and the inner-validation average precision.
    """
    y = np.asarray(y)
    tr, va = inner_split(ts)
    rows = []
    names = list(PARAM_GRID)
    for values in itertools.product(*PARAM_GRID.values()):
        params = dict(zip(names, values))
        model = make_lgbm(weighted, **params)
        model.fit(
            X.loc[tr], y[tr],
            eval_X=(X.loc[va],),
            eval_y=(y[va],),
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        rows.append({
            **params,
            "best_iteration": int(model.best_iteration_),
            "inner_ap": float(model.best_score_["valid_0"]["average_precision"]),
        })
    return pd.DataFrame(rows).sort_values("inner_ap", ascending=False, ignore_index=True)


def fit_final_lgbm(X: pd.DataFrame, y, weighted: bool, best: dict) -> lgb.LGBMClassifier:
    """Refit on the whole training period with the tuned settings."""
    params = {name: best[name] for name in PARAM_GRID}
    model = make_lgbm(weighted, n_estimators=int(best["best_iteration"]), **params)
    return model.fit(X, np.asarray(y))