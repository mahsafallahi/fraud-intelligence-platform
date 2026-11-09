"""XGBoost robustness check for the final model F (Phase 4, step G).

Design (decided 2026-11-09):
- Same 16 FINAL_COLUMNS and the same inner time split as LightGBM (inner_split
  from src.models.train), native categorical support (tree_method "hist").
- No class weights, like F.
- Small grid, mirroring F's: max_depth {4, 6} x min_child_weight {0.25, 2.5}.
  XGBoost's min_child_weight is a sum of hessians, about 0.005 per row at this
  fraud rate, so 0.25 and 2.5 correspond to LightGBM's 50 and 500 rows.
- Early stopping on the inner validation with "aucpr"; the final model is refit
  on the whole training period with the best number of trees.
- It is a check only: F stays the final model unless a person decides otherwise.
"""

import itertools

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.train import EARLY_STOPPING_ROUNDS, SEED, inner_split

XGB_BASE = {
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "enable_categorical": True,
    "eval_metric": "aucpr",
    "random_state": SEED,
    "n_jobs": -1,
}
XGB_GRID = {"max_depth": [4, 6], "min_child_weight": [0.25, 2.5]}


def make_xgb(**params) -> xgb.XGBClassifier:
    """XGBoost classifier with the shared base settings."""
    return xgb.XGBClassifier(**{**XGB_BASE, **params})


def tune_xgb(X: pd.DataFrame, y, ts) -> pd.DataFrame:
    """Try every XGB_GRID combination on the inner split, with early stopping.

    Returns one row per combination, best first: the parameters, the best number
    of trees and the inner-validation average precision ("aucpr").
    """
    y = np.asarray(y)
    tr, va = inner_split(ts)
    rows = []
    names = list(XGB_GRID)
    for values in itertools.product(*XGB_GRID.values()):
        params = dict(zip(names, values))
        model = make_xgb(early_stopping_rounds=EARLY_STOPPING_ROUNDS, **params)
        model.fit(X.loc[tr], y[tr], eval_set=[(X.loc[va], y[va])], verbose=False)
        rows.append({
            **params,
            "best_iteration": int(model.best_iteration) + 1,  # a number of trees, as for LightGBM
            "inner_ap": float(model.best_score),
        })
    return pd.DataFrame(rows).sort_values("inner_ap", ascending=False, ignore_index=True)


def fit_final_xgb(X: pd.DataFrame, y, best: dict) -> xgb.XGBClassifier:
    """Refit on the whole training period with the tuned settings."""
    params = {name: best[name] for name in XGB_GRID}
    model = make_xgb(n_estimators=int(best["best_iteration"]), **params)
    return model.fit(X, np.asarray(y), verbose=False)
