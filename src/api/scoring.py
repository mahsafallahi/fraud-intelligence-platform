"""Score one transaction with model F (Phase 6).

No feature logic lives here: the snapshot store returns the model frame (the
card's earlier transactions + the new one), build_features() from
src/models/features.py turns it into features exactly as offline, and the
new row is scored by the bundle's booster.

Outputs:
- raw_score: F's log-odds (LightGBM raw_score=True)
- probability_raw: F's probability; the alert rule is probability_raw >= threshold
- probability_calibrated: the Platt calibrator applied to probability_raw
- reasons: the (at most) three largest POSITIVE SHAP contributions (LightGBM
  pred_contrib, log-odds), with the feature's value; categories as names.
  Negative contributions argue against fraud, so they are not reasons.
"""

import math
from dataclasses import dataclass

import numpy as np

from src.api.bundle import Bundle
from src.api.history import SnapshotStore
from src.models.features import build_features

N_REASONS = 3
PREDICT_PARAMS = {"num_threads": 1}  # one row: threads cost more than they save


@dataclass(frozen=True)
class Score:
    raw_score: float
    probability_raw: float
    probability_calibrated: float
    alert: bool
    reasons: list


def _plain(value):
    """A JSON-friendly feature value: None for missing, int for whole numbers, names for categories."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    return str(value)


def new_row_features(store: SnapshotStore, columns: list, **txn):
    """Features of the new transaction (a one-row DataFrame in model column order)."""
    frame = store.model_frame(**txn)
    is_new = frame.pop("is_new").to_numpy()
    features = build_features(frame)
    return features.loc[frame.index[is_new], columns]


def score_features(bundle: Bundle, X) -> Score:
    booster = bundle.booster
    raw = float(booster.predict(X, raw_score=True, **PREDICT_PARAMS)[0])
    prob = float(booster.predict(X, **PREDICT_PARAMS)[0])
    calibrated = float(bundle.calibrator.predict(np.array([prob]))[0])
    contrib = booster.predict(X, pred_contrib=True, **PREDICT_PARAMS)[0, :-1]  # last = base value

    order = np.argsort(-contrib, kind="stable")  # ties keep model column order
    reasons = []
    for i in order[:N_REASONS]:
        if contrib[i] <= 0:
            break
        col = bundle.columns[i]
        reasons.append({"feature": col, "value": _plain(X.iloc[0][col]),
                        "contribution": float(contrib[i])})
    return Score(raw_score=raw, probability_raw=prob, probability_calibrated=calibrated,
                 alert=bool(prob >= bundle.threshold), reasons=reasons)


def score_transaction(bundle: Bundle, store: SnapshotStore, *, card_key, trans_ts, amt,
                      category, merch_lat, merch_long) -> Score:
    X = new_row_features(store, bundle.columns, card_key=card_key, trans_ts=trans_ts, amt=amt,
                         category=category, merch_lat=merch_lat, merch_long=merch_long)
    return score_features(bundle, X)
