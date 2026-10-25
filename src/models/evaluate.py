"""Evaluation tools for the fraud models (Phase 4, step 4).

Design (decided 2026-10-25):
- Ranking: average precision (PR-AUC), threshold-free.
- Equal workload: a model is compared with the hand-made rule at the SAME number
  of alerts in the same period (the rule raises about 67 alerts per day),
  because an analyst team has a fixed capacity; a 0.5 threshold has no
  business meaning.
- Money: share of fraud dollars caught, not only the count of fraud cases.
- Uncertainty: 95% intervals from a bootstrap that resamples whole CARDS, not
  rows. A card's fraud forms one short episode, so rows are not independent and
  a row-level bootstrap would give intervals that are too narrow.
- All comparisons run on the validation period; the test period is used once.
All functions are pure (arrays in, numbers out) and unit-tested.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

RULE_AMOUNT = 136.33  # top-decile amount boundary found in Phase 3


def rule_baseline(features: pd.DataFrame) -> pd.Series:
    """Phase 3 rule: night (22:00-03:59) AND amount above $136.33 -> 1, else 0."""
    flagged = (features["is_night"] == 1) & (features["amt"] > RULE_AMOUNT)
    return flagged.astype("int8").rename("rule")


def _check_scores(score) -> np.ndarray:
    score = np.asarray(score, dtype="float64")
    if np.isnan(score).any():
        raise ValueError(f"{int(np.isnan(score).sum())} missing scores")
    return score


def pr_auc(y, score) -> float:
    """Average precision: area under the precision-recall curve."""
    return float(average_precision_score(np.asarray(y), _check_scores(score)))


def top_n_mask(score, n: int) -> np.ndarray:
    """Boolean mask of the n highest scores (ties broken by row order)."""
    score = _check_scores(score)
    if not 0 <= n <= len(score):
        raise ValueError(f"n must be between 0 and {len(score)}, got {n}")
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(-score, kind="stable")[:n]] = True
    return mask


def alert_metrics(y, flagged, amt) -> dict:
    """What a set of alerts achieves: cases and dollars of fraud caught."""
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    amt = np.asarray(amt, dtype="float64")
    caught = y & flagged
    n_alerts, n_fraud = int(flagged.sum()), int(y.sum())
    fraud_dollars = amt[y].sum()
    return {
        "alerts": n_alerts,
        "fraud_caught": int(caught.sum()),
        "recall": caught.sum() / n_fraud if n_fraud else np.nan,
        "precision": caught.sum() / n_alerts if n_alerts else np.nan,
        "dollars_caught": float(amt[caught].sum()),
        "dollar_recall": amt[caught].sum() / fraud_dollars if fraud_dollars > 0 else np.nan,
    }


def card_bootstrap(cards, stat_fn, n_boot: int = 500, seed: int = 42,
                   level: float = 0.95) -> dict:
    """Bootstrap a statistic by resampling whole cards with replacement.

    stat_fn receives an array of row positions (rows of a card drawn twice
    appear twice) and returns a number. Returns the full-data estimate, a
    percentile interval and the bootstrap samples.
    """
    codes, uniques = pd.factorize(np.asarray(cards))
    if (codes < 0).any():
        raise ValueError("missing card keys")
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=len(uniques))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    blocks = [order[starts[c]:starts[c] + counts[c]] for c in range(len(uniques))]

    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.integers(0, len(uniques), size=len(uniques))
        samples[b] = stat_fn(np.concatenate([blocks[c] for c in drawn]))

    tail = 100 * (1 - level) / 2
    low, high = np.nanpercentile(samples, [tail, 100 - tail])
    return {
        "estimate": float(stat_fn(np.arange(len(codes)))),
        "low": float(low),
        "high": float(high),
        "samples": samples,
    }